import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
import timm
from tqdm import tqdm
import logging
from backend.app.config import DATASET_DIR, FINGERSPELLING_MODEL_PATH
from backend.app.models.fingerspell_model import FingerspellCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_fingerspell():
    use_cuda = False
    if torch.cuda.is_available():
        try:
            linear = nn.Linear(129, 256).cuda()
            test_x = torch.randn(32, 30, 129).cuda()
            _ = linear(test_x)
            torch.cuda.synchronize()
            use_cuda = True
        except Exception:
            logger.warning("CUDA is available but throws error. Falling back to CPU.")
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Training Fingerspell CNN on device: {device}")
    
    # 1. Dataset location
    train_dir = DATASET_DIR / "asl_alphabet" / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
    if not train_dir.exists():
        logger.error(f"Training directory not found at {train_dir}")
        return
        
    # 2. Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Dataset
    dataset = datasets.ImageFolder(root=str(train_dir), transform=transform)
    num_classes = len(dataset.classes)
    logger.info(f"Dataset classes: {dataset.classes} (Count: {num_classes})")
    
    # Split: 80% Train, 20% Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Enable cuDNN benchmark for NVIDIA RTX GPU maximum throughput
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Dataloaders - 4 workers to feed GPU faster, pin_memory=False to prevent Windows deadlocks
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=False)
    
    # 4. Model setup
    model = FingerspellCNN(num_classes=num_classes, pretrained=True)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler() # Mixed precision training
    
    # 5. Training loop
    best_loss = float('inf')
    epochs = 15
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)
            
        epoch_loss = train_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1} finished. Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        scheduler.step()
        
        # Save best model + early stopping
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(FINGERSPELLING_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), FINGERSPELLING_MODEL_PATH)
            logger.info(f"Saved new best model checkpoint to {FINGERSPELLING_MODEL_PATH}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    train_fingerspell()
