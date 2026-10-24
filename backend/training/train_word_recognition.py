"""
Word-level Sign Recognition Transformer Training

Key changes from v1:
- Filter to classes with >= 5 samples (2926 classes, 22395 samples)
- Data augmentation: temporal jitter, Gaussian noise, speed perturbation
- Label smoothing for better generalization with few samples
- Learning rate warmup then cosine decay
- Gradient clipping for stability
"""
import os
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
from collections import Counter
from backend.app.config import DATASET_DIR, WORD_MODEL_PATH, WORD_LABELS_PATH
from backend.app.models.word_model import WordTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_SAMPLES_PER_CLASS = 8


def get_random_rotation_matrix(max_angle_degrees=15):
    """Generates a random 3D rotation matrix for landmark keypoints."""
    angles = np.random.uniform(-max_angle_degrees, max_angle_degrees, 3)
    rads = np.radians(angles)
    cx, cy, cz = np.cos(rads)
    sx, sy, sz = np.sin(rads)
    
    # 3D rotation matrix (pitch, yaw, roll)
    R = np.array([
        [cy*cz, -cy*sz, sy],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy],
        [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy]
    ], dtype=np.float32)
    return R


class AugmentedLandmarkDataset(Dataset):
    """Landmark dataset with temporal and spatial augmentations for state-of-the-art generalisation."""
    
    def __init__(self, landmarks: np.ndarray, labels: list, augment: bool = True):
        self.landmarks = landmarks.astype(np.float32)
        self.labels = labels
        self.augment = augment
        self.seq_len = landmarks.shape[1]  # 30
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = self.landmarks[idx].copy()  # (30, 129)
        
        if self.augment:
            # 1. 3D Geometric Rotation & Translation (Moderate angles and small translation)
            if np.random.random() < 0.5:
                R = get_random_rotation_matrix(8)  # max 8 degrees
                shift = np.random.normal(0, 0.02, (1, 3)).astype(np.float32)  # max 0.02 shift
                
                # Rotate & shift nose
                seq[:, 0:3] = np.dot(seq[:, 0:3], R.T) + shift
                
                # Rotate & shift LHand if active
                lhand = seq[:, 3:66].reshape(-1, 21, 3)
                if not np.all(lhand == 0):
                    seq[:, 3:66] = (np.dot(lhand, R.T) + shift).reshape(-1, 63)
                    
                # Rotate & shift RHand if active
                rhand = seq[:, 66:129].reshape(-1, 21, 3)
                if not np.all(rhand == 0):
                    seq[:, 66:129] = (np.dot(rhand, R.T) + shift).reshape(-1, 63)
            
            # 2. Soft Hand Dropout (Spatial masking)
            if np.random.random() < 0.02:
                if np.random.random() < 0.5:
                    seq[:, 3:66] = 0
                else:
                    seq[:, 66:129] = 0
            
            # 3. Temporal Frame Masking (Temporal Cutout)
            if np.random.random() < 0.05:
                mask_len = np.random.randint(1, 3)
                start_idx = np.random.randint(0, self.seq_len - mask_len)
                seq[start_idx:start_idx+mask_len] = 0
 
            # 4. Light Gaussian noise on landmarks
            noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
            seq = seq + noise
            
            # 5. Temporal shift (real translation, no wrap-around): pad/truncate to avoid velocity spikes
            if np.random.random() < 0.3:
                shift_val = np.random.randint(-2, 3)
                if shift_val != 0:
                    if shift_val > 0:
                        seq = np.concatenate([np.zeros((shift_val, seq.shape[1]), dtype=np.float32), seq[:-shift_val]], axis=0)
                    else:
                        seq = np.concatenate([seq[-shift_val:], np.zeros((-shift_val, seq.shape[1]), dtype=np.float32)], axis=0)
            
            # 6. Random speed perturbation (stretch/compress time)
            if np.random.random() < 0.3:
                speed = np.random.uniform(0.8, 1.2)
                indices = np.linspace(0, self.seq_len - 1, int(self.seq_len * speed))
                indices = np.clip(indices, 0, self.seq_len - 1).astype(int)
                seq_aug = seq[indices]
                # Pad/truncate back to original length
                if len(seq_aug) >= self.seq_len:
                    seq = seq_aug[:self.seq_len]
                else:
                    pad = np.zeros((self.seq_len - len(seq_aug), seq.shape[1]), dtype=np.float32)
                    seq = np.concatenate([seq_aug, pad], axis=0)
            
            # 7. Random scale (simulate distance variation)
            if np.random.random() < 0.2:
                scale = np.random.uniform(0.9, 1.1)
                seq = seq * scale
                
        # --- Upgrade 129-dim to 190-dim ---
        # 129 format: Nose (3), LHand (63), RHand (63)
        # 190 format: Pose (24), LHand (63), RHand (63), Face (40)
        pose_nose = seq[:, 0:3]
        left_hand = seq[:, 3:66]
        right_hand = seq[:, 66:129]
        
        # Pad pose to 24 dims. We put nose in the first 3 dims (though it's supposed to be shoulder 11).
        # Since this is just for adapting old datasets, we zero pad the rest.
        pose_padded = np.zeros((self.seq_len, 24), dtype=np.float32)
        pose_padded[:, 0:3] = pose_nose
        face_padded = np.zeros((self.seq_len, 40), dtype=np.float32)
        
        seq_190 = np.concatenate([pose_padded, left_hand, right_hand, face_padded], axis=1)
        
        # --- Compute Kinematics (Velocity & Acceleration) to get 570 dims ---
        velocity = np.zeros_like(seq_190)
        velocity[1:] = seq_190[1:] - seq_190[:-1]
        
        acceleration = np.zeros_like(seq_190)
        acceleration[2:] = velocity[2:] - velocity[1:-1]
        
        seq_570 = np.concatenate([seq_190, velocity, acceleration], axis=1)

        return torch.tensor(seq_570, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def train_word_model():
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
    logger.info(f"Training Word Transformer on device: {device}")
    
    # 1. Load pickle data
    pkl_path = DATASET_DIR / "new_citizen_landmarks_190.pkl"
    if not pkl_path.exists():
        pkl_path = DATASET_DIR / "merged_landmarks_129.pkl"
    if not pkl_path.exists():
        logger.error("No landmark PKL file found. Training aborted.")
        return
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
        
    X = d["X"]       # numpy array shape (N, 30, 190)
    y = d["y"]       # list of string labels
    words = d["words"]  # list of unique classes
    
    logger.info(f"Raw data loaded: {X.shape[0]} samples, {len(words)} classes")
    
    # --- Rigorous Data Cleaning Block ---
    logger.info("Starting rigorous data cleaning to discard null, empty, dummy, or static samples...")
    valid_indices = []
    
    for i in tqdm(range(len(y)), desc="Cleaning data"):
        label = y[i]
        seq = X[i]
        
        # 1. Reject if label is empty, null, or dummy
        if not label or str(label).strip() == "" or str(label).lower() in ["null", "nan", "none", "dummy", "empty", "test"]:
            continue
            
        # 2. Reject if array has NaNs or Infs
        if np.isnan(seq).any() or np.isinf(seq).any():
            continue
            
        # 3. Reject if sequence is all zeros
        if np.all(seq == 0):
            continue
            
        # 4. Reject if sequence is completely static (frozen over time)
        # If std dev across time axis is extremely small, it means no movement
        if np.std(seq, axis=0).sum() < 1e-4:
            continue
            
        valid_indices.append(i)
        
    X = X[valid_indices]
    y = [y[i] for i in valid_indices]
    words = list(set(y))
    logger.info(f"Data cleaning completed. Retained {len(valid_indices)} fully clean and valid samples.")
    # ------------------------------------
    
    # 2. Filter to classes with enough samples
    counts = Counter(y)
    valid_classes = sorted([cls for cls, cnt in counts.items() if cnt >= MIN_SAMPLES_PER_CLASS])
    valid_set = set(valid_classes)
    
    # Filter data
    mask = [label in valid_set for label in y]
    X_filtered = X[mask]
    y_filtered = [label for label, m in zip(y, mask) if m]
    
    # Rebuild class list and mapping
    filtered_words = valid_classes
    word_to_id = {word: i for i, word in enumerate(filtered_words)}
    y_ids = np.array([word_to_id[label] for label in y_filtered])
    
    num_classes = len(filtered_words)
    logger.info(f"Filtered data: {len(y_ids)} samples, {num_classes} classes (min {MIN_SAMPLES_PER_CLASS} samples each)")
    
    # Save word labels manifest
    os.makedirs(os.path.dirname(WORD_LABELS_PATH), exist_ok=True)
    with open(WORD_LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(filtered_words, f, indent=4)
    logger.info(f"Saved word labels list to {WORD_LABELS_PATH}")
    
    # 3. Dataset and Loader setup
    full_dataset = AugmentedLandmarkDataset(X_filtered, y_ids, augment=True)
    
    # 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset_noaug = AugmentedLandmarkDataset(X_filtered, y_ids, augment=False)
    val_indices = val_dataset.indices
    val_dataset_actual = torch.utils.data.Subset(val_dataset_noaug, val_indices)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset_actual)}")
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset_actual, batch_size=64, shuffle=False, 
                            num_workers=num_workers, pin_memory=False)
    
    # 4. Model setup
    model = WordTransformer(num_classes=num_classes, d_model=256, nhead=4, 
                            num_layers=4, dim_feedforward=512, dropout=0.3)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Label smoothing helps with few-shot classes
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Warmup + cosine decay
    epochs = 200
    patience = 30
    total_steps = len(train_loader) * epochs  # max epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # 5. Training loop
    best_val_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for sequences, labels in loop:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type="cuda" if use_cuda else "cpu"):
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total, 
                             lr=optimizer.param_groups[0]['lr'])
            
        epoch_loss = train_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_top5_correct = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                with autocast(device_type="cuda" if use_cuda else "cpu"):
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * sequences.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, dim=1)
                val_top5_correct += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))
                
        epoch_val_loss = val_loss / len(val_dataset_actual)
        epoch_val_acc = 100.0 * val_correct / val_total
        epoch_val_top5 = 100.0 * val_top5_correct / val_total
        
        logger.info(f"Epoch {epoch+1} finished. "
                     f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
                     f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, "
                     f"Val Top-5: {epoch_val_top5:.2f}%")
        
        # Save based on best val accuracy (more meaningful than loss with label smoothing)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), WORD_MODEL_PATH)
            logger.info(f"Saved new best model checkpoint to {WORD_MODEL_PATH} (Val Acc: {epoch_val_acc:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_val_acc:.2f}%")
                break

if __name__ == "__main__":
    train_word_model()
