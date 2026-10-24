import torch
import torch.nn as nn
import timm

class FingerspellCNN(nn.Module):
    def __init__(self, num_classes: int = 29, pretrained: bool = False):
        super(FingerspellCNN, self).__init__()
        # Load backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        # Define classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, 3, 224, 224)
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
