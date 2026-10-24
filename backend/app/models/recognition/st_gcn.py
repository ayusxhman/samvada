import torch
import torch.nn as nn
from backend.app.models.recognition.base import BaseSignRecognitionModel

class STGCNRecognitionModel(BaseSignRecognitionModel):
    """
    Spatial-Temporal Graph Convolutional Network (ST-GCN) Candidate Architecture.
    """
    def __init__(self, input_dim: int = 630, hidden_dim: int = 256, num_classes: int = 2000, dropout: float = 0.3):
        super().__init__(input_dim, num_classes)
        self.spatial_gcn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4:
            B, W, T, D = x.shape
            x = x.view(B * W * T, D)
            feat = self.spatial_gcn(x)
            feat = feat.view(B, W * T, -1).transpose(1, 2) # (B, Hidden, Length)
        else:
            B, T, D = x.shape
            x = x.view(B * T, D)
            feat = self.spatial_gcn(x)
            feat = feat.view(B, T, -1).transpose(1, 2)

        t_out = self.temporal_conv(feat)
        pooled = torch.mean(t_out, dim=2)
        return self.classifier(pooled)
