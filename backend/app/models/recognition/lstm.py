import torch
import torch.nn as nn
from backend.app.models.recognition.base import BaseSignRecognitionModel

class LSTMRecognitionModel(BaseSignRecognitionModel):
    def __init__(self, input_dim: int = 630, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 2000, dropout: float = 0.3):
        super().__init__(input_dim, num_classes)
        self.fc_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Handle 4D (Batch, Windows, Time, Dim) or 3D (Batch, Time, Dim)
        if x.dim() == 4:
            B, W, T, D = x.shape
            x = x.view(B, W * T, D)
        
        proj = torch.relu(self.fc_proj(x))
        out, _ = self.lstm(proj)
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)
