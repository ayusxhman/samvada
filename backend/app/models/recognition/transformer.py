import torch
import torch.nn as nn
from backend.app.models.recognition.base import BaseSignRecognitionModel

class TransformerRecognitionModel(BaseSignRecognitionModel):
    def __init__(self, input_dim: int = 630, d_model: int = 256, nhead: int = 4, num_layers: int = 3, num_classes: int = 2000, dropout: float = 0.2):
        super().__init__(input_dim, num_classes)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4:
            B, W, T, D = x.shape
            x = x.view(B, W * T, D)
        
        B, L, D = x.shape
        emb = self.embedding(x) + self.pos_encoder[:, :L, :]
        out = self.transformer_encoder(emb)
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)
