import torch
import torch.nn as nn
from backend.app.models.recognition.base import BaseSignRecognitionModel

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, groups=d_model), # Depthwise
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1), # Pointwise
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN 1
        x = x + 0.5 * self.ffn1(self.norm1(x))
        # Self-Attention
        attn_out, _ = self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x))
        x = x + attn_out
        # Conv
        c_in = self.norm3(x).transpose(1, 2)
        c_out = self.conv(c_in).transpose(1, 2)
        x = x + c_out
        # FFN 2
        x = x + 0.5 * self.ffn2(self.norm4(x))
        return x

class ConformerRecognitionModel(BaseSignRecognitionModel):
    def __init__(self, input_dim: int = 630, d_model: int = 256, nhead: int = 4, num_blocks: int = 2, num_classes: int = 2000, dropout: float = 0.2):
        super().__init__(input_dim, num_classes)
        self.proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model=d_model, nhead=nhead, dropout=dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4:
            B, W, T, D = x.shape
            x = x.view(B, W * T, D)

        h = self.proj(x)
        for block in self.blocks:
            h = block(h)

        pooled = torch.mean(h, dim=1)
        return self.classifier(pooled)
