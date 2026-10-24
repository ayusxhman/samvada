import torch
import torch.nn as nn
from backend.app.models.recognition.base import BaseSignRecognitionModel

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        return torch.relu(out + res)

class TCNRecognitionModel(BaseSignRecognitionModel):
    def __init__(self, input_dim: int = 630, num_channels: list = [256, 256, 256], num_classes: int = 2000, dropout: float = 0.2):
        super().__init__(input_dim, num_classes)
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=3, dilation=dilation, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if x.dim() == 4:
            B, W, T, D = x.shape
            x = x.view(B, W * T, D)
        
        # (Batch, Time, Dim) -> (Batch, Dim, Time)
        x_trans = x.transpose(1, 2)
        out = self.tcn(x_trans)
        pooled = torch.mean(out, dim=2)
        return self.classifier(pooled)
