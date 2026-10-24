import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class SentenceTransformer(nn.Module):
    """
    Continuous Sign Language Recognition (CSLR) Model.
    Processes variable-length sequence of landmark frames and outputs
    per-frame probabilities across the vocabulary for CTC decoding.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 570,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 1D Convolution for local temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expected: [batch_size, seq_len, input_dim]
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        
        # Transpose for Transformer (batch_first=True): [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # Positional Encoding & Transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Output Logits for CTC Loss (No Softmax here, raw logits)
        # Shape: [batch_size, seq_len, num_classes]
        logits = self.fc_out(x)
        return logits
