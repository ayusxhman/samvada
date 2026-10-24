import torch
import torch.nn as nn
import torch.nn.functional as F

class WordBiGRU(nn.Module):
    """
    Bidirectional GRU with self-attention for word-level sign language recognition.
    Excellent generalization on few-shot landmark sequence datasets.
    """
    def __init__(self, num_classes: int = 2000, input_dim: int = 570, hidden_dim: int = 256, 
                 num_layers: int = 4, dropout: float = 0.3, d_model: int = None,
                 nhead: int = None, dim_feedforward: int = None, **kwargs):
        super(WordBiGRU, self).__init__()
        
        actual_hidden_dim = d_model if d_model is not None else hidden_dim
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, actual_hidden_dim),
            nn.LayerNorm(actual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=actual_hidden_dim,
            hidden_size=actual_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-Attention layer
        self.attention = nn.Sequential(
            nn.Linear(actual_hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(actual_hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(actual_hidden_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Input shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x) # -> [batch_size, seq_len, hidden_dim]
        
        gru_out, _ = self.gru(x) # -> [batch_size, seq_len, hidden_dim * 2]
        
        # Compute self-attention weights
        attn_weights = self.attention(gru_out) # -> [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Dynamic context pooling over temporal dimension
        context = torch.sum(attn_weights * gru_out, dim=1) # -> [batch_size, hidden_dim * 2]
        
        # Classify
        logits = self.fc(context) # -> [batch_size, num_classes]
        return logits

# Alias for backward compatibility
WordTransformer = WordBiGRU


