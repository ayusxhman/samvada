import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSignRecognitionModel(nn.Module, ABC):
    """
    Common Base Interface for all Sign Language Recognition Models.
    Guarantees interchangeable evaluation across candidate architectures.
    """
    def __init__(self, input_dim: int = 630, num_classes: int = 2000):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:  x of shape (Batch, Windows, Time, Input_Dim) or (Batch, Time, Input_Dim)
                mask of shape (Batch, Windows, Time) or (Batch, Time)
        Output: logits of shape (Batch, Num_Classes)
        """
        pass

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
