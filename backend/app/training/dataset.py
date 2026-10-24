import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset, DataLoader

from backend.app.services.dataset_preprocessor import DatasetPreprocessor, SequenceConfig, AugmentationConfig, DATASET_ROOT

logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """
    Native PyTorch Dataset for Sign Language Recognition.
    Loads tensors, applies windowing/padding, attention mask generation, and optional data augmentations.
    """
    def __init__(
        self,
        sample_ids: List[str],
        preprocessor: Optional[DatasetPreprocessor] = None,
        transform: bool = False,
        normalize: bool = True
    ):
        self.sample_ids = sample_ids
        self.preprocessor = preprocessor or DatasetPreprocessor()
        self.transform = transform
        self.normalize = normalize
        
        # Load normalization stats if requested
        self.norm_mean = None
        self.norm_std = None
        norm_path = DATASET_ROOT.parent / "normalization_stats.json"
        if normalize and norm_path.exists():
            with open(norm_path, "r") as fp:
                stats = json.load(fp)
                if stats.get("mean"):
                    self.norm_mean = np.array(stats["mean"], dtype=np.float32)
                    self.norm_std = np.array(stats["std"], dtype=np.float32)

        # Index valid samples and annotations
        self.data_samples: List[Dict[str, Any]] = []
        for sid in self.sample_ids:
            is_valid, _ = self.preprocessor.validate_sample(sid)
            if is_valid:
                ann_path = DATASET_ROOT / sid / "annotations.json"
                with open(ann_path, "r") as fp:
                    ann = json.load(fp)
                self.data_samples.append(ann)

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data_samples[idx]
        session_id = sample["session_id"]
        tensor_path = DATASET_ROOT / session_id / "tensors.npy"

        raw_tensor = np.load(tensor_path)  # (T, D)

        # Apply Standard Scaling Normalization
        if self.normalize and self.norm_mean is not None and self.norm_std is not None:
            raw_tensor = (raw_tensor - self.norm_mean) / self.norm_std

        # Apply Augmentations
        if self.transform:
            raw_tensor = self.preprocessor.apply_augmentation(raw_tensor)

        # Generate Sliding Windows and Attention Masks
        windows, masks = self.preprocessor.generate_sliding_windows(raw_tensor)

        return {
            "session_id": session_id,
            "tensor": torch.tensor(windows, dtype=torch.float32),      # (Num_Windows, Window_Len, D)
            "mask": torch.tensor(masks, dtype=torch.float32),          # (Num_Windows, Window_Len)
            "sequence_length": torch.tensor(len(raw_tensor), dtype=torch.long)
        }


def sign_language_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom PyTorch Collate Function to handle mixed sequence lengths and window batches seamlessly.
    """
    session_ids = [b["session_id"] for b in batch]
    seq_lengths = torch.stack([b["sequence_length"] for b in batch])

    # Pad window counts across batch
    max_windows = max(b["tensor"].size(0) for b in batch)
    feature_dim = batch[0]["tensor"].size(2)
    window_len = batch[0]["tensor"].size(1)

    padded_tensors = []
    padded_masks = []

    for b in batch:
        t = b["tensor"]  # (W, Win_Len, D)
        m = b["mask"]    # (W, Win_Len)
        w_cnt = t.size(0)

        if w_cnt < max_windows:
            pad_w = max_windows - w_cnt
            t_pad = torch.zeros((pad_w, window_len, feature_dim), dtype=torch.float32)
            m_pad = torch.zeros((pad_w, window_len), dtype=torch.float32)
            t = torch.cat([t, t_pad], dim=0)
            m = torch.cat([m, m_pad], dim=0)

        padded_tensors.append(t)
        padded_masks.append(m)

    return {
        "session_ids": session_ids,
        "tensors": torch.stack(padded_tensors),  # (Batch_Size, Max_Windows, Window_Len, Feature_Dim)
        "masks": torch.stack(padded_masks),      # (Batch_Size, Max_Windows, Window_Len)
        "sequence_lengths": seq_lengths
    }


def create_dataloader(
    sample_ids: List[str],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: bool = False
) -> DataLoader:
    dataset = SignLanguageDataset(sample_ids=sample_ids, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sign_language_collate_fn,
        pin_memory=True
    )
