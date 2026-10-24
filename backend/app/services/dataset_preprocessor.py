import os
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

DATASET_ROOT = Path("dataset/recordings")
SPLITS_ROOT = Path("dataset/splits")
SPLITS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class AugmentationConfig:
    enable_rotation: bool = False
    enable_translation: bool = False
    enable_scaling: bool = False
    enable_temporal_stretch: bool = False
    enable_frame_drop: bool = False
    enable_gaussian_noise: bool = False
    rotation_deg_max: float = 10.0
    translation_max: float = 0.05
    scale_min_max: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.005
    frame_drop_prob: float = 0.05


@dataclass
class SequenceConfig:
    window_length: int = 30
    stride: int = 2
    overlap_pct: float = 0.93
    padding: str = "zero"  # "zero", "edge"
    support_variable_length: bool = True


class DatasetPreprocessor:
    """
    Production Dataset Preparation Subsystem.
    Performs:
    1. Dataset Indexing & Corrupted Sample Filtering
    2. Sequence Sliding Window & Padding Generation
    3. Dataset-wide Mean/Std Standard Scaling Normalization
    4. Configurable Data Augmentation Engine
    5. Signer-Independent & Stratified Dataset Splitting
    """
    def __init__(self, sequence_config: Optional[SequenceConfig] = None, aug_config: Optional[AugmentationConfig] = None):
        self.seq_config = sequence_config or SequenceConfig()
        self.aug_config = aug_config or AugmentationConfig()

    def index_dataset(self) -> List[Dict[str, Any]]:
        manifest_path = DATASET_ROOT.parent / "dataset_manifest.json"
        if not manifest_path.exists():
            return []
        with open(manifest_path, "r") as fp:
            return json.load(fp)

    def validate_sample(self, sample_id: str) -> Tuple[bool, List[str]]:
        sample_dir = DATASET_ROOT / sample_id
        errors = []

        if not sample_dir.exists():
            return False, [f"Directory {sample_dir} does not exist"]

        required_files = ["landmarks.json", "tensors.npy", "structured_features.npz", "annotations.json"]
        for f in required_files:
            if not (sample_dir / f).exists():
                errors.append(f"Missing file: {f}")

        if errors:
            return False, errors

        try:
            tensors = np.load(sample_dir / "tensors.npy")
            if len(tensors) == 0:
                errors.append("Empty tensors.npy file")
            if np.isnan(tensors).any() or np.isinf(tensors).any():
                errors.append("Corrupted tensors.npy containing NaNs or Infs")
        except Exception as e:
            errors.append(f"Failed to load tensors.npy: {str(e)}")

        return (len(errors) == 0), errors

    def compute_dataset_normalization(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_tensors = []
        for s in samples:
            sample_id = s["session_id"]
            tensor_path = DATASET_ROOT / sample_id / "tensors.npy"
            if tensor_path.exists():
                arr = np.load(tensor_path)
                if len(arr) > 0:
                    all_tensors.append(arr)

        if not all_tensors:
            return {"mean": [], "std": [], "num_samples": 0}

        concat = np.concatenate(all_tensors, axis=0)  # (Total_Frames, Feature_Dim)
        mean = np.mean(concat, axis=0)
        std = np.std(concat, axis=0) + 1e-8

        norm_stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_dim": int(concat.shape[1]),
            "total_frames_indexed": int(concat.shape[0]),
            "num_samples": len(samples)
        }

        with open(DATASET_ROOT.parent / "normalization_stats.json", "w") as fp:
            json.dump(norm_stats, fp, indent=2)

        return norm_stats

    def apply_augmentation(self, tensor: np.ndarray) -> np.ndarray:
        aug_tensor = tensor.copy()

        if self.aug_config.enable_gaussian_noise:
            noise = np.random.normal(0.0, self.aug_config.noise_std, size=aug_tensor.shape).astype(np.float32)
            aug_tensor += noise

        if self.aug_config.enable_scaling:
            scale = np.random.uniform(*self.aug_config.scale_min_max)
            aug_tensor *= scale

        if self.aug_config.enable_frame_drop and len(aug_tensor) > 10:
            mask = np.random.rand(len(aug_tensor)) > self.aug_config.frame_drop_prob
            if np.any(mask):
                aug_tensor = aug_tensor[mask]

        return aug_tensor

    def generate_sliding_windows(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, D = tensor.shape
        W = self.seq_config.window_length
        S = self.seq_config.stride

        if T < W:
            pad_len = W - T
            if self.seq_config.padding == "zero":
                pad = np.zeros((pad_len, D), dtype=np.float32)
            else:
                pad = np.tile(tensor[-1:], (pad_len, 1))
            padded_tensor = np.concatenate([tensor, pad], axis=0)
            mask = np.concatenate([np.ones(T, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
            return np.expand_dims(padded_tensor, 0), np.expand_dims(mask, 0)

        windows = []
        masks = []
        for start in range(0, T - W + 1, S):
            end = start + W
            windows.append(tensor[start:end])
            masks.append(np.ones(W, dtype=np.float32))

        return np.array(windows, dtype=np.float32), np.array(masks, dtype=np.float32)

    def create_splits(self, samples: List[Dict[str, Any]], split_type: str = "signer_independent", train_pct: float = 0.8) -> Dict[str, List[str]]:
        valid_samples = [s["session_id"] for s in samples if self.validate_sample(s["session_id"])[0]]

        if split_type == "signer_independent":
            signer_map = {}
            for s in samples:
                sid = s["session_id"]
                if sid in valid_samples:
                    signer = s.get("signer_id", "signer_01")
                    signer_map.setdefault(signer, []).append(sid)

            signers = list(signer_map.keys())
            train_signers = signers[:max(1, int(len(signers) * train_pct))]
            val_test_signers = signers[len(train_signers):]

            train_ids = [sid for signer in train_signers for sid in signer_map[signer]]
            val_ids = [sid for signer in val_test_signers for sid in signer_map[signer]]
            test_ids = val_ids.copy()
        else:
            np.random.shuffle(valid_samples)
            n = len(valid_samples)
            n_train = int(n * train_pct)
            n_val = int(n * 0.1)
            train_ids = valid_samples[:n_train]
            val_ids = valid_samples[n_train:n_train + n_val]
            test_ids = valid_samples[n_train + n_val:]

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "split_type": split_type
        }

        with open(SPLITS_ROOT / "dataset_splits.json", "w") as fp:
            json.dump(splits, fp, indent=2)

        return splits

dataset_preprocessor = DatasetPreprocessor()
