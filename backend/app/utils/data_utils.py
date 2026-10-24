import pickle
import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def load_pickle(file_path: Path) -> Any:
    """
    Safely loads a pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}")
        return None

def load_json(file_path: Path) -> Dict[str, Any]:
    """
    Safely loads a JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def normalize_sequence(landmarks: np.ndarray, target_len: int = 30) -> np.ndarray:
    """
    Pads or truncates landmark sequence length to target_len.
    """
    seq_len = landmarks.shape[0]
    if seq_len == target_len:
        return landmarks
    elif seq_len < target_len:
        # Zero-pad
        padding = np.zeros((target_len - seq_len, landmarks.shape[1]), dtype=landmarks.dtype)
        return np.concatenate([landmarks, padding], axis=0)
    else:
        # Interpolate/downsample
        indices = np.linspace(0, seq_len - 1, target_len, dtype=int)
        return landmarks[indices]
