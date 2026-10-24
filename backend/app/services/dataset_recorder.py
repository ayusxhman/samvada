import os
import json
import time
import uuid
import logging
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DATASET_ROOT = Path("dataset/recordings")
DATASET_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class FrameSample:
    frame_index: int
    timestamp: float
    pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_palm_normal: List[float]
    right_palm_normal: List[float]
    struct_features: Dict[str, Any]
    flat_tensor: np.ndarray
    detector_confidence: float
    has_lh: bool
    has_rh: bool


class DatasetRecorderSession:
    """
    Manages a single recording session for multimodal sign language datasets.
    """
    def __init__(self, sentence_label: str, gloss_labels: List[str], signer_id: str = "signer_01"):
        self.session_id = f"sample_{uuid.uuid4().hex[:8]}"
        self.sentence_label = sentence_label
        self.gloss_labels = gloss_labels
        self.signer_id = signer_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.status = "recording"  # "recording", "paused", "stopped", "discarded"
        self.frames: List[FrameSample] = []
        self.quality_warnings: List[str] = []

    def add_frame(self, frame_sample: FrameSample):
        if self.status != "recording":
            return
        self.frames.append(frame_sample)

    def pause(self):
        if self.status == "recording":
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self.status = "recording"

    def stop_and_save(self) -> Dict[str, Any]:
        self.status = "stopped"
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        sample_dir = DATASET_ROOT / self.session_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        num_frames = len(self.frames)
        if num_frames == 0:
            return {"error": "No frames recorded", "session_id": self.session_id}

        # Quality Control Analysis
        lh_missing_cnt = sum(1 for f in self.frames if not f.has_lh)
        rh_missing_cnt = sum(1 for f in self.frames if not f.has_rh)
        missing_rate = (lh_missing_cnt + rh_missing_cnt) / (2.0 * num_frames)

        quality_score = max(0.0, 100.0 - (missing_rate * 50.0))
        warnings = []
        if missing_rate > 0.3:
            warnings.append("High missing landmark rate detected (>30%)")
        if num_frames < 10:
            warnings.append("Very short recording duration (<10 frames)")

        # 1. Landmarks JSON
        landmarks_data = []
        for f in self.frames:
            landmarks_data.append({
                "frame_index": f.frame_index,
                "timestamp": f.timestamp,
                "has_lh": f.has_lh,
                "has_rh": f.has_rh,
                "left_hand": f.left_hand.tolist(),
                "right_hand": f.right_hand.tolist(),
                "pose": f.pose.tolist(),
                "left_palm_normal": f.left_palm_normal,
                "right_palm_normal": f.right_palm_normal,
                "confidence": f.detector_confidence
            })

        with open(sample_dir / "landmarks.json", "w") as fp:
            json.dump(landmarks_data, fp, indent=2)

        # 2. Tensors NPY & Structured NPZ
        flat_tensors_arr = np.array([f.flat_tensor for f in self.frames], dtype=np.float32)
        np.save(sample_dir / "tensors.npy", flat_tensors_arr)

        np.savez_compressed(
            sample_dir / "structured_features.npz",
            lh=np.array([f.left_hand for f in self.frames], dtype=np.float32),
            rh=np.array([f.right_hand for f in self.frames], dtype=np.float32),
            pose=np.array([f.pose for f in self.frames], dtype=np.float32)
        )

        # 3. Annotations & Metadata JSON
        annotations = {
            "session_id": self.session_id,
            "sentence_label": self.sentence_label,
            "gloss_labels": self.gloss_labels,
            "signer_id": self.signer_id,
            "num_frames": num_frames,
            "duration_sec": round(duration, 3),
            "fps": round(num_frames / duration if duration > 0 else 0.0, 2),
            "quality_score": round(quality_score, 1),
            "warnings": warnings,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(sample_dir / "annotations.json", "w") as fp:
            json.dump(annotations, fp, indent=2)

        # Update dataset manifest
        manifest_path = DATASET_ROOT.parent / "dataset_manifest.json"
        manifest = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as fp:
                    manifest = json.load(fp)
            except Exception:
                manifest = []
        manifest.append(annotations)

        with open(manifest_path, "w") as fp:
            json.dump(manifest, fp, indent=2)

        return annotations


class DatasetRecorderService:
    def __init__(self):
        self.active_sessions: Dict[str, DatasetRecorderSession] = {}

    def start_session(self, sentence_label: str, gloss_labels: List[str], signer_id: str = "signer_01") -> str:
        session = DatasetRecorderSession(sentence_label, gloss_labels, signer_id)
        self.active_sessions[session.session_id] = session
        return session.session_id

    def record_frame(self, session_id: str, frame_sample: FrameSample):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].add_frame(frame_sample)

    def pause_session(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].pause()

    def resume_session(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].resume()

    def stop_session(self, session_id: str) -> Dict[str, Any]:
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            return session.stop_and_save()
        return {"error": "Session not found"}

    def discard_session(self, session_id: str):
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

dataset_recorder = DatasetRecorderService()
