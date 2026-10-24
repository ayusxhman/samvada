import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent / "tasks_models"
HAND_MODEL_PATH = str(TASKS_DIR / "hand_landmarker.task")
POSE_MODEL_PATH = str(TASKS_DIR / "pose_landmarker.task")

class LandmarkExtractor:
    def __init__(self):
        # MediaPipe loading takes 2-3s. We lazily initialize these on first inference.
        self.hand_landmarker = None
        self.pose_landmarker = None

        # 8 Upper Body Pose Indices (matching dataset contract)
        self.pose_indices = [0, 12, 13, 14, 15, 16, 23, 24]
        
    def _initialize_models(self):
        if self.hand_landmarker is not None:
            return
            
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    def extract_landmarks_from_frame(self, frame) -> np.ndarray:
        """
        Processes a frame using MediaPipe Tasks Vision API (Hand + Pose).
        Returns a 190-dim landmark feature array matching existing contract.
        """
        self._initialize_models()
        import mediapipe as mp
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Execute Hand & Pose Landmarkers
        hand_result = self.hand_landmarker.detect(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)

        pose_feats = np.zeros((8, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        face_feats = np.zeros((20, 2), dtype=np.float32)

        # Extract Pose
        shoulder_mid = np.zeros(3, dtype=np.float32)
        shoulder_width = 1.0

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            lm = pose_result.pose_landmarks[0]
            left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
            right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width < 1e-5:
                shoulder_width = 1.0

            for i, idx in enumerate(self.pose_indices):
                if idx < len(lm):
                    pose_feats[i] = [lm[idx].x, lm[idx].y, lm[idx].z]

        # Extract Hands with handedness distinction
        if hand_result.hand_landmarks and hand_result.handedness:
            for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower()
                target_arr = left_hand if label == "left" else right_hand
                for i, lm in enumerate(hand_lms):
                    target_arr[i] = [lm.x, lm.y, lm.z]

        # Apply Torso-Centered Spatial Normalization
        def normalize_3d(arr):
            mask = np.any(arr != 0, axis=1)
            arr[mask] = (arr[mask] - shoulder_mid) / shoulder_width
            return arr

        pose_feats = normalize_3d(pose_feats)
        left_hand = normalize_3d(left_hand)
        right_hand = normalize_3d(right_hand)

        # Zero out non-nose pose and face to preserve dataset alignment
        pose_feats[1:] = 0.0

        features = np.concatenate([
            pose_feats.flatten(),
            left_hand.flatten(),
            right_hand.flatten(),
            face_feats.flatten()
        ])
        return features

    def compute_kinematics(self, sequence: np.ndarray) -> np.ndarray:
        T, D = sequence.shape
        if T < 2:
            return np.concatenate([sequence, np.zeros_like(sequence), np.zeros_like(sequence)], axis=1)
        velocity = np.zeros_like(sequence)
        velocity[1:] = sequence[1:] - sequence[:-1]
        acceleration = np.zeros_like(sequence)
        acceleration[2:] = velocity[2:] - velocity[1:-1]
        return np.concatenate([sequence, velocity, acceleration], axis=1)

    def close(self):
        self.hand_landmarker.close()
        self.pose_landmarker.close()

landmark_extractor = LandmarkExtractor()
