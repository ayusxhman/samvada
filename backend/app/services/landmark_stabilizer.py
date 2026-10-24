import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

class AdaptiveOneEuroFilter:
    """
    Velocity-adaptive low-pass 1-Euro filter for 3D landmark trajectory smoothing.
    Dynamically scales cutoff frequency based on landmark velocity:
    - Low velocity (precision static poses): heavy filtering (min_cutoff) -> zero jitter.
    - High velocity (rapid motion sweeps): high cutoff -> zero phase lag.
    """
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def filter(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        x_arr = np.array(x, dtype=np.float32)
        if self.x_prev is None or self.t_prev is None or self.x_prev.shape != x_arr.shape:
            self.x_prev = x_arr.copy()
            self.dx_prev = np.zeros_like(x_arr)
            self.t_prev = timestamp
            return self.x_prev

        dt = max(timestamp - self.t_prev, 1e-4)
        self.t_prev = timestamp

        # Derivative calculation
        dx = (x_arr - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Adaptive cutoff scaling based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(dt, cutoff)
        x_hat = a * x_arr + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> np.ndarray:
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class LandmarkStabilizer:
    """
    Independent Landmark Stabilization Layer.
    Performs:
    1. Torso-Centered Spatial Normalization
    2. Adaptive 1-Euro Velocity-Scaled Filtering
    3. Gram-Schmidt Palm Normal Vector Estimation
    4. Kinematic Velocity & Acceleration Tracking
    """
    def __init__(self):
        self.left_hand_filter = AdaptiveOneEuroFilter(min_cutoff=0.5, beta=0.005)
        self.right_hand_filter = AdaptiveOneEuroFilter(min_cutoff=0.5, beta=0.005)
        self.pose_filter = AdaptiveOneEuroFilter(min_cutoff=0.5, beta=0.005)
        
        # Last calculated palm normals
        self.last_left_palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.last_right_palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def compute_palm_normal(self, hand_landmarks: np.ndarray) -> np.ndarray:
        """
        Computes 3D Palm Normal Unit Vector using Gram-Schmidt Orthogonalization
        over Wrist (0), Index MCP (5), and Pinky MCP (17).
        """
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)

        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17]

        u = index_mcp - wrist
        v = pinky_mcp - wrist

        normal = np.cross(u, v)
        norm = np.linalg.norm(normal)

        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (normal / norm).astype(np.float32)

    def normalize_torso(self, pose_landmarks: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Translates all 3D points relative to torso (shoulder midpoint)
        and scales by shoulder width.
        Returns (norm_pose, norm_lh, norm_rh, shoulder_width).
        """
        if pose_landmarks is not None and len(pose_landmarks) >= 13:
            left_shoulder = pose_landmarks[11] if len(pose_landmarks) > 11 else np.zeros(3)
            right_shoulder = pose_landmarks[12] if len(pose_landmarks) > 12 else np.zeros(3)
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width < 1e-4:
                shoulder_width = 1.0
        else:
            shoulder_mid = np.zeros(3, dtype=np.float32)
            shoulder_width = 1.0

        def transform(arr):
            if arr is None or len(arr) == 0:
                return np.zeros_like(arr) if arr is not None else np.zeros((0, 3), dtype=np.float32)
            mask = np.any(arr != 0, axis=1)
            res = arr.copy()
            res[mask] = (arr[mask] - shoulder_mid) / shoulder_width
            return res

        norm_pose = transform(pose_landmarks)
        norm_lh = transform(left_hand)
        norm_rh = transform(right_hand)
        return norm_pose, norm_lh, norm_rh, shoulder_width

    def stabilize(
        self,
        raw_left_hand: np.ndarray,
        raw_right_hand: np.ndarray,
        raw_pose: np.ndarray,
        timestamp: float
    ) -> dict:
        """
        Main stabilization entry point. Takes raw 3D landmark arrays and timestamp,
        applies filtering, spatial normalization, and palm normal calculation.
        """
        # 1. Torso Normalization
        norm_pose, norm_lh, norm_rh, s_width = self.normalize_torso(raw_pose, raw_left_hand, raw_right_hand)

        # 2. Adaptive 1-Euro Low-Pass Filtering
        stab_lh = self.left_hand_filter.filter(norm_lh, timestamp) if len(norm_lh) > 0 and np.any(norm_lh != 0) else norm_lh
        stab_rh = self.right_hand_filter.filter(norm_rh, timestamp) if len(norm_rh) > 0 and np.any(norm_rh != 0) else norm_rh
        stab_pose = self.pose_filter.filter(norm_pose, timestamp) if len(norm_pose) > 0 and np.any(norm_pose != 0) else norm_pose

        # 3. Palm Normal Vectors
        lh_normal = self.compute_palm_normal(stab_lh)
        rh_normal = self.compute_palm_normal(stab_rh)

        self.last_left_palm_normal = lh_normal
        self.last_right_palm_normal = rh_normal

        return {
            "left_hand": stab_lh,
            "right_hand": stab_rh,
            "pose": stab_pose,
            "left_palm_normal": lh_normal.tolist(),
            "right_palm_normal": rh_normal.tolist(),
            "shoulder_width": s_width,
            "timestamp": timestamp
        }

    def reset(self):
        self.left_hand_filter.reset()
        self.right_hand_filter.reset()
        self.pose_filter.reset()

landmark_stabilizer = LandmarkStabilizer()
