import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class FeatureConfig:
    enable_coordinates: bool = True
    enable_angles: bool = True
    enable_palm_features: bool = True
    enable_bone_features: bool = True
    enable_finger_features: bool = True
    enable_motion_features: bool = True
    enable_body_features: bool = True


class FeatureExtractorEngine:
    """
    Modular, Configurable Feature Extraction Engine for Sign Language Recognition.
    Generates structured feature objects across 7 independent feature groups:
    1. Normalized 3D Coordinates (Hands + Pose)
    2. Joint Flexion / Extension Angles (MCP, PIP, DIP)
    3. Palm Features (Palm normal vector, Roll/Pitch/Yaw)
    4. Bone Features (Bone direction vectors & lengths)
    5. Finger Features (Finger curl, spread & thumb opposition)
    6. Kinematic Motion Features (Velocity & Acceleration)
    7. Body-Relative Features (Distance vectors to chest/shoulders)
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Connections for finger bone pairs: (parent_idx, child_idx)
        self.finger_bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
        ]
        
        # Triplets for joint angle calculation: (proximal, joint, distal)
        self.angle_triplets = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Thumb
            (0, 5, 6), (5, 6, 7), (6, 7, 8),       # Index
            (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Middle
            (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
            (0, 17, 18), (17, 18, 19), (18, 19, 20) # Pinky
        ]

        # Previous frame state for velocity and acceleration tracking
        self._prev_struct_lh: Optional[np.ndarray] = None
        self._prev_struct_rh: Optional[np.ndarray] = None
        self._prev_vel_lh: Optional[np.ndarray] = None
        self._prev_vel_rh: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None

    def reset_history(self):
        self._prev_struct_lh = None
        self._prev_struct_rh = None
        self._prev_vel_lh = None
        self._prev_vel_rh = None
        self._prev_time = None

    @staticmethod
    def _compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def compute_hand_angles(self, hand_pts: np.ndarray) -> np.ndarray:
        if hand_pts is None or len(hand_pts) < 21 or not np.any(hand_pts):
            return np.zeros(15, dtype=np.float32)

        angles = []
        for p, j, d in self.angle_triplets:
            v1 = hand_pts[p] - hand_pts[j]
            v2 = hand_pts[d] - hand_pts[j]
            angles.append(self._compute_angle(v1, v2))
        return np.array(angles, dtype=np.float32)

    def compute_palm_features(self, hand_pts: np.ndarray) -> np.ndarray:
        if hand_pts is None or len(hand_pts) < 21 or not np.any(hand_pts):
            return np.zeros(6, dtype=np.float32)

        wrist = hand_pts[0]
        index_mcp = hand_pts[5]
        pinky_mcp = hand_pts[17]

        u = index_mcp - wrist
        v = pinky_mcp - wrist

        normal = np.cross(u, v)
        norm = np.linalg.norm(normal)
        normal_unit = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        roll = np.arctan2(normal_unit[1], normal_unit[2])
        pitch = np.arctan2(-normal_unit[0], np.sqrt(normal_unit[1]**2 + normal_unit[2]**2))
        yaw = np.arctan2(u[1], u[0])

        return np.array([*normal_unit, roll, pitch, yaw], dtype=np.float32)

    def compute_bone_features(self, hand_pts: np.ndarray) -> np.ndarray:
        if hand_pts is None or len(hand_pts) < 21 or not np.any(hand_pts):
            return np.zeros(20 * 4, dtype=np.float32)

        bone_feats = []
        for parent, child in self.finger_bones:
            vec = hand_pts[child] - hand_pts[parent]
            length = np.linalg.norm(vec)
            unit_vec = vec / length if length > 1e-6 else np.zeros(3)
            bone_feats.extend([*unit_vec, length])
        return np.array(bone_feats, dtype=np.float32)

    def compute_finger_features(self, hand_pts: np.ndarray) -> np.ndarray:
        if hand_pts is None or len(hand_pts) < 21 or not np.any(hand_pts):
            return np.zeros(10, dtype=np.float32)

        wrist = hand_pts[0]
        tips = [4, 8, 12, 16, 20]
        mcps = [1, 5, 9, 13, 17]

        curls = []
        for tip, mcp in zip(tips, mcps):
            dist_wrist_tip = np.linalg.norm(hand_pts[tip] - wrist)
            max_len = np.linalg.norm(hand_pts[mcp] - wrist) + np.linalg.norm(hand_pts[tip] - hand_pts[mcp])
            curls.append(dist_wrist_tip / max_len if max_len > 1e-6 else 0.0)

        thumb_tip = hand_pts[4]
        oppositions = [np.linalg.norm(hand_pts[tip] - thumb_tip) for tip in [8, 12, 16, 20]]

        v_index = hand_pts[8] - hand_pts[5]
        v_pinky = hand_pts[20] - hand_pts[17]
        spread = self._compute_angle(v_index, v_pinky)

        return np.array([*curls, *oppositions, spread], dtype=np.float32)

    def extract_structured(self, left_hand: np.ndarray, right_hand: np.ndarray, pose: np.ndarray, timestamp: float) -> dict:
        """
        Produces a clean, structured feature object first before tensor flattening.
        """
        struct = {}

        if self.config.enable_coordinates:
            struct["coordinates"] = {
                "left_hand": left_hand,
                "right_hand": right_hand,
                "pose": pose
            }

        if self.config.enable_angles:
            struct["angles"] = {
                "left_hand": self.compute_hand_angles(left_hand),
                "right_hand": self.compute_hand_angles(right_hand)
            }

        if self.config.enable_palm_features:
            struct["palm_features"] = {
                "left_hand": self.compute_palm_features(left_hand),
                "right_hand": self.compute_palm_features(right_hand)
            }

        if self.config.enable_bone_features:
            struct["bone_features"] = {
                "left_hand": self.compute_bone_features(left_hand),
                "right_hand": self.compute_bone_features(right_hand)
            }

        if self.config.enable_finger_features:
            struct["finger_features"] = {
                "left_hand": self.compute_finger_features(left_hand),
                "right_hand": self.compute_finger_features(right_hand)
            }

        if self.config.enable_motion_features:
            dt = max(timestamp - self._prev_time, 1e-4) if self._prev_time is not None else 1.0 / 30.0
            
            lh_active = left_hand is not None and len(left_hand) == 21 and np.any(left_hand)
            rh_active = right_hand is not None and len(right_hand) == 21 and np.any(right_hand)

            vel_lh = (left_hand - self._prev_struct_lh) / dt if self._prev_struct_lh is not None and self._prev_struct_lh.shape == left_hand.shape and lh_active else np.zeros_like(left_hand)
            acc_lh = (vel_lh - self._prev_vel_lh) / dt if self._prev_vel_lh is not None and self._prev_vel_lh.shape == vel_lh.shape else np.zeros_like(left_hand)

            vel_rh = (right_hand - self._prev_struct_rh) / dt if self._prev_struct_rh is not None and self._prev_struct_rh.shape == right_hand.shape and rh_active else np.zeros_like(right_hand)
            acc_rh = (vel_rh - self._prev_vel_rh) / dt if self._prev_vel_rh is not None and self._prev_vel_rh.shape == vel_rh.shape else np.zeros_like(right_hand)

            struct["motion_features"] = {
                "left_hand_velocity": vel_lh,
                "left_hand_acceleration": acc_lh,
                "right_hand_velocity": vel_rh,
                "right_hand_acceleration": acc_rh
            }

            self._prev_struct_lh = left_hand.copy() if lh_active else None
            self._prev_struct_rh = right_hand.copy() if rh_active else None
            self._prev_vel_lh = vel_lh.copy()
            self._prev_vel_rh = vel_rh.copy()
            self._prev_time = timestamp

        if self.config.enable_body_features:
            shoulder_mid = (pose[0] + pose[1]) / 2.0 if pose is not None and len(pose) >= 2 else np.zeros(3)
            struct["body_features"] = {
                "lh_wrist_to_chest": left_hand[0] - shoulder_mid if left_hand is not None and len(left_hand) > 0 else np.zeros(3),
                "rh_wrist_to_chest": right_hand[0] - shoulder_mid if right_hand is not None and len(right_hand) > 0 else np.zeros(3)
            }

        return struct

    def flatten_structured(self, struct_obj: dict) -> np.ndarray:
        """
        Flattens the structured feature object into a 1D tensor for machine learning models.
        """
        flat_parts = []

        if "coordinates" in struct_obj:
            c = struct_obj["coordinates"]
            flat_parts.extend([c["left_hand"].flatten(), c["right_hand"].flatten(), c["pose"].flatten()])

        if "angles" in struct_obj:
            a = struct_obj["angles"]
            flat_parts.extend([a["left_hand"], a["right_hand"]])

        if "palm_features" in struct_obj:
            p = struct_obj["palm_features"]
            flat_parts.extend([p["left_hand"], p["right_hand"]])

        if "bone_features" in struct_obj:
            b = struct_obj["bone_features"]
            flat_parts.extend([b["left_hand"], b["right_hand"]])

        if "finger_features" in struct_obj:
            f = struct_obj["finger_features"]
            flat_parts.extend([f["left_hand"], f["right_hand"]])

        if "motion_features" in struct_obj:
            m = struct_obj["motion_features"]
            flat_parts.extend([
                m["left_hand_velocity"].flatten(),
                m["left_hand_acceleration"].flatten(),
                m["right_hand_velocity"].flatten(),
                m["right_hand_acceleration"].flatten()
            ])

        if "body_features" in struct_obj:
            bd = struct_obj["body_features"]
            flat_parts.extend([bd["lh_wrist_to_chest"], bd["rh_wrist_to_chest"]])

        return np.concatenate([p for p in flat_parts if len(p) > 0]).astype(np.float32)

feature_engine = FeatureExtractorEngine()
