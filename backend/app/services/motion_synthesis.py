"""
MotionSynthesisService — AI VRM Bone Quaternion Timeline Engine
=================================================================
Converts 3D MediaPipe landmark datasets (WLASL & Merged) into real-time
VRM humanoid bone quaternion animation timelines for arbitrary words & sentences.

Responsibilities:
1. Loads 141-dim / 129-dim landmark PKL datasets at startup.
2. Converts 3D keypoint arrays to VRM bone quaternions using analytical IK.
3. Caches per-word quaternion timelines for instant (<1ms) lookup.
4. Concatenates gloss sequences with 4-frame SLERP overlap blending into smooth sentence timelines.
5. Falls back to procedural ASL fingerspelling for unmapped glosses.
"""

import pickle
import logging
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.app.config import DATASET_DIR
from backend.app.services.procedural_poses import generate_procedural_fingerspell_timeline

logger = logging.getLogger(__name__)

# Import analytical IK utilities from existing training script
from backend.training.generate_vrma_from_video import (
    landmarks_to_bone_rotations,
    smooth_rotations,
    quat_slerp,
    quat_identity,
    VRM_BONES
)

MERGED_141_PATH = DATASET_DIR / "merged_landmarks_141.pkl"
MERGED_129_PATH = DATASET_DIR / "merged_landmarks_129.pkl"

class MotionSynthesisService:
    def __init__(self):
        self.word_timelines: Dict[str, dict] = {}
        self.raw_dataset_map: Dict[str, np.ndarray] = {}
        self.fps = 30
        self.initialized = False
        self.load_and_index_datasets()
        import threading
        threading.Thread(target=self._precompile_background, daemon=True).start()

    def _convert_141_array_to_frames(self, arr_30x141: np.ndarray) -> List[dict]:
        """
        Converts a 30-frame 141-dim landmark array into frame_data dicts
        expected by landmarks_to_bone_rotations().
        
        141-dim layout:
          [0:3]     -> Nose (x, y, z)
          [3:66]    -> Left Hand (21 keypoints * 3D = 63 values)
          [66:129]  -> Right Hand (21 keypoints * 3D = 63 values)
          [129:141] -> Pose Joints (Left Shoulder, Right Shoulder, Left Elbow, Right Elbow = 4 * 3D = 12 values)
        """
        frames = []
        for i in range(arr_30x141.shape[0]):
            row = arr_30x141[i]
            
            # Reconstruct 33-pose landmark list (zero-initialized)
            pose_lms = [[0.0, 0.0, 0.0] for _ in range(33)]
            
            # Nose
            pose_lms[0] = row[0:3].tolist()
            
            # Upper body pose
            if row.shape[0] >= 141:
                pose_lms[11] = row[129:132].tolist() # Left Shoulder
                pose_lms[12] = row[132:135].tolist() # Right Shoulder
                pose_lms[13] = row[135:138].tolist() # Left Elbow
                pose_lms[14] = row[138:141].tolist() # Right Elbow
            else:
                # Default shoulders if missing
                pose_lms[11] = [-0.2, 0.4, 0.0]
                pose_lms[12] = [0.2, 0.4, 0.0]
                pose_lms[13] = [-0.3, 0.1, 0.0]
                pose_lms[14] = [0.3, 0.1, 0.0]

            # Reconstruct wrists from hand landmark 0
            lh_raw = row[3:66].reshape(21, 3)
            rh_raw = row[66:129].reshape(21, 3)
            
            pose_lms[15] = lh_raw[0].tolist() # Left Wrist
            pose_lms[16] = rh_raw[0].tolist() # Right Wrist
            
            # Approximate hips relative to shoulders
            pose_lms[23] = [pose_lms[11][0], pose_lms[11][1] + 0.5, pose_lms[11][2]]
            pose_lms[24] = [pose_lms[12][0], pose_lms[12][1] + 0.5, pose_lms[12][2]]

            frame_dict = {
                "pose": pose_lms,
                "left_hand": lh_raw.tolist(),
                "right_hand": rh_raw.tolist(),
            }
            frames.append(frame_dict)
            
        return frames

    def _clamp_quaternion(self, q: List[float], bone_name: str) -> List[float]:
        """Validates and clamps quaternions to physiological human joint limits."""
        # Convert quaternion to euler (XYZ order)
        x, y, z, w = q
        # Norm check
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Calculate euler angles
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        ex = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            ey = math.copysign(math.pi / 2, sinp)
        else:
            ey = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        ez = math.atan2(siny_cosp, cosy_cosp)

        # Apply specific physiological bone clamps (Fix #1 corrected ranges)
        if bone_name == "leftUpperArm":
            ex = max(math.radians(-90), min(math.radians(90), ex))
            ey = max(math.radians(-90), min(math.radians(90), ey))
            ez = max(math.radians(-110), min(math.radians(45), ez))
        elif bone_name == "rightUpperArm":
            ex = max(math.radians(-90), min(math.radians(90), ex))
            ey = max(math.radians(-90), min(math.radians(90), ey))
            ez = max(math.radians(-45), min(math.radians(110), ez))
        elif bone_name in ["leftLowerArm", "rightLowerArm"]:
            ex = max(math.radians(-150), min(math.radians(10), ex))
            ey = max(math.radians(-90), min(math.radians(90), ey))
            ez = max(math.radians(-90), min(math.radians(90), ez))

        # Reconvert euler (XYZ) back to quaternion
        cx = math.cos(ex * 0.5)
        sx = math.sin(ex * 0.5)
        cy = math.cos(ey * 0.5)
        sy = math.sin(ey * 0.5)
        cz = math.cos(ez * 0.5)
        sz = math.sin(ez * 0.5)

        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
        
        return [qx, qy, qz, qw]

    def _compute_frame_nmm(self, word: str, frame_idx: int, total_frames: int, fd: dict) -> dict:
        """Computes Non-Manual Grammatical Markers (NMM) for a single frame.
        
        Includes:
        - Eyebrow raise/lowering for syntactic functions (topicalization, requests, intensifiers, wh-questions).
        - Gaze tracking toward active signing space (horizontal shift and vertical elevation).
        - Subtle mouth posture & jaw relaxation matching grammatical roles.
        - Grammatical closure blinks at syntactic phrase boundaries.
        """
        w_clean = word.lower().strip()
        bs = {}
        
        # 1. Spatial Gaze Tracking Toward Active Signing Space
        pose = fd.get("pose", [[0.0, 0.0, 0.0]] * 33)
        if len(pose) >= 17:
            ls = pose[11]
            rs = pose[12]
            lw = pose[15]
            rw = pose[16]
            
            mid_s_x = (ls[0] + rs[0]) / 2.0
            x_shift = ((lw[0] - mid_s_x) + (rw[0] - mid_s_x)) / 2.0
            y_elev = -((lw[1] - ls[1]) + (rw[1] - rs[1])) / 2.0
            
            # Horizontal gaze tracking
            if x_shift > 0.02:
                bs["lookRight"] = round(float(np.clip((x_shift - 0.02) * 1.5, 0.0, 0.35)), 4)
            elif x_shift < -0.02:
                bs["lookLeft"] = round(float(np.clip((-x_shift - 0.02) * 1.5, 0.0, 0.35)), 4)
                
            # Vertical gaze tracking
            if y_elev > 0.10:
                bs["lookUp"] = round(float(np.clip((y_elev - 0.10) * 0.8, 0.0, 0.25)), 4)
            elif y_elev < -0.10:
                bs["lookDown"] = round(float(np.clip((-y_elev - 0.10) * 0.8, 0.0, 0.25)), 4)

        # 2. Syntactic Eyebrow & Mouth Postures
        if w_clean in ["please", "thank", "thank you", "help", "book", "chair", "yes", "welcome", "hello", "good"]:
            # Polite requests / Affirmation / Topicalization -> subtle eyebrow raise & relaxed mouth
            bs["surprised"] = 0.16
            bs["relaxed"] = 0.12
            bs["ih"] = 0.08
        elif w_clean in ["very", "much", "why", "what", "where", "who", "how", "big", "fast", "hard", "no"]:
            # Intensifiers / Wh-questions / Magnitude -> subtle eyebrow furrowing & magnitude jaw opening
            bs["angry"] = 0.15
            bs["aa"] = 0.14
            bs["oh"] = 0.12
        elif w_clean in ["drink", "eat", "water", "cup", "food"]:
            # Ingestion / Reception verbs -> subtle lip rounding
            bs["relaxed"] = 0.08
            bs["ou"] = 0.15
            bs["oh"] = 0.10
        else:
            # Neutral sign alertness & jaw relaxation
            bs["relaxed"] = 0.06

        # 3. Grammatical Closure Blinking at Syntactic Phrase Boundaries
        # Eyes remain open (0.0) during active articulation; subtle closure blink occurs at phrase boundary (last 20%)
        if total_frames > 0 and (frame_idx / float(total_frames)) >= 0.80:
            progress = (frame_idx - 0.80 * total_frames) / (0.20 * total_frames)
            bs["blink"] = round(float(math.sin(progress * math.pi) * 0.25), 4)
        else:
            bs["blink"] = 0.0
            
        return bs

    def _convert_array_to_timeline(self, sample_arr: np.ndarray, word: str) -> dict:
        """Converts raw 3D landmark array (30 frames) to VRM quaternion timeline."""
        if sample_arr.shape[1] >= 141:
            frame_dicts = self._convert_141_array_to_frames(sample_arr)
        else:
            # For 129-dim, pad pose joints with defaults
            padded = np.zeros((sample_arr.shape[0], 141), dtype=np.float32)
            padded[:, :129] = sample_arr
            frame_dicts = self._convert_141_array_to_frames(padded)

        # Run analytical IK per frame
        raw_quats_list = [landmarks_to_bone_rotations(fd) for fd in frame_dicts]
        
        # Smooth quaternions across time
        smoothed_quats = smooth_rotations(raw_quats_list, window=5)
        
        timeline_frames = []
        duration = len(smoothed_quats) / float(self.fps)
        
        for i, quat_map in enumerate(smoothed_quats):
            # Validate and clamp all bone quaternions
            clamped_map = {}
            for bone, q in quat_map.items():
                clamped_map[bone] = self._clamp_quaternion(q, bone)
                
            timeline_frames.append({
                "t": round(i / float(self.fps), 4),
                "bones": clamped_map,
                "blendshapes": self._compute_frame_nmm(word, i, len(smoothed_quats), frame_dicts[i])
            })
            
        return {
            "fps": self.fps,
            "duration": max(0.5, duration),
            "frames": timeline_frames
        }

    def load_and_index_datasets(self):
        """Loads PKL landmark datasets synchronously into fast lookup index."""
        try:
            logger.info("Initializing MotionSynthesisService — Indexing raw landmark arrays...")
            
            # 1. Try loading 141-dim dataset
            if MERGED_141_PATH.exists():
                with open(MERGED_141_PATH, "rb") as f:
                    data = pickle.load(f)
                    X_arr = data.get("X")
                    sample_labels = data.get("y", [])
                    
                    if X_arr is not None and len(sample_labels) == len(X_arr):
                        for idx, word_label in enumerate(sample_labels):
                            word_clean = str(word_label).lower().strip()
                            if word_clean not in self.raw_dataset_map:
                                self.raw_dataset_map[word_clean] = X_arr[idx] # shape (30, 141)
            
            # 2. Try loading 129-dim dataset for any missing words
            if MERGED_129_PATH.exists():
                with open(MERGED_129_PATH, "rb") as f:
                    data = pickle.load(f)
                    X_arr = data.get("X")
                    sample_labels = data.get("y", [])
                    
                    if X_arr is not None and len(sample_labels) == len(X_arr):
                        for idx, word_label in enumerate(sample_labels):
                            word_clean = str(word_label).lower().strip()
                            if word_clean not in self.raw_dataset_map:
                                self.raw_dataset_map[word_clean] = X_arr[idx] # shape (30, 129)

            self.initialized = True
            logger.info(f"Successfully indexed {len(self.raw_dataset_map)} unique sign word raw arrays.")
        except Exception as e:
            logger.error(f"Error indexing MotionSynthesisService: {e}")
            self.initialized = False

    def _precompile_background(self):
        """Precompiles quaternion timelines in background thread without blocking."""
        try:
            for word_clean, sample_arr in list(self.raw_dataset_map.items()):
                if word_clean not in self.word_timelines:
                    self.word_timelines[word_clean] = self._convert_array_to_timeline(sample_arr, word_clean)
            logger.info(f"Successfully precompiled {len(self.word_timelines)} VRM quaternion timelines in background.")
        except Exception as e:
            logger.error(f"Error during background precompilation: {e}")

    def get_word_timeline(self, word: str) -> dict:
        """Returns the quaternion timeline for a single word, or procedural fingerspelling if unknown."""
        word_clean = word.lower().strip()
        if word_clean in self.word_timelines:
            return self.word_timelines[word_clean]
        
        # Fast on-demand IK conversion if word exists in raw dataset map
        if hasattr(self, 'raw_dataset_map') and word_clean in self.raw_dataset_map:
            timeline = self._convert_array_to_timeline(self.raw_dataset_map[word_clean], word_clean)
            self.word_timelines[word_clean] = timeline
            return timeline
        
        # Fallback: Procedural ASL fingerspelling
        return generate_procedural_fingerspell_timeline(word, fps=self.fps)

    def _get_frame_velocity(self, f1: dict, f2: dict) -> float:
        """Calculates total angular displacement (degrees) across arm joints between two frames."""
        total_deg = 0.0
        arm_bones = ["leftShoulder", "rightShoulder", "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm", "leftHand", "rightHand"]
        for b in arm_bones:
            q1 = f1["bones"].get(b, quat_identity())
            q2 = f2["bones"].get(b, quat_identity())
            dot = min(1.0, max(-1.0, abs(sum(a*c for a, c in zip(q1, q2)))))
            total_deg += math.degrees(2 * math.acos(dot))
        return round(total_deg, 2)

    def _trim_inactive_boundary_frames(self, frames: List[dict], is_end: bool = False, threshold: float = 1.5, max_trim_ratio: float = 0.3) -> List[dict]:
        """Trims leading or trailing inactive frames (e.g. static rest holds or idling drop-downs)."""
        n = len(frames)
        max_trim = int(n * max_trim_ratio)
        if max_trim <= 0:
            return frames
        
        if is_end:
            trim_count = 0
            for i in range(n - 1, n - 1 - max_trim, -1):
                if self._get_frame_velocity(frames[i-1], frames[i]) < threshold:
                    trim_count += 1
                else:
                    break
            return frames[:n - trim_count] if trim_count > 0 else frames
        else:
            trim_count = 0
            for i in range(0, max_trim):
                if self._get_frame_velocity(frames[i], frames[i+1]) < threshold:
                    trim_count += 1
                else:
                    break
            return frames[trim_count:] if trim_count > 0 else frames

    def generate_sentence_timeline(self, gloss_seq: List[str]) -> dict:
        """
        Concatenates a sequence of glosses into one continuous animation timeline
        using spherical spline co-articulation and boundary inactivity trimming.
        """
        if not gloss_seq:
            return generate_procedural_fingerspell_timeline("A", fps=self.fps)

        word_timelines = [self.get_word_timeline(gloss) for gloss in gloss_seq]
        
        # 1. Clean individual word timelines by trimming inactive boundary frames
        cleaned_words = []
        for idx, wt in enumerate(word_timelines):
            w_frames = wt.get("frames", [])
            if not w_frames:
                continue
            # Preserve natural intro of 1st word and natural outro of last word
            trim_start = (idx > 0)
            trim_end = (idx < len(word_timelines) - 1)
            
            frames = w_frames
            if trim_start:
                frames = self._trim_inactive_boundary_frames(frames, is_end=False, threshold=1.5, max_trim_ratio=0.3)
            if trim_end:
                frames = self._trim_inactive_boundary_frames(frames, is_end=True, threshold=1.5, max_trim_ratio=0.3)
            cleaned_words.append(frames)
            
        if not cleaned_words:
            return generate_procedural_fingerspell_timeline("A", fps=self.fps)

        # 2. Stitch words with moving-trajectory spherical polynomial co-articulation
        stitched_frames = []
        current_time = 0.0
        overlap_frames = 8

        for w_idx, w_frames in enumerate(cleaned_words):
            if w_idx == 0:
                for f in w_frames:
                    stitched_frames.append({
                        "t": round(current_time, 4),
                        "bones": dict(f["bones"]),
                        "blendshapes": dict(f.get("blendshapes", {}))
                    })
                    current_time += 1.0 / float(self.fps)
            else:
                prev_len = len(stitched_frames)
                k = min(overlap_frames, prev_len, len(w_frames))
                
                if k > 0:
                    start_replace_idx = prev_len - k
                    for b_i in range(k):
                        u = (b_i + 1) / float(k + 1)
                        # Minimum-jerk polynomial (C^1 and C^2 velocity/acceleration continuity)
                        blend_factor = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
                        
                        frame_a = stitched_frames[start_replace_idx + b_i]
                        frame_b = w_frames[b_i]
                        
                        blended_bones = {}
                        all_bones = set(frame_a["bones"].keys()).union(set(frame_b["bones"].keys()))
                        for b_name in all_bones:
                            qa = frame_a["bones"].get(b_name, quat_identity())
                            qb = frame_b["bones"].get(b_name, qa)
                            blended_bones[b_name] = quat_slerp(qa, qb, blend_factor)
                            
                        blended_bs = {}
                        bs_a = frame_a.get("blendshapes", {})
                        bs_b = frame_b.get("blendshapes", {})
                        all_bs = set(bs_a.keys()).union(set(bs_b.keys()))
                        for bs_name in all_bs:
                            va = bs_a.get(bs_name, 0.0)
                            vb = bs_b.get(bs_name, 0.0)
                            blended_bs[bs_name] = round(float(va + (vb - va) * blend_factor), 4)
                        stitched_frames[start_replace_idx + b_i]["blendshapes"] = blended_bs
                        stitched_frames[start_replace_idx + b_i]["bones"] = blended_bones
                            
                    remaining_frames = w_frames[k:]
                else:
                    remaining_frames = w_frames
                    
                for f in remaining_frames:
                    stitched_frames.append({
                        "t": round(current_time, 4),
                        "bones": dict(f["bones"]),
                        "blendshapes": dict(f.get("blendshapes", {}))
                    })
                    current_time += 1.0 / float(self.fps)

        total_duration = round(len(stitched_frames) / float(self.fps), 3)
        return {
            "fps": self.fps,
            "duration": max(0.5, total_duration),
            "frames": stitched_frames
        }

motion_synthesis_service = MotionSynthesisService()
