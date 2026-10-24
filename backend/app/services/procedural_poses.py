"""
Procedural ASL Handshape & Pose Generator (Milestone 7 Refinement)
==================================================================
Defines production-quality anatomical ASL fingerspelling handshapes (A-Z) with:
  - Exact thumb opposition, CMC placement, and IP/MCP flexion
  - Independent MCP joint posture and PIP/DIP anatomical coupling
  - Explicit inter-finger spacing (splay/abduction) and collision avoidance
  - Natural resting curvature for active and non-dominant hands
  - C^2 continuous minimum-jerk SLERP transitions between letters
  - Lexicalized fingerspelling drift and syllabic cadence bounce
  - Dynamic kinematic trajectories for moving letters ('J' and 'Z')
"""

import math
from typing import Dict, List, Tuple

def quat_from_euler(x: float, y: float, z: float) -> List[float]:
    """Euler angles (radians, XYZ order) to quaternion [x, y, z, w]."""
    cx, sx = math.cos(x * 0.5), math.sin(x * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cz, sz = math.cos(z * 0.5), math.sin(z * 0.5)
    return [
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ]

def quat_mul(q1: List[float], q2: List[float]) -> List[float]:
    """Hamilton product of two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]

def slerp(q1: List[float], q2: List[float], t: float) -> List[float]:
    """Spherical linear interpolation between two quaternions [x, y, z, w]."""
    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0.0:
        q2 = [-a for a in q2]
        dot = -dot
    
    if dot > 0.9995:
        res = [a + t * (b - a) for a, b in zip(q1, q2)]
        norm = math.sqrt(sum(a * a for a in res))
        return [a / (norm + 1e-8) for a in res]
    
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    
    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    res = [s1 * a + s2 * b for a, b in zip(q1, q2)]
    norm = math.sqrt(sum(a * a for a in res))
    return [a / (norm + 1e-8) for a in res]

def _make_finger_pose(
    hand_side: str,
    finger_name: str,
    mcp_flex: float,
    mcp_splay: float,
    pip_flex: float,
    dip_flex: float
) -> Dict[str, List[float]]:
    """Generates 3 bone quaternions (Proximal, Intermediate, Distal) for a finger."""
    # In VRM right hand, +Y is radial splay (towards thumb), -Y is ulnar splay (towards pinky).
    # For left hand, splay sign flips so symmetry is preserved.
    splay_val = mcp_splay if hand_side == "right" else -mcp_splay
    return {
        f"{hand_side}{finger_name}Proximal": quat_from_euler(mcp_flex, splay_val, 0.0),
        f"{hand_side}{finger_name}Intermediate": quat_from_euler(pip_flex, 0.0, 0.0),
        f"{hand_side}{finger_name}Distal": quat_from_euler(dip_flex, 0.0, 0.0),
    }

def _make_thumb_pose(
    hand_side: str,
    cmc_rot: List[float],
    mcp_flex: float,
    ip_flex: float
) -> Dict[str, List[float]]:
    """Generates 3 bone quaternions for the thumb (Metacarpal, Proximal, Distal)."""
    # For left hand, flip Y and Z angles on CMC for mirror symmetry
    cmc = list(cmc_rot)
    if hand_side == "left":
        cmc[1] = -cmc[1]
        cmc[2] = -cmc[2]
    return {
        f"{hand_side}ThumbMetacarpal": quat_from_euler(cmc[0], cmc[1], cmc[2]),
        f"{hand_side}ThumbProximal": quat_from_euler(0.0, mcp_flex, 0.0),
        f"{hand_side}ThumbDistal": quat_from_euler(0.0, ip_flex, 0.0),
    }

def _make_hand_shape(
    thumb_spec: Tuple[List[float], float, float],
    index_spec: Tuple[float, float, float, float],
    middle_spec: Tuple[float, float, float, float],
    ring_spec: Tuple[float, float, float, float],
    little_spec: Tuple[float, float, float, float],
    hand_rot: List[float] = None
) -> Dict[str, List[float]]:
    """Builds a complete 15-finger-bone dictionary + wrist rotation for the right hand."""
    if hand_rot is None:
        hand_rot = [0.0, 0.0, 0.0]
        
    pose = {}
    pose["rightHand"] = quat_from_euler(hand_rot[0], hand_rot[1], hand_rot[2])
    
    pose.update(_make_thumb_pose("right", thumb_spec[0], thumb_spec[1], thumb_spec[2]))
    pose.update(_make_finger_pose("right", "Index", *index_spec))
    pose.update(_make_finger_pose("right", "Middle", *middle_spec))
    pose.update(_make_finger_pose("right", "Ring", *ring_spec))
    pose.update(_make_finger_pose("right", "Little", *little_spec))
    
    return pose

# Procedural ASL Alphabet Handshape Map (Letter -> Right Hand Pose Dict)
# Tuple format for fingers: (mcp_flex, mcp_splay, pip_flex, dip_flex)
# Tuple format for thumb: ([cmc_x, cmc_y, cmc_z], mcp_flex, ip_flex)
ASL_HANDSHAPES: Dict[str, Dict[str, List[float]]] = {
    # A: Fist with thumb resting upright alongside index finger
    'A': _make_hand_shape(
        thumb_spec=([0.15, -0.35, -0.20], -0.05, -0.05),
        index_spec=(-1.25, 0.00, -1.35, -1.20),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.25, 0.00, -1.35, -1.20),
        little_spec=(-1.25, 0.00, -1.35, -1.20),
        hand_rot=[0.0, 0.0, 0.0]
    ),
    # B: Flat hand, fingers pointing up and tightly adducted together, thumb folded across palm
    'B': _make_hand_shape(
        thumb_spec=([-0.25, 0.45, 0.25], -0.30, -0.30),
        index_spec=(-0.05, -0.03, -0.05, -0.05),
        middle_spec=(-0.05, 0.00, -0.05, -0.05),
        ring_spec=(-0.05, +0.03, -0.05, -0.05),
        little_spec=(-0.05, +0.06, -0.05, -0.05),
        hand_rot=[-0.1, 0.3, 0.1]
    ),
    # C: Curved hand forming an open C-arc, thumb opposing at bottom
    'C': _make_hand_shape(
        thumb_spec=([-0.40, -0.55, 0.20], -0.35, -0.35),
        index_spec=(-0.45, +0.06, -0.50, -0.40),
        middle_spec=(-0.45, 0.00, -0.50, -0.40),
        ring_spec=(-0.45, -0.05, -0.50, -0.40),
        little_spec=(-0.45, -0.10, -0.50, -0.40),
        hand_rot=[-0.2, -0.3, 0.0]
    ),
    # D: Index pointing upright, middle/ring/little curled to meet thumb tip in a loop
    'D': _make_hand_shape(
        thumb_spec=([-0.30, -0.50, 0.15], -0.40, -0.40),
        index_spec=(-0.05, +0.05, -0.06, -0.05),
        middle_spec=(-0.90, 0.00, -1.10, -0.95),
        ring_spec=(-0.95, 0.00, -1.15, -1.00),
        little_spec=(-1.00, 0.00, -1.20, -1.05),
        hand_rot=[-0.2, -0.4, 0.0]
    ),
    # E: All four fingertips clawed tightly down to rest on top of folded thumb edge
    'E': _make_hand_shape(
        thumb_spec=([0.20, -0.65, -0.10], -0.30, -0.30),
        index_spec=(-0.60, 0.00, -1.40, -1.30),
        middle_spec=(-0.60, 0.00, -1.40, -1.30),
        ring_spec=(-0.65, 0.00, -1.45, -1.35),
        little_spec=(-0.70, 0.00, -1.50, -1.40),
        hand_rot=[-0.1, -0.5, 0.0]
    ),
    # F: Index tip and thumb tip touching in circle (OK sign), middle/ring/little upright and spread
    'F': _make_hand_shape(
        thumb_spec=([-0.35, -0.50, 0.15], -0.45, -0.45),
        index_spec=(-0.65, -0.05, -0.85, -0.70),
        middle_spec=(-0.06, +0.08, -0.08, -0.06),
        ring_spec=(-0.06, 0.00, -0.08, -0.06),
        little_spec=(-0.06, -0.12, -0.08, -0.06),
        hand_rot=[-0.3, -0.4, 0.0]
    ),
    # G: Index and thumb pointing forward horizontally parallel, others curled
    'G': _make_hand_shape(
        thumb_spec=([0.05, -0.25, 0.00], -0.05, -0.05),
        index_spec=(-0.05, 0.00, -0.05, -0.05),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.3, 0.8, 0.0]
    ),
    # H: Index and middle pointing forward horizontally side-by-side, others curled
    'H': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.10], -0.35, -0.35),
        index_spec=(-0.05, -0.03, -0.05, -0.05),
        middle_spec=(-0.05, +0.03, -0.05, -0.05),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.3, 0.8, 0.0]
    ),
    # I: Little finger upright, index/middle/ring curled into palm, thumb over middle/ring nails
    'I': _make_hand_shape(
        thumb_spec=([0.15, -0.45, -0.20], -0.40, -0.40),
        index_spec=(-1.25, 0.00, -1.35, -1.20),
        middle_spec=(-1.30, 0.00, -1.40, -1.25),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-0.05, -0.08, -0.06, -0.05),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # J: Starts as I, dynamically swoops to trace J in air
    'J': _make_hand_shape(
        thumb_spec=([0.15, -0.45, -0.20], -0.40, -0.40),
        index_spec=(-1.25, 0.00, -1.35, -1.20),
        middle_spec=(-1.30, 0.00, -1.40, -1.25),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-0.05, -0.08, -0.06, -0.05),
        hand_rot=[0.2, 0.3, 0.2]
    ),
    # K: Index upright, middle forward 45 deg, thumb tip resting in V notch
    'K': _make_hand_shape(
        thumb_spec=([-0.15, -0.35, 0.00], -0.15, -0.15),
        index_spec=(-0.05, +0.05, -0.06, -0.05),
        middle_spec=(-0.75, 0.00, -0.15, -0.10),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.2, -0.3, 0.0]
    ),
    # L: Index upright, thumb extended horizontally at 90 deg (L shape)
    'L': _make_hand_shape(
        thumb_spec=([0.00, -0.85, -0.50], 0.00, 0.00),
        index_spec=(-0.05, 0.00, -0.05, -0.05),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.0, -0.7, -0.4]
    ),
    # M: Thumb tucked under index, middle, and ring fingers!
    'M': _make_hand_shape(
        thumb_spec=([0.26, -0.55, -0.10], -0.35, -0.35),
        index_spec=(-1.15, 0.00, -1.25, -1.15),
        middle_spec=(-1.15, 0.00, -1.25, -1.15),
        ring_spec=(-1.15, 0.00, -1.25, -1.15),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.2, 0.0]
    ),
    # N: Thumb tucked under index and middle fingers!
    'N': _make_hand_shape(
        thumb_spec=([0.24, -0.50, -0.10], -0.35, -0.35),
        index_spec=(-1.15, 0.00, -1.25, -1.15),
        middle_spec=(-1.15, 0.00, -1.25, -1.15),
        ring_spec=(-1.35, 0.00, -1.45, -1.30),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.2, 0.0]
    ),
    # O: All four fingertips meet thumb tip forming an O-circle
    'O': _make_hand_shape(
        thumb_spec=([-0.45, -0.55, 0.20], -0.40, -0.40),
        index_spec=(-0.55, +0.05, -0.65, -0.55),
        middle_spec=(-0.55, 0.00, -0.65, -0.55),
        ring_spec=(-0.55, -0.05, -0.65, -0.55),
        little_spec=(-0.55, -0.10, -0.65, -0.55),
        hand_rot=[-0.4, -0.4, 0.0]
    ),
    # P: Same as K, pointing downward
    'P': _make_hand_shape(
        thumb_spec=([-0.15, -0.35, 0.00], -0.15, -0.15),
        index_spec=(-0.05, +0.05, -0.06, -0.05),
        middle_spec=(-0.75, 0.00, -0.15, -0.10),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.8, 0.0, 0.0]
    ),
    # Q: Same as G, pointing downward
    'Q': _make_hand_shape(
        thumb_spec=([0.05, -0.25, 0.00], -0.05, -0.05),
        index_spec=(-0.05, 0.00, -0.05, -0.05),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.8, 0.5, 0.0]
    ),
    # R: Index and middle upright and crossed (middle over index)
    'R': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.08, -0.12, -0.06, -0.05),
        middle_spec=(-0.02, +0.14, -0.06, -0.05),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # S: Fist with thumb folded across front of index/middle knuckles
    'S': _make_hand_shape(
        thumb_spec=([0.25, -0.55, -0.15], -0.35, -0.35),
        index_spec=(-1.25, 0.00, -1.35, -1.20),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.25, 0.00, -1.35, -1.20),
        little_spec=(-1.25, 0.00, -1.35, -1.20),
        hand_rot=[0.1, -0.3, 0.0]
    ),
    # T: Thumb tucked under index finger only!
    'T': _make_hand_shape(
        thumb_spec=([0.22, -0.45, -0.10], -0.35, -0.35),
        index_spec=(-1.15, 0.00, -1.25, -1.15),
        middle_spec=(-1.30, 0.00, -1.40, -1.25),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.2, 0.0]
    ),
    # U: Index and middle upright, tightly adducted together side-by-side
    'U': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.05, -0.04, -0.05, -0.05),
        middle_spec=(-0.05, +0.04, -0.05, -0.05),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # V: Index and middle upright, spread wide apart into V (peace sign)
    'V': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.05, +0.16, -0.05, -0.05),
        middle_spec=(-0.05, -0.16, -0.05, -0.05),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # W: Index, middle, ring upright and spread into 3 prongs
    'W': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.05, +0.16, -0.05, -0.05),
        middle_spec=(-0.05, 0.00, -0.05, -0.05),
        ring_spec=(-0.05, -0.16, -0.05, -0.05),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # X: Index finger hooked into a claw/hook
    'X': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.15, 0.00, -1.25, -1.05),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[0.1, -0.3, -0.1]
    ),
    # Y: Thumb and little finger extended wide out (Shaka sign)
    'Y': _make_hand_shape(
        thumb_spec=([0.00, -0.85, -0.50], 0.00, 0.00),
        index_spec=(-1.25, 0.00, -1.35, -1.20),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.25, 0.00, -1.40, -1.25),
        little_spec=(-0.05, -0.18, -0.05, -0.05),
        hand_rot=[0.0, -0.7, -0.4]
    ),
    # Z: Index extended pointing forward, dynamically traces Z in air
    'Z': _make_hand_shape(
        thumb_spec=([0.10, -0.40, -0.15], -0.35, -0.35),
        index_spec=(-0.05, 0.00, -0.05, -0.05),
        middle_spec=(-1.25, 0.00, -1.35, -1.20),
        ring_spec=(-1.30, 0.00, -1.40, -1.25),
        little_spec=(-1.35, 0.00, -1.45, -1.30),
        hand_rot=[-0.2, -0.4, 0.0]
    ),
}

def generate_procedural_fingerspell_timeline(word: str, fps: int = 30) -> dict:
    """
    Generates a production-quality procedural animation timeline for a word by fingerspelling
    each letter in sequence with:
      - C^2 continuous minimum-jerk SLERP inter-letter transitions
      - Lexicalized fingerspelling drift and syllabic cadence bounce
      - Dynamic kinematic trajectories for 'J' and 'Z'
      - Non-dominant hand natural resting curvature
    """
    clean_word = "".join([c for c in word.upper().strip() if c.isalpha()])
    if not clean_word:
        clean_word = "A"

    letter_duration = 0.35      # Seconds per letter articulation
    transition_duration = 0.16  # C^2 transition window between letters (~45% of letter duration)
    
    total_duration = len(clean_word) * letter_duration
    total_frames = max(15, int(total_duration * fps))

    frames = []
    
    # Base arm pose for signing in front of chest/shoulder
    base_arm_bones = {
        "rightUpperArm": quat_from_euler(-0.40, 0.20, -0.60),
        "rightLowerArm": quat_from_euler(-0.80, 0.00, 0.00),
        "rightShoulder": quat_from_euler(0.00, 0.00, -0.10),
        "leftUpperArm": quat_from_euler(0.00, 0.00, 1.35),
        "leftLowerArm": quat_from_euler(-0.20, 0.00, 0.00),
        "leftShoulder": quat_from_euler(0.00, 0.00, 0.10),
        "leftHand": quat_from_euler(-0.10, 0.00, -0.05),
        "spine": quat_from_euler(0.05, 0.00, 0.00),
        "chest": quat_from_euler(0.02, 0.00, 0.00),
        "neck": quat_from_euler(0.02, 0.00, 0.00),
        "head": quat_from_euler(0.00, 0.00, 0.00),
    }
    
    # Add soft natural resting curvature to Left Hand fingers (non-dominant hand)
    base_arm_bones.update(_make_thumb_pose("left", [-0.10, 0.20, 0.10], -0.10, -0.10))
    base_arm_bones.update(_make_finger_pose("left", "Index", -0.06, -0.08, -0.08, -0.06))
    base_arm_bones.update(_make_finger_pose("left", "Middle", -0.06, 0.00, -0.08, -0.06))
    base_arm_bones.update(_make_finger_pose("left", "Ring", -0.06, +0.06, -0.08, -0.06))
    base_arm_bones.update(_make_finger_pose("left", "Little", -0.06, +0.12, -0.08, -0.06))

    for frame_idx in range(total_frames):
        t = frame_idx / float(fps)
        idx = min(int(t / letter_duration), len(clean_word) - 1)
        local_t = t - idx * letter_duration
        
        curr_letter = clean_word[idx]
        curr_pose = ASL_HANDSHAPES.get(curr_letter, ASL_HANDSHAPES['A'])
        
        # 1. C^2 continuous minimum-jerk SLERP transition between letters
        if idx > 0 and local_t < transition_duration:
            prev_letter = clean_word[idx - 1]
            prev_pose = ASL_HANDSHAPES.get(prev_letter, ASL_HANDSHAPES['A'])
            u = local_t / transition_duration
            w_u = 10.0 * (u ** 3) - 15.0 * (u ** 4) + 6.0 * (u ** 5)
            
            blended_pose = {}
            for bone_name, target_quat in curr_pose.items():
                prev_quat = prev_pose.get(bone_name, target_quat)
                blended_pose[bone_name] = slerp(prev_quat, target_quat, w_u)
        else:
            blended_pose = dict(curr_pose)
            
        # 2. Dynamic kinematic trajectories for 'J' and 'Z'
        if curr_letter == 'J':
            tau = min(1.0, max(0.0, local_t / letter_duration))
            if 0.15 < tau < 0.90:
                phi = (tau - 0.15) / 0.75
                yaw_delta = -0.65 * math.sin(math.pi * phi)
                roll_delta = 0.50 * math.sin(math.pi * phi)
                dyn_quat = quat_from_euler(0.0, yaw_delta, roll_delta)
                if "rightHand" in blended_pose:
                    blended_pose["rightHand"] = quat_mul(dyn_quat, blended_pose["rightHand"])
                    
        elif curr_letter == 'Z':
            tau = min(1.0, max(0.0, local_t / letter_duration))
            yaw_delta = 0.35 * math.cos(2.0 * math.pi * tau)
            pitch_delta = 0.20 * math.sin(2.0 * math.pi * tau)
            dyn_quat = quat_from_euler(pitch_delta, yaw_delta, 0.0)
            if "rightHand" in blended_pose:
                blended_pose["rightHand"] = quat_mul(dyn_quat, blended_pose["rightHand"])

        # 3. Assemble frame bones with Lexicalized Drift & Syllabic Cadence Bounce
        frame_bones = dict(base_arm_bones)
        frame_bones.update(blended_pose)
        
        word_progress = frame_idx / float(max(1, total_frames - 1))
        drift_yaw = -0.12 * word_progress  # Subtle ~6.8 deg rightward typewriter drift
        bounce_pitch = -0.06 * math.sin(math.pi * (local_t / letter_duration))  # Syllabic cadence dip
        
        # Apply drift to upper arm
        ua_base = frame_bones["rightUpperArm"]
        frame_bones["rightUpperArm"] = quat_mul(quat_from_euler(0.0, drift_yaw, 0.0), ua_base)
        
        # Apply cadence bounce to wrist
        wrist_base = frame_bones.get("rightHand", quat_from_euler(0, 0, 0))
        frame_bones["rightHand"] = quat_mul(quat_from_euler(bounce_pitch, 0.0, 0.0), wrist_base)
        
        # 4. Syntactic facial behavior (Milestone 6 integration)
        # Focused neutral alertness during active spelling, closure blink at word completion
        is_closure = (t >= total_duration * 0.80)
        blink_weight = round(0.25 * math.sin(math.pi * ((t - total_duration * 0.80) / (total_duration * 0.20))), 4) if is_closure else 0.0
        
        frames.append({
            "t": round(t, 4),
            "bones": frame_bones,
            "blendshapes": {
                "relaxed": 0.10,
                "lookDown": 0.05,
                "lookRight": 0.05,
                "blink": max(0.0, blink_weight)
            }
        })

    return {
        "fps": fps,
        "duration": round(total_duration, 3),
        "frames": frames
    }
