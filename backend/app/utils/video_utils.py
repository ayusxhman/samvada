import cv2
import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)) -> List[str]:
    """
    Extracts frames from video and saves them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_paths = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized = cv2.resize(frame, target_size)
        frame_name = f"frame_{count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        
        cv2.imwrite(frame_path, resized)
        frame_paths.append(frame_path)
        count += 1
        
    cap.release()
    logger.info(f"Extracted {count} frames from {video_path}")
    return frame_paths

def get_video_duration(video_path: str) -> float:
    """
    Returns duration of video in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        return frame_count / fps
    return 0.0
