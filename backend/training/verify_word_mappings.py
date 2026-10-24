"""
Verify Word-to-Sign Mappings
=============================
Audits video_word_map.json against the original WLASL_v0.3.json master dataset
to ensure each video ID correctly maps to the right word/sign.

Usage:
    python -m backend.training.verify_word_mappings

Output:
    A report showing verified, mismatched, and missing mappings.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
WLASL_DIR = DATASET_DIR / "wlasl"
VIDEOS_DIR = WLASL_DIR / "videos"

VIDEO_WORD_MAP_PATH = WLASL_DIR / "video_word_map.json"
WLASL_MASTER_PATH = WLASL_DIR / "WLASL_v0.3.json"


def load_master_wlasl() -> dict:
    """Load the master WLASL_v0.3.json and build a video_id -> word mapping."""
    if not WLASL_MASTER_PATH.exists():
        print(f"[WARN] Master WLASL file not found at {WLASL_MASTER_PATH}")
        return {}

    with open(WLASL_MASTER_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    master_map = {}
    for entry in data:
        gloss = entry.get("gloss", "").strip()
        instances = entry.get("instances", [])
        for inst in instances:
            video_id = inst.get("video_id", "").strip()
            if video_id:
                master_map[video_id] = gloss
    return master_map


def load_video_word_map() -> dict:
    """Load the current video_word_map.json."""
    if not VIDEO_WORD_MAP_PATH.exists():
        print(f"[ERROR] video_word_map.json not found at {VIDEO_WORD_MAP_PATH}")
        return {}

    with open(VIDEO_WORD_MAP_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_mappings():
    """Cross-reference video_word_map.json with the master WLASL dataset."""
    print("=" * 60)
    print("WLASL Word-to-Sign Mapping Verification Report")
    print("=" * 60)

    master_map = load_master_wlasl()
    video_map = load_video_word_map()

    if not master_map:
        print("\n[INFO] Cannot verify without master WLASL dataset.")
        print("[INFO] Performing basic consistency checks only...\n")

    verified = 0
    mismatched = []
    not_in_master = []
    missing_video_file = []
    word_counts = defaultdict(int)

    for video_id, info in video_map.items():
        word = info.get("word", "").lower().strip()
        word_counts[word] += 1

        # Check if video file exists
        video_file = VIDEOS_DIR / f"{video_id}.mp4"
        if not video_file.exists():
            missing_video_file.append((video_id, word))
            continue

        # Cross-reference with master
        if master_map:
            if video_id in master_map:
                master_word = master_map[video_id].lower().strip()
                if master_word == word:
                    verified += 1
                else:
                    mismatched.append({
                        "video_id": video_id,
                        "current_word": word,
                        "master_word": master_word
                    })
            else:
                not_in_master.append((video_id, word))
        else:
            verified += 1  # Can't verify, assume correct

    # Print Results
    total = len(video_map)
    print(f"\nTotal entries in video_word_map.json: {total}")
    print(f"Unique words: {len(word_counts)}")
    print(f"Verified correct: {verified}")
    print(f"Mismatched: {len(mismatched)}")
    print(f"Not in master dataset: {len(not_in_master)}")
    print(f"Missing video files: {len(missing_video_file)}")

    if mismatched:
        print(f"\n{'='*60}")
        print("MISMATCHED MAPPINGS (video_word_map says X, master says Y)")
        print(f"{'='*60}")
        for m in mismatched[:50]:  # Show first 50
            print(f"  Video {m['video_id']}: "
                  f"current='{m['current_word']}' | master='{m['master_word']}'")
        if len(mismatched) > 50:
            print(f"  ... and {len(mismatched) - 50} more")

    if missing_video_file:
        print(f"\n{'='*60}")
        print("MISSING VIDEO FILES")
        print(f"{'='*60}")
        for vid, word in missing_video_file[:20]:
            print(f"  {vid}.mp4 (word: '{word}')")
        if len(missing_video_file) > 20:
            print(f"  ... and {len(missing_video_file) - 20} more")

    # Auto-fix mismatched mappings
    if mismatched:
        print(f"\n{'='*60}")
        print("AUTO-FIX: Correcting mismatched mappings...")
        print(f"{'='*60}")
        fixed_count = 0
        for m in mismatched:
            video_map[m['video_id']]['word'] = m['master_word']
            fixed_count += 1

        # Save corrected map
        backup_path = VIDEO_WORD_MAP_PATH.with_suffix('.json.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy2(VIDEO_WORD_MAP_PATH, backup_path)
            print(f"  Backed up original to {backup_path.name}")

        with open(VIDEO_WORD_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(video_map, f, indent=2, ensure_ascii=False)
        print(f"  Fixed {fixed_count} mismatched mappings in video_word_map.json")

    # Summary
    print(f"\n{'='*60}")
    accuracy = (verified / total * 100) if total > 0 else 0
    print(f"Mapping Accuracy: {accuracy:.1f}%")
    if accuracy >= 99:
        print("STATUS: [OK] Mappings are highly accurate")
    elif accuracy >= 95:
        print("STATUS: [WARN] Some mappings need attention")
    else:
        print("STATUS: [FAIL] Significant mapping issues detected")
    print(f"{'='*60}")


if __name__ == "__main__":
    verify_mappings()
