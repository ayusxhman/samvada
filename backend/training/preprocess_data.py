import os
import json
import logging
import hashlib
from pathlib import Path
from collections import defaultdict
from backend.app.config import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_file_hash(file_path: Path) -> str:
    """
    Computes MD5 hash of a file to check for exact duplicates quickly.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def deduplicate_images(directory: Path):
    """
    Finds and deletes exact duplicate images in the dataset.
    """
    import time
    logger.info(f"Scanning for duplicate images in: {directory}")
    
    # 1. Group files by size first (extremely fast metadata check)
    size_map = defaultdict(list)
    total_files = 0
    start_time = time.time()
    last_print = start_time

    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = Path(root) / f
                total_files += 1
                try:
                    size = file_path.stat().st_size
                    size_map[size].append(file_path)
                except Exception as e:
                    logger.error(f"Error checking size of {file_path}: {e}")
                
                now = time.time()
                if now - last_print >= 1.0:
                    logger.info(f"Scanning directory: found {total_files} files...")
                    last_print = now
                    
    # 2. Only hash files that share the exact same size
    hashes = defaultdict(list)
    duplicates_removed = 0
    scanned_hashes = 0
    
    logger.info(f"Scanning complete. Found {total_files} files. Grouping candidate duplicates...")
    
    for size, paths in size_map.items():
        if len(paths) > 1:
            for file_path in paths:
                scanned_hashes += 1
                try:
                    f_hash = get_file_hash(file_path)
                    hashes[f_hash].append(file_path)
                except Exception as e:
                    logger.error(f"Error hashing {file_path}: {e}")
                    
                now = time.time()
                if now - last_print >= 1.0:
                    logger.info(f"Hashing duplicates: processed {scanned_hashes} candidate files...")
                    last_print = now

    # Remove duplicates
    for f_hash, paths in hashes.items():
        if len(paths) > 1:
            # Keep the first path, delete the rest
            for dup_path in paths[1:]:
                try:
                    os.remove(dup_path)
                    duplicates_removed += 1
                except Exception as e:
                    logger.error(f"Error removing duplicate {dup_path}: {e}")

    logger.info(f"Deduplication complete. Scanned: {total_files}, Hashed candidates: {scanned_hashes}, Removed duplicates: {duplicates_removed}")
    return duplicates_removed

def main():
    # 1. Deduplicate ASL Alphabet dataset
    alphabet_dir = DATASET_DIR / "asl_alphabet" / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
    if alphabet_dir.exists():
        deduplicate_images(alphabet_dir)
    else:
        logger.warning(f"Alphabet directory not found at {alphabet_dir}")

    # 2. Deduplicate sign_alphabet_real
    real_alphabet_dir = DATASET_DIR / "sign_alphabet_real"
    if real_alphabet_dir.exists():
        deduplicate_images(real_alphabet_dir)

    # 3. Deduplicate signlan_img
    signlan_dir = DATASET_DIR / "signlan_img"
    if signlan_dir.exists():
        deduplicate_images(signlan_dir)

if __name__ == "__main__":
    main()
