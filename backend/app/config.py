import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent # c:\Users\Ayushman\Desktop\projectx
DATASET_DIR = BASE_DIR / "dataset"
TRAINED_MODELS_DIR = BASE_DIR / "backend" / "trained_models"

# Dataset Specific Paths
WLASL_VIDEOS_DIR = DATASET_DIR / "wlasl" / "videos"
ANIMATION_CLIPS_DIR = DATASET_DIR / "animation_clips"
VRM_FILE_PATH = BASE_DIR / "eric1.vrm"

# Metadata Files
VALID_WORDS_PATH = DATASET_DIR / "valid_words.json"
SYNONYMS_MAP_PATH = DATASET_DIR / "complex_synonyms_map.json"
OFFLINE_TRANSLATION_DICT_PATH = DATASET_DIR / "offline_translation_dict.json"

# Models
FINGERSPELLING_MODEL_PATH = TRAINED_MODELS_DIR / "fingerspell_model.pth"
WORD_MODEL_PATH = TRAINED_MODELS_DIR / "word_model.pth"
WORD_LABELS_PATH = TRAINED_MODELS_DIR / "word_labels.json"
SENTENCE_MODEL_PATH = TRAINED_MODELS_DIR / "sentence_model.pth"

# Comma-separated browser origins permitted to call authenticated API routes.
# Local development remains usable without exposing credentialed API access to
# arbitrary origins. Deployments must set FRONTEND_ORIGINS explicitly.
FRONTEND_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
)

# Ensure trained models dir exists
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

# Supported Languages Mapping
# Code to Name mapping
SUPPORTED_LANGUAGES = {
    # Indian Regional Languages
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mni": "Manipuri",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "doi": "Dogri",
    "mai": "Maithili",
    "sat": "Santali",
    
    # Global/World Languages
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "fi": "Finnish",
    "no": "Norwegian",
    "da": "Danish",
    "he": "Hebrew",
    "fa": "Persian",
    "sw": "Swahili",
    "tl": "Filipino"
}
