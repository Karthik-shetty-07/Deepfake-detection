"""
Configuration settings for the Deepfake Detection API.
Optimized for sharper detection.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "weights"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# File constraints
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Video processing - more frames = better temporal analysis
FRAMES_TO_EXTRACT = 48
FACE_SIZE = (224, 224)
MIN_FACE_CONFIDENCE = 0.8  # Lowered to catch more faces

# Model settings
CNN_MODEL = "xception"  # Options: xception, efficientnet
USE_GPU = True  # Auto-fallback to CPU if unavailable

# Detection thresholds (tighter for sharper detection)
THRESHOLDS = {
    "real": 0.45,            # < 45% = Real
    "low_confidence": 0.72,  # 45-72% = Low Confidence Deepfake
    "probable": 0.88,        # 72-88% = Most Probably Deepfake
    # > 88% = Definitely a Deepfake
}

# API settings
API_VERSION = "2.0.0"
API_TITLE = "Deepfake Detection API"
CORS_ORIGINS = ["*"]  # Configure for production

# Processing timeouts
MAX_PROCESSING_TIME = 120  # seconds (heavier analysis)

