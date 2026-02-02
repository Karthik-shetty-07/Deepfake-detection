"""
Configuration settings for the Deepfake Detection API.
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
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Video processing
FRAMES_TO_EXTRACT = 32  # Number of frames to analyze
FACE_SIZE = (224, 224)  # Face crop size for CNN
MIN_FACE_CONFIDENCE = 0.9

# Model settings
CNN_MODEL = "xception"  # Options: xception, efficientnet
USE_GPU = True  # Auto-fallback to CPU if unavailable

# Detection thresholds
THRESHOLDS = {
    "real": 0.50,           # < 50% = Real
    "low_confidence": 0.80,  # 50-80% = Low Confidence Deepfake
    "probable": 0.92,        # 80-92% = Most Probably Deepfake
    # > 92% = Definitely a Deepfake
}

# API settings
API_VERSION = "1.0.0"
API_TITLE = "Deepfake Detection API"
CORS_ORIGINS = ["*"]  # Configure for production

# Processing timeouts
MAX_PROCESSING_TIME = 60  # seconds
