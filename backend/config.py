"""
Configuration settings for the Deepfake Detection API.
Optimized for sharper detection.

All deployment-sensitive settings are read from environment variables
with safe defaults for local development.
"""
import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "weights"

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# File constraints
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------------------------------------------------------------------------
# Video processing – more frames = better temporal analysis
# ---------------------------------------------------------------------------
FRAMES_TO_EXTRACT = 48
FACE_SIZE = (224, 224)
MIN_FACE_CONFIDENCE = 0.8  # Lowered to catch more faces

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
CNN_MODEL = "xception"  # Options: xception, efficientnet
USE_GPU = True  # Auto-fallback to CPU if unavailable

# ---------------------------------------------------------------------------
# Detection thresholds (tighter for sharper detection)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "real": 0.45,            # < 45% = Real
    "low_confidence": 0.72,  # 45-72% = Low Confidence Deepfake
    "probable": 0.88,        # 72-88% = Most Probably Deepfake
    # > 88% = Definitely a Deepfake
}

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_VERSION = "2.0.0"
API_TITLE = "Deepfake Detection API"
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# CORS – default "*" for dev; set a comma-separated list in production
_cors_env = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",")] if _cors_env != "*" else ["*"]

# ---------------------------------------------------------------------------
# Processing timeouts
# ---------------------------------------------------------------------------
MAX_PROCESSING_TIME = 120  # seconds (heavier analysis)
