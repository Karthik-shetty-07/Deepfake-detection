"""
Deepfake Detection API - Main FastAPI Application (Optimized for Render)

Provides endpoints for:
- Video upload and analysis
- Real-time processing status
- Health checks

Optimized with lazy loading to minimize memory footprint during deployment.
"""
import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles

from config import (
    UPLOAD_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS,
    API_TITLE, API_VERSION, CORS_ORIGINS, FRAMES_TO_EXTRACT,
    FACE_SIZE, MIN_FACE_CONFIDENCE, DEBUG
)
from video_processor import VideoProcessor, validate_video_file

logger = logging.getLogger(__name__)

# Upload streaming chunk size (64 KB)
_UPLOAD_CHUNK_SIZE = 64 * 1024


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisStatus(BaseModel):
    """Status of an analysis task."""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[Dict] = None
    created_at: str
    completed_at: Optional[str] = None


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    task_id: str
    final_score: float
    classification: str
    classification_label: str
    confidence: float
    component_scores: Dict[str, float]
    processing_time: float
    video_info: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    gpu_available: bool


# ============================================================================
# Global State
# ============================================================================

# Task storage (in production, use Redis or database)
tasks: Dict[str, AnalysisStatus] = {}

# Model registry - stores None until models are actually needed
_model_cache = {
    "cnn": None,
    "frequency": None,
    "temporal": None,
    "ensemble": None,
    "video_processor": None
}

# Track if GPU is available (checked once)
_gpu_available = None


def check_gpu_availability():
    """Check if GPU is available (cached result)."""
    global _gpu_available
    if _gpu_available is None:
        try:
            import torch
            _gpu_available = torch.cuda.is_available()
        except ImportError:
            _gpu_available = False
    return _gpu_available


def get_device():
    """Get device for model loading."""
    return "cuda" if check_gpu_availability() else "cpu"


# ============================================================================
# Lazy Model Loading Functions
# ============================================================================

def get_cnn_detector():
    """Lazy load CNN detector only when needed."""
    if _model_cache["cnn"] is None:
        logger.info("Loading CNN detector...")
        from models import CNNDetector
        _model_cache["cnn"] = CNNDetector(device=get_device())
        logger.info("CNN detector loaded")
    return _model_cache["cnn"]


def get_frequency_analyzer():
    """Lazy load frequency analyzer only when needed."""
    if _model_cache["frequency"] is None:
        logger.info("Loading frequency analyzer...")
        from models import FrequencyAnalyzer
        _model_cache["frequency"] = FrequencyAnalyzer()
        logger.info("Frequency analyzer loaded")
    return _model_cache["frequency"]


def get_temporal_model():
    """Lazy load temporal model only when needed."""
    if _model_cache["temporal"] is None:
        logger.info("Loading temporal model...")
        from models import TemporalModel
        _model_cache["temporal"] = TemporalModel(device=get_device())
        logger.info("Temporal model loaded")
    return _model_cache["temporal"]


def get_ensemble_detector():
    """Lazy load ensemble detector only when needed."""
    if _model_cache["ensemble"] is None:
        logger.info("Loading ensemble detector...")
        from models import EnsembleDetector
        _model_cache["ensemble"] = EnsembleDetector()
        logger.info("Ensemble detector loaded")
    return _model_cache["ensemble"]


def get_video_processor():
    """Lazy load video processor only when needed."""
    if _model_cache["video_processor"] is None:
        logger.info("Loading video processor...")
        _model_cache["video_processor"] = VideoProcessor(
            frames_to_extract=FRAMES_TO_EXTRACT,
            face_size=FACE_SIZE,
            min_face_confidence=MIN_FACE_CONFIDENCE,
            device=get_device()
        )
        logger.info("Video processor loaded")
    return _model_cache["video_processor"]


def cleanup_models():
    """Cleanup loaded models to free memory."""
    global _model_cache
    for key in _model_cache:
        if _model_cache[key] is not None:
            del _model_cache[key]
            _model_cache[key] = None

    # Force garbage collection
    import gc
    gc.collect()

    # Clear CUDA cache if available
    if check_gpu_availability():
        import torch
        torch.cuda.empty_cache()

    logger.info("Models cleaned up, memory freed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup - DO NOT load models, just prepare directories
    logger.info("Starting up - using lazy loading for models")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown - cleanup any loaded models
    logger.info("Shutting down...")
    cleanup_models()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-powered deepfake detection API using hybrid CNN + Frequency + Temporal analysis (Optimized)",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helpers
# ============================================================================

def _validate_uuid(value: str) -> bool:
    """Return True if *value* is a valid UUID v4 string."""
    try:
        uuid.UUID(value, version=4)
        return True
    except (ValueError, AttributeError):
        return False


def _safe_error_detail(message: str) -> str:
    """In production, strip potentially sensitive details from errors."""
    if DEBUG:
        return message
    # Only return the first sentence / a generic message
    return "An internal error occurred. Please try again."


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=any(model is not None for model in _model_cache.values()),
        gpu_available=check_gpu_availability()
    )


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and analyze a video for deepfake detection.

    - **file**: Video file (max 50MB, formats: mp4, avi, mov, mkv, webm)

    Returns a task ID that can be used to check status and get results.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Save uploaded file — stream in chunks to avoid loading the
    # entire upload into memory at once.
    file_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    try:
        total_written = 0
        async with aiofiles.open(file_path, 'wb') as out_file:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_FILE_SIZE_BYTES:
                    # Remove partial file before raising
                    await out_file.close()
                    cleanup_video(str(file_path))
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB"
                    )
                await out_file.write(chunk)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=_safe_error_detail(f"Failed to save file: {e}"))

    # Validate video file
    is_valid, message = validate_video_file(str(file_path))
    if not is_valid:
        cleanup_video(str(file_path))
        raise HTTPException(status_code=400, detail=message)

    # Create task status
    tasks[task_id] = AnalysisStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="Video uploaded, starting analysis...",
        created_at=datetime.now().isoformat()
    )

    # Start background processing
    background_tasks.add_task(process_video_task, task_id, str(file_path))

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Video uploaded successfully. Processing started."
    }


@app.get("/api/status/{task_id}", response_model=AnalysisStatus)
async def get_status(task_id: str):
    """
    Get the status of an analysis task.

    - **task_id**: Task ID returned from /api/analyze
    """
    if not _validate_uuid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """
    Get the full result of a completed analysis.

    - **task_id**: Task ID returned from /api/analyze
    """
    if not _validate_uuid(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format")

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]

    if task.status != "completed":
        return {
            "task_id": task_id,
            "status": task.status,
            "message": task.message,
            "result": None
        }

    return {
        "task_id": task_id,
        "status": "completed",
        "result": task.result
    }


@app.post("/api/cleanup")
async def cleanup_endpoint():
    """
    Manual endpoint to cleanup models and free memory.
    Useful after processing to reduce memory usage.
    """
    cleanup_models()
    return {"status": "success", "message": "Models cleaned up, memory freed"}


# ============================================================================
# Background Processing
# ============================================================================

async def process_video_task(task_id: str, video_path: str):
    """
    Background task to process a video.

    This runs the full analysis pipeline with lazy-loaded models:
    1. Extract frames and detect faces
    2. Run CNN analysis
    3. Run frequency analysis
    4. Run temporal analysis
    5. Ensemble fusion and classification
    """
    import time
    start_time = time.time()

    try:
        # Update status
        tasks[task_id].status = "processing"
        tasks[task_id].progress = 0.1
        tasks[task_id].message = "Extracting frames and detecting faces..."

        # Process video (lazy loads video processor if needed)
        video_processor = get_video_processor()
        video_info = video_processor.get_video_info(video_path)
        face_images = video_processor.get_face_images(video_path)

        if len(face_images) == 0:
            tasks[task_id].status = "failed"
            tasks[task_id].message = "No faces detected in video"
            cleanup_video(video_path)
            return

        tasks[task_id].progress = 0.3
        tasks[task_id].message = f"Analyzing {len(face_images)} faces with CNN..."

        # Run CNN analysis (lazy loads CNN detector if needed)
        cnn_detector = get_cnn_detector()
        cnn_result = cnn_detector.analyze_video_frames(face_images)

        tasks[task_id].progress = 0.5
        tasks[task_id].message = "Running frequency analysis..."

        # Run frequency analysis (lazy loads frequency analyzer if needed)
        freq_analyzer = get_frequency_analyzer()
        freq_result = freq_analyzer.analyze_video_frames(face_images)

        tasks[task_id].progress = 0.7
        tasks[task_id].message = "Analyzing temporal patterns..."

        # Run temporal analysis (lazy loads temporal model if needed)
        temp_model = get_temporal_model()
        if len(cnn_result.get("features", [])) > 0:
            temp_result = temp_model.analyze_video_features(cnn_result["features"])
        else:
            temp_result = {"mean_score": 0.5, "temporal_consistency": 1.0, "anomaly_count": 0}

        tasks[task_id].progress = 0.9
        tasks[task_id].message = "Computing final classification..."

        # Ensemble fusion (lazy loads ensemble detector if needed)
        ensemble = get_ensemble_detector()
        final_result = ensemble.analyze(cnn_result, freq_result, temp_result)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build result
        result = {
            "final_score": round(final_result.final_score * 100, 1),
            "classification": final_result.classification.value,
            "classification_label": final_result.classification_label,
            "confidence": round(final_result.confidence * 100, 1),
            "component_scores": {
                "cnn": round(final_result.cnn_score * 100, 1),
                "frequency": round(final_result.frequency_score * 100, 1),
                "temporal": round(final_result.temporal_score * 100, 1)
            },
            "processing_time": round(processing_time, 2),
            "frames_analyzed": len(face_images),
            "video_info": {
                "duration": round(video_info.get("duration", 0), 2),
                "frame_count": video_info.get("frame_count", 0),
                "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
            }
        }

        # Update task
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].message = "Analysis complete"
        tasks[task_id].result = result
        tasks[task_id].completed_at = datetime.now().isoformat()

        logger.info("Task %s completed in %.2fs", task_id, processing_time)

    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].message = f"Analysis failed: {str(e)}"
        logger.exception("Task %s failed", task_id)

    finally:
        # Cleanup
        cleanup_video(video_path)


def cleanup_video(video_path: str):
    """Remove uploaded video file."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        logger.warning("Failed to cleanup %s: %s", video_path, e)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)