"""
Deepfake Detection API - Main FastAPI Application

Provides endpoints for:
- Video upload and analysis
- Real-time processing status
- Health checks
"""
import os
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles

from config import (
    UPLOAD_DIR, MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS,
    API_TITLE, API_VERSION, CORS_ORIGINS, FRAMES_TO_EXTRACT,
    FACE_SIZE, MIN_FACE_CONFIDENCE
)
from video_processor import VideoProcessor, validate_video_file
from models import CNNDetector, FrequencyAnalyzer, TemporalModel, EnsembleDetector


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

# Model instances (lazy loaded)
models = {
    "cnn": None,
    "frequency": None,
    "temporal": None,
    "ensemble": None,
    "video_processor": None
}


def load_models():
    """Load all ML models."""
    global models
    
    if models["cnn"] is None:
        print("[API] Loading models...")
        
        # Check GPU availability
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[API] Using device: {device}")
        
        models["cnn"] = CNNDetector(device=device)
        models["frequency"] = FrequencyAnalyzer()
        models["temporal"] = TemporalModel(device=device)
        models["ensemble"] = EnsembleDetector()
        models["video_processor"] = VideoProcessor(
            frames_to_extract=FRAMES_TO_EXTRACT,
            face_size=FACE_SIZE,
            min_face_confidence=MIN_FACE_CONFIDENCE,
            device=device
        )
        
        print("[API] All models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    load_models()
    yield
    # Shutdown
    print("[API] Shutting down...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="AI-powered deepfake detection API using hybrid CNN + Frequency + Temporal analysis",
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
# API Endpoints
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    import torch
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=models["cnn"] is not None,
        gpu_available=torch.cuda.is_available()
    )


@app.post("/api/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and analyze a video for deepfake detection.
    
    - **file**: Video file (max 20MB, formats: mp4, avi, mov, mkv, webm)
    
    Returns a task ID that can be used to check status and get results.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size from content-length if available
    # (actual size check happens after upload)
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    
    try:
        # Stream file to disk
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB"
                )
            
            await out_file.write(content)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Validate video file
    is_valid, message = validate_video_file(str(file_path))
    if not is_valid:
        os.remove(file_path)
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
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks[task_id]


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """
    Get the full result of a completed analysis.
    
    - **task_id**: Task ID returned from /api/analyze
    """
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


# ============================================================================
# Background Processing
# ============================================================================

async def process_video_task(task_id: str, video_path: str):
    """
    Background task to process a video.
    
    This runs the full analysis pipeline:
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
        
        # Process video
        video_processor = models["video_processor"]
        video_info = video_processor.get_video_info(video_path)
        face_images = video_processor.get_face_images(video_path)
        
        if len(face_images) == 0:
            tasks[task_id].status = "failed"
            tasks[task_id].message = "No faces detected in video"
            cleanup_video(video_path)
            return
        
        tasks[task_id].progress = 0.3
        tasks[task_id].message = f"Analyzing {len(face_images)} faces with CNN..."
        
        # Run CNN analysis
        cnn_detector = models["cnn"]
        cnn_result = cnn_detector.analyze_video_frames(face_images)
        
        tasks[task_id].progress = 0.5
        tasks[task_id].message = "Running frequency analysis..."
        
        # Run frequency analysis
        freq_analyzer = models["frequency"]
        freq_result = freq_analyzer.analyze_video_frames(face_images)
        
        tasks[task_id].progress = 0.7
        tasks[task_id].message = "Analyzing temporal patterns..."
        
        # Run temporal analysis
        temp_model = models["temporal"]
        if len(cnn_result.get("features", [])) > 0:
            temp_result = temp_model.analyze_video_features(cnn_result["features"])
        else:
            temp_result = {"mean_score": 0.5, "temporal_consistency": 1.0, "anomaly_count": 0}
        
        tasks[task_id].progress = 0.9
        tasks[task_id].message = "Computing final classification..."
        
        # Ensemble fusion
        ensemble = models["ensemble"]
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
        
        print(f"[API] Task {task_id} completed in {processing_time:.2f}s")
        
    except Exception as e:
        tasks[task_id].status = "failed"
        tasks[task_id].message = f"Analysis failed: {str(e)}"
        print(f"[API] Task {task_id} failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_video(video_path)


def cleanup_video(video_path: str):
    """Remove uploaded video file."""
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        print(f"[API] Failed to cleanup {video_path}: {e}")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
