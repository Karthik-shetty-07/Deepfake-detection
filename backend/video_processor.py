"""
Video Processor for Deepfake Detection.

Handles:
- Video frame extraction
- Face detection and cropping
- Frame sampling strategies
- Preprocessing pipeline
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass
import tempfile
import os

# Face detection using facenet-pytorch's MTCNN
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[Warning] facenet-pytorch not available, using OpenCV face detector")

# Alternative: MediaPipe face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class FaceData:
    """Container for extracted face data."""
    face_image: np.ndarray
    frame_index: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float


class VideoProcessor:
    """
    Video processing pipeline for deepfake detection.
    
    Extracts frames, detects faces, and prepares data
    for the ML models.
    """
    
    def __init__(
        self,
        frames_to_extract: int = 32,
        face_size: Tuple[int, int] = (224, 224),
        min_face_confidence: float = 0.9,
        device: str = "cpu"
    ):
        self.frames_to_extract = frames_to_extract
        self.face_size = face_size
        self.min_face_confidence = min_face_confidence
        self.device = device
        
        # Initialize face detector
        self._init_face_detector()
        
        print(f"[Video Processor] Initialized - extracting {frames_to_extract} frames")
    
    def _init_face_detector(self):
        """Initialize the face detection model."""
        if MTCNN_AVAILABLE:
            import torch
            device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            self.face_detector = MTCNN(
                image_size=160,
                margin=40,
                min_face_size=50,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device,
                keep_all=True  # Detect all faces
            )
            self.detector_type = "mtcnn"
        elif MEDIAPIPE_AVAILABLE:
            self.face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            self.detector_type = "mediapipe"
        else:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            self.detector_type = "opencv"
        
        print(f"[Video Processor] Using face detector: {self.detector_type}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0.0
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info
    
    def extract_frames(
        self,
        video_path: str,
        strategy: str = "uniform"
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video using specified strategy.
        
        Args:
            video_path: Path to video file
            strategy: "uniform" for evenly spaced, "keyframe" for scene changes
            
        Yields:
            Tuples of (frame_index, frame_image)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return
        
        # Determine which frames to extract
        if strategy == "uniform":
            # Evenly spaced frames
            if total_frames <= self.frames_to_extract:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames / self.frames_to_extract
                frame_indices = [int(i * step) for i in range(self.frames_to_extract)]
        else:
            # Default to uniform
            step = max(1, total_frames // self.frames_to_extract)
            frame_indices = list(range(0, total_frames, step))[:self.frames_to_extract]
        
        # Extract selected frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                yield frame_idx, frame
        
        cap.release()
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of (bbox, confidence) tuples
        """
        if self.detector_type == "mtcnn":
            return self._detect_faces_mtcnn(frame)
        elif self.detector_type == "mediapipe":
            return self._detect_faces_mediapipe(frame)
        else:
            return self._detect_faces_opencv(frame)
    
    def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces using MTCNN."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.face_detector.detect(rgb_frame)
        
        if boxes is None:
            return []
        
        results = []
        for box, prob in zip(boxes, probs):
            if prob >= self.min_face_confidence:
                x1, y1, x2, y2 = [int(b) for b in box]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                results.append((bbox, float(prob)))
        
        return results
    
    def _detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                
                if confidence >= self.min_face_confidence:
                    faces.append(((x, y, width, height), float(confidence)))
        
        return faces
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # OpenCV doesn't provide confidence, so use 1.0
        return [(tuple(face), 1.0) for face in faces]
    
    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.3
    ) -> np.ndarray:
        """
        Crop and resize face from frame with margin.
        
        Args:
            frame: Full frame image
            bbox: Face bounding box (x, y, w, h)
            margin: Margin around face as fraction
            
        Returns:
            Cropped and resized face image
        """
        h, w = frame.shape[:2]
        x, y, fw, fh = bbox
        
        # Add margin
        margin_x = int(fw * margin)
        margin_y = int(fh * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)
        
        # Crop face
        face = frame[y1:y2, x1:x2]
        
        # Resize to target size
        if face.size > 0:
            face = cv2.resize(face, self.face_size)
        else:
            face = np.zeros((self.face_size[1], self.face_size[0], 3), dtype=np.uint8)
        
        return face
    
    def process_video(self, video_path: str) -> List[FaceData]:
        """
        Complete video processing pipeline.
        
        Extracts frames, detects faces, and returns cropped faces
        with metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of FaceData objects
        """
        all_faces = []
        
        # Get video info
        info = self.get_video_info(video_path)
        print(f"[Video Processor] Processing video: {info['frame_count']} frames, "
              f"{info['duration']:.1f}s duration")
        
        # Extract and process frames
        for frame_idx, frame in self.extract_frames(video_path):
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Crop each detected face
            for bbox, confidence in faces:
                face_image = self.crop_face(frame, bbox)
                
                face_data = FaceData(
                    face_image=face_image,
                    frame_index=frame_idx,
                    bbox=bbox,
                    confidence=confidence
                )
                all_faces.append(face_data)
        
        print(f"[Video Processor] Extracted {len(all_faces)} faces from video")
        
        return all_faces
    
    def get_face_images(self, video_path: str) -> List[np.ndarray]:
        """
        Simple interface to get just the face images.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of face images
        """
        face_data_list = self.process_video(video_path)
        return [fd.face_image for fd in face_data_list]


def validate_video_file(file_path: str, max_size_mb: float = 20.0) -> Tuple[bool, str]:
    """
    Validate a video file for processing.
    
    Args:
        file_path: Path to video file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check file exists
    if not os.path.exists(file_path):
        return False, "File not found"
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB limit"
    
    # Check if it's a valid video
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return False, "Cannot open video file - invalid format"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if frame_count == 0:
        return False, "Video has no frames"
    
    if fps <= 0:
        return False, "Invalid video FPS"
    
    return True, f"Valid video: {frame_count} frames, {fps:.1f} FPS"
