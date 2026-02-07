"""
CNN-based Deepfake Detector using XceptionNet.

XceptionNet is highly effective for deepfake detection due to its
depthwise separable convolutions that capture manipulation artifacts.

OPTIMIZED: Models are loaded lazily only when needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class XceptionNet(nn.Module):
    """
    XceptionNet model adapted for binary deepfake classification.
    Uses pre-trained weights and adds a custom classification head.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load pre-trained Xception from timm
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classification head for deepfake detection
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # [real, fake] logits
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)


class CNNDetector:
    """
    CNN-based deepfake detector using XceptionNet.
    
    Analyzes individual frames for manipulation artifacts including:
    - Blending boundaries
    - Texture inconsistencies
    - Unnatural facial features
    - Compression artifacts specific to deepfakes
    
    OPTIMIZED: Model is loaded lazily on first use.
    """
    
    def __init__(self, device: Optional[str] = None, pretrained: bool = True):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Model will be loaded lazily
        self.model = None
        self.pretrained = pretrained
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Xception input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        print(f"[CNN Detector] Initialized (lazy loading enabled) on {self.device}")
    
    def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self.model is None:
            print(f"[CNN Detector] Loading XceptionNet model...")
            self.model = XceptionNet(pretrained=self.pretrained)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[CNN Detector] Model loaded successfully")
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face image for the model.
        
        Args:
            face_image: BGR face crop from OpenCV
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_rgb = face_image[:, :, ::-1]
        else:
            face_rgb = face_image
        
        # Convert to PIL
        pil_image = Image.fromarray(face_rgb.astype(np.uint8))
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    @torch.no_grad()
    def analyze_frame(self, face_image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Analyze a single face image for deepfake artifacts.
        
        Args:
            face_image: BGR face crop
            
        Returns:
            Tuple of (fake_probability, feature_vector)
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Preprocess
        tensor = self.preprocess_face(face_image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        
        # Extract features for ensemble
        features = self.model.get_features(tensor)
        
        # Return fake probability and features
        fake_prob = probs[0, 1].item()
        feature_vec = features[0].cpu().numpy()
        
        return fake_prob, feature_vec
    
    @torch.no_grad()
    def analyze_batch(self, face_images: List[np.ndarray]) -> List[Tuple[float, np.ndarray]]:
        """
        Analyze a batch of face images efficiently.
        
        Args:
            face_images: List of BGR face crops
            
        Returns:
            List of (fake_probability, feature_vector) tuples
        """
        if not face_images:
            return []
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Preprocess all images
        tensors = [self.preprocess_face(img) for img in face_images]
        batch = torch.stack(tensors).to(self.device)
        
        # Get predictions
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        
        # Extract features
        features = self.model.get_features(batch)
        
        # Collect results
        results = []
        for i in range(len(face_images)):
            fake_prob = probs[i, 1].item()
            feature_vec = features[i].cpu().numpy()
            results.append((fake_prob, feature_vec))
        
        return results
    
    def analyze_video_frames(self, faces: List[np.ndarray], batch_size: int = 8) -> dict:
        """
        Analyze all faces extracted from a video.
        
        Args:
            faces: List of face crops from video frames
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with analysis results
        """
        all_probs = []
        all_features = []
        
        # Process in batches
        for i in range(0, len(faces), batch_size):
            batch = faces[i:i + batch_size]
            results = self.analyze_batch(batch)
            
            for prob, feat in results:
                all_probs.append(prob)
                all_features.append(feat)
        
        if not all_probs:
            return {
                "mean_score": 0.5,
                "max_score": 0.5,
                "min_score": 0.5,
                "std_score": 0.0,
                "frame_scores": [],
                "features": np.array([])
            }
        
        probs_array = np.array(all_probs)
        features_array = np.stack(all_features) if all_features else np.array([])
        
        return {
            "mean_score": float(np.mean(probs_array)),
            "max_score": float(np.max(probs_array)),
            "min_score": float(np.min(probs_array)),
            "std_score": float(np.std(probs_array)),
            "frame_scores": all_probs,
            "features": features_array
        }