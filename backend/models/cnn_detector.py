"""
CNN-based Deepfake Detector using XceptionNet / EfficientNet.

Supports two backbones:
- XceptionNet: Depthwise separable convolutions for manipulation artifacts
- EfficientNet-B4: Higher capacity with better generalization

OPTIMIZED: Models are loaded lazily only when needed.
Trained weights are automatically loaded from weights/ directory.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


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
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
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


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 model for deepfake detection.
    Higher capacity than XceptionNet with better generalization.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0
        )
        
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )
        
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
        return self.backbone(x)


class CNNDetector:
    """
    CNN-based deepfake detector with multiple backbone support.
    
    Analyzes individual frames for manipulation artifacts including:
    - Blending boundaries
    - Texture inconsistencies
    - Unnatural facial features
    - Compression artifacts specific to deepfakes
    
    Improvements:
    - Supports both XceptionNet and EfficientNet-B4 backbones
    - Automatic trained weights loading
    - Test-Time Augmentation (TTA) for sharper detection
    - BatchNorm in classifier head for better calibration
    """
    
    def __init__(self, device: Optional[str] = None, pretrained: bool = True, 
                 model_type: str = "xception"):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Model will be loaded lazily
        self.model = None
        self.pretrained = pretrained
        self.model_type = model_type
        
        # Input size depends on model
        self.input_size = 299 if model_type == "xception" else 380
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        # TTA transforms for sharper detection
        self.tta_transforms = [
            self.transform,  # Original
            transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            transforms.Compose([
                transforms.Resize((int(self.input_size * 1.1), int(self.input_size * 1.1))),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
        ]
        
        logger.info("Initialized (%s, lazy loading) on %s", model_type, self.device)
    
    def _ensure_model_loaded(self):
        """Load the model if not already loaded, including trained weights."""
        if self.model is None:
            logger.info("Loading %s model...", self.model_type)
            
            if self.model_type == "efficientnet":
                self.model = EfficientNetDetector(pretrained=self.pretrained)
            else:
                self.model = XceptionNet(pretrained=self.pretrained)
            
            self.model = self.model.to(self.device)
            
            # Try to load trained weights
            self._load_trained_weights()
            
            self.model.eval()
            logger.info("Model loaded successfully")
    
    def _load_trained_weights(self):
        """Attempt to load trained weights from the weights directory."""
        weights_dir = Path(__file__).parent.parent / "weights"
        weights_path = weights_dir / "trained_model.pth"
        
        if weights_path.exists():
            try:
                checkpoint = torch.load(weights_path, map_location=self.device,
                                       weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    acc = checkpoint.get('best_acc', 'unknown')
                    logger.info("Loaded trained weights (accuracy: %s)", acc)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                    logger.info("Loaded trained weights")
            except Exception as e:
                logger.warning("Could not load trained weights: %s", e)
                logger.info("Using ImageNet pre-trained weights")
        else:
            logger.info("No trained weights found, using ImageNet pre-trained")
    
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
    
    def preprocess_face_tta(self, face_image: np.ndarray) -> List[torch.Tensor]:
        """Preprocess with test-time augmentation for sharper results."""
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_rgb = face_image[:, :, ::-1]
        else:
            face_rgb = face_image
        
        pil_image = Image.fromarray(face_rgb.astype(np.uint8))
        
        tensors = [t(pil_image) for t in self.tta_transforms]
        return tensors
    
    @torch.no_grad()
    def analyze_frame(self, face_image: np.ndarray, use_tta: bool = False) -> Tuple[float, np.ndarray]:
        """
        Analyze a single face image for deepfake artifacts.
        
        Args:
            face_image: BGR face crop
            use_tta: Whether to use test-time augmentation
            
        Returns:
            Tuple of (fake_probability, feature_vector)
        """
        self._ensure_model_loaded()
        
        if use_tta:
            # TTA: average predictions across augmentations
            tta_tensors = self.preprocess_face_tta(face_image)
            all_probs = []
            
            for tensor in tta_tensors:
                tensor = tensor.unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs[0, 1].item())
            
            fake_prob = np.mean(all_probs)
            
            # Features from original
            tensor = tta_tensors[0].unsqueeze(0).to(self.device)
            features = self.model.get_features(tensor)
            feature_vec = features[0].cpu().numpy()
        else:
            tensor = self.preprocess_face(face_image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            features = self.model.get_features(tensor)
            
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
    
    def analyze_video_frames(self, faces: List[np.ndarray], batch_size: int = 8,
                             use_tta: bool = True) -> dict:
        """
        Analyze all faces extracted from a video.
        
        Uses TTA on a subset of frames for sharper detection while
        keeping processing time reasonable.
        
        Args:
            faces: List of face crops from video frames
            batch_size: Batch size for processing
            use_tta: Whether to use TTA on key frames
            
        Returns:
            Dictionary with analysis results
        """
        all_probs = []
        all_features = []
        
        # Process in batches (no TTA for speed)
        for i in range(0, len(faces), batch_size):
            batch = faces[i:i + batch_size]
            results = self.analyze_batch(batch)
            
            for prob, feat in results:
                all_probs.append(prob)
                all_features.append(feat)
        
        # TTA on key frames (first, middle, last, max-score)
        if use_tta and len(faces) >= 3:
            key_indices = [0, len(faces) // 2, len(faces) - 1]
            
            # Also add the frame with highest initial score
            if all_probs:
                max_idx = int(np.argmax(all_probs))
                if max_idx not in key_indices:
                    key_indices.append(max_idx)
            
            tta_probs = []
            for idx in key_indices:
                if idx < len(faces):
                    tta_prob, _ = self.analyze_frame(faces[idx], use_tta=True)
                    tta_probs.append(tta_prob)
            
            # Blend TTA results with batch results
            if tta_probs:
                tta_mean = np.mean(tta_probs)
                batch_mean = np.mean(all_probs) if all_probs else 0.5
                # Weighted: TTA results are more reliable
                blended_mean = 0.4 * batch_mean + 0.6 * tta_mean
            else:
                blended_mean = np.mean(all_probs) if all_probs else 0.5
        else:
            blended_mean = np.mean(all_probs) if all_probs else 0.5
        
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
        
        # Use blended mean that includes TTA for sharper detection
        return {
            "mean_score": float(blended_mean),
            "max_score": float(np.max(probs_array)),
            "min_score": float(np.min(probs_array)),
            "std_score": float(np.std(probs_array)),
            "median_score": float(np.median(probs_array)),
            "p90_score": float(np.percentile(probs_array, 90)),
            "frame_scores": all_probs,
            "features": features_array
        }