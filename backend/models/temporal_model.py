"""
Temporal Consistency Model for Deepfake Detection.

Uses Transformer architecture to analyze temporal patterns in video:
- Facial expression consistency
- Blinking patterns
- Movement continuity
- Frame-to-frame coherence

OPTIMIZED: Model is loaded lazily only when needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class TemporalTransformer(nn.Module):
    """
    Transformer model for temporal analysis of face sequences.
    
    Detects temporal inconsistencies in deepfake videos:
    - Unnatural transitions between frames
    - Inconsistent facial expressions
    - Abnormal blinking patterns
    """
    
    def __init__(
        self,
        input_dim: int = 2048,  # Feature dimension from CNN
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [real, fake]
        )
        
        # Temporal difference analyzer
        self.diff_analyzer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (classification_logits, temporal_anomaly_scores)
        """
        # Project input
        x = self.input_projection(x)  # [batch, seq, d_model]
        
        # Transpose for transformer: [seq, batch, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)  # [seq, batch, d_model]
        
        # Global pooling for classification
        pooled = encoded.mean(dim=0)  # [batch, d_model]
        logits = self.classifier(pooled)  # [batch, 2]
        
        # Compute temporal differences for anomaly detection
        encoded_t = encoded.transpose(0, 1)  # [batch, seq, d_model]
        diffs = encoded_t[:, 1:, :] - encoded_t[:, :-1, :]  # [batch, seq-1, d_model]
        anomaly_scores = self.diff_analyzer(diffs).squeeze(-1)  # [batch, seq-1]
        
        return logits, anomaly_scores


class TemporalModel:
    """
    Temporal consistency analyzer for deepfake detection.
    
    Analyzes sequences of face features to detect:
    - Temporal inconsistencies
    - Unnatural transitions
    - Missing/abnormal blink patterns
    
    OPTIMIZED: Model is loaded lazily on first use.
    """
    
    def __init__(self, input_dim: int = 2048, device: Optional[str] = None):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Model will be loaded lazily
        self.model = None
        self.input_dim = input_dim
        
        # Statistics for feature normalization
        self.feature_mean = None
        self.feature_std = None
        
        print(f"[Temporal Model] Initialized (lazy loading enabled) on {self.device}")
    
    def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self.model is None:
            print(f"[Temporal Model] Loading Transformer model...")
            self.model = TemporalTransformer(input_dim=self.input_dim)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[Temporal Model] Model loaded successfully")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for better model performance."""
        if self.feature_mean is None:
            self.feature_mean = np.mean(features, axis=(0, 1), keepdims=True)
            self.feature_std = np.std(features, axis=(0, 1), keepdims=True) + 1e-8
        
        return (features - self.feature_mean) / self.feature_std
    
    @torch.no_grad()
    def analyze_sequence(self, features: np.ndarray) -> Dict[str, any]:
        """
        Analyze a sequence of frame features.
        
        Args:
            features: Feature array [seq_len, feature_dim]
            
        Returns:
            Dictionary with temporal analysis results
        """
        if len(features) < 2:
            return {
                "fake_score": 0.5,
                "temporal_consistency": 1.0,
                "anomaly_frames": []
            }
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Normalize features
        features_norm = self.normalize_features(features[np.newaxis, ...])
        
        # Convert to tensor
        tensor = torch.FloatTensor(features_norm).to(self.device)
        
        # Get predictions
        logits, anomaly_scores = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        
        fake_prob = probs[0, 1].item()
        anomaly_scores = anomaly_scores[0].cpu().numpy()
        
        # Compute temporal consistency (inverse of anomaly)
        temporal_consistency = 1.0 - np.mean(anomaly_scores)
        
        # Find anomalous frames
        anomaly_threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
        anomaly_frames = np.where(anomaly_scores > anomaly_threshold)[0].tolist()
        
        return {
            "fake_score": fake_prob,
            "temporal_consistency": float(temporal_consistency),
            "anomaly_frames": anomaly_frames,
            "anomaly_scores": anomaly_scores.tolist()
        }
    
    def analyze_with_statistical_features(self, features: np.ndarray) -> Dict[str, any]:
        """
        Combine model-based and statistical temporal analysis.
        
        Adds heuristic-based features that are effective for deepfake detection.
        """
        # Get model-based analysis
        model_result = self.analyze_sequence(features)
        
        # Compute additional statistical features
        stat_features = self._compute_statistical_features(features)
        
        # Combine scores (weighted average)
        combined_score = (
            0.6 * model_result["fake_score"] +
            0.4 * stat_features["statistical_score"]
        )
        
        return {
            **model_result,
            "statistical_features": stat_features,
            "combined_score": float(combined_score)
        }
    
    def analyze_video_features(self, features: np.ndarray) -> Dict[str, any]:
        """
        Full temporal analysis for video features.
        
        Args:
            features: Feature array from CNN [num_frames, feature_dim]
            
        Returns:
            Complete temporal analysis results
        """
        if len(features) == 0:
            return {
                "mean_score": 0.5,
                "temporal_consistency": 1.0,
                "anomaly_count": 0
            }
        
        result = self.analyze_with_statistical_features(features)
        
        return {
            "mean_score": result["combined_score"],
            "model_score": result["fake_score"],
            "temporal_consistency": result["temporal_consistency"],
            "anomaly_count": len(result["anomaly_frames"]),
            "anomaly_frames": result["anomaly_frames"],
            "statistical_score": result["statistical_features"]["statistical_score"]
        }
    
    def _compute_statistical_features(self, features: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical features for temporal analysis.
        
        These hand-crafted features capture temporal patterns
        that are effective for deepfake detection.
        """
        if len(features) < 2:
            return {"statistical_score": 0.5}
        
        # Frame-to-frame differences
        diffs = np.diff(features, axis=0)
        diff_norms = np.linalg.norm(diffs, axis=1)
        
        # Statistics of temporal differences
        mean_diff = np.mean(diff_norms)
        std_diff = np.std(diff_norms)
        max_diff = np.max(diff_norms)
        
        # Coefficient of variation (normalized measure of variability)
        cv = std_diff / (mean_diff + 1e-8)
        
        # Sudden change detection
        sudden_changes = np.sum(diff_norms > (mean_diff + 3 * std_diff))
        
        # Autocorrelation (smoothness of changes)
        if len(diff_norms) > 1:
            autocorr = np.corrcoef(diff_norms[:-1], diff_norms[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Compute heuristic fake score
        score = 0.5
        
        # High variability is suspicious
        if cv > 0.5:
            score += 0.1
        
        # Sudden changes are suspicious
        if sudden_changes > 2:
            score += min(0.2, sudden_changes * 0.05)
        
        # Very low autocorrelation (erratic movements) is suspicious
        if autocorr < 0.3:
            score += 0.1
        
        # Very high autocorrelation (unnaturally smooth) is also suspicious
        if autocorr > 0.95:
            score += 0.05
        
        return {
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "max_diff": float(max_diff),
            "cv": float(cv),
            "sudden_changes": int(sudden_changes),
            "autocorrelation": float(autocorr),
            "statistical_score": float(max(0.0, min(1.0, score)))
        }