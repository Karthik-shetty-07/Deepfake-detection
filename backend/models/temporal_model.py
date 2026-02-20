"""
Temporal Consistency Model for Deepfake Detection.

Uses Transformer architecture to analyze temporal patterns in video:
- Facial expression consistency
- Blinking patterns
- Movement continuity
- Frame-to-frame coherence

Improved with:
- Better positional encoding
- Optical flow-inspired difference features
- More robust statistical analysis
- Weighted temporal scoring
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import math

logger = logging.getLogger(__name__)


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
        return x + self.pe[:x.size(0)]


class TemporalTransformer(nn.Module):
    """
    Transformer model for temporal analysis of face sequences.
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
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
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
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
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x, mask=mask)
        
        # Global pooling for classification
        pooled = encoded.mean(dim=0)
        logits = self.classifier(pooled)
        
        # Temporal differences for anomaly detection
        encoded_t = encoded.transpose(0, 1)
        diffs = encoded_t[:, 1:, :] - encoded_t[:, :-1, :]
        anomaly_scores = self.diff_analyzer(diffs).squeeze(-1)
        
        return logits, anomaly_scores


class TemporalModel:
    """
    Temporal consistency analyzer for deepfake detection.
    
    Analyzes sequences of face features to detect:
    - Temporal inconsistencies
    - Unnatural transitions
    - Missing/abnormal blink patterns
    
    Improved with better statistical features and weighted scoring.
    """
    
    def __init__(self, input_dim: int = 2048, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.input_dim = input_dim
        self.feature_mean = None
        self.feature_std = None
        
        logger.info("Initialized (lazy loading) on %s", self.device)
    
    def _ensure_model_loaded(self):
        if self.model is None:
            logger.info("Loading Transformer model...")
            self.model = TemporalTransformer(input_dim=self.input_dim)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None:
            self.feature_mean = np.mean(features, axis=(0, 1), keepdims=True)
            self.feature_std = np.std(features, axis=(0, 1), keepdims=True) + 1e-8
        return (features - self.feature_mean) / self.feature_std
    
    @torch.no_grad()
    def analyze_sequence(self, features: np.ndarray) -> Dict[str, any]:
        if len(features) < 2:
            return {
                "fake_score": 0.5,
                "temporal_consistency": 1.0,
                "anomaly_frames": []
            }
        
        self._ensure_model_loaded()
        
        features_norm = self.normalize_features(features[np.newaxis, ...])
        tensor = torch.FloatTensor(features_norm).to(self.device)
        
        logits, anomaly_scores = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        
        fake_prob = probs[0, 1].item()
        anomaly_scores = anomaly_scores[0].cpu().numpy()
        
        temporal_consistency = 1.0 - np.mean(anomaly_scores)
        
        # Find anomalous frames using adaptive thresholding
        if len(anomaly_scores) > 2:
            anomaly_threshold = np.mean(anomaly_scores) + 1.5 * np.std(anomaly_scores)
        else:
            anomaly_threshold = 0.7
        anomaly_frames = np.where(anomaly_scores > anomaly_threshold)[0].tolist()
        
        return {
            "fake_score": fake_prob,
            "temporal_consistency": float(temporal_consistency),
            "anomaly_frames": anomaly_frames,
            "anomaly_scores": anomaly_scores.tolist()
        }
    
    def analyze_with_statistical_features(self, features: np.ndarray) -> Dict[str, any]:
        """Combine model-based and statistical temporal analysis."""
        model_result = self.analyze_sequence(features)
        stat_features = self._compute_statistical_features(features)
        
        # Weighted combination
        combined_score = (
            0.5 * model_result["fake_score"] +
            0.5 * stat_features["statistical_score"]
        )
        
        return {
            **model_result,
            "statistical_features": stat_features,
            "combined_score": float(combined_score)
        }
    
    def analyze_video_features(self, features: np.ndarray) -> Dict[str, any]:
        """Full temporal analysis for video features."""
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
        Enhanced statistical features for temporal analysis.
        """
        if len(features) < 2:
            return {"statistical_score": 0.5}
        
        # Frame-to-frame differences
        diffs = np.diff(features, axis=0)
        diff_norms = np.linalg.norm(diffs, axis=1)
        
        # Basic statistics
        mean_diff = np.mean(diff_norms)
        std_diff = np.std(diff_norms)
        max_diff = np.max(diff_norms)
        
        # Coefficient of variation
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
        
        # Jitter: second-order differences (acceleration)
        if len(features) >= 3:
            second_diffs = np.diff(features, n=2, axis=0)
            jitter = float(np.mean(np.linalg.norm(second_diffs, axis=1)))
        else:
            jitter = 0.0
        
        # Feature-space trajectory smoothness
        if len(features) >= 3:
            # Cosine similarity between consecutive difference vectors
            cos_sims = []
            for i in range(len(diffs) - 1):
                d1 = diffs[i]
                d2 = diffs[i + 1]
                norm1 = np.linalg.norm(d1)
                norm2 = np.linalg.norm(d2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_sim = np.dot(d1, d2) / (norm1 * norm2)
                    cos_sims.append(cos_sim)
            trajectory_smoothness = float(np.mean(cos_sims)) if cos_sims else 0.0
        else:
            trajectory_smoothness = 0.0
        
        # Compute heuristic fake score
        score = 0.5
        
        # High variability is suspicious
        if cv > 0.5:
            score += 0.10
        elif cv > 0.35:
            score += 0.05
        
        # Sudden changes are suspicious
        if sudden_changes > 2:
            score += min(0.20, sudden_changes * 0.05)
        elif sudden_changes > 0:
            score += 0.05
        
        # Very low autocorrelation (erratic) is suspicious
        if autocorr < 0.2:
            score += 0.10
        
        # Very high autocorrelation (unnaturally smooth) is suspicious
        if autocorr > 0.95:
            score += 0.08
        
        # High jitter suggests frame-level manipulation
        if jitter > 0 and mean_diff > 0:
            jitter_ratio = jitter / (mean_diff + 1e-8)
            if jitter_ratio > 1.5:
                score += 0.08
        
        # Low trajectory smoothness = erratic direction changes
        if trajectory_smoothness < 0.0:
            score += 0.08
        
        return {
            "mean_diff": float(mean_diff),
            "std_diff": float(std_diff),
            "max_diff": float(max_diff),
            "cv": float(cv),
            "sudden_changes": int(sudden_changes),
            "autocorrelation": float(autocorr),
            "jitter": float(jitter),
            "trajectory_smoothness": float(trajectory_smoothness),
            "statistical_score": float(max(0.0, min(1.0, score)))
        }