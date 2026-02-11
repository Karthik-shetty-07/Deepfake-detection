"""
Ensemble Detector for Deepfake Detection.

Combines outputs from:
- CNN (XceptionNet/EfficientNet) - Frame-level artifacts
- Frequency Analyzer - FFT/DCT/Channel patterns
- Temporal Model - Sequence consistency

Improved with:
- Adaptive weight adjustment based on signal strength
- Percentile-based scoring for robustness
- Better calibration curves
- Multi-indicator boosting
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DetectionResult(Enum):
    """Classification categories based on confidence thresholds."""
    REAL = "real"
    LOW_CONFIDENCE = "low_confidence_deepfake"
    PROBABLE = "most_probably_deepfake"
    DEFINITE = "definitely_deepfake"


@dataclass
class AnalysisResult:
    """Complete analysis result from the ensemble detector."""
    final_score: float
    classification: DetectionResult
    classification_label: str
    cnn_score: float
    frequency_score: float
    temporal_score: float
    confidence: float
    details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "final_score": self.final_score,
            "classification": self.classification.value,
            "classification_label": self.classification_label,
            "confidence": self.confidence,
            "component_scores": {
                "cnn": self.cnn_score,
                "frequency": self.frequency_score,
                "temporal": self.temporal_score
            },
            "details": self.details
        }


class EnsembleDetector:
    """
    Ensemble deepfake detector combining multiple analysis methods.
    
    Improved fusion strategy:
    1. Collect + calibrate scores from all components
    2. Adaptive weight adjustment based on signal reliability
    3. Multi-indicator boosting for high-confidence detection
    4. Percentile-based robustness
    5. Sharper thresholds with better separation
    """
    
    # Default weights (tuned for balanced detection)
    DEFAULT_WEIGHTS = {
        "cnn": 0.50,        # CNN is most reliable for general detection
        "frequency": 0.25,  # Frequency catches GAN artifacts
        "temporal": 0.25    # Temporal catches video-specific issues
    }
    
    # Detection thresholds - tighter for sharper classification
    THRESHOLDS = {
        "real": 0.45,            # < 45% = Real
        "low_confidence": 0.72,  # 45-72% = Low Confidence Deepfake
        "probable": 0.88,        # 72-88% = Most Probably Deepfake
        # > 88% = Definitely a Deepfake
    }
    
    LABELS = {
        DetectionResult.REAL: "✅ Real - This appears to be an authentic video",
        DetectionResult.LOW_CONFIDENCE: "⚠️ Low Confidence Deepfake - Some suspicious patterns detected",
        DetectionResult.PROBABLE: "🔶 Most Probably Deepfake - Strong manipulation indicators found",
        DetectionResult.DEFINITE: "🔴 Definitely a Deepfake - Clear evidence of manipulation"
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"[Ensemble Detector] Initialized with weights: {self.weights}")
    
    def fuse_scores(
        self,
        cnn_result: Dict,
        frequency_result: Dict,
        temporal_result: Dict
    ) -> Tuple[float, Dict]:
        """Fuse scores from all components with adaptive weighting."""
        
        # Extract primary scores
        cnn_score = cnn_result.get("mean_score", 0.5)
        freq_score = frequency_result.get("mean_score", 0.5)
        temp_score = temporal_result.get("mean_score", 0.5)
        
        # Calibrate scores
        cnn_calibrated = self._calibrate_cnn_score(cnn_score, cnn_result)
        freq_calibrated = self._calibrate_freq_score(freq_score, frequency_result)
        temp_calibrated = self._calibrate_temp_score(temp_score, temporal_result)
        
        # Adaptive weight adjustment
        weights = self._compute_adaptive_weights(
            cnn_calibrated, freq_calibrated, temp_calibrated,
            cnn_result, frequency_result, temporal_result
        )
        
        # Weighted fusion
        fused_score = (
            weights["cnn"] * cnn_calibrated +
            weights["frequency"] * freq_calibrated +
            weights["temporal"] * temp_calibrated
        )
        
        # Multi-indicator boosting
        strong_indicators = sum([
            cnn_calibrated > 0.70,
            freq_calibrated > 0.65,
            temp_calibrated > 0.65
        ])
        
        if strong_indicators >= 3:
            fused_score = min(1.0, fused_score * 1.15)  # 15% boost for unanimous
        elif strong_indicators >= 2:
            fused_score = min(1.0, fused_score * 1.08)  # 8% boost
        
        # Suppression: if all components say real, push down
        weak_indicators = sum([
            cnn_calibrated < 0.35,
            freq_calibrated < 0.40,
            temp_calibrated < 0.40
        ])
        
        if weak_indicators >= 3:
            fused_score = fused_score * 0.85  # Suppress false positives
        
        # Handle edge cases
        fused_score = self._handle_edge_cases(
            fused_score, cnn_result, frequency_result, temporal_result
        )
        
        details = {
            "cnn_raw": cnn_score,
            "cnn_calibrated": cnn_calibrated,
            "frequency_raw": freq_score,
            "frequency_calibrated": freq_calibrated,
            "temporal_raw": temp_score,
            "temporal_calibrated": temp_calibrated,
            "strong_indicators": strong_indicators,
            "adaptive_weights": weights,
        }
        
        return fused_score, details
    
    def classify(self, score: float) -> Tuple[DetectionResult, str]:
        """Classify based on score thresholds."""
        if score < self.THRESHOLDS["real"]:
            result = DetectionResult.REAL
        elif score < self.THRESHOLDS["low_confidence"]:
            result = DetectionResult.LOW_CONFIDENCE
        elif score < self.THRESHOLDS["probable"]:
            result = DetectionResult.PROBABLE
        else:
            result = DetectionResult.DEFINITE
        
        return result, self.LABELS[result]
    
    def analyze(
        self,
        cnn_result: Dict,
        frequency_result: Dict,
        temporal_result: Dict
    ) -> AnalysisResult:
        """Complete analysis combining all components."""
        fused_score, details = self.fuse_scores(
            cnn_result, frequency_result, temporal_result
        )
        
        classification, label = self.classify(fused_score)
        
        confidence = self._compute_confidence(
            fused_score, cnn_result, frequency_result, temporal_result
        )
        
        return AnalysisResult(
            final_score=fused_score,
            classification=classification,
            classification_label=label,
            cnn_score=cnn_result.get("mean_score", 0.5),
            frequency_score=frequency_result.get("mean_score", 0.5),
            temporal_score=temporal_result.get("mean_score", 0.5),
            confidence=confidence,
            details=details
        )
    
    def _compute_adaptive_weights(
        self,
        cnn_cal: float, freq_cal: float, temp_cal: float,
        cnn_result: Dict, frequency_result: Dict, temporal_result: Dict
    ) -> Dict[str, float]:
        """
        Compute adaptive weights based on signal reliability.
        Components with higher confidence get more weight.
        """
        weights = self.weights.copy()
        
        # Boost CNN weight if it has low variance (consistent across frames)
        cnn_std = cnn_result.get("std_score", 0.5)
        if cnn_std < 0.1:
            weights["cnn"] *= 1.2  # Consistent CNN = more reliable
        elif cnn_std > 0.3:
            weights["cnn"] *= 0.85  # High variance = less reliable
        
        # Boost frequency weight if it detects clear anomalies
        if frequency_result.get("high_freq_anomaly", False):
            weights["frequency"] *= 1.3
        
        # Boost temporal weight if enough frames were analyzed
        anomaly_count = temporal_result.get("anomaly_count", 0)
        if anomaly_count > 3:
            weights["temporal"] *= 1.2
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _calibrate_cnn_score(self, score: float, result: Dict) -> float:
        """Calibrate CNN score with percentile-based features."""
        # High variance suggests inconsistent manipulation
        if "std_score" in result and result["std_score"] > 0.15:
            score = min(1.0, score + 0.05)
        
        # P90 score: if top percentile is very high, boost
        p90 = result.get("p90_score", score)
        if p90 > 0.85:
            score = max(score, p90 * 0.75)
        
        # Max score: if max is very high, it's a strong signal
        max_score = result.get("max_score", score)
        if max_score > 0.9:
            score = max(score, max_score * 0.7)
        
        # Median vs mean divergence: suggests outlier manipulation frames  
        median = result.get("median_score", score)
        if abs(score - median) > 0.15:
            score = min(1.0, score + 0.03)
        
        return score
    
    def _calibrate_freq_score(self, score: float, result: Dict) -> float:
        """Calibrate frequency score with new features."""
        if result.get("high_freq_anomaly", False):
            score = min(1.0, score + 0.12)
        
        # Channel inconsistency is a strong signal
        chan_incon = result.get("mean_channel_inconsistency", 0.0)
        if chan_incon > 0.05:
            score = min(1.0, score + min(0.12, chan_incon * 1.5))
        
        return score
    
    def _calibrate_temp_score(self, score: float, result: Dict) -> float:
        """Calibrate temporal score."""
        consistency = result.get("temporal_consistency", 1.0)
        
        if consistency < 0.6:
            score = min(1.0, score + 0.12)
        elif consistency < 0.75:
            score = min(1.0, score + 0.06)
        
        anomaly_count = result.get("anomaly_count", 0)
        if anomaly_count > 3:
            score = min(1.0, score + min(0.15, anomaly_count * 0.03))
        
        return score
    
    def _handle_edge_cases(
        self, score: float,
        cnn_result: Dict, frequency_result: Dict, temporal_result: Dict
    ) -> float:
        """Handle edge cases in score fusion."""
        num_frames = len(cnn_result.get("frame_scores", [1]))
        
        if num_frames < 5:
            score = 0.3 * 0.5 + 0.7 * score
        elif num_frames < 10:
            score = 0.1 * 0.5 + 0.9 * score
        
        return max(0.0, min(1.0, score))
    
    def _compute_confidence(
        self, score: float,
        cnn_result: Dict, frequency_result: Dict, temporal_result: Dict
    ) -> float:
        """
        Compute confidence in the detection result.
        Higher when components agree and score is clearly in a category.
        """
        scores = [
            cnn_result.get("mean_score", 0.5),
            frequency_result.get("mean_score", 0.5),
            temporal_result.get("mean_score", 0.5)
        ]
        agreement = 1.0 - np.std(scores) * 2
        
        # Distance from nearest threshold
        threshold_distances = [abs(score - t) for t in self.THRESHOLDS.values()]
        min_dist = min(threshold_distances)
        clarity = min(1.0, min_dist * 5)  # Further from threshold = more clear
        
        # Frame count factor
        num_frames = len(cnn_result.get("frame_scores", [1]))
        frame_factor = min(1.0, num_frames / 15)
        
        confidence = (0.35 * max(0, agreement) + 0.35 * clarity + 0.30 * frame_factor)
        
        return max(0.0, min(1.0, confidence))
