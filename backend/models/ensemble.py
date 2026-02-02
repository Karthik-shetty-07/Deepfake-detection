"""
Ensemble Detector for Deepfake Detection.

Combines outputs from:
- CNN (XceptionNet) - Frame-level artifacts
- Frequency Analyzer - FFT/DCT patterns
- Temporal Model - Sequence consistency

Uses weighted fusion with learned calibration.
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
    
    Fusion strategy:
    1. Collect scores from CNN, Frequency, and Temporal analyzers
    2. Apply component-specific calibration
    3. Weighted fusion with learned/tuned weights
    4. Post-processing for edge cases
    5. Classification based on thresholds
    """
    
    # Default weights (tuned for balanced detection)
    DEFAULT_WEIGHTS = {
        "cnn": 0.50,        # CNN is most reliable for general detection
        "frequency": 0.25,  # Frequency analysis catches GAN artifacts
        "temporal": 0.25    # Temporal catches video-specific issues
    }
    
    # Detection thresholds
    THRESHOLDS = {
        "real": 0.50,           # < 50% = Real
        "low_confidence": 0.80,  # 50-80% = Low Confidence Deepfake
        "probable": 0.92,        # 80-92% = Most Probably Deepfake
        # > 92% = Definitely a Deepfake
    }
    
    # Classification labels for display
    LABELS = {
        DetectionResult.REAL: "✅ Real - This appears to be an authentic video",
        DetectionResult.LOW_CONFIDENCE: "⚠️ Low Confidence Deepfake - Some suspicious patterns detected",
        DetectionResult.PROBABLE: "🔶 Most Probably Deepfake - Strong manipulation indicators found",
        DetectionResult.DEFINITE: "🔴 Definitely a Deepfake - Clear evidence of manipulation"
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ensemble detector.
        
        Args:
            weights: Optional custom weights for component fusion
        """
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        
        # Ensure weights are normalized
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"[Ensemble Detector] Initialized with weights: {self.weights}")
    
    def fuse_scores(
        self,
        cnn_result: Dict,
        frequency_result: Dict,
        temporal_result: Dict
    ) -> Tuple[float, Dict]:
        """
        Fuse scores from all components.
        
        Args:
            cnn_result: Output from CNN detector
            frequency_result: Output from frequency analyzer
            temporal_result: Output from temporal model
            
        Returns:
            Tuple of (fused_score, details_dict)
        """
        # Extract primary scores
        cnn_score = cnn_result.get("mean_score", 0.5)
        freq_score = frequency_result.get("mean_score", 0.5)
        temp_score = temporal_result.get("mean_score", 0.5)
        
        # Apply score calibration
        cnn_calibrated = self._calibrate_cnn_score(cnn_score, cnn_result)
        freq_calibrated = self._calibrate_freq_score(freq_score, frequency_result)
        temp_calibrated = self._calibrate_temp_score(temp_score, temporal_result)
        
        # Weighted fusion
        fused_score = (
            self.weights["cnn"] * cnn_calibrated +
            self.weights["frequency"] * freq_calibrated +
            self.weights["temporal"] * temp_calibrated
        )
        
        # Post-processing: boost score if multiple strong indicators
        strong_indicators = sum([
            cnn_calibrated > 0.75,
            freq_calibrated > 0.70,
            temp_calibrated > 0.70
        ])
        
        if strong_indicators >= 2:
            fused_score = min(1.0, fused_score * 1.1)  # 10% boost
        
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
            "cnn_details": cnn_result,
            "frequency_details": frequency_result,
            "temporal_details": temporal_result
        }
        
        return fused_score, details
    
    def classify(self, score: float) -> Tuple[DetectionResult, str]:
        """
        Classify based on score thresholds.
        
        Args:
            score: Fused detection score [0, 1]
            
        Returns:
            Tuple of (DetectionResult, label_string)
        """
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
        """
        Complete analysis combining all components.
        
        Args:
            cnn_result: Output from CNN detector
            frequency_result: Output from frequency analyzer
            temporal_result: Output from temporal model
            
        Returns:
            AnalysisResult with complete detection information
        """
        # Fuse scores
        fused_score, details = self.fuse_scores(
            cnn_result, frequency_result, temporal_result
        )
        
        # Classify
        classification, label = self.classify(fused_score)
        
        # Compute confidence (how certain we are about the result)
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
    
    def _calibrate_cnn_score(self, score: float, result: Dict) -> float:
        """
        Calibrate CNN score based on additional signals.
        
        Adjusts score based on:
        - Score variance across frames
        - Max score (worst case)
        """
        # High variance suggests inconsistent manipulation
        if "std_score" in result and result["std_score"] > 0.15:
            score = min(1.0, score + 0.05)
        
        # If max score is very high, boost overall
        if "max_score" in result and result["max_score"] > 0.9:
            score = max(score, result["max_score"] * 0.8)
        
        return score
    
    def _calibrate_freq_score(self, score: float, result: Dict) -> float:
        """
        Calibrate frequency score.
        
        High-frequency anomalies are strong indicators.
        """
        if result.get("high_freq_anomaly", False):
            score = min(1.0, score + 0.15)
        
        return score
    
    def _calibrate_temp_score(self, score: float, result: Dict) -> float:
        """
        Calibrate temporal score.
        
        Low temporal consistency is a red flag.
        """
        consistency = result.get("temporal_consistency", 1.0)
        
        if consistency < 0.7:
            score = min(1.0, score + 0.1)
        
        # Many anomaly frames is suspicious
        anomaly_count = result.get("anomaly_count", 0)
        if anomaly_count > 3:
            score = min(1.0, score + min(0.15, anomaly_count * 0.03))
        
        return score
    
    def _handle_edge_cases(
        self,
        score: float,
        cnn_result: Dict,
        frequency_result: Dict,
        temporal_result: Dict
    ) -> float:
        """
        Handle edge cases in score fusion.
        
        - Very short videos get uncertainty penalty
        - Single-frame analysis is less reliable
        """
        # Check if we have enough data
        num_frames = len(cnn_result.get("frame_scores", [1]))
        
        if num_frames < 5:
            # Push towards uncertainty for very short videos
            score = 0.3 * 0.5 + 0.7 * score
        
        return max(0.0, min(1.0, score))
    
    def _compute_confidence(
        self,
        score: float,
        cnn_result: Dict,
        frequency_result: Dict,
        temporal_result: Dict
    ) -> float:
        """
        Compute confidence in the detection result.
        
        Higher confidence when:
        - Components agree with each other
        - Score is clearly in a category (not near thresholds)
        - Enough frames were analyzed
        """
        # Component agreement
        scores = [
            cnn_result.get("mean_score", 0.5),
            frequency_result.get("mean_score", 0.5),
            temporal_result.get("mean_score", 0.5)
        ]
        agreement = 1.0 - np.std(scores) * 2  # Lower std = higher agreement
        
        # Distance from thresholds
        threshold_distances = [
            abs(score - t) for t in self.THRESHOLDS.values()
        ]
        clarity = min(threshold_distances) * 10  # Closer to threshold = less clear
        
        # Number of frames
        num_frames = len(cnn_result.get("frame_scores", [1]))
        frame_factor = min(1.0, num_frames / 20)  # More frames = more confidence
        
        # Combine factors
        confidence = (0.4 * agreement + 0.3 * (1.0 - clarity) + 0.3 * frame_factor)
        
        return max(0.0, min(1.0, confidence))
