"""
Frequency Domain Analysis for Deepfake Detection.

Analyzes images in the frequency domain to detect:
- GAN-generated artifacts (characteristic frequency patterns)
- Unnatural compression artifacts
- Blending boundary inconsistencies
"""
import numpy as np
from scipy import fftpack
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict
import cv2


class FrequencyAnalyzer:
    """
    Frequency-based deepfake detector using FFT and DCT analysis.
    
    Key insights:
    - GANs leave characteristic fingerprints in frequency domain
    - Deepfakes often have unusual high-frequency patterns
    - Face swapping creates boundary artifacts visible in spectrum
    """
    
    def __init__(self):
        print("[Frequency Analyzer] Initialized")
    
    def compute_fft_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute FFT-based features from an image.
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary of frequency features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size for consistent analysis
        gray = cv2.resize(gray, (256, 256))
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log transform for better visualization and analysis
        magnitude_log = np.log1p(magnitude_spectrum)
        
        # Compute radial profile (azimuthally averaged power spectrum)
        radial_profile = self._compute_radial_profile(magnitude_log)
        
        # Extract features
        center = len(radial_profile) // 2
        
        # High frequency energy ratio
        low_freq = np.sum(radial_profile[:center // 2])
        mid_freq = np.sum(radial_profile[center // 2:center])
        high_freq = np.sum(radial_profile[center:])
        total_energy = low_freq + mid_freq + high_freq + 1e-10
        
        # Spectral entropy
        normalized_spectrum = magnitude_log / (np.sum(magnitude_log) + 1e-10)
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        
        # Peak detection for GAN artifacts
        peaks = self._detect_spectral_peaks(radial_profile)
        
        return {
            "high_freq_ratio": high_freq / total_energy,
            "mid_freq_ratio": mid_freq / total_energy,
            "low_freq_ratio": low_freq / total_energy,
            "spectral_entropy": spectral_entropy,
            "spectral_flatness": self._compute_spectral_flatness(radial_profile),
            "peak_count": len(peaks),
            "radial_profile": radial_profile.tolist()
        }
    
    def compute_dct_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute DCT-based features for GAN fingerprint detection.
        
        DCT is particularly effective at detecting JPEG artifacts
        and GAN-specific patterns.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Compute 2D DCT
        dct_result = fftpack.dct(fftpack.dct(gray.astype(np.float32).T, norm='ortho').T, norm='ortho')
        
        # Analyze DCT coefficients
        abs_dct = np.abs(dct_result)
        
        # Block-based DCT analysis (8x8 blocks like JPEG)
        block_energies = []
        for i in range(0, 256, 8):
            for j in range(0, 256, 8):
                block = abs_dct[i:i+8, j:j+8]
                # AC energy (exclude DC component)
                ac_energy = np.sum(block[1:, :]) + np.sum(block[0, 1:])
                dc_energy = block[0, 0]
                block_energies.append(ac_energy / (dc_energy + 1e-10))
        
        block_energies = np.array(block_energies)
        
        # DCT coefficient statistics
        return {
            "dct_ac_dc_ratio": float(np.mean(block_energies)),
            "dct_variance": float(np.var(block_energies)),
            "dct_high_freq_energy": float(np.sum(abs_dct[128:, :]) + np.sum(abs_dct[:, 128:])),
            "dct_sparsity": float(np.sum(abs_dct < 1.0) / abs_dct.size)
        }
    
    def analyze_face(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Perform complete frequency analysis on a face image.
        
        Args:
            face_image: BGR face crop
            
        Returns:
            Dictionary with all frequency features and fake score
        """
        # Get FFT features
        fft_features = self.compute_fft_features(face_image)
        
        # Get DCT features
        dct_features = self.compute_dct_features(face_image)
        
        # Compute anomaly score based on frequency analysis
        fake_score = self._compute_fake_score(fft_features, dct_features)
        
        return {
            "fft_features": fft_features,
            "dct_features": dct_features,
            "fake_score": fake_score
        }
    
    def analyze_video_frames(self, faces: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze all faces from video frames.
        
        Args:
            faces: List of face crops
            
        Returns:
            Aggregated frequency analysis results
        """
        if not faces:
            return {
                "mean_score": 0.5,
                "max_score": 0.5,
                "scores": [],
                "high_freq_anomaly": False
            }
        
        scores = []
        fft_high_freq_ratios = []
        
        for face in faces:
            result = self.analyze_face(face)
            scores.append(result["fake_score"])
            fft_high_freq_ratios.append(result["fft_features"]["high_freq_ratio"])
        
        scores_array = np.array(scores)
        
        # Detect if there's consistent high-frequency anomaly
        # (indicative of GAN artifacts)
        mean_high_freq = np.mean(fft_high_freq_ratios)
        high_freq_anomaly = mean_high_freq > 0.4  # Threshold for abnormal HF content
        
        return {
            "mean_score": float(np.mean(scores_array)),
            "max_score": float(np.max(scores_array)),
            "std_score": float(np.std(scores_array)),
            "scores": scores,
            "high_freq_anomaly": high_freq_anomaly,
            "mean_high_freq_ratio": float(mean_high_freq)
        }
    
    def _compute_radial_profile(self, image: np.ndarray) -> np.ndarray:
        """Compute the radially averaged profile of an image."""
        center = np.array(image.shape) // 2
        y, x = np.indices(image.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Radial binning
        r_max = int(np.sqrt(center[0]**2 + center[1]**2))
        radial_mean = np.zeros(r_max)
        
        for i in range(r_max):
            mask = r == i
            if np.any(mask):
                radial_mean[i] = np.mean(image[mask])
        
        return radial_mean
    
    def _compute_spectral_flatness(self, spectrum: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).
        Higher values indicate more noise-like (flat) spectrum.
        """
        spectrum = spectrum + 1e-10  # Avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean
    
    def _detect_spectral_peaks(self, profile: np.ndarray) -> List[int]:
        """Detect peaks in the radial profile that may indicate GAN artifacts."""
        # Smooth the profile
        smoothed = gaussian_filter(profile, sigma=3)
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                # Check if peak is significant
                if smoothed[i] > np.mean(smoothed) + np.std(smoothed):
                    peaks.append(i)
        
        return peaks
    
    def _compute_fake_score(self, fft_features: Dict, dct_features: Dict) -> float:
        """
        Compute a fake score based on frequency analysis.
        
        This is a heuristic-based score combining multiple indicators.
        Higher scores indicate more likelihood of being fake.
        """
        score = 0.5  # Start neutral
        
        # High frequency ratio indicator
        # Deepfakes often have unusual high-frequency patterns
        hf_ratio = fft_features["high_freq_ratio"]
        if hf_ratio > 0.35:
            score += 0.15
        elif hf_ratio < 0.15:
            score += 0.1  # Too smooth can also be suspicious
        
        # Spectral entropy
        # Natural images have characteristic entropy levels
        entropy = fft_features["spectral_entropy"]
        if entropy < 10 or entropy > 15:
            score += 0.1
        
        # DCT analysis
        # GAN images often have unusual AC/DC ratios
        ac_dc = dct_features["dct_ac_dc_ratio"]
        if ac_dc > 2.0 or ac_dc < 0.3:
            score += 0.1
        
        # Spectral flatness
        # Overly flat or peaked spectrum is suspicious
        flatness = fft_features["spectral_flatness"]
        if flatness > 0.8 or flatness < 0.2:
            score += 0.05
        
        # Peak count (GAN fingerprints)
        if fft_features["peak_count"] > 5:
            score += 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
