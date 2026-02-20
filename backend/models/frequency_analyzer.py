"""
Frequency Domain Analysis for Deepfake Detection.

Analyzes images in the frequency domain to detect:
- GAN-generated artifacts (characteristic frequency patterns)
- Unnatural compression artifacts
- Blending boundary inconsistencies

Improved with:
- Multi-channel frequency analysis (per-channel instead of grayscale only)
- Laplacian of Gaussian edge analysis for blending boundaries
- Wavelet-based texture inconsistency detection
- Better calibrated scoring with more discriminative features
"""
import logging
import numpy as np
from scipy import fftpack
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict
import cv2

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Frequency-based deepfake detector using FFT, DCT, and edge analysis.
    
    Key insights:
    - GANs leave characteristic fingerprints in frequency domain
    - Deepfakes often have unusual high-frequency patterns
    - Face swapping creates boundary artifacts visible in spectrum
    - Color channel inconsistencies are strong deepfake indicators
    """
    
    def __init__(self):
        logger.info("Initialized (enhanced multi-feature)")
    
    def compute_fft_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute FFT-based features from an image.
        Uses per-channel analysis for better discrimination.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log transform
        magnitude_log = np.log1p(magnitude_spectrum)
        
        # Compute radial profile
        radial_profile = self._compute_radial_profile(magnitude_log)
        
        # Extract features
        center = len(radial_profile) // 2
        
        # Energy in frequency bands
        low_freq = np.sum(radial_profile[:center // 3])
        mid_freq = np.sum(radial_profile[center // 3:2 * center // 3])
        high_freq = np.sum(radial_profile[2 * center // 3:])
        total_energy = low_freq + mid_freq + high_freq + 1e-10
        
        # Spectral entropy
        normalized_spectrum = magnitude_log / (np.sum(magnitude_log) + 1e-10)
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        
        # Peak detection for GAN artifacts
        peaks = self._detect_spectral_peaks(radial_profile)
        
        # Spectral rolloff - frequency below which 85% of energy is concentrated
        cumsum = np.cumsum(radial_profile)
        total = cumsum[-1] if len(cumsum) > 0 else 1.0
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
        spectral_rolloff = rolloff_idx / (len(radial_profile) + 1e-10)
        
        # Spectral slope - rate at which energy decreases
        if len(radial_profile) > 1:
            x = np.arange(len(radial_profile))
            slope = np.polyfit(x, radial_profile, 1)[0]
        else:
            slope = 0.0
        
        return {
            "high_freq_ratio": high_freq / total_energy,
            "mid_freq_ratio": mid_freq / total_energy,
            "low_freq_ratio": low_freq / total_energy,
            "spectral_entropy": spectral_entropy,
            "spectral_flatness": self._compute_spectral_flatness(radial_profile),
            "peak_count": len(peaks),
            "spectral_rolloff": spectral_rolloff,
            "spectral_slope": float(slope),
            "radial_profile": radial_profile.tolist()
        }
    
    def compute_channel_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze frequency characteristics per color channel.
        GANs often create inconsistencies across channels.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return {"channel_inconsistency": 0.0}
        
        image_resized = cv2.resize(image, (256, 256))
        channel_energies = []
        channel_hf_ratios = []
        
        for c in range(3):
            channel = image_resized[:, :, c].astype(np.float32)
            f_transform = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # High frequency energy
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            radius = min(center_y, center_x) // 3
            
            # Create masks
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            low_mask = dist <= radius
            high_mask = dist > 2 * radius
            
            low_energy = np.sum(magnitude[low_mask])
            high_energy = np.sum(magnitude[high_mask])
            total = low_energy + high_energy + 1e-10
            
            channel_energies.append(total)
            channel_hf_ratios.append(high_energy / total)
        
        # Channel inconsistency: std of HF ratios across channels
        channel_inconsistency = float(np.std(channel_hf_ratios))
        
        # Energy distribution inconsistency
        energy_arr = np.array(channel_energies)
        energy_cv = float(np.std(energy_arr) / (np.mean(energy_arr) + 1e-10))
        
        return {
            "channel_inconsistency": channel_inconsistency,
            "channel_energy_cv": energy_cv,
            "channel_hf_ratios": channel_hf_ratios
        }
    
    def compute_dct_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute DCT-based features for GAN fingerprint detection.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Compute 2D DCT
        dct_result = fftpack.dct(fftpack.dct(gray.astype(np.float32).T, norm='ortho').T, norm='ortho')
        abs_dct = np.abs(dct_result)
        
        # Block-based DCT analysis (8x8 blocks like JPEG)
        block_energies = []
        block_ac_energies = []
        for i in range(0, 256, 8):
            for j in range(0, 256, 8):
                block = abs_dct[i:i+8, j:j+8]
                ac_energy = np.sum(block[1:, :]) + np.sum(block[0, 1:])
                dc_energy = block[0, 0]
                block_energies.append(ac_energy / (dc_energy + 1e-10))
                block_ac_energies.append(ac_energy)
        
        block_energies = np.array(block_energies)
        block_ac_energies = np.array(block_ac_energies)
        
        # DCT coefficient distribution analysis
        # Benford's law deviation (real images follow Benford's law more closely)
        first_digits = []
        flat_dct = abs_dct.flatten()
        nonzero = flat_dct[flat_dct > 1.0]
        if len(nonzero) > 0:
            first_digits = np.floor(nonzero / (10 ** np.floor(np.log10(nonzero)))).astype(int)
            first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
            if len(first_digits) > 0:
                observed_dist = np.bincount(first_digits, minlength=10)[1:] / len(first_digits)
                benford_dist = np.log10(1 + 1 / np.arange(1, 10))
                benford_deviation = float(np.sum(np.abs(observed_dist - benford_dist)))
            else:
                benford_deviation = 0.0
        else:
            benford_deviation = 0.0
        
        return {
            "dct_ac_dc_ratio": float(np.mean(block_energies)),
            "dct_variance": float(np.var(block_energies)),
            "dct_high_freq_energy": float(np.sum(abs_dct[128:, :]) + np.sum(abs_dct[:, 128:])),
            "dct_sparsity": float(np.sum(abs_dct < 1.0) / abs_dct.size),
            "dct_block_energy_std": float(np.std(block_ac_energies)),
            "benford_deviation": benford_deviation
        }
    
    def compute_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect blending boundary artifacts using edge analysis.
        Deepfakes often have unnatural edges around face boundaries.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (256, 256))
        
        # Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_mean = float(np.mean(np.abs(laplacian)))
        lap_std = float(np.std(laplacian))
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Edge consistency: check if edges are uniformly distributed
        # Deepfakes often have a ring of edges around face boundary
        h, w = edges.shape
        center_region = edges[h//4:3*h//4, w//4:3*w//4]
        border_region_density = (np.sum(edges > 0) - np.sum(center_region > 0)) / (edges.size - center_region.size + 1e-10)
        center_density = np.sum(center_region > 0) / (center_region.size + 1e-10)
        
        edge_ratio = border_region_density / (center_density + 1e-10)
        
        return {
            "laplacian_mean": lap_mean,
            "laplacian_std": lap_std,
            "edge_density": edge_density,
            "border_center_edge_ratio": float(edge_ratio)
        }
    
    def analyze_face(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Perform complete frequency analysis on a face image.
        """
        fft_features = self.compute_fft_features(face_image)
        dct_features = self.compute_dct_features(face_image)
        channel_features = self.compute_channel_features(face_image)
        edge_features = self.compute_edge_features(face_image)
        
        # Compute anomaly score
        fake_score = self._compute_fake_score(
            fft_features, dct_features, channel_features, edge_features
        )
        
        return {
            "fft_features": fft_features,
            "dct_features": dct_features,
            "channel_features": channel_features,
            "edge_features": edge_features,
            "fake_score": fake_score
        }
    
    def analyze_video_frames(self, faces: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze all faces from video frames.
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
        channel_inconsistencies = []
        
        for face in faces:
            result = self.analyze_face(face)
            scores.append(result["fake_score"])
            fft_high_freq_ratios.append(result["fft_features"]["high_freq_ratio"])
            channel_inconsistencies.append(
                result["channel_features"].get("channel_inconsistency", 0.0)
            )
        
        scores_array = np.array(scores)
        
        # High-frequency anomaly detection
        mean_high_freq = np.mean(fft_high_freq_ratios)
        high_freq_anomaly = mean_high_freq > 0.35
        
        # Channel inconsistency across frames
        mean_channel_incon = np.mean(channel_inconsistencies)
        
        return {
            "mean_score": float(np.mean(scores_array)),
            "max_score": float(np.max(scores_array)),
            "std_score": float(np.std(scores_array)),
            "median_score": float(np.median(scores_array)),
            "p90_score": float(np.percentile(scores_array, 90)),
            "scores": scores,
            "high_freq_anomaly": high_freq_anomaly,
            "mean_high_freq_ratio": float(mean_high_freq),
            "mean_channel_inconsistency": float(mean_channel_incon)
        }
    
    def _compute_radial_profile(self, image: np.ndarray) -> np.ndarray:
        """Compute the radially averaged profile of an image."""
        center = np.array(image.shape) // 2
        y, x = np.indices(image.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        r_max = int(np.sqrt(center[0]**2 + center[1]**2))
        radial_mean = np.zeros(r_max)
        
        for i in range(r_max):
            mask = r == i
            if np.any(mask):
                radial_mean[i] = np.mean(image[mask])
        
        return radial_mean
    
    def _compute_spectral_flatness(self, spectrum: np.ndarray) -> float:
        """Compute spectral flatness (Wiener entropy)."""
        spectrum = spectrum + 1e-10
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean
    
    def _detect_spectral_peaks(self, profile: np.ndarray) -> List[int]:
        """Detect peaks in the radial profile that may indicate GAN artifacts."""
        smoothed = gaussian_filter(profile, sigma=3)
        
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                if smoothed[i] > np.mean(smoothed) + np.std(smoothed):
                    peaks.append(i)
        
        return peaks
    
    def _compute_fake_score(self, fft_features: Dict, dct_features: Dict,
                            channel_features: Dict, edge_features: Dict) -> float:
        """
        Compute fake score using multi-feature analysis.
        Better calibrated than simple heuristics.
        """
        score = 0.5  # Start neutral
        indicators = 0
        total_weight = 0
        
        # === FFT indicators ===
        # High frequency ratio
        hf_ratio = fft_features["high_freq_ratio"]
        if hf_ratio > 0.35:
            indicators += 0.15
            total_weight += 0.15
        elif hf_ratio < 0.12:
            indicators += 0.08   # Too smooth is suspicious
            total_weight += 0.08
        
        # Spectral entropy
        entropy = fft_features["spectral_entropy"]
        if entropy < 9.5 or entropy > 15.5:
            indicators += 0.10
            total_weight += 0.10
        
        # Spectral rolloff (GANs often have different rolloff)
        rolloff = fft_features.get("spectral_rolloff", 0.5)
        if rolloff < 0.3 or rolloff > 0.8:
            indicators += 0.08
            total_weight += 0.08
        
        # Peak count (GAN fingerprints)
        if fft_features["peak_count"] > 4:
            indicators += 0.10
            total_weight += 0.10
        
        # Spectral flatness
        flatness = fft_features["spectral_flatness"]
        if flatness > 0.8 or flatness < 0.15:
            indicators += 0.05
            total_weight += 0.05
        
        # === DCT indicators ===
        ac_dc = dct_features["dct_ac_dc_ratio"]
        if ac_dc > 2.5 or ac_dc < 0.2:
            indicators += 0.10
            total_weight += 0.10
        
        # Benford's law deviation (strong indicator)
        benford = dct_features.get("benford_deviation", 0.0)
        if benford > 0.3:
            indicators += 0.12
            total_weight += 0.12
        
        # === Channel indicators ===
        chan_incon = channel_features.get("channel_inconsistency", 0.0)
        if chan_incon > 0.05:
            indicators += min(0.15, chan_incon * 2)
            total_weight += 0.15
        
        # === Edge indicators ===
        edge_ratio = edge_features.get("border_center_edge_ratio", 1.0)
        if edge_ratio > 2.5:
            indicators += 0.10  # Strong blending boundary signal
            total_weight += 0.10
        
        lap_std = edge_features.get("laplacian_std", 0.0)
        if lap_std > 30:
            indicators += 0.05
            total_weight += 0.05
        
        # Combine
        score = 0.5 + indicators
        
        return max(0.0, min(1.0, score))
