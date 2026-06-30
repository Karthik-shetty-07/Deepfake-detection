"""
Deepfake Detection ML Models Package

Lazy-loaded to minimize startup time and memory on free-tier hosting.
Classes are only imported when first accessed via __getattr__.
"""


def __getattr__(name):
    """Lazy import model classes only when accessed."""
    if name == "CNNDetector":
        from .cnn_detector import CNNDetector
        return CNNDetector
    elif name == "FrequencyAnalyzer":
        from .frequency_analyzer import FrequencyAnalyzer
        return FrequencyAnalyzer
    elif name == "TemporalModel":
        from .temporal_model import TemporalModel
        return TemporalModel
    elif name == "EnsembleDetector":
        from .ensemble import EnsembleDetector
        return EnsembleDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CNNDetector",
    "FrequencyAnalyzer",
    "TemporalModel",
    "EnsembleDetector",
]
