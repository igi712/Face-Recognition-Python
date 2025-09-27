"""Quality assessment components mirroring Jetson Nano stack."""

from .blur import BlurAnalyzer
from .live import LiveDetector

__all__ = ["BlurAnalyzer", "LiveDetector"]
