"""Detector backends mirroring Jetson Nano modules."""

from .retina import RetinaDetector
from .mtcnn import MTcnnDetector

__all__ = ["RetinaDetector", "MTcnnDetector"]
