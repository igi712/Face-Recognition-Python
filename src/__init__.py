#!/usr/bin/env python3
"""
Python Face Recognition System
Init file for the src package

Created: 2025
"""

from .face_detector import FaceDetector, FaceObject
from .face_features import FaceFeatureExtractor
from .face_database import FaceDatabase
from .face_quality import FaceQualityAssessment
from .config_manager import ConfigManager

__version__ = "1.0.0"
__author__ = "Face Recognition Python Team"

__all__ = [
    'FaceDetector',
    'FaceObject', 
    'FaceFeatureExtractor',
    'FaceDatabase',
    'FaceQualityAssessment',
    'ConfigManager'
]