#!/usr/bin/env python3
"""
Face Feature Extraction Module - Python Implementation
Based on Face-Recognition-Jetson-Nano ArcFace implementation
Using deep learning models for face feature extraction

Created: 2025
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import os

class FaceFeatureExtractor:
    """Face feature extraction using ArcFace-like models"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize face feature extractor
        Args:
            model_path: Path to the feature extraction model
        """
        self.net = None
        self.feature_dim = 128  # Feature vector dimension
        self.input_size = (112, 112)  # Standard face input size for ArcFace
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("Warning: No feature extraction model provided. Using simple features.")
            self.use_simple_features = True
    
    def _load_model(self, model_path: str):
        """Load the feature extraction model"""
        try:
            # Try to load ONNX model first
            if model_path.endswith('.onnx'):
                self.net = cv2.dnn.readNetFromONNX(model_path)
            # Try Caffe model
            elif model_path.endswith('.caffemodel'):
                prototxt_path = model_path.replace('.caffemodel', '.prototxt')
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            # Try TensorFlow model
            elif model_path.endswith('.pb'):
                self.net = cv2.dnn.readNetFromTensorflow(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            print(f"Loaded feature extraction model: {model_path}")
            self.use_simple_features = False
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            print("Falling back to simple feature extraction")
            self.use_simple_features = True
    
    def extract_feature(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from face image
        Args:
            face_image: Cropped and aligned face image
        Returns:
            Feature vector as numpy array
        """
        if self.use_simple_features:
            return self._extract_simple_features(face_image)
        else:
            return self._extract_deep_features(face_image)
    
    def _extract_deep_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract features using deep learning model"""
        # Preprocess the face image
        face_processed = self._preprocess_face(face_image)
        
        # Create blob and run inference
        blob = cv2.dnn.blobFromImage(face_processed, 1.0/255.0, self.input_size, (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        feature = self.net.forward()
        
        # Normalize feature vector
        feature = feature.flatten()
        feature = self._normalize_feature(feature)
        
        return feature
    
    def _extract_simple_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract simple features as fallback (HOG-like features)
        This is a simplified version for demonstration
        """
        # Resize to standard size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert to grayscale
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
        
        # Extract histogram features from different regions
        features = []
        
        # Divide face into regions and extract histograms
        h, w = face_gray.shape
        regions = [
            (0, 0, w//2, h//2),        # Top-left
            (w//2, 0, w, h//2),        # Top-right
            (0, h//2, w//2, h),        # Bottom-left
            (w//2, h//2, w, h),        # Bottom-right
            (w//4, h//4, 3*w//4, 3*h//4)  # Center
        ]
        
        for x1, y1, x2, y2 in regions:
            region = face_gray[y1:y2, x1:x2]
            hist = cv2.calcHist([region], [0], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # Add some texture features
        # Sobel edges
        sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_hist_x = cv2.calcHist([np.uint8(np.abs(sobelx))], [0], None, [8], [0, 256])
        edge_hist_y = cv2.calcHist([np.uint8(np.abs(sobely))], [0], None, [8], [0, 256])
        
        features.extend(edge_hist_x.flatten())
        features.extend(edge_hist_y.flatten())
        
        # Normalize and pad/truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        features = self._normalize_feature(features)
        
        # Pad or truncate to desired dimension
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
        
        return features
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for feature extraction
        Args:
            face_image: Input face image
        Returns:
            Preprocessed face image
        """
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32)
        
        # Convert BGR to RGB if needed
        if len(face_normalized.shape) == 3:
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
        
        return face_normalized
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector (L2 normalization)
        Args:
            feature: Input feature vector
        Returns:
            Normalized feature vector
        """
        norm = np.linalg.norm(feature)
        if norm == 0:
            return feature
        return feature / norm
    
    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors
        Args:
            feature1: First feature vector
            feature2: Second feature vector
        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def align_face(self, face_image: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Align face based on landmarks
        Args:
            face_image: Input face image
            landmarks: Facial landmarks [(x1,y1), (x2,y2), ...]
        Returns:
            Aligned face image
        """
        if len(landmarks) < 2:
            # No landmarks available, just resize
            return cv2.resize(face_image, self.input_size)
        
        # Use eye positions for alignment
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        
        # Calculate center point
        center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        h, w = face_image.shape[:2]
        aligned = cv2.warpAffine(face_image, M, (w, h))
        
        # Resize to standard size
        aligned = cv2.resize(aligned, self.input_size)
        
        return aligned