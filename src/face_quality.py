#!/usr/bin/env python3
"""
Face Quality Assessment Module - Python Implementation
Based on Face-Recognition-Jetson-Nano project
Includes blur detection and liveness detection

Created: 2025
"""

import cv2
import numpy as np
from typing import Tuple

class FaceQualityAssessment:
    """Face quality assessment for blur and liveness detection"""

    def __init__(self, fast_mode: bool = False):
        """Initialize face quality assessment"""
        self.fast_mode = fast_mode
        self.blur_threshold = -25.0  # More positive = sharper image
        self.liveness_threshold = 0.75 if fast_mode else 0.93

        # Simple liveness detector (in practice, you'd use a trained model)
        self.use_simple_liveness = True
        self._analysis_max_side = 96 if fast_mode else 160
        self._lbp_stride = 2 if fast_mode else 1
    
    def assess_blur(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Assess face image blur using Laplacian variance
        Args:
            face_image: Input face image
        Returns:
            (blur_score, is_sharp) - Higher score = sharper image
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Convert to log scale (similar to original implementation)
        blur_score = np.log(laplacian_var) if laplacian_var > 0 else -100
        
        is_sharp = blur_score > self.blur_threshold
        
        return blur_score, is_sharp
    
    def assess_liveness(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Assess face liveness (anti-spoofing)
        Args:
            face_image: Input face image
        Returns:
            (liveness_score, is_live) - Higher score = more likely to be live
        """
        if self.use_simple_liveness:
            return self._simple_liveness_detection(face_image)
        else:
            # In practice, you would use a trained anti-spoofing model
            return self._model_liveness_detection(face_image)
    
    def _simple_liveness_detection(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Simple liveness detection based on image properties
        This is a basic implementation - real systems use trained models
        """
        # Convert to different color spaces for analysis
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        else:
            gray = face_image
            hsv = cv2.cvtColor(cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        if self.fast_mode:
            return self._fast_liveness_detection(gray, hsv)

        # Feature 1: Texture analysis
        # Calculate local binary patterns
        lbp_score = self._calculate_lbp_uniformity(gray)
        
        # Feature 2: Color distribution analysis
        color_score = self._analyze_color_distribution(hsv)
        
        # Feature 3: Edge density
        edge_score = self._calculate_edge_density(gray)
        
        # Feature 4: Frequency domain analysis
        freq_score = self._analyze_frequency_domain(gray)
        
        # Combine features (simple weighted average)
        weights = [0.3, 0.2, 0.25, 0.25]
        liveness_score = (
            weights[0] * lbp_score + 
            weights[1] * color_score + 
            weights[2] * edge_score + 
            weights[3] * freq_score
        )
        
        is_live = liveness_score > self.liveness_threshold
        
        return liveness_score, is_live
    
    def _calculate_lbp_uniformity(self, gray_image: np.ndarray) -> float:
        """Calculate local binary pattern uniformity"""
        gray_work = self._resize_for_analysis(gray_image)
        h, w = gray_work.shape
        if h < 16 or w < 16:
            return 0.5  # Default score for small images
        
        # Simple LBP-like calculation
        patterns = []
        for y in range(1, h - 1, self._lbp_stride):
            for x in range(1, w - 1, self._lbp_stride):
                center = gray_work[y, x]
                pattern = 0
                
                # 8-neighborhood
                neighbors = [
                    gray_work[y-1, x-1], gray_work[y-1, x], gray_work[y-1, x+1],
                    gray_work[y, x+1], gray_work[y+1, x+1], gray_work[y+1, x],
                    gray_work[y+1, x-1], gray_work[y, x-1]
                ]
                
                for i, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << i)
                
                patterns.append(pattern)
        
        if len(patterns) == 0:
            return 0.5
        
        # Calculate uniformity (simplified)
        unique_patterns = len(set(patterns))
        uniformity = 1.0 - (unique_patterns / len(patterns))
        
        return uniformity
    
    def _analyze_color_distribution(self, hsv_image: np.ndarray) -> float:
        """Analyze color distribution for skin-like colors"""
        # Extract hue and saturation channels
        h_channel = hsv_image[:, :, 0]
        s_channel = hsv_image[:, :, 1]
        
        # Define skin hue range (approximate)
        skin_hue_low, skin_hue_high = 0, 30  # Red-orange range
        
        # Count pixels in skin hue range
        skin_mask = (h_channel >= skin_hue_low) & (h_channel <= skin_hue_high)
        skin_ratio = np.sum(skin_mask) / (hsv_image.shape[0] * hsv_image.shape[1])
        
        # Analyze saturation distribution
        sat_mean = np.mean(s_channel)
        sat_std = np.std(s_channel)
        
        # Combine metrics
        color_score = (skin_ratio * 0.6) + (sat_mean / 255.0 * 0.2) + (min(sat_std / 50.0, 1.0) * 0.2)
        
        return min(color_score, 1.0)
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density"""
        gray_work = self._resize_for_analysis(gray_image)

        # Sobel edge detection
        sobelx = cv2.Sobel(gray_work, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_work, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold edges
        edge_threshold = 30
        edges = magnitude > edge_threshold
        
        # Calculate edge density
        edge_density = np.sum(edges) / (gray_work.shape[0] * gray_work.shape[1])
        
        # Normalize to 0-1 range
        return min(edge_density * 10, 1.0)  # Scale factor for typical face edge density
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> float:
        """Analyze frequency domain characteristics"""
        # Apply FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Analyze high frequency content
        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        
        # Define high frequency region (outer ring)
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) // 6)**2
        
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        total_energy = np.mean(magnitude_spectrum)
        
        if total_energy == 0:
            return 0.5
        
        # High frequency ratio
        hf_ratio = high_freq_energy / total_energy
        
        # Normalize (typical range for real faces)
        freq_score = min(hf_ratio / 0.3, 1.0)
        
        return freq_score

    def _fast_liveness_detection(self, gray_image: np.ndarray, hsv_image: np.ndarray) -> Tuple[float, bool]:
        """Faster approximate liveness detection for real-time mode."""
        gray_small, hsv_small = self._resize_pair_for_fast_mode(gray_image, hsv_image)

        # Contrast-based dynamic texture measure
        contrast = float(np.clip(np.std(gray_small) / 64.0, 0.0, 1.0))

        # Edge density on reduced resolution
        edge_score = self._calculate_edge_density(gray_small)

        # Skin tone distribution remains informative even at reduced scale
        color_score = self._analyze_color_distribution(hsv_small)

        liveness_score = 0.45 * contrast + 0.35 * edge_score + 0.20 * color_score
        is_live = liveness_score > self.liveness_threshold

        return liveness_score, is_live

    def _resize_for_analysis(self, gray_image: np.ndarray) -> np.ndarray:
        """Resize grayscale image for faster analysis when in fast mode."""
        if not self.fast_mode:
            return gray_image

        h, w = gray_image.shape[:2]
        max_side = max(h, w)
        if max_side <= self._analysis_max_side:
            return gray_image

        scale = self._analysis_max_side / float(max_side)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        return cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _resize_pair_for_fast_mode(
        self, gray_image: np.ndarray, hsv_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fast_mode:
            return gray_image, hsv_image

        h, w = gray_image.shape[:2]
        max_side = max(h, w)
        if max_side <= self._analysis_max_side:
            return gray_image, hsv_image

        scale = self._analysis_max_side / float(max_side)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        gray_small = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        hsv_small = cv2.resize(hsv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray_small, hsv_small
    
    def _model_liveness_detection(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Model-based liveness detection
        In practice, this would use a trained anti-spoofing model
        """
        # Placeholder for model-based liveness detection
        # You would load and use a trained model here
        
        # For now, return simple detection result
        return self._simple_liveness_detection(face_image)
    
    def assess_face_angle(self, landmarks: list, max_angle: float = 10.0) -> Tuple[float, bool]:
        """
        Assess face angle based on landmarks
        Args:
            landmarks: Facial landmarks
            max_angle: Maximum allowed angle in degrees
        Returns:
            (angle, is_frontal) - Face angle and whether it's frontal enough
        """
        if len(landmarks) < 2:
            return 0.0, True  # Assume frontal if no landmarks
        
        # Use eye positions to calculate face angle
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        if dx == 0:
            angle = 90.0 if dy > 0 else -90.0
        else:
            angle = np.arctan2(dy, dx) * 180.0 / np.pi
        
        is_frontal = abs(angle) <= max_angle
        
        return angle, is_frontal
    
    def comprehensive_quality_check(
        self,
        face_image: np.ndarray,
        landmarks: list = None,
        *,
        require_liveness: bool = True,
        require_blur: bool = True,
    ) -> dict:
        """
        Perform comprehensive quality assessment
        Args:
            face_image: Input face image
            landmarks: Facial landmarks (optional)
            require_liveness: Whether to perform liveness estimation
            require_blur: Whether to perform blur estimation
        Returns:
            Dictionary with quality assessment results
        """
        results = {}
        
        # Blur assessment
        if require_blur:
            blur_score, is_sharp = self.assess_blur(face_image)
        else:
            blur_score, is_sharp = 0.0, True
        results['blur_score'] = blur_score
        results['is_sharp'] = is_sharp
        
        # Liveness assessment
        if require_liveness:
            live_score, is_live = self.assess_liveness(face_image)
        else:
            live_score, is_live = 1.0, True
        results['liveness_score'] = live_score
        results['is_live'] = is_live
        
        # Face angle assessment (if landmarks available)
        if landmarks:
            angle, is_frontal = self.assess_face_angle(landmarks)
            results['face_angle'] = angle
            results['is_frontal'] = is_frontal
        else:
            results['face_angle'] = 0.0
            results['is_frontal'] = True
        
        # Overall quality score
        quality_components = []
        if is_sharp or not require_blur:
            quality_components.append(0.4)
        if is_live or not require_liveness:
            quality_components.append(0.4)
        if results['is_frontal']:
            quality_components.append(0.2)
        
        results['overall_quality'] = sum(quality_components)
        results['is_good_quality'] = results['overall_quality'] >= 0.6
        
        return results