#!/usr/bin/env python3
"""
Advanced Face Feature Extraction - ArcFace/MobileFaceNet Implementation
Based on Face-Recognition-Jetson-Nano ArcFace implementation
Supports ONNX models for high-performance inference

Created: 2025
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, Union
import json

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .ncnn_models import NCNNMobileFaceNet, NCNNModelFactory
    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False

class ArcFaceExtractor:
    """
    ArcFace feature extractor using ONNX runtime
    Compatible with MobileFaceNet and other ArcFace models
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 input_size: Tuple[int, int] = (112, 112),
                 use_ncnn: bool = True,
                 use_fallback: bool = True,
                 use_zscore_norm: bool = False):
        """
        Initialize ArcFace feature extractor
        Args:
            model_path: Path to ONNX model file or "ncnn" for NCNN models
            input_size: Input image size (width, height)
            use_ncnn: Whether to try NCNN models first (recommended)
            use_fallback: Whether to use fallback features if model fails to load
            use_zscore_norm: Whether to use Z-score normalization (like Jetson Nano) instead of L2
        """
        self.model_path = model_path
        self.input_size = input_size
        self.feature_dim = 128  # Standard ArcFace feature dimension
        self.session = None
        self.input_name = None
        self.output_name = None
        self.ncnn_model = None
        self.use_ncnn = use_ncnn
        self.use_fallback = use_fallback
        self.use_zscore_norm = use_zscore_norm
        
        # Model normalization parameters (ImageNet standard)
        self.mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        self.std = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        
        # Try different model types in priority order
        self.use_fallback = True
        
        # 1. Try NCNN models first (best performance on Jetson Nano)
        if use_ncnn and NCNN_AVAILABLE and (model_path is None or model_path == "ncnn"):
            if self._load_ncnn_model():
                self.use_fallback = False
                return
        
        # 2. Try ONNX model if provided
        if model_path and model_path != "ncnn" and os.path.exists(model_path):
            if self._load_onnx_model(model_path):
                self.use_fallback = False
                return
        
        # 3. Fallback to improved feature extraction
        if self.use_fallback:
            print("Using enhanced fallback feature extraction")
    
    def _load_ncnn_model(self) -> bool:
        """Load NCNN MobileFaceNet model"""
        if not NCNN_AVAILABLE:
            return False
        
        try:
            self.ncnn_model = NCNNModelFactory.create_mobilefacenet()
            if self.ncnn_model and not self.ncnn_model.use_fallback:
                try:
                    print("✅ NCNN MobileFaceNet loaded successfully")
                except Exception:
                    print("NCNN MobileFaceNet loaded successfully")
                return True
            else:
                try:
                    print("⚠️  NCNN model loading failed")
                except Exception:
                    print("NCNN model loading failed")
                return False
        except Exception as e:
            try:
                print(f"⚠️  Error loading NCNN model: {e}")
            except Exception:
                print(f"Error loading NCNN model: {e}")
            return False
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model for inference"""
        if not ONNX_AVAILABLE:
            print("Error: ONNX Runtime not available. Install with: pip install onnxruntime")
            return False
        
        try:
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape to verify
            input_shape = self.session.get_inputs()[0].shape
            try:
                print(f"✅ ArcFace ONNX model loaded: {model_path}")
            except Exception:
                print(f"ArcFace ONNX model loaded: {model_path}")
            print(f"   Input shape: {input_shape}")
            print(f"   Input name: {self.input_name}")
            print(f"   Output name: {self.output_name}")
            
            return True
            
        except Exception as e:
            print(f"Error loading ONNX model {model_path}: {e}")
            return False
    
    def extract_feature(self, face_image: np.ndarray, landmarks: Optional[list] = None) -> np.ndarray:
        """
        Extract face feature vector
        Args:
            face_image: Input face image
            landmarks: Optional facial landmarks for alignment
        Returns:
            Normalized feature vector
        """
        # Try NCNN model first (highest priority)
        if self.ncnn_model and not self.ncnn_model.use_fallback:
            try:
                return self.ncnn_model.extract_feature(face_image, landmarks)
            except Exception as e:
                try:
                    print(f"NCNN extraction failed: {e}, trying ONNX...")
                except Exception:
                    print(f"NCNN extraction failed: {e}, trying ONNX...")
        
        # Try ONNX model
        if self.session:
            try:
                # Preprocess face image
                preprocessed = self._preprocess_face(face_image, landmarks)
                
                # Run inference
                inputs = {self.input_name: preprocessed}
                outputs = self.session.run([self.output_name], inputs)
                feature = outputs[0].flatten()
                
                # Apply normalization based on configuration
                if self.use_zscore_norm:
                    feature = self._zscore_normalize(feature)
                else:
                    feature = self._l2_normalize(feature)
                
                return feature.astype(np.float32)
            except Exception as e:
                try:
                    print(f"ONNX extraction failed: {e}, trying fallback...")
                except Exception:
                    print(f"ONNX extraction failed: {e}, trying fallback...")
        
        # Fallback feature extraction or error
        if self.use_fallback:
            return self._extract_fallback_features(face_image)
        else:
            raise RuntimeError("No ArcFace model available and fallback is disabled. "
                             "Please install NCNN models or provide a valid ONNX model path.")
    
    def _preprocess_face(self, face_image: np.ndarray, landmarks: Optional[list] = None) -> np.ndarray:
        """
        Preprocess face image for ArcFace model
        Args:
            face_image: Input face image
            landmarks: Optional landmarks for alignment
        Returns:
            Preprocessed image ready for inference
        """
        # Align face if landmarks available
        if landmarks and len(landmarks) >= 2:
            aligned_face = self._align_face(face_image, landmarks)
        else:
            aligned_face = face_image
        
        # Resize to model input size
        resized = cv2.resize(aligned_face, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = (rgb_image.astype(np.float32) - self.mean) / self.std
        
        # Convert to CHW format (Channels, Height, Width)
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch_image = np.expand_dims(chw_image, axis=0)
        
        return batch_image
    
    def _align_face(self, face_image: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Align face based on eye landmarks
        Args:
            face_image: Input face image
            landmarks: Facial landmarks [(x1,y1), (x2,y2), ...]
        Returns:
            Aligned face image
        """
        if len(landmarks) < 2:
            return face_image
        
        # Use eye positions for alignment (landmarks 0 and 1 are typically eyes)
        left_eye = np.array(landmarks[0], dtype=np.float32)
        right_eye = np.array(landmarks[1], dtype=np.float32)
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        
        # Calculate center point between eyes
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # Apply rotation
        height, width = face_image.shape[:2]
        aligned = cv2.warpAffine(face_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        return aligned
    
    def _l2_normalize(self, feature: np.ndarray) -> np.ndarray:
        """
        L2 normalization of feature vector
        Args:
            feature: Input feature vector
        Returns:
            L2 normalized feature vector
        """
        norm = np.linalg.norm(feature)
        if norm == 0:
            return feature
        return feature / norm
    
    def _zscore_normalize(self, feature: np.ndarray) -> np.ndarray:
        """
        Z-score normalization (mean=0, std=1) like Jetson Nano implementation
        Args:
            feature: Input feature vector
        Returns:
            Z-score normalized feature vector
        """
        mean_val = np.mean(feature)
        std_val = np.std(feature)
        if std_val == 0:
            return feature - mean_val  # Subtract mean only if std is zero
        return (feature - mean_val) / std_val
    
    def _extract_fallback_features(self, face_image: np.ndarray) -> np.ndarray:
        """
        Fallback feature extraction when ArcFace model not available
        Uses improved histogram and texture features
        """
        # Resize to standard size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Convert to grayscale
        if len(face_resized.shape) == 3:
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_resized
        
        features = []
        
        # 1. Local Binary Pattern features
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 2. Gradient features
        grad_features = self._extract_gradient_features(gray)
        features.extend(grad_features)
        
        # 3. Regional histogram features
        hist_features = self._extract_regional_histograms(gray)
        features.extend(hist_features)
        
        # Convert to numpy array and normalize
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to desired dimension
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
        
        # L2 normalize
        features = self._l2_normalize(features)
        
        return features
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> list:
        """Extract Local Binary Pattern features"""
        h, w = gray_image.shape
        lbp_features = []
        
        # Calculate LBP for 3x3 regions
        for y in range(1, h-1, 4):  # Skip pixels for efficiency
            for x in range(1, w-1, 4):
                center = gray_image[y, x]
                pattern = 0
                
                # 8-connected neighbors
                neighbors = [
                    gray_image[y-1, x-1], gray_image[y-1, x], gray_image[y-1, x+1],
                    gray_image[y, x+1], gray_image[y+1, x+1], gray_image[y+1, x],
                    gray_image[y+1, x-1], gray_image[y, x-1]
                ]
                
                for i, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << i)
                
                lbp_features.append(pattern / 255.0)  # Normalize
        
        return lbp_features
    
    def _extract_gradient_features(self, gray_image: np.ndarray) -> list:
        """Extract gradient-based features"""
        # Sobel gradients
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and orientation
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Histogram of gradients (simplified HOG)
        mag_hist, _ = np.histogram(magnitude.flatten(), bins=8, range=(0, 255))
        ori_hist, _ = np.histogram(orientation.flatten(), bins=8, range=(-np.pi, np.pi))
        
        # Normalize histograms
        mag_hist = mag_hist.astype(np.float32) / np.sum(mag_hist) if np.sum(mag_hist) > 0 else mag_hist.astype(np.float32)
        ori_hist = ori_hist.astype(np.float32) / np.sum(ori_hist) if np.sum(ori_hist) > 0 else ori_hist.astype(np.float32)
        
        return mag_hist.tolist() + ori_hist.tolist()
    
    def _extract_regional_histograms(self, gray_image: np.ndarray) -> list:
        """Extract histogram features from different face regions"""
        h, w = gray_image.shape
        features = []
        
        # Divide face into regions
        regions = [
            (0, 0, w//2, h//2),          # Top-left
            (w//2, 0, w, h//2),          # Top-right
            (0, h//2, w//2, h),          # Bottom-left
            (w//2, h//2, w, h),          # Bottom-right
            (w//4, h//4, 3*w//4, 3*h//4) # Center region
        ]
        
        for x1, y1, x2, y2 in regions:
            region = gray_image[y1:y2, x1:x2]
            hist, _ = np.histogram(region.flatten(), bins=16, range=(0, 256))
            hist = hist.astype(np.float32) / np.sum(hist) if np.sum(hist) > 0 else hist.astype(np.float32)
            features.extend(hist.tolist())
        
        return features
    
    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors
        Args:
            feature1: First feature vector
            feature2: Second feature vector
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Ensure features are normalized
        f1_norm = np.linalg.norm(feature1)
        f2_norm = np.linalg.norm(feature2)
        
        if f1_norm == 0 or f2_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(feature1, feature2) / (f1_norm * f2_norm)
        
        # Clamp to valid range
        return float(np.clip(similarity, -1.0, 1.0))

class MobileFaceNetExtractor(ArcFaceExtractor):
    """
    MobileFaceNet feature extractor (specialized version of ArcFace)
    Optimized for mobile/edge deployment
    """
    
    def __init__(self, model_path: Optional[str] = None, use_zscore_norm: bool = False):
        """Initialize MobileFaceNet extractor"""
        super().__init__(model_path, input_size=(112, 112), use_zscore_norm=use_zscore_norm)
        self.feature_dim = 128  # MobileFaceNet standard output
        
        print("MobileFaceNet extractor initialized")

def download_arcface_model(model_name: str = "mobilefacenet", output_dir: str = "models/") -> str:
    """
    Download pre-trained ArcFace model
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
    Returns:
        Path to downloaded model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Model URLs (you can add more models here)
    model_urls = {
        "mobilefacenet": {
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "filename": "mobilefacenet.onnx",
            "description": "MobileFaceNet ONNX model"
        },
        "arcface_r100": {
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", 
            "filename": "arcface_r100.onnx",
            "description": "ArcFace ResNet100 model"
        }
    }
    
    if model_name not in model_urls:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(model_urls.keys())}")
        return ""
    
    model_info = model_urls[model_name]
    model_path = os.path.join(output_dir, model_info["filename"])
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return model_path
    
    print(f"To download {model_info['description']}:")
    print(f"1. Go to: {model_info['url']}")
    print(f"2. Extract and place the .onnx file as: {model_path}")
    print(f"3. Or use the conversion script to convert from C++ models")
    
    return ""

def convert_ncnn_to_onnx(ncnn_param_path: str, ncnn_bin_path: str, output_path: str) -> bool:
    """
    Convert NCNN model to ONNX format
    Args:
        ncnn_param_path: Path to .param file
        ncnn_bin_path: Path to .bin file  
        output_path: Output ONNX file path
    Returns:
        True if conversion successful
    """
    try:
        # This would require ncnn2onnx converter
        print(f"To convert NCNN to ONNX:")
        print(f"1. Install ncnn2onnx: pip install ncnn2onnx")
        print(f"2. Run: ncnn2onnx {ncnn_param_path} {ncnn_bin_path} {output_path}")
        print(f"3. Or use online conversion tools")
        
        return False
    except Exception as e:
        print(f"Error converting NCNN to ONNX: {e}")
        return False

if __name__ == "__main__":
    # Test the ArcFace extractor
    print("Testing ArcFace Feature Extractor")
    print("=" * 40)
    
    # Initialize extractor (will use fallback if no model)
    extractor = ArcFaceExtractor()
    
    # Create test image
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Extract features
    feature = extractor.extract_feature(test_img)
    print(f"Feature shape: {feature.shape}")
    print(f"Feature norm: {np.linalg.norm(feature):.3f}")
    print(f"Feature range: [{feature.min():.3f}, {feature.max():.3f}]")
    
    # Test similarity
    feature2 = extractor.extract_feature(test_img)
    similarity = ArcFaceExtractor.cosine_similarity(feature, feature2)
    print(f"Self-similarity: {similarity:.3f}")
    
    print("\n💡 To use real ArcFace model:")
    print("1. Download model: python -c \"from src.face_features_arcface import download_arcface_model; download_arcface_model()\"")
    print("2. Or convert from C++ models in ../Face-Recognition-Jetson-Nano/models/")
    print("3. Then: extractor = ArcFaceExtractor('models/mobilefacenet.onnx')")