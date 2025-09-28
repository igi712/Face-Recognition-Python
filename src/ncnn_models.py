#!/usr/bin/env python3
"""
NCNN Model Loader - Direct NCNN model usage for Python
Loads NCNN models directly without conversion to ONNX

Created: 2025
"""

import os
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, List

try:
    import ncnn
    NCNN_AVAILABLE = True
    try:
        print("✅ NCNN Python bindings available")
    except Exception:
        print("NCNN Python bindings available")
except ImportError:
    NCNN_AVAILABLE = False
    try:
        print("❌ NCNN Python bindings not available")
    except Exception:
        print("NCNN Python bindings not available")
    print("   Install with: pip install ncnn")

class NCNNMobileFaceNet:
    """MobileFaceNet using NCNN backend"""
    
    def __init__(self, param_path: str, bin_path: str):
        """
        Initialize MobileFaceNet with NCNN
        Args:
            param_path: Path to .param file
            bin_path: Path to .bin file
        """
        self.param_path = param_path
        self.bin_path = bin_path
        self.net = None
        self.input_size = (112, 112)
        self.feature_dim = 128
        
        if NCNN_AVAILABLE:
            self._load_model()
        else:
            print("⚠️  NCNN not available, using fallback")
            self.use_fallback = True
    
    def _load_model(self):
        """Load NCNN model"""
        try:
            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = False  # Use CPU for compatibility
            
            # Load model
            ret1 = self.net.load_param(self.param_path)
            ret2 = self.net.load_model(self.bin_path)
            
            if ret1 == 0 and ret2 == 0:
                print(f"✅ MobileFaceNet NCNN model loaded successfully")
                print(f"   Param: {self.param_path}")
                print(f"   Model: {self.bin_path}")
                self.use_fallback = False
            else:
                print(f"❌ Failed to load NCNN model (ret1={ret1}, ret2={ret2})")
                self.use_fallback = True
                
        except Exception as e:
            print(f"❌ Error loading NCNN model: {e}")
            self.use_fallback = True
    
    def extract_feature(self, face_image: np.ndarray, landmarks: Optional[List] = None) -> np.ndarray:
        """
        Extract face features using NCNN MobileFaceNet (C++ compatible)
        Args:
            face_image: Input face image (BGR format)
            landmarks: Facial landmarks (optional)
        Returns:
            Z-Score normalized feature vector (same as C++)
        """
        if not NCNN_AVAILABLE or self.use_fallback:
            return self._fallback_features(face_image)
        
        try:
            # Preprocess image (align if landmarks available)
            if landmarks and len(landmarks) >= 2:
                preprocessed = self._align_face(face_image, landmarks)
            else:
                preprocessed = face_image
            
            # Resize to 112x112 (standard for MobileFaceNet)
            resized = cv2.resize(preprocessed, (112, 112), interpolation=cv2.INTER_LINEAR)
            
            # Create NCNN Mat directly from BGR image (no RGB conversion like C++)
            input_mat = ncnn.Mat.from_pixels(resized, ncnn.Mat.PixelType.PIXEL_BGR, 
                                           resized.shape[1], resized.shape[0])
            
            # No normalization at input level (handled by model)
            
            # Create extractor and run inference
            ex = self.net.create_extractor()
            ex.set_light_mode(True)  # Same as C++
            ex.input("data", input_mat)  # C++ blob name
            
            # Extract features from fc1 layer (same as C++)
            ret, output_mat = ex.extract("fc1")
            
            if ret == 0:
                # Convert to numpy array
                feature = np.array(output_mat).flatten()
                
                # Apply Z-Score normalization (exactly like C++)
                feature_normalized = self._zscore_normalize(feature)
                
                return feature_normalized.astype(np.float32)
            else:
                print(f"⚠️  NCNN inference failed (ret={ret}), using fallback")
                return self._fallback_features(face_image)
                
        except Exception as e:
            print(f"⚠️  Error in NCNN inference: {e}, using fallback")
            return self._fallback_features(face_image)
    
    def _zscore_normalize(self, feature: np.ndarray) -> np.ndarray:
        """
        Z-Score normalization (exactly like C++ TArcFace::Zscore)
        Mean = 0, Std = 1
        Args:
            feature: Input feature vector
        Returns:
            Z-Score normalized feature
        """
        mean = np.mean(feature)
        std = np.std(feature)
        
        if std == 0:
            return feature  # Avoid division by zero
        
        return (feature - mean) / std
    
    def _preprocess_face(self, face_image: np.ndarray, landmarks: Optional[List] = None) -> np.ndarray:
        """Preprocess face for MobileFaceNet (C++ compatible)"""
        # Align face if landmarks available
        if landmarks and len(landmarks) >= 2:
            aligned_face = self._align_face(face_image, landmarks)
        else:
            aligned_face = face_image
        
        # Resize to model input size (same as C++)
        resized = cv2.resize(aligned_face, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Keep BGR format (no RGB conversion like C++)
        return resized
    
    def _align_face(self, face_image: np.ndarray, landmarks: List) -> np.ndarray:
        """Simple face alignment based on eye positions"""
        if len(landmarks) < 2:
            return face_image
        
        # Use first two landmarks as eyes
        left_eye = np.array(landmarks[0], dtype=np.float32)
        right_eye = np.array(landmarks[1], dtype=np.float32)
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.arctan2(dy, dx) * 180.0 / np.pi
        
        # Calculate center
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Rotate image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        height, width = face_image.shape[:2]
        aligned = cv2.warpAffine(face_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        
        return aligned
    
    def _fallback_features(self, face_image: np.ndarray) -> np.ndarray:
        """Fallback feature extraction when NCNN not available"""
        # Use the improved fallback from ArcFace extractor
        from .face_features_arcface import ArcFaceExtractor
        extractor = ArcFaceExtractor()
        return extractor._extract_fallback_features(face_image)

class NCNNModelFactory:
    """Factory for creating NCNN models"""
    
    @staticmethod
    def create_mobilefacenet(models_dir: str = "models") -> NCNNMobileFaceNet:
        """Create MobileFaceNet NCNN model"""
        param_path = os.path.join(models_dir, "mobilefacenet", "mobilefacenet.param")
        bin_path = os.path.join(models_dir, "mobilefacenet", "mobilefacenet.bin")
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            print(f"❌ MobileFaceNet model files not found:")
            print(f"   {param_path}")
            print(f"   {bin_path}")
            return None
        
        return NCNNMobileFaceNet(param_path, bin_path)
    
    @staticmethod
    def list_available_models(models_dir: str = "models") -> dict:
        """List all available NCNN models"""
        models = {}
        
        if not os.path.exists(models_dir):
            return models
        
        for category in os.listdir(models_dir):
            category_path = os.path.join(models_dir, category)
            if not os.path.isdir(category_path):
                continue
            
            models[category] = []
            
            for file in os.listdir(category_path):
                if file.endswith('.param'):
                    model_name = file[:-6]  # Remove .param
                    bin_file = file.replace('.param', '.bin')
                    bin_path = os.path.join(category_path, bin_file)
                    
                    if os.path.exists(bin_path):
                        models[category].append({
                            'name': model_name,
                            'param': os.path.join(category_path, file),
                            'bin': bin_path
                        })
        
        return models

def test_ncnn_models():
    """Test NCNN model loading and inference"""
    print("🧪 Testing NCNN Models")
    print("=" * 40)
    
    # List available models
    models = NCNNModelFactory.list_available_models()
    print(f"Found models in {len(models)} categories:")
    for category, model_list in models.items():
        print(f"  📁 {category}: {len(model_list)} models")
        for model in model_list:
            print(f"    • {model['name']}")
    
    # Test MobileFaceNet
    print(f"\n🚀 Testing MobileFaceNet...")
    mobilefacenet = NCNNModelFactory.create_mobilefacenet()
    
    if mobilefacenet:
        # Create test image
        test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # Extract features
        feature = mobilefacenet.extract_feature(test_img)
        
        print(f"✅ MobileFaceNet test successful!")
        print(f"   Feature shape: {feature.shape}")
        print(f"   Feature norm: {np.linalg.norm(feature):.3f}")
        print(f"   Feature range: [{feature.min():.3f}, {feature.max():.3f}]")
        
        return True
    else:
        print(f"❌ MobileFaceNet test failed")
        return False

def install_ncnn_instructions():
    """Show instructions for installing NCNN Python bindings"""
    print("📦 NCNN Python Installation Instructions")
    print("=" * 50)
    print("1. Install pre-built wheel (recommended):")
    print("   pip install ncnn")
    print()
    print("2. Or build from source:")
    print("   git clone https://github.com/Tencent/ncnn")
    print("   cd ncnn")
    print("   mkdir build && cd build")
    print("   cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_PYTHON=ON ..")
    print("   make -j4")
    print("   cd python")
    print("   pip install .")
    print()
    print("3. Alternative using conda:")
    print("   conda install -c conda-forge ncnn")

if __name__ == "__main__":
    if not NCNN_AVAILABLE:
        install_ncnn_instructions()
    else:
        test_ncnn_models()