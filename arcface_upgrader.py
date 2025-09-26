#!/usr/bin/env python3
"""
ArcFace Upgrade Tool - Convert existing database to ArcFace features
Allows testing and migration from legacy features to ArcFace

Created: 2025
"""

import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.face_database import FaceDatabase
from src.face_detector import FaceDetector
from src.face_features_arcface import ArcFaceExtractor, download_arcface_model

class ArcFaceUpgrader:
    """Tool for upgrading face database to use ArcFace features"""
    
    def __init__(self, arcface_model_path: str = None):
        """Initialize the upgrader"""
        self.arcface_model_path = arcface_model_path
        self.detector = FaceDetector()
        
        print("🚀 ArcFace Database Upgrader")
        print("=" * 50)
    
    def test_arcface_model(self) -> bool:
        """Test if ArcFace model works correctly"""
        print("\n📋 Testing ArcFace Model...")
        
        try:
            # Initialize ArcFace extractor
            extractor = ArcFaceExtractor(self.arcface_model_path)
            
            # Create test image
            test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            # Test feature extraction
            feature = extractor.extract_feature(test_img)
            
            print(f"✅ ArcFace model test successful!")
            print(f"   Feature dimension: {len(feature)}")
            print(f"   Feature norm: {np.linalg.norm(feature):.3f}")
            print(f"   Model type: {'ONNX' if not extractor.use_fallback else 'Fallback'}")
            
            return True
            
        except Exception as e:
            print(f"❌ ArcFace model test failed: {e}")
            return False
    
    def compare_extractors(self, image_path: str) -> None:
        """Compare legacy vs ArcFace feature extraction on a real image"""
        print(f"\n🔍 Comparing extractors on: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        # Detect face
        faces = self.detector.detect_faces(image)
        if len(faces) == 0:
            print("❌ No faces detected in image")
            return
        
        face_obj = faces[0]  # Use first face
        
        # Extract face image from the detected face
        if hasattr(face_obj, 'face_image'):
            face_img = face_obj.face_image
        else:
            # Extract face from original image using rect
            if face_obj.rect is not None:
                x, y, w, h = face_obj.rect
                face_img = image[y:y+h, x:x+w]
            else:
                print("❌ No face rectangle found")
                return
        
        print(f"   Face detected: {face_img.shape}")
        
        # Initialize both extractors
        legacy_db = FaceDatabase(use_arcface=False)
        arcface_db = FaceDatabase(use_arcface=True, arcface_model_path=self.arcface_model_path)
        
        # Extract features using both methods
        print("   Extracting legacy features...")
        legacy_feature = legacy_db.legacy_extractor.extract_feature(face_img)
        
        print("   Extracting ArcFace features...")
        if arcface_db.use_arcface:
            arcface_feature = arcface_db.arcface_extractor.extract_feature(face_img)
        else:
            arcface_feature = arcface_db.legacy_extractor.extract_feature(face_img)
        
        # Compare results
        print("\n📊 Comparison Results:")
        print(f"   Legacy features:  {len(legacy_feature)} dimensions")
        print(f"   ArcFace features: {len(arcface_feature)} dimensions")
        print(f"   Legacy norm:      {np.linalg.norm(legacy_feature):.3f}")
        print(f"   ArcFace norm:     {np.linalg.norm(arcface_feature):.3f}")
        
        # Test self-similarity
        legacy_sim = np.dot(legacy_feature, legacy_feature) / (np.linalg.norm(legacy_feature) ** 2)
        if arcface_db.use_arcface:
            arcface_sim = ArcFaceExtractor.cosine_similarity(arcface_feature, arcface_feature)
        else:
            arcface_sim = np.dot(arcface_feature, arcface_feature) / (np.linalg.norm(arcface_feature) ** 2)
        
        print(f"   Legacy self-sim:  {legacy_sim:.3f}")
        print(f"   ArcFace self-sim: {arcface_sim:.3f}")
    
    def upgrade_database(self, source_db_path: str, target_db_path: str, 
                        images_folder: str) -> bool:
        """
        Upgrade existing database from legacy to ArcFace features
        Args:
            source_db_path: Path to existing legacy database
            target_db_path: Path for new ArcFace database
            images_folder: Folder containing original images
        Returns:
            True if upgrade successful
        """
        print(f"\n🔄 Upgrading Database...")
        print(f"   Source: {source_db_path}")
        print(f"   Target: {target_db_path}")
        print(f"   Images: {images_folder}")
        
        if not os.path.exists(source_db_path):
            print(f"❌ Source database not found: {source_db_path}")
            return False
        
        if not os.path.exists(images_folder):
            print(f"❌ Images folder not found: {images_folder}")
            return False
        
        try:
            # Load existing database to get names
            legacy_db = FaceDatabase(source_db_path, use_arcface=False)
            legacy_db.load_database()
            
            print(f"   Legacy database loaded: {len(legacy_db.face_names)} people")
            
            # Create new ArcFace database
            arcface_db = FaceDatabase(target_db_path, use_arcface=True, 
                                     arcface_model_path=self.arcface_model_path)
            arcface_db.clear_database()
            
            # Process each person's images
            success_count = 0
            total_count = 0
            
            for person_name in legacy_db.face_names:
                print(f"   Processing {person_name}...")
                person_folder = os.path.join(images_folder, person_name)
                
                if os.path.exists(person_folder):
                    # Process all images in person folder
                    image_files = [f for f in os.listdir(person_folder) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    for image_file in image_files:
                        image_path = os.path.join(person_folder, image_file)
                        total_count += 1
                        
                        # Load and detect face in image
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        # Detect face
                        faces = self.detector.detect_faces(image)
                        if len(faces) == 0:
                            continue
                        
                        # Get face image
                        face_obj = faces[0]
                        if hasattr(face_obj, 'face_image'):
                            face_image = face_obj.face_image
                        else:
                            x, y, w, h = face_obj.rect
                            face_image = image[y:y+h, x:x+w]
                        
                        # Add to ArcFace database
                        if arcface_db.add_face(person_name, face_image, face_obj.landmark):
                            success_count += 1
                
                else:
                    print(f"     Warning: No folder found for {person_name}")
            
            # Save new database
            arcface_db.save_database()
            
            print(f"\n✅ Database upgrade completed!")
            print(f"   Successfully processed: {success_count}/{total_count} images")
            print(f"   New database saved: {target_db_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Database upgrade failed: {e}")
            return False
    
    def benchmark_recognition(self, database_path: str, test_images_folder: str) -> None:
        """
        Benchmark recognition accuracy with different feature types
        Args:
            database_path: Path to database
            test_images_folder: Folder with test images
        """
        print(f"\n🏁 Benchmarking Recognition Performance...")
        
        if not os.path.exists(test_images_folder):
            print(f"❌ Test images folder not found: {test_images_folder}")
            return
        
        # Test both legacy and ArcFace
        for use_arcface in [False, True]:
            feature_type = "ArcFace" if use_arcface else "Legacy"
            print(f"\n📊 Testing {feature_type} Features:")
            
            # Initialize database
            db = FaceDatabase(database_path, use_arcface=use_arcface, 
                            arcface_model_path=self.arcface_model_path if use_arcface else None)
            db.load_database()
            
            if len(db.face_names) == 0:
                print("   ❌ No faces in database")
                continue
            
            # Test recognition on sample images
            correct = 0
            total = 0
            confidences = []
            
            for person_name in db.face_names[:3]:  # Test first 3 people
                person_folder = os.path.join(test_images_folder, person_name)
                if not os.path.exists(person_folder):
                    continue
                
                image_files = [f for f in os.listdir(person_folder)[:2]  # Test 2 images per person
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for image_file in image_files:
                    image_path = os.path.join(person_folder, image_file)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Detect face
                    faces = self.detector.detect_faces(image)
                    if len(faces) == 0:
                        continue
                    
                    # Get face image
                    face_obj = faces[0]
                    if hasattr(face_obj, 'face_image'):
                        face_image = face_obj.face_image
                        landmarks = face_obj.landmark
                    else:
                        x, y, w, h = face_obj.rect
                        face_image = image[y:y+h, x:x+w]
                        landmarks = face_obj.landmark
                    
                    # Recognize face
                    name_idx, confidence = db.recognize_face(face_image, landmarks)
                    
                    total += 1
                    if name_idx >= 0 and db.face_names[name_idx] == person_name:
                        correct += 1
                        confidences.append(confidence)
            
            # Print results
            if total > 0:
                accuracy = (correct / total) * 100
                avg_confidence = np.mean(confidences) if confidences else 0
                print(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
                print(f"   Avg Confidence: {avg_confidence:.3f}")
            else:
                print("   No test images processed")

def main():
    parser = argparse.ArgumentParser(description="ArcFace Database Upgrader")
    parser.add_argument("--model", type=str, help="Path to ArcFace ONNX model")
    parser.add_argument("--test", action="store_true", help="Test ArcFace model")
    parser.add_argument("--compare", type=str, help="Compare extractors on image")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade database to ArcFace")
    parser.add_argument("--source", type=str, default="face_database_mobilefacenet.json", help="Source database")
    parser.add_argument("--target", type=str, default="face_database_mobilefacenet.json", help="Target database")
    parser.add_argument("--images", type=str, default="images_processed/", help="Images folder")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark recognition")
    parser.add_argument("--download", action="store_true", help="Show model download info")
    
    args = parser.parse_args()
    
    upgrader = ArcFaceUpgrader(args.model)
    
    if args.download:
        print("📥 ArcFace Model Download Information:")
        print("=" * 50)
        download_arcface_model("mobilefacenet")
        return
    
    if args.test:
        upgrader.test_arcface_model()
        return
    
    if args.compare:
        upgrader.compare_extractors(args.compare)
        return
    
    if args.upgrade:
        upgrader.upgrade_database(args.source, args.target, args.images)
        return
    
    if args.benchmark:
        upgrader.benchmark_recognition(args.source, args.images)
        return
    
    # Default: show help
    parser.print_help()
    print("\n💡 Quick Start:")
    print("  1. Test fallback features: python arcface_upgrader.py --test")
    print("  2. Compare on image: python arcface_upgrader.py --compare images_processed/Danu/img_1.jpg")
    print("  3. Upgrade database: python arcface_upgrader.py --upgrade")
    print("  4. Benchmark performance: python arcface_upgrader.py --benchmark")

if __name__ == "__main__":
    main()