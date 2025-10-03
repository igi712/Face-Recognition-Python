#!/usr/bin/env python3
"""
Face Database Management Module - Python Implementation
Based on Face-Recognition-Jetson-Nano project
Manages face database with feature vectors and names
Enhanced with ArcFace support

Created: 2025
"""

import os
import json
import base64
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from .face_features import FaceFeatureExtractor
try:
    from .face_features_arcface import ArcFaceExtractor
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False

class FaceDatabase:
    """Face database for storing and matching face features"""
    
    def __init__(self, database_path: str = "face_database_mobilefacenet.json", max_items: int = 2000,
                 use_arcface: bool = True, arcface_model_path: Optional[str] = None, use_zscore_norm: bool = True):
        """
        Initialize face database
        Args:
            database_path: Path to save/load database
            max_items: Maximum number of faces in database
            use_arcface: Whether to use ArcFace features (more accurate)
            arcface_model_path: Path to ArcFace ONNX model file
            use_zscore_norm: Whether to use Z-score normalization (like Jetson Nano) - DEFAULT TRUE
        """
        self.database_path = database_path
        self.max_items = max_items
        self.use_arcface = use_arcface and ARCFACE_AVAILABLE
        self.use_zscore_norm = use_zscore_norm
        
        # Initialize feature extractors
        self.legacy_extractor = FaceFeatureExtractor()
        if self.use_arcface:
            self.arcface_extractor = ArcFaceExtractor(arcface_model_path, use_zscore_norm=use_zscore_norm)
            print("✅ Using ArcFace feature extraction" + (" (Z-score norm - Jetson Nano compatible)" if use_zscore_norm else " (L2 norm)"))
            # ✅ PAMERAN: Threshold dinaikkan ke 0.6 untuk accuracy
            # Z-score normalization menghasilkan range similarity yang berbeda
            # Jetson Nano asli: 0.5, tapi kita naikkan ke 0.6 untuk lebih strict
            self.similarity_threshold = 0.55 if use_zscore_norm else 0.5  # Stricter untuk Z-score (0.6 → 0.7)
        else:
            self.arcface_extractor = None
            self.similarity_threshold = 0.8  # Lower threshold for legacy features
            if use_arcface and not ARCFACE_AVAILABLE:
                print("⚠️  ArcFace requested but not available, using legacy features")

        self.faces = {}  # type: Dict[str, List[List[float]]]  # {name: [feature_vectors]}
        self.face_names: List[str] = []  # List of names

        self.load_database()
    
    def set_feature_extractor(self, extractor: FaceFeatureExtractor):
        """Set the feature extractor to use"""
        self.feature_extractor = extractor
    
    def add_face(self, name: str, face_image: np.ndarray, landmarks: List = None) -> bool:
        """
        Add a face to the database
        Args:
            name: Name of the person
            face_image: Cropped face image
            landmarks: Facial landmarks (optional)
        Returns:
            True if added successfully
        """        
        # Check if database is full
        total_faces = sum(len(features) for features in self.faces.values())
        if total_faces >= self.max_items:
            print(f"Database full (max {self.max_items} faces)")
            return False
        
        try:
            # Extract features using appropriate extractor
            if self.use_arcface and self.arcface_extractor:
                feature = self.arcface_extractor.extract_feature(face_image, landmarks)
            else:
                # Align face if landmarks available
                if landmarks and len(landmarks) >= 2:
                    face_aligned = self.legacy_extractor.align_face(face_image, landmarks)
                else:
                    face_aligned = cv2.resize(face_image, (112, 112))  # Standard size
                
                feature = self.legacy_extractor.extract_feature(face_aligned)
            
            # Add to database
            if name not in self.faces:
                self.faces[name] = []
                self.face_names.append(name)
            
            self.faces[name].append(feature.tolist())
            
            feature_type = "ArcFace" if self.use_arcface else "Legacy"
            print(f"Added {feature_type} features for {name} (total: {len(self.faces[name])} faces)")
            return True
            
        except Exception as e:
            print(f"Error adding face for '{name}': {e}")
            return False
    
    def add_face_from_file(self, name: str, image_path: str) -> bool:
        """
        Add a face from image file
        Args:
            name: Name of the person
            image_path: Path to image file
        Returns:
            True if added successfully
        """
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        return self.add_face(name, image)
    
    def recognize_face(self, face_image: np.ndarray, landmarks: List = None, 
                      threshold: Optional[float] = None) -> Tuple[int, float]:
        """
        Recognize a face against the database
        Args:
            face_image: Cropped face image
            landmarks: Facial landmarks (optional)
            threshold: Minimum similarity threshold (uses default if None)
        Returns:
            (name_index, confidence) - (-1, 0.0) if not recognized
        """        
        if len(self.faces) == 0:
            return -1, 0.0  # Empty database
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.similarity_threshold
        
        try:
            # Extract features using appropriate extractor
            if self.use_arcface and self.arcface_extractor:
                query_feature = self.arcface_extractor.extract_feature(face_image, landmarks)
            else:
                # Align face if landmarks available
                if landmarks and len(landmarks) >= 2:
                    face_aligned = self.legacy_extractor.align_face(face_image, landmarks)
                else:
                    face_aligned = cv2.resize(face_image, (112, 112))
                
                query_feature = self.legacy_extractor.extract_feature(face_aligned)
            
            best_match_idx = -1
            best_similarity = 0.0
            
            # Compare with all faces in database
            for name_idx, name in enumerate(self.face_names):
                face_features = self.faces[name]
                
                for stored_feature in face_features:
                    stored_feature_np = np.array(stored_feature, dtype=np.float32)
                    
                    # Calculate similarity
                    if self.use_arcface:
                        similarity = ArcFaceExtractor.cosine_similarity(query_feature, stored_feature_np)
                    else:
                        similarity = self._calculate_similarity(query_feature, stored_feature_np)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = name_idx
            
            # Check threshold
            if best_similarity >= threshold:
                # ✅ DEBUG: Log match details
                # matched_name = self.face_names[best_match_idx]
                # print(f"🎯 MATCH: {matched_name} | Confidence: {best_similarity:.4f} | Threshold: {threshold:.2f}")
                return best_match_idx, best_similarity
            else:
                # ✅ DEBUG: Log rejection
                # if best_similarity > 0:
                #     print(f"❌ REJECT: Best match confidence {best_similarity:.4f} < threshold {threshold:.2f}")
                return -1, 0.0
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return -1, 0.0
    
    def _calculate_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        # Normalize features
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(feature1, feature2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def _cosine_similarity_vec(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Return cosine similarity between two vectors, guarding against zero norms."""
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def _deduplicate_person(self, name: str, similarity_threshold: float) -> int:
        """Remove near-duplicate feature vectors for a single person."""
        features = self.faces.get(name)
        if not features or len(features) < 2:
            return 0

        kept_vectors: List[np.ndarray] = []
        kept_payloads: List[List[float]] = []
        removed = 0

        for payload in features:
            vec = np.asarray(payload, dtype=np.float32)

            is_duplicate = False
            for kept_vec in kept_vectors:
                if self._cosine_similarity_vec(vec, kept_vec) >= similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                removed += 1
            else:
                kept_vectors.append(vec)
                kept_payloads.append([float(x) for x in vec])

        if removed:
            self.faces[name] = kept_payloads

        return removed

    def deduplicate(self, similarity_threshold: float = 0.98) -> Dict[str, int]:
        """
        Deduplicate feature vectors for every person in the database.

        Args:
            similarity_threshold: Cosine similarity (0-1). Entries equal or above are treated
                as duplicates and removed, keeping the first occurrence.

        Returns:
            Dict mapping person names to the number of entries removed.
        """
        summary: Dict[str, int] = {}

        for name in list(self.faces.keys()):
            removed = self._deduplicate_person(name, similarity_threshold)
            if removed:
                summary[name] = removed

        # Clean up face_names if any person lost every embedding (should be rare).
        empty_names = [name for name, entries in self.faces.items() if not entries]
        for name in empty_names:
            if name in self.faces:
                del self.faces[name]
            if name in self.face_names:
                self.face_names.remove(name)

        return summary
    
    def get_name(self, name_index: int) -> str:
        """Get name by index"""
        if 0 <= name_index < len(self.face_names):
            return self.face_names[name_index]
        return "Unknown"
    
    def remove_person(self, name: str) -> bool:
        """
        Remove all faces of a person from database
        Args:
            name: Name of person to remove
        Returns:
            True if removed successfully
        """
        if name in self.faces:
            del self.faces[name]
            if name in self.face_names:
                self.face_names.remove(name)
            print(f"Removed {name} from database")
            return True
        return False
    
    def clear_database(self):
        """Clear all faces from database"""
        self.faces = {}
        self.face_names = []
        print("Database cleared")
    
    def save_database(self) -> bool:
        """
        Save database to file
        Returns:
            True if saved successfully
        """
        try:
            database_data = {
                'face_names': self.face_names,
                'faces': self.faces,
                'max_items': self.max_items
            }
            
            with open(self.database_path, 'w') as f:
                json.dump(database_data, f, indent=2)
            
            print(f"Database saved to {self.database_path}")
            return True
            
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def load_database(self) -> bool:
        """
        Load database from file
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(self.database_path):
            print(f"Database file not found: {self.database_path}")
            return False
        
        try:
            with open(self.database_path, 'r') as f:
                database_data = json.load(f)

            self.face_names = database_data.get('face_names', [])
            raw_faces: Dict[str, List[Any]] = database_data.get('faces', {})
            self.faces = self._normalise_faces(raw_faces)
            self.max_items = database_data.get('max_items', 2000)

            total_faces = sum(len(features) for features in self.faces.values())
            print(f"Database loaded: {len(self.face_names)} people, {total_faces} total faces")
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def _normalise_faces(self, raw_faces: Dict[str, List[Any]]) -> Dict[str, List[List[float]]]:
        """Normalise face feature payloads from different on-disk formats."""

        normalised: Dict[str, List[List[float]]] = {}

        for name, entries in raw_faces.items():
            normalised[name] = []

            for entry in entries:
                feature_vector: Optional[List[float]] = None

                if isinstance(entry, dict):
                    features_payload = entry.get('features')

                    if isinstance(features_payload, list):
                        feature_vector = [float(x) for x in features_payload]
                    elif isinstance(features_payload, str):
                        try:
                            decoded = base64.b64decode(features_payload)
                            feature_array = np.frombuffer(decoded, dtype=np.float32)

                            # If feature_dim provided, trim/pad accordingly
                            feature_dim = entry.get('feature_dim')
                            if feature_dim is not None and feature_array.size >= int(feature_dim):
                                feature_array = feature_array[: int(feature_dim)]

                            feature_vector = feature_array.astype(np.float32).tolist()
                        except (ValueError, TypeError):
                            feature_vector = None
                elif isinstance(entry, (list, tuple)):
                    feature_vector = [float(x) for x in entry]

                if feature_vector:
                    normalised[name].append(feature_vector)

        return normalised
    
    def auto_populate_from_directory(self, images_dir: str) -> int:
        """
        Automatically populate database from image directory
        Expected structure: images_dir/person_name/image.jpg
        Args:
            images_dir: Directory containing person folders with images
        Returns:
            Number of faces added
        """
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return 0
        
        added_count = 0
        
        for person_name in os.listdir(images_dir):
            person_dir = os.path.join(images_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            print(f"Processing {person_name}...")
            
            # Process all images for this person
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                image_path = os.path.join(person_dir, image_file)
                
                if self.add_face_from_file(person_name, image_path):
                    added_count += 1
                    print(f"  Added {image_file}")
        
        print(f"Auto-populated database with {added_count} faces")
        return added_count
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        total_faces = sum(len(features) for features in self.faces.values())
        
        stats = {
            'total_people': len(self.face_names),
            'total_faces': total_faces,
            'max_items': self.max_items,
            'usage_percent': (total_faces / self.max_items) * 100 if self.max_items > 0 else 0
        }
        
        # Per-person statistics
        person_stats = {}
        for name in self.face_names:
            person_stats[name] = len(self.faces.get(name, []))
        
        stats['person_stats'] = person_stats
        
        return stats
    
    def print_statistics(self):
        """Print database statistics"""
        stats = self.get_statistics()
        
        print("=== Face Database Statistics ===")
        print(f"Total People: {stats['total_people']}")
        print(f"Total Faces: {stats['total_faces']}")
        print(f"Database Usage: {stats['usage_percent']:.1f}%")
        print("\nPer-person breakdown:")
        
        for name, count in stats['person_stats'].items():
            print(f"  {name}: {count} faces")