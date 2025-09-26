#!/usr/bin/env python3
"""
Face Recognition Main Application - Python Implementation
Based on Face-Recognition-Jetson-Nano project
Enhanced with ArcFace support for superior accuracy

Created: 2025
"""

import sys
import os
import cv2
import argparse
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_detector import FaceDetector, FaceObject
from src.face_features import FaceFeatureExtractor
from src.face_database import FaceDatabase
from src.face_quality import FaceQualityAssessment

class FaceRecognitionSystem:
    """Main face recognition system with ArcFace support"""
    
    def __init__(self, config: dict = None, use_arcface: bool = True, arcface_model_path: str = None):
        """
        Initialize face recognition system
        Args:
            config: Configuration dictionary
            use_arcface: Whether to use ArcFace features (more accurate)
            arcface_model_path: Path to ArcFace ONNX model
        """
        # Default configuration
        self.config = {
            'min_face_size': 90,
            'face_threshold': 0.5,
            'recognition_threshold': 0.5, 
            'liveness_threshold': 0.93,
            'max_blur': -25.0,
            'max_angle': 10.0,
            'max_database_items': 2000,
            'show_landmarks': True,
            'show_legend': True,
            'enable_liveness': True,
            'enable_blur_filter': True,
            'auto_add_faces': False,
            'database_path': 'face_database_mobilefacenet.json',
            'images_directory': 'images',
            'use_arcface': use_arcface,
            'arcface_model_path': arcface_model_path,
            'detection_downscale': 1.0,
            'detection_frame_size': (320, 240),
            'quality_interval': 1,
            'fast_mode': False,
            'quality_cache_bucket': 16,
            'opencv_threads': None,
        }
        
        # Update with provided config
        if config:
            self.config.update(config)

        detection_frame_override = bool(config and 'detection_frame_size' in config)

        self._overrides = self.config.pop('_overrides', {}) if isinstance(self.config.get('_overrides'), dict) else {}
        self._overrides.setdefault('detection_frame_size', detection_frame_override)

        # Avoid conflicting overrides from CLI/fast-mode metadata
        if detection_frame_override and not self._overrides.get('detection_frame_size'):
            self._overrides['detection_frame_size'] = True

        if self.config.get('fast_mode'):
            if not self._overrides.get('detection_downscale') and self.config.get('detection_downscale', 1.0) == 1.0:
                self.config['detection_downscale'] = 0.6
            if not self._overrides.get('enable_liveness'):
                self.config['enable_liveness'] = False
            if not self._overrides.get('enable_blur_filter'):
                self.config['enable_blur_filter'] = False
            if not self._overrides.get('show_landmarks'):
                self.config['show_landmarks'] = False
            if not self._overrides.get('quality_interval'):
                self.config['quality_interval'] = max(self.config.get('quality_interval', 1), 3)
            if not self._overrides.get('detection_frame_size'):
                self.config['detection_frame_size'] = None

        self.config['quality_interval'] = max(1, int(self.config.get('quality_interval', 1)))
        downscale = float(self.config.get('detection_downscale', 1.0))
        if downscale <= 0:
            downscale = 1.0
        self.config['detection_downscale'] = max(min(downscale, 1.0), 0.1)
        self.config['quality_cache_bucket'] = max(4, int(self.config.get('quality_cache_bucket', 16)))
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.feature_extractor = FaceFeatureExtractor()
        self.face_database = FaceDatabase(
            database_path=self.config['database_path'],
            max_items=self.config['max_database_items'],
            use_arcface=self.config['use_arcface'],
            arcface_model_path=self.config['arcface_model_path']
        )
        self.quality_assessor = FaceQualityAssessment(
            fast_mode=self.config.get('fast_mode', False)
        )

        self.frame_count = 0
        self._quality_cache: Dict[Tuple[int, int, int, int], Tuple[dict, int]] = {}

        try:
            cv2.setUseOptimized(True)
            threads = self.config.get('opencv_threads')
            if threads is not None:
                threads_int = int(threads)
                if threads_int >= 0:
                    cv2.setNumThreads(threads_int)
        except Exception:
            pass
        
        # Performance tracking
        self.fps_buffer = [0.0] * 16
        self.fps_index = 0
        
        feature_type = "ArcFace" if self.config['use_arcface'] else "Legacy"
        print(f"Face Recognition System initialized with {feature_type} features")
        self.face_database.print_statistics()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[FaceObject]]:
        """
        Process a single frame for face recognition
        Args:
            frame: Input frame
        Returns:
            (processed_frame, detected_faces)
        """
        self.frame_count += 1

        detection_resize = None
        downscale = self.config.get('detection_downscale', 1.0)

        if self.face_detector.use_retina:
            if downscale < 1.0:
                detection_resize = downscale
        else:
            target_size = self.config.get('detection_frame_size')
            if isinstance(target_size, (tuple, list)) and len(target_size) >= 2 and target_size[0] and target_size[1]:
                frame_h, frame_w = frame.shape[:2]
                target_w = max(1, int(target_size[0]))
                target_h = max(1, int(target_size[1]))
                scale_x = target_w / float(frame_w) if frame_w > 0 else 1.0
                scale_y = target_h / float(frame_h) if frame_h > 0 else 1.0
                scale_x = min(scale_x, 1.0)
                scale_y = min(scale_y, 1.0)
                if scale_x < 1.0 or scale_y < 1.0:
                    detection_resize = (scale_x, scale_y)
            if detection_resize is None and downscale < 1.0:
                detection_resize = downscale

        faces = self.face_detector.detect_faces(frame, resize_factor=detection_resize)
        
        # Filter faces by minimum size
        faces = self.face_detector.filter_faces(
            faces,
            self.config['min_face_size'],
            self.config.get('face_threshold', 0.0),
        )
        
        # Process each face
        for face in faces:
            self._process_single_face(frame, face)
        
        # Draw results
        result_frame = self._draw_results(frame, faces)
        
        self._cleanup_quality_cache()

        return result_frame, faces
    
    def _process_single_face(self, frame: np.ndarray, face: FaceObject):
        """Process a single detected face"""
        x, y, w, h = face.rect
        
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return

        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        if face_region.size == 0:
            return

        relative_landmarks: Optional[List[Tuple[float, float]]] = None
        if face.landmark:
            relative_landmarks = []
            for lx, ly in face.landmark:
                rel_x = float(lx - x)
                rel_y = float(ly - y)
                rel_x = float(np.clip(rel_x, 0.0, float(w)))
                rel_y = float(np.clip(rel_y, 0.0, float(h)))
                relative_landmarks.append((rel_x, rel_y))
        
        quality_interval = self.config.get('quality_interval', 1)
        cache_key = self._make_face_cache_key(face)
        quality_results = None

        if cache_key and quality_interval > 1:
            cached = self._quality_cache.get(cache_key)
            if cached and (self.frame_count - cached[1]) < quality_interval:
                quality_results = cached[0]

        if quality_results is None:
            quality_results = self.quality_assessor.comprehensive_quality_check(
                face_region,
                relative_landmarks or face.landmark,
                require_liveness=False,
                require_blur=self.config.get('enable_blur_filter', True)
            )
            if cache_key:
                self._quality_cache[cache_key] = (quality_results, self.frame_count)
        else:
            self._quality_cache[cache_key] = (quality_results, self.frame_count)
        
        # Update face object with quality info
        face.angle = quality_results['face_angle']

        # Face recognition before tiny check to mirror Jetson flow
        name_index, confidence = self.face_database.recognize_face(
            face_region,
            relative_landmarks or face.landmark,
            self.config['recognition_threshold']
        )
        
        face.name_index = name_index
        face.name_prob = confidence
        face.live_prob = quality_results.get('liveness_score', 1.0)

        if h < self.config['min_face_size']:
            if face.name_index < 0:
                face.name_index = -1
            face.color = 2
            return

        needs_liveness = (
            self.config.get('enable_liveness', True)
            and face.name_index >= 0
        )

        if needs_liveness:
            live_score, is_live = self.quality_assessor.assess_liveness(face_region)
            face.live_prob = live_score
            quality_results['liveness_score'] = live_score
            quality_results['is_live'] = is_live
            if not is_live:
                face.name_index = -3
                face.color = 3
                return
        else:
            quality_results.setdefault('liveness_score', face.live_prob)
            quality_results.setdefault('is_live', True)

        if name_index >= 0:
            face.color = 0
        else:
            face.color = 1
            if self.config['auto_add_faces'] and quality_results['is_good_quality']:
                self._auto_add_face(face_region, relative_landmarks or face.landmark)

    def _make_face_cache_key(self, face: FaceObject) -> Optional[Tuple[int, int, int, int]]:
        if not face.rect:
            return None
        bucket = self.config.get('quality_cache_bucket', 16)
        x, y, w, h = face.rect
        return (
            int(x // bucket),
            int(y // bucket),
            int(w // bucket),
            int(h // bucket)
        )

    def _cleanup_quality_cache(self):
        if not self._quality_cache:
            return
        quality_interval = self.config.get('quality_interval', 1)
        max_age = max(quality_interval * 6, 10)
        cutoff = self.frame_count - max_age
        keys_to_delete = [key for key, (_, frame_idx) in self._quality_cache.items() if frame_idx < cutoff]
        for key in keys_to_delete:
            self._quality_cache.pop(key, None)
    
    def _auto_add_face(self, face_region: np.ndarray, landmarks: List):
        """Automatically add unknown face to database"""
        # Generate name for unknown person
        timestamp = int(time.time())
        unknown_name = f"Unknown_{timestamp}"
        
        if self.face_database.add_face(unknown_name, face_region, landmarks):
            print(f"Auto-added face: {unknown_name}")
    
    def _draw_results(self, frame: np.ndarray, faces: List[FaceObject]) -> np.ndarray:
        """Draw face recognition results on frame"""
        result_frame = frame.copy()
        
        for face in faces:
            self._draw_single_face(result_frame, face)
        
        # Draw legend if enabled
        if self.config['show_legend']:
            self._draw_legend(result_frame, faces)
        
        return result_frame
    
    def _draw_single_face(self, frame: np.ndarray, face: FaceObject):
        """Draw a single face detection result"""
        x, y, w, h = face.rect
        
        # Draw rectangle
        colors = {
            0: (255, 255, 255),  # White - recognized
            1: (80, 255, 255),   # Yellow - stranger
            2: (255, 237, 178),  # Blue - too tiny
            3: (127, 127, 255),  # Red - fake
        }
        color = colors.get(face.color, (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw landmarks if enabled
        if self.config['show_landmarks'] and face.landmark:
            for landmark in face.landmark:
                cv2.circle(frame, landmark, 2, (0, 255, 255), -1)
        
        # Draw label
        label = self._get_face_label(face)
        self._draw_label(frame, label, (x, y), color)
    
    def _get_face_label(self, face: FaceObject) -> str:
        """Get label text for face"""
        if face.name_index == -1:
            return "Stranger"
        elif face.name_index == -2:
            return "Too tiny"
        elif face.name_index == -3:
            return "Fake!"
        elif face.name_index >= 0:
            return self.face_database.get_name(face.name_index)
        else:
            return "Unknown"
    
    def _draw_label(self, frame: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw text label on frame"""
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Adjust position if near edges
        if y < text_height + baseline:
            y = text_height + baseline
        if x + text_width > frame.shape[1]:
            x = frame.shape[1] - text_width
        
        # Draw background rectangle
        cv2.rectangle(frame, (x, y - text_height - baseline), 
                     (x + text_width, y + baseline), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    
    def _draw_legend(self, frame: np.ndarray, faces: List[FaceObject]):
        """Draw information legend on frame"""
        if not faces:
            return
        
        face = faces[0]  # Use first face for legend
        y_offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (180, 180, 0)
        
        # Draw face information
        cv2.putText(frame, f"Angle: {face.angle:.1f}°", 
                   (10, y_offset), font, font_scale, color)
        y_offset += 20
        
        cv2.putText(frame, f"Face prob: {face.face_prob:.4f}", 
                   (10, y_offset), font, font_scale, color)
        y_offset += 20
        
        cv2.putText(frame, f"Name prob: {face.name_prob:.4f}", 
                   (10, y_offset), font, font_scale, color)
        y_offset += 20
        
        if self.config['enable_liveness']:
            if face.color == 2:  # Too tiny
                cv2.putText(frame, "Live prob: ??", 
                           (10, y_offset), font, font_scale, color)
            else:
                cv2.putText(frame, f"Live prob: {face.live_prob:.4f}", 
                           (10, y_offset), font, font_scale, color)
    
    def update_fps(self, fps: float):
        """Update FPS tracking"""
        self.fps_buffer[self.fps_index] = fps
        self.fps_index = (self.fps_index + 1) % len(self.fps_buffer)
    
    def get_average_fps(self) -> float:
        """Get average FPS"""
        return sum(self.fps_buffer) / len(self.fps_buffer)
    
    def add_person_from_directory(self, person_name: str, image_directory: str) -> int:
        """Add a person's faces from directory"""
        return self.face_database.auto_populate_from_directory(image_directory)
    
    def save_database(self):
        """Save face database"""
        self.face_database.save_database()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Face Recognition System - Python Implementation with ArcFace Support",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('input', type=str, nargs='?', default='0',
                       help='Input source (webcam index, image, or video file)')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--database', type=str, default='face_database_mobilefacenet.json',
                       help='Face database file path')
    parser.add_argument('--populate', type=str, help='Auto-populate database from directory')
    parser.add_argument('--threshold', type=float, help='Recognition threshold (auto-set based on feature type)')
    parser.add_argument('--min-size', type=int, default=90,
                       help='Minimum face size in pixels')
    parser.add_argument('--auto-add', action='store_true',
                       help='Automatically add unknown faces')
    parser.add_argument('--show-landmarks', dest='show_landmarks', action='store_const', const=True,
                       help='Draw facial landmarks')
    parser.add_argument('--hide-landmarks', dest='show_landmarks', action='store_const', const=False,
                       help='Do not draw facial landmarks (speed boost)')
    parser.add_argument('--show-legend', dest='show_legend', action='store_true',
                       help='Show information legend (default)')
    parser.add_argument('--hide-legend', dest='show_legend', action='store_false',
                       help='Hide information legend')
    parser.add_argument('--enable-liveness', dest='enable_liveness', action='store_true',
                       help='Enable liveness detection (default)')
    parser.add_argument('--disable-liveness', dest='enable_liveness', action='store_false',
                       help='Disable liveness detection for speed')
    parser.add_argument('--enable-blur-filter', dest='enable_blur', action='store_true',
                       help='Enable blur-based quality filtering (default)')
    parser.add_argument('--disable-blur-filter', dest='enable_blur', action='store_false',
                       help='Disable blur filtering for speed')
    parser.add_argument('--detection-downscale', type=float, default=1.0,
                       help='Resize factor (<1.0) applied before detection to boost FPS')
    parser.add_argument('--quality-interval', type=int, default=1,
                       help='Run blur/liveness checks every N frames (>=1)')
    parser.add_argument('--fast', action='store_true',
                       help='Enable preset speed optimizations (changes several settings)')
    parser.add_argument('--opencv-threads', type=int,
                       help='Override OpenCV thread count (0 lets OpenCV decide)')
    
    # ArcFace options
    parser.add_argument('--use-arcface', action='store_true', default=True,
                       help='Use ArcFace features (default: True, more accurate)')
    parser.add_argument('--use-legacy', action='store_true',
                       help='Force use of legacy features (less accurate)')
    parser.add_argument('--arcface-model', type=str,
                       help='Path to ArcFace ONNX model file')
    
    parser.set_defaults(show_legend=True, enable_liveness=False, enable_blur=True)

    args = parser.parse_args()

    provided_flags = {arg.split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}

    def flag_provided(*names: str) -> bool:
        return any(name in provided_flags for name in names)

    overrides = {
        'show_landmarks': flag_provided('--show-landmarks', '--hide-landmarks'),
        'enable_liveness': flag_provided('--disable-liveness', '--enable-liveness'),
        'enable_blur_filter': flag_provided('--disable-blur-filter', '--enable-blur-filter'),
        'detection_downscale': flag_provided('--detection-downscale'),
        'quality_interval': flag_provided('--quality-interval'),
    }

    input_source_hint = args.input
    is_camera_input = input_source_hint.isdigit()

    if not overrides['enable_liveness']:
        args.enable_liveness = is_camera_input

    if args.fast:
        if not overrides['detection_downscale'] and args.detection_downscale == 1.0:
            args.detection_downscale = 0.6
        if not overrides['quality_interval']:
            args.quality_interval = max(args.quality_interval, 3)
        if not overrides['enable_liveness']:
            args.enable_liveness = False
        if not overrides['enable_blur_filter']:
            args.enable_blur = False
        if not overrides['show_landmarks'] and args.show_landmarks is None:
            args.show_landmarks = False
    
    # Determine feature type
    use_arcface = args.use_arcface and not args.use_legacy
    if args.use_legacy:
        use_arcface = False
        print("🔄 Using legacy features (forced by --use-legacy)")
    elif use_arcface:
        print("🚀 Using ArcFace features (more accurate)")
    
    # Auto-set threshold based on feature type
    if args.threshold is None:
        threshold = 0.4 if use_arcface else 0.8
    else:
        threshold = args.threshold
    
    # Configuration
    config = {
        'database_path': args.database,
        'recognition_threshold': threshold,
        'min_face_size': args.min_size,
        'show_legend': args.show_legend,
        'enable_liveness': args.enable_liveness,
        'auto_add_faces': args.auto_add,
        'use_arcface': use_arcface,
        'arcface_model_path': args.arcface_model,
        'enable_blur_filter': args.enable_blur,
        'detection_downscale': args.detection_downscale,
        'quality_interval': args.quality_interval,
        'fast_mode': args.fast,
        'opencv_threads': args.opencv_threads,
    }

    if args.show_landmarks is not None:
        config['show_landmarks'] = args.show_landmarks

    config['_overrides'] = overrides
    
    # Initialize system
    face_recognition = FaceRecognitionSystem(config, use_arcface, args.arcface_model)
    
    # Auto-populate database if requested
    if args.populate:
        print(f"Auto-populating database from {args.populate}")
        face_recognition.add_person_from_directory("", args.populate)
        face_recognition.save_database()
    
    # Determine input source
    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)  # Webcam
        is_camera = True
    else:
        is_camera = False
    
    # Open input source
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open input source: {input_source}")
        return
    
    print("Face Recognition System started. Press 'q' to quit, 's' to save database.")
    
    # Main processing loop
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    print("Error reading from camera")
                    break
                else:
                    print("End of video file")
                    break
            
            # Process frame
            result_frame, faces = face_recognition.process_frame(frame)
            
            # Calculate and update FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if end_time > start_time else 0
            face_recognition.update_fps(fps)
            
            # Draw FPS
            avg_fps = face_recognition.get_average_fps()
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Face Recognition System', result_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                face_recognition.save_database()
                print("Database saved")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save database
        face_recognition.save_database()
        print("Face Recognition System stopped")

if __name__ == "__main__":
    main()