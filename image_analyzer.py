#!/usr/bin/env python3
"""
Image Preprocessing and Analysis Tool
Analyze current images and provide cropping/formatting utilities

Created: 2025
"""

import os
import cv2
import sys
from typing import Dict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_detector import FaceDetector
from image_processor import ImageProcessor, SUPPORTED_FORMATS, DEFAULT_TARGET_SIZE

class ImageAnalyzer:
    """Analyze and preprocess face images"""
    
    def __init__(self):
        """Initialize image analyzer"""
        self.face_detector = FaceDetector()
        self.target_size = DEFAULT_TARGET_SIZE  # Standard ArcFace input size
        self.supported_formats = list(SUPPORTED_FORMATS)
        
    def analyze_database_images(self, images_dir: str = "images/") -> Dict:
        """
        Analyze all images in database directory
        Returns analysis results
        """
        print("🔍 ANALYZING DATABASE IMAGES")
        print("=" * 50)
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return {}
        
        analysis = {
            'total_images': 0,
            'total_people': 0,
            'issues': [],
            'people_stats': {},
            'format_distribution': {},
            'size_distribution': {},
            'face_detection_stats': {
                'images_with_faces': 0,
                'images_without_faces': 0,
                'multiple_faces': 0,
                'total_faces_detected': 0
            }
        }
        
        # Scan all person directories
        for person_name in os.listdir(images_dir):
            person_dir = os.path.join(images_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            print(f"\n👤 Analyzing: {person_name}")
            analysis['total_people'] += 1
            
            person_stats = {
                'images': 0,
                'formats': {},
                'sizes': {},
                'face_detection': {
                    'with_faces': 0,
                    'without_faces': 0,
                    'multiple_faces': 0,
                    'total_faces': 0
                },
                'issues': []
            }
            
            # Analyze each image
            for image_file in os.listdir(person_dir):
                if not any(image_file.lower().endswith(fmt) for fmt in self.supported_formats):
                    continue
                
                image_path = os.path.join(person_dir, image_file)
                analysis['total_images'] += 1
                person_stats['images'] += 1
                
                # Analyze single image
                img_analysis = self._analyze_single_image(image_path)
                
                if img_analysis:
                    # Format distribution
                    fmt = img_analysis['format']
                    person_stats['formats'][fmt] = person_stats['formats'].get(fmt, 0) + 1
                    analysis['format_distribution'][fmt] = analysis['format_distribution'].get(fmt, 0) + 1
                    
                    # Size distribution
                    size = img_analysis['size']
                    size_key = f"{size[1]}x{size[0]}"  # height x width
                    person_stats['sizes'][size_key] = person_stats['sizes'].get(size_key, 0) + 1
                    analysis['size_distribution'][size_key] = analysis['size_distribution'].get(size_key, 0) + 1
                    
                    # Face detection
                    faces_count = img_analysis['faces_detected']
                    person_stats['face_detection']['total_faces'] += faces_count
                    analysis['face_detection_stats']['total_faces_detected'] += faces_count
                    
                    if faces_count == 0:
                        person_stats['face_detection']['without_faces'] += 1
                        analysis['face_detection_stats']['images_without_faces'] += 1
                        person_stats['issues'].append(f"No face detected: {image_file}")
                    elif faces_count == 1:
                        person_stats['face_detection']['with_faces'] += 1
                        analysis['face_detection_stats']['images_with_faces'] += 1
                    else:
                        person_stats['face_detection']['multiple_faces'] += 1
                        analysis['face_detection_stats']['multiple_faces'] += 1
                        person_stats['issues'].append(f"Multiple faces ({faces_count}): {image_file}")
                    
                    # Check for issues
                    if img_analysis['issues']:
                        person_stats['issues'].extend([f"{image_file}: {issue}" for issue in img_analysis['issues']])
            
            analysis['people_stats'][person_name] = person_stats
            
            # Print person summary
            print(f"   📸 Images: {person_stats['images']}")
            print(f"   👁️  With faces: {person_stats['face_detection']['with_faces']}")
            print(f"   ❌ Without faces: {person_stats['face_detection']['without_faces']}")
            print(f"   👥 Multiple faces: {person_stats['face_detection']['multiple_faces']}")
            if person_stats['issues']:
                print(f"   ⚠️  Issues: {len(person_stats['issues'])}")
        
        self._print_analysis_summary(analysis)
        return analysis
    
    def _analyze_single_image(self, image_path: str) -> Dict:
        """Analyze a single image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'issues': ['Could not load image']}
            
            h, w, c = img.shape
            file_ext = os.path.splitext(image_path)[1].lower()
            
            # Detect faces
            faces = self.face_detector.detect_faces(img)
            
            # Check for issues
            issues = []
            if w < 100 or h < 100:
                issues.append("Too small (< 100px)")
            if w != h:
                issues.append("Not square")
            if (w, h) != self.target_size:
                issues.append(f"Wrong size (should be {self.target_size})")
                
            return {
                'size': (w, h),
                'channels': c,
                'format': file_ext,
                'faces_detected': len(faces),
                'faces': faces,
                'issues': issues
            }
            
        except Exception as e:
            return {'issues': [f'Error analyzing: {str(e)}']}
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print analysis summary"""
        print(f"\n📊 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"👥 Total People: {analysis['total_people']}")
        print(f"📸 Total Images: {analysis['total_images']}")
        print()
        
        # Face detection summary
        fds = analysis['face_detection_stats']
        print("👁️  Face Detection Stats:")
        print(f"   ✅ Images with faces: {fds['images_with_faces']}")
        print(f"   ❌ Images without faces: {fds['images_without_faces']}")
        print(f"   👥 Images with multiple faces: {fds['multiple_faces']}")
        print(f"   🎯 Total faces detected: {fds['total_faces_detected']}")
        
        # Format distribution
        print(f"\n📄 Format Distribution:")
        for fmt, count in analysis['format_distribution'].items():
            print(f"   {fmt}: {count} images")
        
        # Size distribution  
        print(f"\n📐 Size Distribution:")
        for size, count in analysis['size_distribution'].items():
            print(f"   {size}: {count} images")
        
        # Recommendations
        self._print_recommendations(analysis)
    
    def _print_recommendations(self, analysis: Dict):
        """Print recommendations based on analysis"""
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 50)
        
        fds = analysis['face_detection_stats']
        
        # Face detection issues
        if fds['images_without_faces'] > 0:
            print(f"❌ {fds['images_without_faces']} images have no face detected!")
            print("   → Consider removing or replacing these images")
        
        if fds['multiple_faces'] > 0:
            print(f"⚠️  {fds['multiple_faces']} images have multiple faces")
            print("   → Consider cropping to single face")
        
        # Format consistency
        if len(analysis['format_distribution']) > 1:
            print(f"📄 Multiple formats detected: {list(analysis['format_distribution'].keys())}")
            print("   → Consider converting all to .jpg for consistency")
        
        # Size consistency
        if len(analysis['size_distribution']) > 1:
            print(f"📐 Multiple sizes detected")
            print(f"   → Consider resizing all to {self.target_size[0]}x{self.target_size[1]}")
        
        # Overall recommendation
        print(f"\n🚀 FOR BEST ACCURACY:")
        print("   1. Crop all images to face only")
        print(f"   2. Resize to {self.target_size[0]}x{self.target_size[1]} pixels")
        print("   3. Convert to .jpg format")
        print("   4. Remove images without faces")
        print("   5. Split images with multiple faces")
        
        print(f"\n🛠️  Use: python image_processor.py --input images/ --output images_processed/")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Analysis and Preprocessing Tool")
    parser.add_argument('--analyze', type=str, default='images/', help='Directory to analyze')
    parser.add_argument('--process', type=str, help='Directory to process (crop and resize)')
    parser.add_argument('--output', type=str, help='Output directory for processed images')
    
    args = parser.parse_args()
    
    if args.process:
        print("🔧 Processing images...")
        processor = ImageProcessor()
        processor.crop_and_process_images(args.process, args.output)
    else:
        analyzer = ImageAnalyzer()
        print("🔍 Analyzing images...")
        analyzer.analyze_database_images(args.analyze)
        
        print(f"\n💡 TO PROCESS IMAGES:")
        print(f"python {__file__} --process {args.analyze}")

if __name__ == "__main__":
    main()