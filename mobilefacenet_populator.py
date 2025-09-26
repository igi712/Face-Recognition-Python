#!/usr/bin/env python3
"""
MobileFaceNet Database Populator
Populate face_database_mobilefacenet.json with ArcFace/MobileFaceNet features

Created: September 21, 2025
"""

import sys
import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
import argparse
import base64

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_features_arcface import MobileFaceNetExtractor

class MobileFaceNetPopulator:
    """Populate database with MobileFaceNet features"""
    
    def __init__(self, database_path: str = "face_database_mobilefacenet.json"):
        """Initialize populator"""
        self.database_path = database_path
        self.feature_extractor = MobileFaceNetExtractor()
        
        # Initialize empty database structure
        self.database = {
            'faces': {},
            'face_names': [],
            'created_at': '2025-09-21',
            'feature_type': 'MobileFaceNet_ArcFace',
            'total_faces': 0,
            'total_people': 0
        }
        
        print(f"🚀 MobileFaceNet Database Populator")
        print(f"📁 Database: {database_path}")
        print(f"🎯 Feature Type: MobileFaceNet + ArcFace")
    
    def clear_database(self):
        """Clear database"""
        self.database = {
            'faces': {},
            'face_names': [],
            'created_at': '2025-09-21',
            'feature_type': 'MobileFaceNet_ArcFace',
            'total_faces': 0,
            'total_people': 0
        }
        self._save_database()
        print("✅ Database cleared")
    
    def populate_from_directory(self, images_dir: str = "images_processed/"):
        """Populate database from images directory"""
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        print(f"\n🔄 Populating from: {images_dir}")
        print("="*60)
        
        total_processed = 0
        total_errors = 0
        
        # Process each person directory
        for person_name in sorted(os.listdir(images_dir)):
            person_dir = os.path.join(images_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            print(f"\n👤 Processing: {person_name}")
            
            # Add person to database
            if person_name not in self.database['face_names']:
                self.database['face_names'].append(person_name)
            
            person_index = self.database['face_names'].index(person_name)
            
            # Initialize person's face list
            if person_name not in self.database['faces']:
                self.database['faces'][person_name] = []
            
            # Process each image
            person_face_count = 0
            for image_file in sorted(os.listdir(person_dir)):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                image_path = os.path.join(person_dir, image_file)
                
                try:
                    # Load and process image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"   ❌ Failed to load: {image_file}")
                        total_errors += 1
                        continue
                    
                    # Extract features using MobileFaceNet
                    features = self.feature_extractor.extract_feature(img)
                    if features is None:
                        print(f"   ❌ No features: {image_file}")
                        total_errors += 1
                        continue
                    
                    # Convert features to base64 for JSON storage
                    features_b64 = base64.b64encode(features.astype(np.float32).tobytes()).decode('utf-8')
                    
                    # Add to database
                    face_data = {
                        'features': features_b64,
                        'image_file': image_file,
                        'feature_dim': len(features),
                        'feature_norm': float(np.linalg.norm(features)),
                        'person_index': person_index
                    }
                    
                    self.database['faces'][person_name].append(face_data)
                    person_face_count += 1
                    total_processed += 1
                    
                    print(f"   ✅ {image_file} -> {len(features)}D features (norm: {face_data['feature_norm']:.3f})")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {image_file}: {str(e)}")
                    total_errors += 1
            
            print(f"   📊 Added {person_face_count} faces for {person_name}")
        
        # Update database stats
        self.database['total_people'] = len(self.database['face_names'])
        self.database['total_faces'] = total_processed
        
        # Save database
        self._save_database()
        
        print(f"\n🎊 POPULATION COMPLETE!")
        print(f"✅ Processed: {total_processed} faces")
        print(f"❌ Errors: {total_errors}")
        print(f"👥 Total people: {self.database['total_people']}")
        print(f"📊 Database saved to: {self.database_path}")
        
        return total_processed > 0
    
    def _save_database(self):
        """Save database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving database: {str(e)}")
            return False
    
    def print_stats(self):
        """Print database statistics"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r') as f:
                    db = json.load(f)
            else:
                db = self.database
            
            print(f"\n📊 DATABASE STATISTICS")
            print("="*50)
            print(f"📁 Database: {self.database_path}")
            print(f"🎯 Feature Type: {db.get('feature_type', 'Unknown')}")
            print(f"👥 Total People: {db.get('total_people', 0)}")
            print(f"📸 Total Faces: {db.get('total_faces', 0)}")
            print(f"📅 Created: {db.get('created_at', 'Unknown')}")
            
            if 'faces' in db and db['faces']:
                print(f"\n👤 Per-person breakdown:")
                for person_name, faces in db['faces'].items():
                    print(f"   {person_name}: {len(faces)} faces")
            
            print()
        except Exception as e:
            print(f"❌ Error reading database: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="MobileFaceNet Database Populator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--database', type=str, default='face_database_mobilefacenet.json',
                       help='Database file path')
    parser.add_argument('--images', type=str, default='images_processed/',
                       help='Images directory to process')
    parser.add_argument('--clear', action='store_true',
                       help='Clear database before populating')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics only')
    parser.add_argument('--populate', action='store_true', default=True,
                       help='Populate database from images')
    
    args = parser.parse_args()
    
    # Initialize populator
    populator = MobileFaceNetPopulator(args.database)
    
    if args.stats:
        populator.print_stats()
        return
    
    if args.clear:
        populator.clear_database()
    
    if args.populate and not args.stats:
        success = populator.populate_from_directory(args.images)
        if success:
            print(f"\n🚀 Database ready for use with:")
            print(f"   python face_recognition_main.py 0 --database {args.database}")
        else:
            print(f"\n❌ Population failed!")
    
    # Show final stats
    populator.print_stats()

if __name__ == "__main__":
    main()