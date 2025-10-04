#!/usr/bin/env python3
"""
Script untuk rebuild database dengan Z-score normalization (Jetson Nano compatible)

PENTING: Database lama tidak kompatibel dengan normalization baru!
Script ini akan:
1. Backup database lama
2. Populate database baru dengan Z-score normalization
3. Verifikasi hasilnya

Usage:
    python rebuild_database_zscore.py
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.face_database import FaceDatabase
from src.face_detector import FaceDetector
import cv2


def backup_old_database(db_path: str) -> bool:
    """Backup database lama"""
    if not os.path.exists(db_path):
        print(f"ℹ️  No existing database found at {db_path}")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backed up old database to: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to backup database: {e}")
        return False


def rebuild_database(
    images_dir: str = "images_processed",
    db_path: str = "face_database.json"
) -> bool:
    """Rebuild database dengan Z-score normalization"""
    
    print("=" * 60)
    print("🔄 REBUILDING DATABASE WITH Z-SCORE NORMALIZATION")
    print("=" * 60)
    
    # Backup database lama jika ada
    backup_old_database(db_path)
    
    # Hapus database lama
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"🗑️  Removed old database: {db_path}")
        except Exception as e:
            print(f"❌ Failed to remove old database: {e}")
            return False
    
    # Cek direktori images
    if not os.path.isdir(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("ℹ️  Please process images first using:")
        print("   python -m image_processor")
        return False
    
    # Initialize components dengan Z-score normalization
    print("\n📦 Initializing with Z-score normalization (Jetson Nano compatible)...")
    face_db = FaceDatabase(
        database_path=db_path,
        use_arcface=True,
        use_zscore_norm=True  # ✅ Z-score seperti Jetson Nano
    )
    
    face_detector = FaceDetector()
    
    # Populate database
    print(f"\n📂 Processing images from: {images_dir}")
    print("-" * 60)
    
    processed_count = 0
    error_count = 0
    
    for person_name in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        print(f"\n👤 Processing: {person_name}")
        person_processed = 0
        
        for image_file in os.listdir(person_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path = os.path.join(person_dir, image_file)
            
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"   ⚠️  Could not load: {image_file}")
                    error_count += 1
                    continue
                
                # Detect face and landmarks
                faces = face_detector.detect_faces(image)
                if not faces or len(faces) == 0:
                    print(f"   ⚠️  No face detected: {image_file}")
                    error_count += 1
                    continue
                
                # Use largest face if multiple detected
                if len(faces) > 1:
                    faces.sort(key=lambda f: f.rect[2] * f.rect[3], reverse=True)
                
                face = faces[0]
                x, y, w, h = face.rect
                
                # Crop face region
                face_region = image[y:y+h, x:x+w]
                
                # Add to database with landmarks
                success = face_db.add_face(person_name, face_region, face.landmark)
                
                if success:
                    processed_count += 1
                    person_processed += 1
                    print(f"   ✅ {image_file}")
                else:
                    error_count += 1
                    print(f"   ❌ Failed to add: {image_file}")
                    
            except Exception as e:
                error_count += 1
                print(f"   ❌ Error processing {image_file}: {e}")
        
        if person_processed > 0:
            print(f"   📊 Added {person_processed} images for {person_name}")
    
    # Save database
    print("\n💾 Saving database...")
    face_db.save_database()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 REBUILD COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Errors: {error_count} images")
    print(f"💾 Database saved to: {db_path}")
    print(f"📝 Normalization: Z-score (Jetson Nano compatible)")
    print(f"🎯 Recommended threshold: 0.35")
    print("=" * 60)
    
    # Verification
    print("\n🔍 VERIFICATION")
    face_db.print_statistics()
    
    return processed_count > 0


def main():
    """Main entry point"""
    
    print("\n" + "=" * 60)
    print("🚀 DATABASE REBUILD SCRIPT")
    print("=" * 60)
    print("This script will rebuild the face database with Z-score")
    print("normalization to match Jetson Nano implementation.")
    print("\nIMPORTANT:")
    print("- Old database will be backed up automatically")
    print("- This uses images from 'images_processed' directory")
    print("- Process images first if you haven't already")
    print("=" * 60)
    
    response = input("\n❓ Continue with rebuild? (y/n): ").lower().strip()
    if response != 'y':
        print("❌ Rebuild cancelled")
        return
    
    # Rebuild database
    success = rebuild_database()
    
    if success:
        print("\n✅ Database rebuild successful!")
        print("\n📋 NEXT STEPS:")
        print("1. Test with image:")
        print("   python face_recognition_main.py test.jpg")
        print("\n2. Test with video:")
        print("   python -m app.cli video --source pakmarwan.mp4")
        print("\n3. Expected confidence: 0.6 - 0.9 (like Jetson Nano!)")
    else:
        print("\n❌ Database rebuild failed!")
        print("Please check the errors above and try again.")


if __name__ == "__main__":
    main()
