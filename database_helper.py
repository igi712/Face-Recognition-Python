#!/usr/bin/env python3
"""
Database Operations Helper
Simple commands for all database operations

Created: September 21, 2025
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run command with description"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    result = os.system(cmd)
    if result == 0:
        print(f"✅ Success: {description}")
    else:
        print(f"❌ Failed: {description}")
    print("-" * 60)
    return result == 0

def main():
    """Main function - show all available commands"""
    print("🗄️  FACE DATABASE OPERATIONS")
    print("=" * 80)
    
    print("\n📋 AVAILABLE COMMANDS:")
    
    print("\n🔧 1. MOBILEFACENET DATABASE OPERATIONS:")
    print("   # Clear dan populate MobileFaceNet database")
    print("   python mobilefacenet_populator.py --clear --images images/")
    print("")
    print("   # Populate dari images/ (tanpa clear)")
    print("   python mobilefacenet_populator.py --images images/")
    print("")
    print("   # Populate dari images_processed/ (yang sudah dioptimasi)")
    print("   python mobilefacenet_populator.py --clear --images images_processed/")
    print("")
    print("   # Cek statistics")
    print("   python mobilefacenet_populator.py --stats")
    
    print("\n🖼️  2. IMAGE PROCESSING:")
    print("   # Analisis images untuk cek kualitas")
    print("   python image_analyzer.py --analyze images/")
    print("")
    print("   # Process images (crop, resize, optimize)")
    print("   python image_analyzer.py --process images/ --output images_processed/")
    
    print("\n🎮 3. FACE RECOGNITION:")
    print("   # Run dengan MobileFaceNet database")
    print("   python face_recognition_main.py 0 --database face_database_mobilefacenet.json")
    print("")
    print("   # Run dengan default database")
    print("   python face_recognition_main.py 0")
    
    print("\n📊 4. DATABASE MANAGEMENT:")
    print("   # General database operations")
    print("   python db_manager.py --clear --populate images/")
    print("   python db_manager.py --stats")
    
    print("\n" + "=" * 80)
    print("🚀 RECOMMENDED WORKFLOW:")
    print("1. python image_analyzer.py --analyze images/")
    print("2. python image_analyzer.py --process images/ --output images_processed/")  
    print("3. python mobilefacenet_populator.py --clear --images images_processed/")
    print("4. python face_recognition_main.py 0 --database face_database_mobilefacenet.json")
    print("=" * 80)

if __name__ == "__main__":
    main()