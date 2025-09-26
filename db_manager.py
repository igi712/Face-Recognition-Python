#!/usr/bin/env python3
"""
Database Manager Script
Easy-to-use script for managing the face recognition database

Created: 2025
"""

import sys
import os
import argparse
from typing import Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_database import FaceDatabase
from src.face_features import FaceFeatureExtractor
from src.config_manager import ConfigManager

class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self, database_path: str = "face_database_mobilefacenet.json"):
        """Initialize database manager"""
        self.database_path = database_path
        self.config = ConfigManager()
        
        # Initialize components
        self.feature_extractor = FaceFeatureExtractor()
        self.database = FaceDatabase(
            database_path=database_path,
            max_items=self.config.get('max_database_items', 2000)
        )
        self.database.set_feature_extractor(self.feature_extractor)
        
        print(f"Database Manager initialized")
        print(f"Database path: {database_path}")
    
    def clear_database(self):
        """Clear all faces from database"""
        print("\n🗑️  Clearing database...")
        self.database.clear_database()
        self.database.save_database()
        print("✅ Database cleared successfully!")
    
    def populate_database(self, images_dir: str):
        """Populate database from images directory"""
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        print(f"\n📁 Populating database from: {images_dir}")
        added_count = self.database.auto_populate_from_directory(images_dir)
        
        if added_count > 0:
            self.database.save_database()
            print(f"✅ Successfully added {added_count} faces to database!")
        else:
            print("⚠️  No faces were added. Check if images directory contains person folders with photos.")
        
        return added_count > 0
    
    def show_statistics(self):
        """Show database statistics"""
        print("\n📊 Database Statistics:")
        self.database.print_statistics()
    
    def backup_database(self, backup_path: Optional[str] = None):
        """Create backup of database"""
        import shutil
        from datetime import datetime
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"face_database_backup_{timestamp}.json"
        
        try:
            shutil.copy2(self.database_path, backup_path)
            print(f"✅ Database backed up to: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        import shutil
        
        if not os.path.exists(backup_path):
            print(f"❌ Backup file not found: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, self.database_path)
            # Reload database
            self.database = FaceDatabase(
                database_path=self.database_path,
                max_items=self.config.get('max_database_items', 2000)
            )
            self.database.set_feature_extractor(self.feature_extractor)
            print(f"✅ Database restored from: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Error restoring database: {e}")
            return False
    
    def remove_person(self, person_name: str):
        """Remove a specific person from database"""
        if self.database.remove_person(person_name):
            self.database.save_database()
            print(f"✅ Removed {person_name} from database")
        else:
            print(f"❌ Person '{person_name}' not found in database")
    
    def list_people(self):
        """List all people in database"""
        stats = self.database.get_statistics()
        
        if stats['total_people'] == 0:
            print("\n📝 Database is empty")
            return
        
        print(f"\n👥 People in database ({stats['total_people']} total):")
        for name, count in stats['person_stats'].items():
            print(f"  • {name}: {count} face{'s' if count != 1 else ''}")

    def deduplicate_database(self, similarity_threshold: float = 0.98):
        """Remove near-duplicate face embeddings per person."""
        print(f"\n🧹 Deduplicating database (threshold={similarity_threshold:.3f})")
        summary = self.database.deduplicate(similarity_threshold)

        if not summary:
            print("ℹ️  No duplicates found. Database unchanged.")
            return

        total_removed = sum(summary.values())
        for name, count in sorted(summary.items(), key=lambda item: item[0].lower()):
            print(f"  • {name}: removed {count} duplicate{'s' if count != 1 else ''}")

        self.database.save_database()
        print(f"✅ Deduplication complete. Removed {total_removed} entries in total.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Face Recognition Database Manager",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python db_manager.py --clear                    # Clear database
  python db_manager.py --populate images/         # Add faces from images/
  python db_manager.py --clear --populate images/ # Clear and repopulate
  python db_manager.py --stats                    # Show statistics
  python db_manager.py --backup                   # Create backup
  python db_manager.py --list                     # List all people
  python db_manager.py --remove "person_name"     # Remove specific person
        """
    )
    
    parser.add_argument('--database', type=str, default='face_database_mobilefacenet.json',
                       help='Database file path')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all faces from database')
    parser.add_argument('--populate', type=str,
                       help='Populate database from images directory')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--backup', type=str, nargs='?', const=True,
                       help='Create database backup (optional: specify filename)')
    parser.add_argument('--restore', type=str,
                       help='Restore database from backup file')
    parser.add_argument('--remove', type=str,
                       help='Remove specific person from database')
    parser.add_argument('--list', action='store_true',
                       help='List all people in database')
    parser.add_argument('--dedupe', type=float, nargs='?', const=0.98,
                       help='Remove near-duplicate face embeddings (optional threshold, default 0.98)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize database manager
    db_manager = DatabaseManager(args.database)
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode(db_manager)
        return
    
    # Handle command line arguments
    actions_performed = False
    
    if args.clear:
        confirm = input("⚠️  Are you sure you want to clear the database? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            db_manager.clear_database()
            actions_performed = True
        else:
            print("❌ Clear operation cancelled")
    
    if args.populate:
        db_manager.populate_database(args.populate)
        actions_performed = True
    
    if args.backup:
        if isinstance(args.backup, str):
            db_manager.backup_database(args.backup)
        else:
            db_manager.backup_database()
        actions_performed = True
    
    if args.restore:
        db_manager.restore_database(args.restore)
        actions_performed = True
    
    if args.remove:
        db_manager.remove_person(args.remove)
        actions_performed = True

    if args.dedupe is not None:
        threshold = args.dedupe if args.dedupe > 0 else 0.98
        db_manager.deduplicate_database(threshold)
        actions_performed = True

    if args.list:
        db_manager.list_people()
        actions_performed = True
    
    if args.stats or not actions_performed:
        db_manager.show_statistics()

def interactive_mode(db_manager: DatabaseManager):
    """Interactive mode for database management"""
    print("\n🎯 Interactive Database Manager")
    print("=" * 40)
    
    while True:
        print("\nChoose an action:")
        print("1. Show database statistics")
        print("2. List people in database")
        print("3. Clear database")
        print("4. Populate from images directory")
        print("5. Clear and repopulate")
        print("6. Remove specific person")
        print("7. Create backup")
        print("8. Restore from backup")
        print("9. Deduplicate near-duplicate faces")
        print("10. Exit")
        
        try:
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == '1':
                db_manager.show_statistics()
            
            elif choice == '2':
                db_manager.list_people()
            
            elif choice == '3':
                confirm = input("⚠️  Are you sure you want to clear the database? (y/N): ")
                if confirm.lower() in ['y', 'yes']:
                    db_manager.clear_database()
                else:
                    print("❌ Clear operation cancelled")
            
            elif choice == '4':
                images_dir = input("Enter images directory path [default: images/]: ").strip()
                if not images_dir:
                    images_dir = "images/"
                db_manager.populate_database(images_dir)
            
            elif choice == '5':
                images_dir = input("Enter images directory path [default: images/]: ").strip()
                if not images_dir:
                    images_dir = "images/"
                confirm = input("⚠️  This will clear the database and repopulate. Continue? (y/N): ")
                if confirm.lower() in ['y', 'yes']:
                    db_manager.clear_database()
                    db_manager.populate_database(images_dir)
                else:
                    print("❌ Operation cancelled")
            
            elif choice == '6':
                db_manager.list_people()
                person_name = input("Enter person name to remove: ").strip()
                if person_name:
                    confirm = input(f"⚠️  Remove '{person_name}' from database? (y/N): ")
                    if confirm.lower() in ['y', 'yes']:
                        db_manager.remove_person(person_name)
                    else:
                        print("❌ Remove operation cancelled")
            
            elif choice == '7':
                backup_name = input("Enter backup filename [optional]: ").strip()
                if backup_name:
                    db_manager.backup_database(backup_name)
                else:
                    db_manager.backup_database()
            
            elif choice == '8':
                backup_file = input("Enter backup file path: ").strip()
                if backup_file:
                    db_manager.restore_database(backup_file)
            
            elif choice == '9':
                thresh_input = input("Similarity threshold (default 0.98): ").strip()
                try:
                    threshold = float(thresh_input) if thresh_input else 0.98
                except ValueError:
                    threshold = 0.98
                db_manager.deduplicate_database(threshold)

            elif choice == '10':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-10.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🗃️  Face Recognition Database Manager")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)