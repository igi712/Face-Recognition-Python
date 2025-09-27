"""Utilities for populating MobileFaceNet-compatible databases."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.face_features_arcface import MobileFaceNetExtractor


class MobileFaceNetPopulator:
    """Populate database with MobileFaceNet features."""

    def __init__(self, database_path: str = "face_database_mobilefacenet.json") -> None:
        self.database_path = database_path
        self.feature_extractor = MobileFaceNetExtractor()
        self.database: Dict[str, object] = self._empty_database()
        print("🚀 MobileFaceNet Database Populator")
        print(f"📁 Database: {database_path}")
        print("🎯 Feature Type: MobileFaceNet + ArcFace")

    @staticmethod
    def _empty_database() -> Dict[str, object]:
        return {
            "faces": {},
            "face_names": [],
            "created_at": "2025-09-21",
            "feature_type": "MobileFaceNet_ArcFace",
            "total_faces": 0,
            "total_people": 0,
        }

    def clear_database(self) -> None:
        self.database = self._empty_database()
        self._save_database()
        print("✅ Database cleared")

    def populate_from_directory(self, images_dir: str = "images_processed/") -> bool:
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False

        print(f"\n🔄 Populating from: {images_dir}")
        print("=" * 60)

        total_processed = 0
        total_errors = 0

        for person_name in sorted(os.listdir(images_dir)):
            person_dir = os.path.join(images_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            print(f"\n👤 Processing: {person_name}")

            if person_name not in self.database["face_names"]:
                self.database["face_names"].append(person_name)
            person_index = self.database["face_names"].index(person_name)

            person_faces = self.database["faces"].setdefault(person_name, [])
            person_face_count = 0

            for image_file in sorted(os.listdir(person_dir)):
                if not image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                image_path = os.path.join(person_dir, image_file)

                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"   ❌ Failed to load: {image_file}")
                        total_errors += 1
                        continue

                    features = self.feature_extractor.extract_feature(img)
                    if features is None:
                        print(f"   ❌ No features: {image_file}")
                        total_errors += 1
                        continue

                    features_b64 = base64.b64encode(features.astype(np.float32).tobytes()).decode("utf-8")

                    face_data: Dict[str, object] = {
                        "features": features_b64,
                        "image_file": image_file,
                        "feature_dim": len(features),
                        "feature_norm": float(np.linalg.norm(features)),
                        "person_index": person_index,
                    }

                    person_faces.append(face_data)
                    person_face_count += 1
                    total_processed += 1

                    print(
                        f"   ✅ {image_file} -> {len(features)}D features (norm: {face_data['feature_norm']:.3f})"
                    )

                except Exception as exc:  # pylint: disable=broad-except
                    print(f"   ❌ Error processing {image_file}: {exc}")
                    total_errors += 1

            print(f"   📊 Added {person_face_count} faces for {person_name}")

        self.database["total_people"] = len(self.database["face_names"])
        self.database["total_faces"] = total_processed
        self._save_database()

        print("\n🎊 POPULATION COMPLETE!")
        print(f"✅ Processed: {total_processed} faces")
        print(f"❌ Errors: {total_errors}")
        print(f"👥 Total people: {self.database['total_people']}")
        print(f"📊 Database saved to: {self.database_path}")

        return total_processed > 0

    def _save_database(self) -> bool:
        try:
            with open(self.database_path, "w", encoding="utf-8") as handle:
                json.dump(self.database, handle, indent=2)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Error saving database: {exc}")
            return False

    def print_stats(self) -> None:
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, "r", encoding="utf-8") as handle:
                    db = json.load(handle)
            else:
                db = self.database

            print("\n📊 DATABASE STATISTICS")
            print("=" * 50)
            print(f"📁 Database: {self.database_path}")
            print(f"🎯 Feature Type: {db.get('feature_type', 'Unknown')}")
            print(f"👥 Total People: {db.get('total_people', 0)}")
            print(f"📸 Total Faces: {db.get('total_faces', 0)}")
            print(f"📅 Created: {db.get('created_at', 'Unknown')}")

            if db.get("faces"):
                print("\n👤 Per-person breakdown:")
                for person_name, faces in db["faces"].items():
                    print(f"   {person_name}: {len(faces)} faces")

            print()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"❌ Error reading database: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MobileFaceNet Database Populator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--database", type=str, default="face_database_mobilefacenet.json", help="Database file path")
    parser.add_argument("--images", type=str, default="images_processed/", help="Images directory to process")
    parser.add_argument("--clear", action="store_true", help="Clear database before populating")
    parser.add_argument("--stats", action="store_true", help="Show database statistics only")
    parser.add_argument("--populate", action="store_true", default=True, help="Populate database from images")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    populator = MobileFaceNetPopulator(args.database)

    if args.stats:
        populator.print_stats()
        return

    if args.clear:
        populator.clear_database()

    if args.populate and not args.stats:
        success = populator.populate_from_directory(args.images)
        if success:
            print(
                "\n🚀 Database ready for use with:\n"
                f"   python -m app.cli 0 --database {args.database}"
            )
        else:
            print("\n❌ Population failed!")

    populator.print_stats()


if __name__ == "__main__":
    main()
