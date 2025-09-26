#!/usr/bin/env python3
"""Image processing utilities for face datasets."""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# Ensure local src modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.face_detector import FaceDetector

SUPPORTED_FORMATS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_TARGET_SIZE: Tuple[int, int] = (112, 112)
DEFAULT_JPEG_QUALITY = 95


class ImageProcessor:
	"""Crop, align, and standardise face images for model ingestion."""

	def __init__(
		self,
		target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
		supported_formats: Sequence[str] = SUPPORTED_FORMATS,
		jpeg_quality: int = DEFAULT_JPEG_QUALITY,
		face_detector: Optional[FaceDetector] = None,
	) -> None:
		self.target_size = target_size
		self.supported_formats = tuple(fmt.lower() for fmt in supported_formats)
		self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
		self.face_detector = face_detector or FaceDetector()

	def crop_and_process_images(
		self,
		images_dir: str = "images",
		output_dir: Optional[str] = "images_processed",
	) -> bool:
		"""Process every supported image under ``images_dir``.

		Returns ``True`` if at least one image is written successfully.
		"""

		if not os.path.isdir(images_dir):
			print(f"❌ Images directory not found: {images_dir}")
			return False

		if not output_dir:
			output_dir = images_dir.rstrip('/\\') + "_processed"

		print("🔧 PROCESSING IMAGES")
		print(f"Input: {images_dir}")
		print(f"Output: {output_dir}")
		print("=" * 50)

		processed_count = 0
		error_count = 0

		os.makedirs(output_dir, exist_ok=True)

		for person_name in os.listdir(images_dir):
			person_dir = os.path.join(images_dir, person_name)
			if not os.path.isdir(person_dir):
				continue

			output_person_dir = os.path.join(output_dir, person_name)
			os.makedirs(output_person_dir, exist_ok=True)

			print(f"\n👤 Processing: {person_name}")

			for image_file in os.listdir(person_dir):
				if not self._is_supported_file(image_file):
					continue

				image_path = os.path.join(person_dir, image_file)

				try:
					success = self._process_single_image(
						image_path=image_path,
						output_dir=output_person_dir,
						output_filename=os.path.splitext(image_file)[0] + '.jpg',
					)

					if success:
						processed_count += 1
						print(f"   ✅ {image_file}")
					else:
						error_count += 1
						print(f"   ❌ {image_file} (no face detected)")

				except Exception as exc:  # pylint: disable=broad-except
					error_count += 1
					print(f"   ❌ {image_file} (error: {exc})")

		print("\n📊 PROCESSING COMPLETE")
		print(f"✅ Processed: {processed_count} images")
		print(f"❌ Errors: {error_count} images")

		return processed_count > 0

	def _is_supported_file(self, filename: str) -> bool:
		return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)

	def _process_single_image(self, image_path: str, output_dir: str, output_filename: str) -> bool:
		"""Detect, crop, and resize a single image before saving it as JPEG."""

		image = cv2.imread(image_path)
		if image is None:
			return False

		faces = self.face_detector.detect_faces(image)
		if not faces:
			return False

		if len(faces) > 1:
			faces.sort(key=lambda face: face.rect[2] * face.rect[3], reverse=True)

		x, y, w, h = faces[0].rect

		padding = int(min(w, h) * 0.2)
		x1 = max(0, x - padding)
		y1 = max(0, y - padding)
		x2 = min(image.shape[1], x + w + padding)
		y2 = min(image.shape[0], y + h + padding)

		face_crop = image[y1:y2, x1:x2]
		face_square = self._pad_to_square(face_crop)
		face_resized = cv2.resize(face_square, self.target_size, interpolation=cv2.INTER_LANCZOS4)

		os.makedirs(output_dir, exist_ok=True)
		output_path = os.path.join(output_dir, output_filename)
		return bool(cv2.imwrite(output_path, face_resized, self.jpeg_params))

	def _pad_to_square(self, crop: np.ndarray) -> np.ndarray:
		"""Pad crop to a square by extending the shorter sides."""

		height, width = crop.shape[:2]
		if height == width:
			return crop

		if height > width:
			pad = height - width
			left = pad // 2
			right = pad - left
			top = bottom = 0
		else:
			pad = width - height
			top = pad // 2
			bottom = pad - top
			left = right = 0

		return cv2.copyMakeBorder(
			crop,
			top,
			bottom,
			left,
			right,
			borderType=cv2.BORDER_REPLICATE,
		)


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(
		description="Crop and resize face images for training.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"--input",
		help="Directory containing raw images",
		default="images",
	)
	parser.add_argument(
		"--output",
		help="Directory to store processed images",
		default="images_processed",
	)
	parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG output quality")

	args = parser.parse_args()

	processor = ImageProcessor(jpeg_quality=args.quality)
	processor.crop_and_process_images(args.input, args.output)


if __name__ == "__main__":
	main()
