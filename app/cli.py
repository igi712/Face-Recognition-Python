#!/usr/bin/env python3
"""CLI entry point for the Face Recognition application."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure src package is importable regardless of entry location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"

from src.face_database import FaceDatabase
from src.face_detector import FaceDetector, FaceObject
from src.face_features import FaceFeatureExtractor
from src.face_quality import FaceQualityAssessment


class FaceRecognitionSystem:
    """Main face recognition system with ArcFace support."""

    def __init__(
        self,
        config: Optional[dict] = None,
        use_arcface: bool = True,
        arcface_model_path: Optional[str] = None,
    ) -> None:
        """Initialize the face recognition system."""
        self.config: Dict[str, object] = {
            "min_face_size": 120,
            "face_threshold": 0.5,
            "recognition_threshold": 0.35,  # Lower threshold untuk Z-score normalization
            "liveness_threshold": 0.93,
            "max_blur": -20.0,
            "max_angle": 15.0,
            "max_database_items": 2000,
            "show_landmarks": True,
            "show_legend": True,
            "enable_liveness": True,
            "enable_blur_filter": True,
            "auto_add_faces": False,
            "database_path": "face_database_mobilefacenet.json",
            "images_directory": "images",
            "use_arcface": use_arcface,
            "arcface_model_path": arcface_model_path,
            "use_zscore_norm": True,  # DEFAULT: Gunakan Z-score seperti Jetson Nano
            "detection_downscale": 0.8,
            "detection_frame_size": (320, 240),
            "quality_interval": 1,
            "fast_mode": False,
            "quality_cache_bucket": 16,
            "opencv_threads": None,
            "recognition_event_cooldown": 1.5,
            "recognition_event_enabled": True,
            "recognition_event_file": str(PROJECT_ROOT / "output" / "recognized_faces.json"),
            "recognition_event_append": True,
            "recognition_event_max_lines": 1000,
        }

        if config:
            self.config.update(config)

        detection_frame_override = bool(config and "detection_frame_size" in config)
        overrides = self.config.pop("_overrides", {}) if isinstance(self.config.get("_overrides"), dict) else {}
        overrides.setdefault("detection_frame_size", detection_frame_override)
        self._overrides = overrides

        if detection_frame_override and not self._overrides.get("detection_frame_size"):
            self._overrides["detection_frame_size"] = True

        if self.config.get("fast_mode"):
            if not self._overrides.get("detection_downscale") and self.config.get("detection_downscale", 1.0) == 1.0:
                self.config["detection_downscale"] = 0.6
            if not self._overrides.get("enable_liveness"):
                self.config["enable_liveness"] = False
            if not self._overrides.get("enable_blur_filter"):
                self.config["enable_blur_filter"] = False
            if not self._overrides.get("show_landmarks"):
                self.config["show_landmarks"] = False
            if not self._overrides.get("quality_interval"):
                self.config["quality_interval"] = max(self.config.get("quality_interval", 1), 3)
            if not self._overrides.get("detection_frame_size"):
                self.config["detection_frame_size"] = None

        self.config["quality_interval"] = max(1, int(self.config.get("quality_interval", 1)))
        downscale = float(self.config.get("detection_downscale", 1.0))
        self.config["detection_downscale"] = max(min(downscale if downscale > 0 else 1.0, 1.0), 0.1)
        self.config["quality_cache_bucket"] = max(4, int(self.config.get("quality_cache_bucket", 16)))

        self.face_detector = FaceDetector()
        self.feature_extractor = FaceFeatureExtractor()
        self.face_database = FaceDatabase(
            database_path=self.config["database_path"],
            max_items=self.config["max_database_items"],
            use_arcface=self.config["use_arcface"],
            arcface_model_path=self.config["arcface_model_path"],
            use_zscore_norm=self.config.get("use_zscore_norm", True),  # Default TRUE seperti Jetson Nano
        )
        self.quality_assessor = FaceQualityAssessment(fast_mode=self.config.get("fast_mode", False))

        self.frame_count = 0
        self._quality_cache: Dict[Tuple[int, int, int, int], Tuple[dict, int]] = {}
        self._recognition_event_cache: Dict[str, float] = {}
        self._recognition_event_file_error_logged = False
        self._recognition_event_trim_error_logged = False

        event_file = self.config.get("recognition_event_file")
        if event_file:
            event_path = Path(event_file)
            if not event_path.is_absolute():
                event_path = (PROJECT_ROOT / event_path).resolve()
            event_path.parent.mkdir(parents=True, exist_ok=True)
            append_mode = bool(self.config.get("recognition_event_append", True))
            self._recognition_event_file = str(event_path)
            self._recognition_event_append_mode = append_mode
            if not append_mode:
                try:
                    event_path.write_text("", encoding="utf-8")
                except Exception as exc:
                    print(f"⚠️ Failed to initialise recognition event file: {exc}", file=sys.stderr)
                    self._recognition_event_file = None
            else:
                try:
                    event_path.touch(exist_ok=True)
                except Exception as exc:
                    print(f"⚠️ Failed to prepare recognition event file: {exc}", file=sys.stderr)
                    self._recognition_event_file = None
        else:
            self._recognition_event_file = None
            self._recognition_event_append_mode = True

        self._recognition_event_max_lines = max(
            0,
            int(self.config.get("recognition_event_max_lines", 0) or 0),
        )

        if self._recognition_event_file:
            mode_str = "append" if self._recognition_event_append_mode else "overwrite"
            max_lines = self._recognition_event_max_lines or "∞"
            print(
                f"📁 Recognition events will be saved to {self._recognition_event_file} "
                f"(mode={mode_str}, max_lines={max_lines})"
            )

        try:
            cv2.setUseOptimized(True)
            threads = self.config.get("opencv_threads")
            if threads is not None:
                threads_int = int(threads)
                if threads_int >= 0:
                    cv2.setNumThreads(threads_int)
        except Exception:  # pragma: no cover - OpenCV thread control may fail silently
            pass

        self.fps_buffer = [0.0] * 16
        self.fps_index = 0

        feature_type = "ArcFace" if self.config["use_arcface"] else "Legacy"
        print(f"Face Recognition System initialized with {feature_type} features")
        self.face_database.print_statistics()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[FaceObject]]:
        """Process a single frame for face recognition."""
        self.frame_count += 1

        detection_resize: Optional[Tuple[float, float]] | None = None
        downscale = self.config.get("detection_downscale", 1.0)

        if self.face_detector.use_retina:
            if downscale < 1.0:
                detection_resize = downscale
        else:
            target_size = self.config.get("detection_frame_size")
            if (
                isinstance(target_size, (tuple, list))
                and len(target_size) >= 2
                and target_size[0]
                and target_size[1]
            ):
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

        faces = self.face_detector.filter_faces(
            faces,
            self.config["min_face_size"],
            self.config.get("face_threshold", 0.0),
        )

        for face in faces:
            self._process_single_face(frame, face)

        result_frame = self._draw_results(frame, faces)
        self._cleanup_quality_cache()
        return result_frame, faces

    def _process_single_face(self, frame: np.ndarray, face: FaceObject) -> None:
        x, y, w, h = face.rect
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return

        face_region = frame[y : y + h, x : x + w]
        if face_region.size == 0:
            return

        relative_landmarks: Optional[List[Tuple[float, float]]] = None
        if face.landmark:
            relative_landmarks = []
            for lx, ly in face.landmark:
                rel_x = float(np.clip(float(lx - x), 0.0, float(w)))
                rel_y = float(np.clip(float(ly - y), 0.0, float(h)))
                relative_landmarks.append((rel_x, rel_y))

        quality_interval = self.config.get("quality_interval", 1)
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
                require_blur=self.config.get("enable_blur_filter", True),
            )
            if cache_key:
                self._quality_cache[cache_key] = (quality_results, self.frame_count)
        else:
            self._quality_cache[cache_key] = (quality_results, self.frame_count)

        face.angle = quality_results["face_angle"]

        name_index, confidence = self.face_database.recognize_face(
            face_region,
            relative_landmarks or face.landmark,
            self.config["recognition_threshold"],
        )

        face.name_index = name_index
        face.name_prob = confidence
        face.live_prob = quality_results.get("liveness_score", 1.0)

        if h < self.config["min_face_size"]:
            if face.name_index < 0:
                face.name_index = -1
            face.color = 2
            return

        needs_liveness = self.config.get("enable_liveness", True) and face.name_index >= 0

        if needs_liveness:
            live_score, is_live = self.quality_assessor.assess_liveness(face_region)
            face.live_prob = live_score
            quality_results["liveness_score"] = live_score
            quality_results["is_live"] = is_live
            if not is_live:
                face.name_index = -3
                face.color = 3
                return
        else:
            quality_results.setdefault("liveness_score", face.live_prob)
            quality_results.setdefault("is_live", True)

        if name_index >= 0:
            face.color = 0
            self._emit_recognition_event(face)
        else:
            face.color = 1
            if self.config["auto_add_faces"] and quality_results["is_good_quality"]:
                self._auto_add_face(face_region, relative_landmarks or face.landmark)

    def _make_face_cache_key(self, face: FaceObject) -> Optional[Tuple[int, int, int, int]]:
        if not face.rect:
            return None
        bucket = self.config.get("quality_cache_bucket", 16)
        x, y, w, h = face.rect
        return (int(x // bucket), int(y // bucket), int(w // bucket), int(h // bucket))

    def _cleanup_quality_cache(self) -> None:
        if not self._quality_cache:
            return
        quality_interval = self.config.get("quality_interval", 1)
        max_age = max(quality_interval * 6, 10)
        cutoff = self.frame_count - max_age
        keys_to_delete = [key for key, (_, frame_idx) in self._quality_cache.items() if frame_idx < cutoff]
        for key in keys_to_delete:
            self._quality_cache.pop(key, None)

    def _auto_add_face(self, face_region: np.ndarray, landmarks: List) -> None:
        timestamp = int(time.time())
        unknown_name = f"Unknown_{timestamp}"
        if self.face_database.add_face(unknown_name, face_region, landmarks):
            print(f"Auto-added face: {unknown_name}")

    def _emit_recognition_event(self, face: FaceObject) -> None:
        if not self.config.get("recognition_event_enabled", True):
            return
        if face.name_index < 0 or not face.rect:
            return

        name = self.face_database.get_name(face.name_index)
        x, y, w, h = face.rect
        cache_key = self._make_face_cache_key(face)
        event_key = f"{name}:{cache_key}"

        cooldown = float(self.config.get("recognition_event_cooldown", 0.0) or 0.0)
        now = time.time()
        last_time = self._recognition_event_cache.get(event_key, -float("inf"))
        if cooldown > 0.0 and (now - last_time) < cooldown:
            return

        liveness_threshold = float(self.config.get("liveness_threshold", 0.0) or 0.0)
        liveness_enabled = bool(self.config.get("enable_liveness", True))
        live_score = float(face.live_prob)
        liveness_passed = True if not liveness_enabled else live_score >= liveness_threshold

        payload = {
            "event": "recognized_face",
            "name": name,
            "frame_index": self.frame_count,
            "timestamp": now,
            "confidence": round(float(face.name_prob), 4),
            "face_probability": round(float(face.face_prob), 4),
            "liveness": {
                "score": round(live_score, 4),
                "passed": liveness_passed,
            },
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },
        }

        print(json.dumps(payload), flush=True)
        self._write_recognition_event(payload)
        self._recognition_event_cache[event_key] = now

    def emit_test_event(self) -> None:
        now = time.time()
        payload = {
            "event": "recognized_face",
            "name": "__test__",
            "frame_index": -1,
            "timestamp": now,
            "confidence": 1.0,
            "face_probability": 1.0,
            "liveness": {"score": 1.0, "passed": True},
            "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
        }
        print(json.dumps(payload), flush=True)
        self._write_recognition_event(payload)

    def _write_recognition_event(self, payload: Dict[str, object]) -> None:
        if not self._recognition_event_file:
            return

        mode = "a"
        try:
            with open(self._recognition_event_file, mode, encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.write("\n")
            self._trim_recognition_event_file()
        except Exception as exc:
            if self._recognition_event_file_error_logged:
                return
            print(
                f"⚠️ Failed to write recognition event to {self._recognition_event_file}: {exc}",
                file=sys.stderr,
            )
            self._recognition_event_file_error_logged = True

    def _trim_recognition_event_file(self) -> None:
        if not self._recognition_event_file or self._recognition_event_max_lines <= 0:
            return

        try:
            with open(self._recognition_event_file, "r+", encoding="utf-8") as fh:
                lines = fh.readlines()
                if len(lines) <= self._recognition_event_max_lines:
                    return
                keep = lines[-self._recognition_event_max_lines :]
                fh.seek(0)
                fh.truncate(0)
                fh.writelines(keep)
        except Exception as exc:
            if self._recognition_event_trim_error_logged:
                return
            print(
                f"⚠️ Failed to trim recognition event file {self._recognition_event_file}: {exc}",
                file=sys.stderr,
            )
            self._recognition_event_trim_error_logged = True

    def _draw_results(self, frame: np.ndarray, faces: List[FaceObject]) -> np.ndarray:
        result_frame = frame.copy()
        for face in faces:
            self._draw_single_face(result_frame, face)
        if self.config["show_legend"]:
            self._draw_legend(result_frame, faces)
        return result_frame

    def _draw_single_face(self, frame: np.ndarray, face: FaceObject) -> None:
        x, y, w, h = face.rect
        colors = {
            0: (255, 255, 255),
            1: (80, 255, 255),
            2: (255, 237, 178),
            3: (127, 127, 255),
        }
        color = colors.get(face.color, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        if self.config["show_landmarks"] and face.landmark:
            for landmark in face.landmark:
                cv2.circle(frame, landmark, 2, (0, 255, 255), -1)

        label = self._get_face_label(face)
        self._draw_label(frame, label, (x, y), color)

    def _get_face_label(self, face: FaceObject) -> str:
        if face.name_index == -1:
            return "Stranger"
        if face.name_index == -2:
            return "Too tiny"
        if face.name_index == -3:
            return "Fake!"
        if face.name_index >= 0:
            return self.face_database.get_name(face.name_index)
        return "Unknown"

    def _draw_label(self, frame: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if y < text_height + baseline:
            y = text_height + baseline
        if x + text_width > frame.shape[1]:
            x = frame.shape[1] - text_width
        cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    def _draw_legend(self, frame: np.ndarray, faces: List[FaceObject]) -> None:
        if not faces:
            return
        face = faces[0]
        y_offset = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (180, 180, 0)
        cv2.putText(frame, f"Angle: {face.angle:.1f}°", (10, y_offset), font, font_scale, color)
        y_offset += 20
        cv2.putText(frame, f"Face prob: {face.face_prob:.4f}", (10, y_offset), font, font_scale, color)
        y_offset += 20
        cv2.putText(frame, f"Name prob: {face.name_prob:.4f}", (10, y_offset), font, font_scale, color)
        y_offset += 20
        if self.config["enable_liveness"]:
            if face.color == 2:
                cv2.putText(frame, "Live prob: ??", (10, y_offset), font, font_scale, color)
            else:
                cv2.putText(frame, f"Live prob: {face.live_prob:.4f}", (10, y_offset), font, font_scale, color)

    def update_fps(self, fps: float) -> None:
        self.fps_buffer[self.fps_index] = fps
        self.fps_index = (self.fps_index + 1) % len(self.fps_buffer)

    def get_average_fps(self) -> float:
        return sum(self.fps_buffer) / len(self.fps_buffer)

    def add_person_from_directory(self, person_name: str, image_directory: str) -> int:
        return self.face_database.auto_populate_from_directory(image_directory)

    def save_database(self) -> None:
        self.face_database.save_database()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face Recognition System - Python Implementation with ArcFace Support",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("input", type=str, nargs="?", default="0", help="Input source (webcam index, image, or video file)")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--database", type=str, default="face_database_mobilefacenet.json", help="Face database file path")
    parser.add_argument("--populate", type=str, help="Auto-populate database from directory")
    parser.add_argument("--threshold", type=float, help="Recognition threshold (auto-set based on feature type)")
    parser.add_argument("--min-size", type=int, default=90, help="Minimum face size in pixels")
    parser.add_argument("--auto-add", action="store_true", help="Automatically add unknown faces")
    parser.add_argument("--show-landmarks", dest="show_landmarks", action="store_const", const=True, help="Draw facial landmarks")
    parser.add_argument("--hide-landmarks", dest="show_landmarks", action="store_const", const=False, help="Do not draw facial landmarks (speed boost)")
    parser.add_argument("--show-legend", dest="show_legend", action="store_true", help="Show information legend (default)")
    parser.add_argument("--hide-legend", dest="show_legend", action="store_false", help="Hide information legend")
    parser.add_argument("--enable-liveness", dest="enable_liveness", action="store_true", help="Enable liveness detection (default)")
    parser.add_argument("--disable-liveness", dest="enable_liveness", action="store_false", help="Disable liveness detection for speed")
    parser.add_argument("--enable-blur-filter", dest="enable_blur", action="store_true", help="Enable blur-based quality filtering (default)")
    parser.add_argument("--disable-blur-filter", dest="enable_blur", action="store_false", help="Disable blur filtering for speed")
    parser.add_argument("--detection-downscale", type=float, default=1.0, help="Resize factor (<1.0) applied before detection to boost FPS")
    parser.add_argument("--quality-interval", type=int, default=1, help="Run blur/liveness checks every N frames (>=1)")
    parser.add_argument("--fast", action="store_true", help="Enable preset speed optimizations (changes several settings)")
    parser.add_argument("--opencv-threads", type=int, help="Override OpenCV thread count (0 lets OpenCV decide)")
    parser.add_argument("--use-arcface", action="store_true", default=True, help="Use ArcFace features (default: True, more accurate)")
    parser.add_argument("--use-legacy", action="store_true", help="Force use of legacy features (less accurate)")
    parser.add_argument("--arcface-model", type=str, help="Path to ArcFace ONNX model file")
    parser.add_argument("--use-zscore-norm", action="store_true", default=True, help="Use Z-score normalization (like Jetson Nano) - DEFAULT")
    parser.add_argument("--use-l2-norm", dest="use_zscore_norm", action="store_false", help="Use L2 normalization instead of Z-score")
    parser.add_argument("--event-file", type=str, help="Write recognized-face JSON events to this file (newline-delimited)")
    parser.add_argument(
        "--event-file-mode",
        type=str,
        choices=["append", "overwrite"],
        default="append",
        help="How to handle the event file when the session starts",
    )
    parser.add_argument(
        "--event-file-max-lines",
        type=int,
        default=1000,
        help="Maximum number of lines to retain in the event file (<=0 disables trimming)",
    )
    parser.add_argument(
        "--emit-test-event",
        action="store_true",
        help="Write a dummy recognized_face event to the log and exit",
    )

    parser.set_defaults(show_legend=True, enable_liveness=False, enable_blur=True)

    args = parser.parse_args()

    provided_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}

    def flag_provided(*names: str) -> bool:
        return any(name in provided_flags for name in names)

    overrides = {
        "show_landmarks": flag_provided("--show-landmarks", "--hide-landmarks"),
        "enable_liveness": flag_provided("--disable-liveness", "--enable-liveness"),
        "enable_blur_filter": flag_provided("--disable-blur-filter", "--enable-blur-filter"),
        "detection_downscale": flag_provided("--detection-downscale"),
        "quality_interval": flag_provided("--quality-interval"),
    }

    input_source_hint = args.input
    is_camera_input = input_source_hint.isdigit()

    if not overrides["enable_liveness"]:
        args.enable_liveness = is_camera_input

    if args.fast:
        if not overrides["detection_downscale"] and args.detection_downscale == 1.0:
            args.detection_downscale = 0.6
        if not overrides["quality_interval"]:
            args.quality_interval = max(args.quality_interval, 3)
        if not overrides["enable_liveness"]:
            args.enable_liveness = False
        if not overrides["enable_blur_filter"]:
            args.enable_blur = False
        if not overrides["show_landmarks"] and args.show_landmarks is None:
            args.show_landmarks = False

    use_arcface = args.use_arcface and not args.use_legacy
    if args.use_legacy:
        use_arcface = False
        print("🔄 Using legacy features (forced by --use-legacy)")
    elif use_arcface:
        print("🚀 Using ArcFace features (more accurate)")

    if args.threshold is None:
        threshold = 0.4 if use_arcface else 0.8
    else:
        threshold = args.threshold

    config = {
        "database_path": args.database,
        "recognition_threshold": threshold,
        "min_face_size": args.min_size,
        "show_legend": args.show_legend,
        "enable_liveness": args.enable_liveness,
        "auto_add_faces": args.auto_add,
        "use_arcface": use_arcface,
        "arcface_model_path": args.arcface_model,
        "use_zscore_norm": args.use_zscore_norm,
        "enable_blur_filter": args.enable_blur,
        "detection_downscale": args.detection_downscale,
        "quality_interval": args.quality_interval,
        "fast_mode": args.fast,
        "opencv_threads": args.opencv_threads,
        "recognition_event_append": args.event_file_mode != "overwrite",
        "recognition_event_max_lines": args.event_file_max_lines,
    }

    if args.event_file is not None:
        config["recognition_event_file"] = args.event_file

    if args.show_landmarks is not None:
        config["show_landmarks"] = args.show_landmarks

    config["_overrides"] = overrides

    face_recognition = FaceRecognitionSystem(config, use_arcface, args.arcface_model)

    if args.emit_test_event:
        face_recognition.emit_test_event()
        print("Test recognition event written.")
        return

    if args.populate:
        print(f"Auto-populating database from {args.populate}")
        face_recognition.add_person_from_directory("", args.populate)
        face_recognition.save_database()

    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)
        is_camera = True
    else:
        is_camera = False

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open input source: {input_source}")
        return

    print("Face Recognition System started. Press 'q' to quit, 's' to save database.")

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    print("Error reading from camera")
                    break
                else:
                    print("End of video file reached")
                    print("\nVideo processing completed!")
                    print("Options:")
                    print("  r - Process another video")
                    print("  q - Quit")
                    print("  s - Save database and quit")
                    
                    while True:
                        try:
                            choice = input("Choose an option (r/q/s): ").strip().lower()
                            if choice == 'r':
                                # Get new video path
                                new_video = input("Enter video file path: ").strip()
                                if new_video and os.path.exists(new_video):
                                    cap.release()
                                    cap = cv2.VideoCapture(new_video)
                                    if cap.isOpened():
                                        print(f"Processing new video: {new_video}")
                                        break
                                    else:
                                        print(f"Could not open video: {new_video}")
                                        cap = cv2.VideoCapture(input_source)  # Reopen original
                                else:
                                    print("Invalid video path or file not found")
                            elif choice == 's':
                                face_recognition.save_database()
                                print("Database saved")
                                return
                            elif choice == 'q':
                                return
                            else:
                                print("Invalid choice. Please enter 'r', 'q', or 's'")
                        except (EOFError, KeyboardInterrupt):
                            print("\nExiting...")
                            return
                    
                    continue  # Continue the main loop with new video

            result_frame, faces = face_recognition.process_frame(frame)

            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if end_time > start_time else 0
            face_recognition.update_fps(fps)

            avg_fps = face_recognition.get_average_fps()
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Recognition System", result_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                face_recognition.save_database()
                print("Database saved")

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_recognition.save_database()
        print("Face Recognition System stopped")


if __name__ == "__main__":
    main()
