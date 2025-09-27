"""Simplified MTCNN-style detector wrapper mirroring TMtCNN."""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple, Union

from .types import FaceObject


class MTcnnDetector:
    """Fallback detector roughly matching the Jetson TMtCNN responsibilities."""

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.input_size = (320, 240)

        self._dnn = None
        self._cascade = None

        if model_path:
            self._load_custom_model(model_path, config_path)
        else:
            self._load_opencv_detector()

    @property
    def is_available(self) -> bool:
        return self._dnn is not None or self._cascade is not None

    def detect(
        self,
        frame: np.ndarray,
        resize_factor: Union[float, Tuple[float, float], None] = 1.0,
    ) -> List[FaceObject]:
        if resize_factor is None:
            scale_x = scale_y = 1.0
        elif isinstance(resize_factor, (tuple, list)):
            scale_x = float(resize_factor[0]) if len(resize_factor) > 0 else 1.0
            scale_y = float(resize_factor[1]) if len(resize_factor) > 1 else scale_x
        else:
            scale_x = scale_y = float(resize_factor)

        scale_x = np.clip(scale_x, 0.0, 1.0) if not np.isnan(scale_x) else 1.0
        scale_y = np.clip(scale_y, 0.0, 1.0) if not np.isnan(scale_y) else 1.0
        scale_x = scale_x if scale_x > 0 else 1.0
        scale_y = scale_y if scale_y > 0 else 1.0

        if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
            working_frame = cv2.resize(frame, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
        else:
            working_frame = frame

        if self._dnn is not None:
            faces = self._detect_with_dnn(working_frame)
        elif self._cascade is not None:
            faces = self._detect_with_cascade(working_frame)
        else:
            return []

        if faces and (not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0)):
            inv_x = 1.0 / scale_x if scale_x > 0 else 1.0
            inv_y = 1.0 / scale_y if scale_y > 0 else 1.0
            for face in faces:
                if not face.rect:
                    continue
                x, y, w, h = face.rect
                x = int(round(x * inv_x))
                y = int(round(y * inv_y))
                w = int(max(round(w * inv_x), 1))
                h = int(max(round(h * inv_y), 1))
                face.rect = (x, y, w, h)

                if face.landmark:
                    scaled = []
                    for lx, ly in face.landmark:
                        scaled.append((int(round(lx * inv_x)), int(round(ly * inv_y))))
                    face.landmark = scaled

        return faces

    # ------------------------------------------------------------------
    # Detector loaders
    # ------------------------------------------------------------------

    def _load_opencv_detector(self) -> None:
        try:
            prototxt_path = "models/opencv_face_detector.pbtxt"
            model_path = "models/opencv_face_detector_uint8.pb"
            self._dnn = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
            print("Loaded OpenCV DNN face detector")
        except Exception:
            print("Warning: Could not load OpenCV face detector. Falling back to Haar cascades.")
            self._dnn = None
            self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _load_custom_model(self, model_path: str, config_path: Optional[str]) -> None:
        try:
            self._dnn = cv2.dnn.readNet(model_path, config_path)
            print(f"Loaded custom face detector from {model_path}")
        except Exception as exc:
            print(f"Error loading custom model: {exc}")
            self._dnn = None
            self._load_opencv_detector()

    # ------------------------------------------------------------------
    # Detection implementations
    # ------------------------------------------------------------------

    def _detect_with_dnn(self, frame: np.ndarray) -> List[FaceObject]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, self.input_size, [104, 117, 123])
        self._dnn.setInput(blob)
        detections = self._dnn.forward()

        faces: List[FaceObject] = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face = FaceObject(
                rect=(x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)),
                face_prob=float(confidence),
            )
            face.landmark = self._estimate_landmarks(face.rect)
            face.angle = self._calculate_angle(face.landmark)
            faces.append(face)

        return faces

    def _detect_with_cascade(self, frame: np.ndarray) -> List[FaceObject]:
        if self._cascade is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        faces: List[FaceObject] = []
        for (x, y, w, h) in detections:
            face = FaceObject(
                rect=(int(x), int(y), int(w), int(h)),
                face_prob=0.8,
            )
            face.landmark = self._estimate_landmarks(face.rect)
            face.angle = self._calculate_angle(face.landmark)
            faces.append(face)

        return faces

    # ------------------------------------------------------------------
    # Landmark helpers (lightweight approximation)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_landmarks(rect: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        x, y, w, h = rect
        return [
            (x + w // 4, y + h // 3),
            (x + 3 * w // 4, y + h // 3),
            (x + w // 2, y + h // 2),
            (x + w // 3, y + 2 * h // 3),
            (x + 2 * w // 3, y + 2 * h // 3),
        ]

    @staticmethod
    def _calculate_angle(landmarks: List[Tuple[int, int]]) -> float:
        if len(landmarks) < 2:
            return 0.0
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        if dx == 0:
            return 90.0 if dy > 0 else -90.0
        return float(np.arctan2(dy, dx) * 180.0 / np.pi)
