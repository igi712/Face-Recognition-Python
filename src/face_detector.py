#!/usr/bin/env python3
"""Face detector orchestration mirroring Jetson module layout."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .detectors import MTcnnDetector, RetinaDetector
from .detectors.types import FaceObject


class FaceDetector:
    """High-level detector that proxies RetinaFace or fallback backends."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        prefer_retina: bool = True,
    ) -> None:
        self._retina = RetinaDetector()
        self._mtcnn = MTcnnDetector(model_path=model_path, config_path=config_path)

        self.use_retina = prefer_retina and self._retina.is_available

    def detect_faces(
        self,
        frame: np.ndarray,
        resize_factor: Union[float, Tuple[float, float], None] = 1.0,
    ) -> List[FaceObject]:
        if frame is None or frame.size == 0:
            return []

        scale_x, scale_y = self._normalize_resize_factor(resize_factor)

        if self.use_retina:
            working_frame = self._resize_frame(frame, scale_x, scale_y)
            faces = self._retina.detect(working_frame)
            return self._rescale_faces(faces, scale_x, scale_y)

        # Delegate scaling to the fallback detector so it can reuse its own heuristics
        return self._mtcnn.detect(frame, resize_factor=resize_factor)

    def filter_faces(
        self,
        faces: List[FaceObject],
        min_face_size: int = 90,
        min_score: float = 0.0,
    ) -> List[FaceObject]:
        filtered_faces: List[FaceObject] = []
        for face in faces:
            if not face.rect:
                continue

            x, y, w, h = face.rect
            if min_score > 0.0 and face.face_prob < min_score:
                continue

            if h < min_face_size:
                face.name_index = -2
                face.color = 2
                continue

            if abs(face.angle) > 30:
                continue

            filtered_faces.append(face)

        return filtered_faces

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_resize_factor(
        resize_factor: Union[float, Tuple[float, float], None]
    ) -> Tuple[float, float]:
        if resize_factor is None:
            return 1.0, 1.0

        if isinstance(resize_factor, (tuple, list)):
            if not resize_factor:
                return 1.0, 1.0
            scale_x = float(resize_factor[0])
            scale_y = float(resize_factor[1]) if len(resize_factor) > 1 else scale_x
        else:
            scale_x = scale_y = float(resize_factor)

        if np.isnan(scale_x) or scale_x <= 0:
            scale_x = 1.0
        if np.isnan(scale_y) or scale_y <= 0:
            scale_y = 1.0

        return min(scale_x, 1.0), min(scale_y, 1.0)

    @staticmethod
    def _resize_frame(frame: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
        if np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0):
            return frame

        return cv2.resize(frame, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _rescale_faces(
        faces: List[FaceObject], scale_x: float, scale_y: float
    ) -> List[FaceObject]:
        if not faces:
            return []

        if np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0):
            return faces

        inv_x = 1.0 / scale_x if scale_x > 0 else 1.0
        inv_y = 1.0 / scale_y if scale_y > 0 else 1.0

        for face in faces:
            if not face.rect:
                continue

            x, y, w, h = face.rect
            face.rect = (
                int(round(x * inv_x)),
                int(round(y * inv_y)),
                int(max(round(w * inv_x), 1)),
                int(max(round(h * inv_y), 1)),
            )

            if face.landmark:
                face.landmark = [
                    (
                        int(round(lx * inv_x)),
                        int(round(ly * inv_y)),
                    )
                    for lx, ly in face.landmark
                ]

        return faces


__all__ = ["FaceDetector", "FaceObject"]