"""Warp and extract helper echoing Jetson's TArcface/TWarp pipeline."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..face_features import FaceFeatureExtractor

try:  # pragma: no cover - optional dependency
    from ..face_features_arcface import ArcFaceExtractor
    ARCFACE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without ArcFace
    ArcFaceExtractor = None  # type: ignore
    ARCFACE_AVAILABLE = False


class WarpAndExtract:
    """Wrapper around alignment + embedding to mirror Jetson contracts."""

    def __init__(
        self,
        *,
        use_arcface: bool = True,
        arcface_model_path: Optional[str] = None,
    ) -> None:
        self.legacy_extractor = FaceFeatureExtractor()
        self.arcface_extractor: Optional[ArcFaceExtractor] = None  # type: ignore[assignment]
        self._arcface_cls = ArcFaceExtractor

        self.use_arcface = bool(use_arcface and ARCFACE_AVAILABLE)
        if self.use_arcface and ArcFaceExtractor is not None:
            self.arcface_extractor = ArcFaceExtractor(arcface_model_path)
        elif use_arcface and not ARCFACE_AVAILABLE:
            print("⚠️  ArcFace requested but not available, using legacy features")
            self.use_arcface = False

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def align(self, face_image: np.ndarray, landmarks: Optional[list] = None) -> np.ndarray:
        if landmarks and len(landmarks) >= 2:
            return self.legacy_extractor.align_face(face_image, landmarks)
        return cv2.resize(face_image, self.legacy_extractor.input_size)

    def extract_feature(
        self, face_image: np.ndarray, landmarks: Optional[list] = None
    ) -> np.ndarray:
        if self.use_arcface and self.arcface_extractor is not None:
            return self.arcface_extractor.extract_feature(face_image, landmarks)

        aligned = self.align(face_image, landmarks)
        return self.legacy_extractor.extract_feature(aligned)

    def similarity(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        if self.use_arcface and self._arcface_cls is not None:
            return float(self._arcface_cls.cosine_similarity(feature_a, feature_b))
        return float(self.legacy_extractor.cosine_similarity(feature_a, feature_b))

    @property
    def feature_dim(self) -> int:
        if self.use_arcface and self.arcface_extractor is not None:
            return getattr(self.arcface_extractor, "feature_dim", 512)
        return self.legacy_extractor.feature_dim

    @property
    def is_arcface(self) -> bool:
        return self.use_arcface
