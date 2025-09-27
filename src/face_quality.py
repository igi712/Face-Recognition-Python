#!/usr/bin/env python3
"""Quality manager composed from blur and liveness analyzers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .quality import BlurAnalyzer, LiveDetector


class FaceQualityAssessment:
    """Coordinator that mirrors Jetson's TBlur + TLive layering."""

    def __init__(self, fast_mode: bool = False) -> None:
        self.fast_mode = fast_mode

        self.blur_analyzer = BlurAnalyzer(threshold=-25.0)
        self.live_detector = LiveDetector(
            fast_mode=fast_mode,
            threshold=0.75 if fast_mode else 0.93,
        )

    # ------------------------------------------------------------------
    # Public API preserved for compatibility
    # ------------------------------------------------------------------

    @property
    def blur_threshold(self) -> float:
        return self.blur_analyzer.threshold

    @property
    def liveness_threshold(self) -> float:
        return self.live_detector.threshold

    def assess_blur(self, face_image) -> Tuple[float, bool]:
        return self.blur_analyzer.assess(face_image)

    def assess_liveness(self, face_image) -> Tuple[float, bool]:
        return self.live_detector.assess(face_image)

    # ------------------------------------------------------------------
    # Higher-level orchestration
    # ------------------------------------------------------------------

    def assess_face_angle(
        self, landmarks: Optional[list], max_angle: float = 10.0
    ) -> Tuple[float, bool]:
        if not landmarks or len(landmarks) < 2:
            return 0.0, True

        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        if dx == 0:
            angle = 90.0 if dy > 0 else -90.0
        else:
            angle = float(np.arctan2(dy, dx) * 180.0 / np.pi)

        return angle, abs(angle) <= max_angle

    def comprehensive_quality_check(
        self,
        face_image,
        landmarks: Optional[list] = None,
        *,
        require_liveness: bool = True,
        require_blur: bool = True,
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}

        if require_blur:
            blur_score, is_sharp = self.assess_blur(face_image)
        else:
            blur_score, is_sharp = 0.0, True
        results["blur_score"] = blur_score
        results["is_sharp"] = is_sharp

        if require_liveness:
            live_score, is_live = self.assess_liveness(face_image)
        else:
            live_score, is_live = 1.0, True
        results["liveness_score"] = live_score
        results["is_live"] = is_live

        angle, is_frontal = self.assess_face_angle(landmarks or [])
        results["face_angle"] = angle
        results["is_frontal"] = is_frontal

        quality_components = []
        if is_sharp or not require_blur:
            quality_components.append(0.4)
        if is_live or not require_liveness:
            quality_components.append(0.4)
        if is_frontal:
            quality_components.append(0.2)

        results["overall_quality"] = float(sum(quality_components))
        results["is_good_quality"] = results["overall_quality"] >= 0.6
        return results


__all__ = ["FaceQualityAssessment"]