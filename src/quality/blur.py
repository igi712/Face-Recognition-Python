"""Blur quality analysis, mirroring Jetson's TBlur."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class BlurAnalyzer:
    """Assess blur using Laplacian variance in log space."""

    def __init__(self, *, threshold: float = -25.0) -> None:
        self.threshold = threshold

    def assess(self, image: np.ndarray) -> Tuple[float, bool]:
        if image is None or image.size == 0:
            return -100.0, False

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = np.log(laplacian_var) if laplacian_var > 0 else -100.0
        is_sharp = blur_score > self.threshold

        return float(blur_score), bool(is_sharp)
