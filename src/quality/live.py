"""Liveness evaluation mirroring Jetson's TLive."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class LiveDetector:
    """Simple heuristic-based liveness detector with fast-mode tweaks."""

    def __init__(
        self,
        *,
        fast_mode: bool = False,
        threshold: float = 0.93,
    ) -> None:
        self.fast_mode = fast_mode
        self.threshold = threshold if not fast_mode else max(0.75, threshold)
        self._analysis_max_side = 96 if fast_mode else 160
        self._lbp_stride = 2 if fast_mode else 1
        self.use_simple_liveness = True

    def assess(self, image: np.ndarray) -> Tuple[float, bool]:
        if image is None or image.size == 0:
            return 0.0, False

        if self.use_simple_liveness:
            return self._simple_liveness(image)
        return self._model_liveness(image)

    # ------------------------------------------------------------------
    # Simple heuristic-based detection
    # ------------------------------------------------------------------

    def _simple_liveness(self, image: np.ndarray) -> Tuple[float, bool]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        if self.fast_mode:
            return self._fast_liveness(gray, hsv)

        lbp_score = self._calculate_lbp_uniformity(gray)
        color_score = self._analyze_color_distribution(hsv)
        edge_score = self._calculate_edge_density(gray)
        freq_score = self._analyze_frequency_domain(gray)

        weights = (0.3, 0.2, 0.25, 0.25)
        liveness_score = (
            weights[0] * lbp_score
            + weights[1] * color_score
            + weights[2] * edge_score
            + weights[3] * freq_score
        )

        return float(liveness_score), bool(liveness_score > self.threshold)

    def _fast_liveness(self, gray: np.ndarray, hsv: np.ndarray) -> Tuple[float, bool]:
        gray_small, hsv_small = self._resize_pair_for_fast_mode(gray, hsv)

        contrast = float(np.clip(np.std(gray_small) / 64.0, 0.0, 1.0))
        edge_score = self._calculate_edge_density(gray_small)
        color_score = self._analyze_color_distribution(hsv_small)

        liveness_score = 0.45 * contrast + 0.35 * edge_score + 0.20 * color_score
        return float(liveness_score), bool(liveness_score > self.threshold)

    def _model_liveness(self, image: np.ndarray) -> Tuple[float, bool]:
        # Placeholder for real anti-spoofing model integration
        return self._simple_liveness(image)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _calculate_lbp_uniformity(self, gray_image: np.ndarray) -> float:
        gray_work = self._resize_for_analysis(gray_image)
        h, w = gray_work.shape
        if h < 16 or w < 16:
            return 0.5

        patterns = []
        for y in range(1, h - 1, self._lbp_stride):
            for x in range(1, w - 1, self._lbp_stride):
                center = gray_work[y, x]
                pattern = 0
                neighbors = [
                    gray_work[y - 1, x - 1],
                    gray_work[y - 1, x],
                    gray_work[y - 1, x + 1],
                    gray_work[y, x + 1],
                    gray_work[y + 1, x + 1],
                    gray_work[y + 1, x],
                    gray_work[y + 1, x - 1],
                    gray_work[y, x - 1],
                ]
                for i, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= 1 << i
                patterns.append(pattern)

        if not patterns:
            return 0.5

        unique_patterns = len(set(patterns))
        uniformity = 1.0 - (unique_patterns / len(patterns))
        return float(np.clip(uniformity, 0.0, 1.0))

    @staticmethod
    def _analyze_color_distribution(hsv_image: np.ndarray) -> float:
        h_channel = hsv_image[:, :, 0]
        s_channel = hsv_image[:, :, 1]

        skin_mask = (h_channel >= 0) & (h_channel <= 30)
        skin_ratio = float(np.sum(skin_mask)) / float(hsv_image.shape[0] * hsv_image.shape[1])

        sat_mean = float(np.mean(s_channel))
        sat_std = float(np.std(s_channel))

        color_score = (
            skin_ratio * 0.6
            + (sat_mean / 255.0) * 0.2
            + min(sat_std / 50.0, 1.0) * 0.2
        )
        return float(min(color_score, 1.0))

    @staticmethod
    def _calculate_edge_density(gray_image: np.ndarray) -> float:
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges = magnitude > 30

        edge_density = float(np.sum(edges)) / float(gray_image.shape[0] * gray_image.shape[1])
        return float(min(edge_density * 10, 1.0))

    @staticmethod
    def _analyze_frequency_domain(gray_image: np.ndarray) -> float:
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        h, w = magnitude_spectrum.shape
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) > (min(h, w) // 6) ** 2

        high_freq_energy = float(np.mean(magnitude_spectrum[mask]))
        total_energy = float(np.mean(magnitude_spectrum))
        if total_energy == 0:
            return 0.5

        hf_ratio = high_freq_energy / total_energy
        return float(min(hf_ratio / 0.3, 1.0))

    def _resize_for_analysis(self, gray_image: np.ndarray) -> np.ndarray:
        if not self.fast_mode:
            return gray_image

        h, w = gray_image.shape[:2]
        max_side = max(h, w)
        if max_side <= self._analysis_max_side:
            return gray_image

        scale = self._analysis_max_side / float(max_side)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        return cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _resize_pair_for_fast_mode(
        self,
        gray_image: np.ndarray,
        hsv_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fast_mode:
            return gray_image, hsv_image

        h, w = gray_image.shape[:2]
        max_side = max(h, w)
        if max_side <= self._analysis_max_side:
            return gray_image, hsv_image

        scale = self._analysis_max_side / float(max_side)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        gray_small = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        hsv_small = cv2.resize(hsv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray_small, hsv_small
