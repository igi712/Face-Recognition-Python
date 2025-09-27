"""RetinaFace detector backend mirroring Jetson's TRetina."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..ncnn_models import NCNN_AVAILABLE
from .types import FaceObject

if NCNN_AVAILABLE:  # pragma: no cover - optional dependency
    import ncnn


class RetinaDetector:
    """Thin wrapper around the NCNN RetinaFace model."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        target_size: int = 640,
        max_size: int = 640,
    ) -> None:
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.target_size = target_size
        self.max_size = max_size
        self.mean_vals = (104.0, 117.0, 123.0)
        self.norm_vals = (1.0, 1.0, 1.0)

        self._net: "ncnn.Net | None" = None
        self._anchor_cache: Dict[Tuple[int, Tuple[float, ...], Tuple[float, ...]], np.ndarray] = {}

        if NCNN_AVAILABLE:
            self._load_model()

    @property
    def is_available(self) -> bool:
        return self._net is not None

    def detect(self, frame: np.ndarray) -> List[FaceObject]:
        if not self._net:
            return []

        img_h, img_w = frame.shape[:2]
        if img_h == 0 or img_w == 0:
            return []

        scale, resized = self._resize_keep_ratio(frame)
        padded, padded_w, padded_h = self._pad_to_stride(resized)

        mat_in = ncnn.Mat.from_pixels(
            padded,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            padded_w,
            padded_h,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self._net.create_extractor()
        ex.set_light_mode(True)
        ex.input("data", mat_in)

        proposals: List[Dict[str, object]] = []
        proposals.extend(
            self._decode_stride(
                ex,
                base_size=16,
                feat_stride=32,
                ratios=(1.0,),
                scales=(32.0, 16.0),
                score_name="face_rpn_cls_prob_reshape_stride32",
                bbox_name="face_rpn_bbox_pred_stride32",
                landmark_name="face_rpn_landmark_pred_stride32",
            )
        )
        proposals.extend(
            self._decode_stride(
                ex,
                base_size=16,
                feat_stride=16,
                ratios=(1.0,),
                scales=(8.0, 4.0),
                score_name="face_rpn_cls_prob_reshape_stride16",
                bbox_name="face_rpn_bbox_pred_stride16",
                landmark_name="face_rpn_landmark_pred_stride16",
            )
        )
        proposals.extend(
            self._decode_stride(
                ex,
                base_size=16,
                feat_stride=8,
                ratios=(1.0,),
                scales=(2.0, 1.0),
                score_name="face_rpn_cls_prob_reshape_stride8",
                bbox_name="face_rpn_bbox_pred_stride8",
                landmark_name="face_rpn_landmark_pred_stride8",
            )
        )

        if not proposals:
            return []

        picks = self._nms(proposals, self.nms_threshold)
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        faces: List[FaceObject] = []
        resized_h, resized_w = resized.shape[:2]

        for prop in picks:
            x0, y0, x1, y1 = prop["bbox"]
            x0 = max(min(x0, resized_w - 1), 0.0) * inv_scale
            y0 = max(min(y0, resized_h - 1), 0.0) * inv_scale
            x1 = max(min(x1, resized_w - 1), 0.0) * inv_scale
            y1 = max(min(y1, resized_h - 1), 0.0) * inv_scale

            face = FaceObject()
            face.face_prob = float(prop["score"])
            face.rect = (
                int(round(x0)),
                int(round(y0)),
                int(max(round(x1 - x0), 1)),
                int(max(round(y1 - y0), 1)),
            )

            landmarks = []
            for lx, ly in prop["landmarks"]:
                lx = max(min(lx, resized_w - 1), 0.0) * inv_scale
                ly = max(min(ly, resized_h - 1), 0.0) * inv_scale
                landmarks.append((int(round(lx)), int(round(ly))))

            face.landmark = landmarks
            face.angle = self._calculate_angle(landmarks)
            faces.append(face)

        return faces

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        module_dir = Path(__file__).resolve().parent.parent
        models_dir = module_dir / "models" / "retina"
        param_path = models_dir / "mnet.25-opt.param"
        bin_path = models_dir / "mnet.25-opt.bin"

        if not (param_path.exists() and bin_path.exists()):
            return

        try:  # pragma: no cover
            net = ncnn.Net()
            net.opt.use_vulkan_compute = False
            net.load_param(str(param_path))
            net.load_model(str(bin_path))
            self._net = net
            print("✅ RetinaFace NCNN detector loaded")
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"⚠️  Failed to load RetinaFace NCNN detector: {exc}")
            self._net = None

    def _resize_keep_ratio(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        img_h, img_w = frame.shape[:2]
        im_min = min(img_w, img_h)
        im_max = max(img_w, img_h)

        scale = float(self.target_size) / float(im_max)
        if scale * im_min > self.max_size:
            scale = float(self.max_size) / float(im_min)

        resized_w = max(int(round(img_w * scale)), 1)
        resized_h = max(int(round(img_h * scale)), 1)

        if scale != 1.0:
            resized = cv2.resize(frame, (resized_w, resized_h))
        else:
            resized = frame.copy()

        return scale, resized

    def _pad_to_stride(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]
        padded_w = int(math.ceil(w / 32.0) * 32)
        padded_h = int(math.ceil(h / 32.0) * 32)

        if padded_w == w and padded_h == h:
            return frame, w, h

        padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded[:h, :w, :] = frame
        return padded, padded_w, padded_h

    def _decode_stride(
        self,
        extractor: "ncnn.Extractor",
        *,
        base_size: int,
        feat_stride: int,
        ratios: Tuple[float, ...],
        scales: Tuple[float, ...],
        score_name: str,
        bbox_name: str,
        landmark_name: str,
    ) -> List[Dict[str, object]]:
        ret, score_blob = extractor.extract(score_name)
        if ret != 0:
            return []
        ret, bbox_blob = extractor.extract(bbox_name)
        if ret != 0:
            return []
        ret, landmark_blob = extractor.extract(landmark_name)
        if ret != 0:
            return []

        anchors = self._get_anchors(base_size, ratios, scales)
        return self._generate_proposals(
            anchors,
            feat_stride,
            score_blob,
            bbox_blob,
            landmark_blob,
        )

    def _get_anchors(
        self,
        base_size: int,
        ratios: Tuple[float, ...],
        scales: Tuple[float, ...],
    ) -> "ncnn.Mat":
        key = (base_size, ratios, scales)
        anchors_np = self._anchor_cache.get(key)
        if anchors_np is None:
            anchors_np = []
            cx = base_size * 0.5
            cy = base_size * 0.5

            for ratio in ratios:
                r_w = round(base_size / math.sqrt(ratio))
                r_h = round(r_w * ratio)

                for scale in scales:
                    rs_w = r_w * scale
                    rs_h = r_h * scale
                    anchors_np.append(
                        [
                            cx - rs_w * 0.5,
                            cy - rs_h * 0.5,
                            cx + rs_w * 0.5,
                            cy + rs_h * 0.5,
                        ]
                    )

            anchors_np = np.array(anchors_np, dtype=np.float32)
            self._anchor_cache[key] = anchors_np

        return ncnn.Mat(anchors_np.copy())

    def _generate_proposals(
        self,
        anchors: "ncnn.Mat",
        feat_stride: int,
        score_blob: "ncnn.Mat",
        bbox_blob: "ncnn.Mat",
        landmark_blob: "ncnn.Mat",
    ) -> List[Dict[str, object]]:
        proposals: List[Dict[str, object]] = []

        feat_w = score_blob.w
        feat_h = score_blob.h
        num_anchors = int(anchors.h)

        for anchor_idx in range(num_anchors):
            anchor = np.array(anchors.row(anchor_idx), dtype=np.float32)
            anchor_w = anchor[2] - anchor[0]
            anchor_h = anchor[3] - anchor[1]

            score_arr = np.array(score_blob.channel(anchor_idx + num_anchors), copy=False).reshape(-1)
            bbox_range = bbox_blob.channel_range(anchor_idx * 4, 4)
            landmark_range = landmark_blob.channel_range(anchor_idx * 10, 10)

            bbox_arr = [
                np.array(bbox_range.channel(c), copy=False).reshape(-1)
                for c in range(4)
            ]
            landmark_arr = [
                np.array(landmark_range.channel(c), copy=False).reshape(-1)
                for c in range(10)
            ]

            anchor_y = float(anchor[1])
            for i in range(feat_h):
                anchor_x = float(anchor[0])
                for j in range(feat_w):
                    index = i * feat_w + j
                    prob = float(score_arr[index])
                    if prob < self.score_threshold:
                        anchor_x += feat_stride
                        continue

                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5

                    dx = float(bbox_arr[0][index])
                    dy = float(bbox_arr[1][index])
                    dw = float(bbox_arr[2][index])
                    dh = float(bbox_arr[3][index])

                    pb_cx = cx + anchor_w * dx
                    pb_cy = cy + anchor_h * dy
                    pb_w = anchor_w * math.exp(dw)
                    pb_h = anchor_h * math.exp(dh)

                    x0 = pb_cx - pb_w * 0.5
                    y0 = pb_cy - pb_h * 0.5
                    x1 = pb_cx + pb_w * 0.5
                    y1 = pb_cy + pb_h * 0.5

                    landmarks = []
                    for k in range(5):
                        lx = cx + (anchor_w + 1.0) * float(landmark_arr[2 * k][index])
                        ly = cy + (anchor_h + 1.0) * float(landmark_arr[2 * k + 1][index])
                        landmarks.append((lx, ly))

                    proposals.append(
                        {
                            "bbox": [x0, y0, x1, y1],
                            "score": prob,
                            "landmarks": landmarks,
                        }
                    )

                    anchor_x += feat_stride
                anchor_y += feat_stride

        return proposals

    def _nms(self, proposals: List[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
        if not proposals:
            return []

        proposals = sorted(proposals, key=lambda x: float(x["score"]), reverse=True)
        picked: List[Dict[str, object]] = []
        areas = []
        for prop in proposals:
            x0, y0, x1, y1 = prop["bbox"]
            areas.append(max(x1 - x0, 0.0) * max(y1 - y0, 0.0))

        suppressed = [False] * len(proposals)

        for i, prop in enumerate(proposals):
            if suppressed[i]:
                continue
            picked.append(prop)

            x0_i, y0_i, x1_i, y1_i = prop["bbox"]
            area_i = areas[i]

            for j in range(i + 1, len(proposals)):
                if suppressed[j]:
                    continue

                x0_j, y0_j, x1_j, y1_j = proposals[j]["bbox"]

                inter_x0 = max(x0_i, x0_j)
                inter_y0 = max(y0_i, y0_j)
                inter_x1 = min(x1_i, x1_j)
                inter_y1 = min(y1_i, y1_j)

                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue

                inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                union_area = area_i + areas[j] - inter_area
                if union_area <= 0:
                    continue
                if inter_area / union_area > threshold:
                    suppressed[j] = True

        return picked

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
        return math.atan2(dy, dx) * 180.0 / math.pi
