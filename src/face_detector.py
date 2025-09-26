#!/usr/bin/env python3
"""
Face Detection Module - Python Implementation
Based on Face-Recognition-Jetson-Nano project
Using MTCNN or RetinaFace for face detection and landmark extraction

Created: 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import math
import os

from .ncnn_models import NCNN_AVAILABLE

if NCNN_AVAILABLE:  # pragma: no cover - optional dependency
    import ncnn

class FaceObject:
    """Face detection result object"""
    def __init__(self):
        self.rect = None  # cv2.Rect (x, y, width, height)
        self.landmark = []  # 5 facial landmarks
        self.face_prob = 0.0  # Face detection confidence
        self.name_prob = 0.0  # Face recognition confidence
        self.live_prob = 0.0  # Liveness detection confidence
        self.angle = 0.0  # Face angle
        self.name_index = -1  # Index in face database
        self.color = 0  # Color coding for visualization
        self.feature = None  # Face feature vector

class FaceDetector:
    """Face detection using OpenCV DNN"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize face detector
        Args:
            model_path: Path to face detection model
            config_path: Path to model configuration
        """
        self.net = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_width = 320
        self.input_height = 240

        self.retina_net: Optional["ncnn.Net"] = None
        self.retina_score_threshold = 0.6
        self.retina_nms_threshold = 0.4
        self.retina_target_size = 640
        self.retina_max_size = 640
        self.retina_mean_vals = (104.0, 117.0, 123.0)
        self.retina_norm_vals = (1.0, 1.0, 1.0)
        self.use_retina = False
        self._retina_anchor_cache: dict = {}

        if NCNN_AVAILABLE:
            self._load_retinaface_detector()

        if not self.use_retina:
            # Initialize with OpenCV's DNN face detector if no model provided
            if model_path is None:
                self._load_opencv_face_detector()
            else:
                self._load_custom_model(model_path, config_path)
    
    def _load_opencv_face_detector(self):
        """Load OpenCV's DNN face detector"""
        try:
            # Try to load the face detection model
            # You can download these from OpenCV's repository
            prototxt_path = "models/opencv_face_detector.pbtxt"
            model_path = "models/opencv_face_detector_uint8.pb"
            
            self.net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
            print("Loaded OpenCV DNN face detector")
        except:
            print("Warning: Could not load OpenCV face detector. Using Haar cascades as fallback.")
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.net = None
    
    def _load_custom_model(self, model_path: str, config_path: str):
        """Load custom face detection model"""
        try:
            self.net = cv2.dnn.readNet(model_path, config_path)
            print(f"Loaded custom face detector from {model_path}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            self._load_opencv_face_detector()
    
    def detect_faces(
        self,
        frame: np.ndarray,
        resize_factor: Union[float, Tuple[float, float], None] = 1.0,
    ) -> List[FaceObject]:
        """
        Detect faces in the input frame
        Args:
            frame: Input image as numpy array
            resize_factor: Optional scaling factor (<1.0 downsamples before detection).
                            Can be a float (uniform) or (fx, fy) tuple for per-axis scaling.
        Returns:
            List of detected face objects
        """
        faces: List[FaceObject] = []

        if resize_factor is None:
            scale_x = scale_y = 1.0
        elif isinstance(resize_factor, (tuple, list)):
            if len(resize_factor) == 0:
                scale_x = scale_y = 1.0
            elif len(resize_factor) == 1:
                scale_x = scale_y = float(resize_factor[0])
            else:
                scale_x = float(resize_factor[0])
                scale_y = float(resize_factor[1])
        else:
            scale_x = scale_y = float(resize_factor)

        if scale_x <= 0 or np.isnan(scale_x):
            scale_x = 1.0
        if scale_y <= 0 or np.isnan(scale_y):
            scale_y = 1.0

        scale_x = min(scale_x, 1.0)
        scale_y = min(scale_y, 1.0)

        if not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0):
            resized_frame = cv2.resize(
                frame,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized_frame = frame

        if self.use_retina and self.retina_net is not None:
            faces = self._detect_with_retina(resized_frame)
        elif self.net is not None:
            faces = self._detect_with_dnn(resized_frame)
        else:
            faces = self._detect_with_cascade(resized_frame)

        if faces and (not np.isclose(scale_x, 1.0) or not np.isclose(scale_y, 1.0)):
            inv_x = 1.0 / scale_x if scale_x > 0 else 1.0
            inv_y = 1.0 / scale_y if scale_y > 0 else 1.0
            self._rescale_faces(faces, inv_x, inv_y)
        
        if not self.use_retina:
            # Extract landmarks using simple heuristics as fallback
            for face in faces:
                face.landmark = self._extract_landmarks(frame, face.rect)
                face.angle = self._calculate_face_angle(face.landmark)
        
        return faces

    @staticmethod
    def _rescale_faces(faces: List[FaceObject], scale_x: float, scale_y: float) -> None:
        """Scale detected face coordinates back to the original frame size."""
        if np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0):
            return

        for face in faces:
            if not face.rect:
                continue

            x, y, w, h = face.rect
            x = int(round(x * scale_x))
            y = int(round(y * scale_y))
            w = int(max(round(w * scale_x), 1))
            h = int(max(round(h * scale_y), 1))
            face.rect = (x, y, w, h)

            if face.landmark:
                scaled_landmarks = []
                for lx, ly in face.landmark:
                    scaled_landmarks.append(
                        (int(round(lx * scale_x)), int(round(ly * scale_y)))
                    )
                face.landmark = scaled_landmarks

    # ------------------------------------------------------------------
    # RetinaFace NCNN detector
    # ------------------------------------------------------------------

    def _load_retinaface_detector(self) -> None:
        """Load the NCNN RetinaFace detector if model files are available."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.abspath(os.path.join(current_dir, "..", "models", "retina"))
        param_path = os.path.join(models_dir, "mnet.25-opt.param")
        bin_path = os.path.join(models_dir, "mnet.25-opt.bin")

        if not (os.path.exists(param_path) and os.path.exists(bin_path)):
            return

        try:  # pragma: no cover - depends on external binary models
            retina_net = ncnn.Net()
            retina_net.opt.use_vulkan_compute = False
            retina_net.load_param(param_path)
            retina_net.load_model(bin_path)
            self.retina_net = retina_net
            self.use_retina = True
            print("✅ RetinaFace NCNN detector loaded")
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"⚠️  Failed to load RetinaFace NCNN detector: {exc}")
            self.retina_net = None
            self.use_retina = False

    def _detect_with_retina(self, frame: np.ndarray) -> List[FaceObject]:
        """Detect faces using the NCNN RetinaFace model."""
        img_h, img_w = frame.shape[:2]

        # Scale image to target size while keeping aspect ratio, then pad to multiple of 32
        im_min = min(img_w, img_h)
        im_max = max(img_w, img_h)
        scale = float(self.retina_target_size) / float(im_max)
        if scale * im_min > self.retina_max_size:
            scale = float(self.retina_max_size) / float(im_min)

        resized_w = max(int(round(img_w * scale)), 1)
        resized_h = max(int(round(img_h * scale)), 1)

        resized = cv2.resize(frame, (resized_w, resized_h)) if scale != 1.0 else frame.copy()

        padded_w = int(math.ceil(resized_w / 32.0) * 32)
        padded_h = int(math.ceil(resized_h / 32.0) * 32)

        padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded[:resized_h, :resized_w, :] = resized

        mat_in = ncnn.Mat.from_pixels(
            padded,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            padded_w,
            padded_h,
        )
        mat_in.substract_mean_normalize(self.retina_mean_vals, self.retina_norm_vals)

        ex = self.retina_net.create_extractor()
        ex.set_light_mode(True)
        ex.input("data", mat_in)

        proposals: List[dict] = []
        proposals.extend(
            self._decode_retina_stride(
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
            self._decode_retina_stride(
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
            self._decode_retina_stride(
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

        final_proposals = self._nms_retina(proposals, self.retina_nms_threshold)

        faces: List[FaceObject] = []
        inv_scale = 1.0 / scale if scale > 0 else 1.0

        for prop in final_proposals:
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
            face.angle = self._calculate_face_angle(face.landmark)
            faces.append(face)

        return faces

    def _decode_retina_stride(
        self,
        extractor: "ncnn.Extractor",
        base_size: int,
        feat_stride: int,
        ratios: Tuple[float, ...],
        scales: Tuple[float, ...],
        score_name: str,
        bbox_name: str,
        landmark_name: str,
    ) -> List[dict]:
        ret, score_blob = extractor.extract(score_name)
        if ret != 0:
            return []
        ret, bbox_blob = extractor.extract(bbox_name)
        if ret != 0:
            return []
        ret, landmark_blob = extractor.extract(landmark_name)
        if ret != 0:
            return []

        anchors = self._get_retina_anchors(base_size, ratios, scales)
        return self._generate_retina_proposals(
            anchors,
            feat_stride,
            score_blob,
            bbox_blob,
            landmark_blob,
        )

    def _get_retina_anchors(
        self,
        base_size: int,
        ratios: Tuple[float, ...],
        scales: Tuple[float, ...],
    ) -> "ncnn.Mat":
        key = (base_size, ratios, scales)
        anchors_np = self._retina_anchor_cache.get(key)
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
            self._retina_anchor_cache[key] = anchors_np

        return ncnn.Mat(anchors_np.copy())

    def _generate_retina_proposals(
        self,
        anchors: "ncnn.Mat",
        feat_stride: int,
        score_blob: "ncnn.Mat",
        bbox_blob: "ncnn.Mat",
        landmark_blob: "ncnn.Mat",
    ) -> List[dict]:
        """Decode RetinaFace outputs into bounding box proposals."""
        proposals: List[dict] = []

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
                    if prob < self.retina_score_threshold:
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

    def _nms_retina(self, proposals: List[dict], threshold: float) -> List[dict]:
        """Apply Non-Maximum Suppression to RetinaFace proposals."""
        if not proposals:
            return []

        proposals = sorted(proposals, key=lambda x: x["score"], reverse=True)
        picked: List[dict] = []
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
    
    def _detect_with_dnn(self, frame: np.ndarray) -> List[FaceObject]:
        """Detect faces using DNN"""
        faces = []
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (self.input_width, self.input_height), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                face = FaceObject()
                face.rect = (x1, y1, x2 - x1, y2 - y1)
                face.face_prob = confidence
                faces.append(face)
        
        return faces
    
    def _detect_with_cascade(self, frame: np.ndarray) -> List[FaceObject]:
        """Detect faces using Haar cascades as fallback"""
        faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_rects = self.cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in face_rects:
            face = FaceObject()
            face.rect = (x, y, w, h)
            face.face_prob = 0.8  # Default confidence for cascade
            faces.append(face)
        
        return faces
    
    def _extract_landmarks(self, frame: np.ndarray, rect: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """
        Extract facial landmarks (simplified version)
        In a full implementation, you would use dlib or similar
        """
        x, y, w, h = rect
        landmarks = []
        
        # Rough estimation of landmark positions
        # Left eye
        landmarks.append((x + w//4, y + h//3))
        # Right eye  
        landmarks.append((x + 3*w//4, y + h//3))
        # Nose
        landmarks.append((x + w//2, y + h//2))
        # Left mouth corner
        landmarks.append((x + w//3, y + 2*h//3))
        # Right mouth corner
        landmarks.append((x + 2*w//3, y + 2*h//3))
        
        return landmarks
    
    def _calculate_face_angle(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate face rotation angle from landmarks"""
        if len(landmarks) < 2:
            return 0.0
        
        # Use eye positions to calculate angle
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        return angle
    
    def filter_faces(
        self,
        faces: List[FaceObject],
        min_face_size: int = 90,
        min_score: float = 0.0,
    ) -> List[FaceObject]:
        """
        Filter faces based on quality criteria
        Args:
            faces: List of detected faces
            min_face_size: Minimum face height in pixels
            min_score: Minimum detector confidence threshold (0-1)
        Returns:
            Filtered list of faces
        """
        filtered_faces = []
        
        for face in faces:
            x, y, w, h = face.rect

            if min_score > 0.0 and face.face_prob < min_score:
                continue
            
            # Check minimum face size
            if h < min_face_size:
                face.name_index = -2  # Too tiny
                face.color = 2
                continue
            
            # Check face angle (optional)
            if abs(face.angle) > 30:  # Face too tilted
                continue
            
            face.face_prob = float(face.face_prob)
            filtered_faces.append(face)
        
        return filtered_faces