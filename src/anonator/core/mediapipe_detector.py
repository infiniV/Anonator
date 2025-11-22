import logging
import os
from pathlib import Path
from typing import List, Optional
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """MediaPipe face detector adapter matching DSFD-Pytorch-Inference interface."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_FILENAME = "blaze_face_short_range.tflite"

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize MediaPipe face detector.

        Args:
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            nms_iou_threshold: NMS IoU threshold for duplicate suppression
            device: Ignored for MediaPipe (CPU only)
            **kwargs: Additional args ignored for compatibility
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Download model if needed
        model_path = self._ensure_model_downloaded()

        # Initialize MediaPipe detector
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=confidence_threshold,
            min_suppression_threshold=nms_iou_threshold
        )

        self.detector = vision.FaceDetector.create_from_options(options)
        self._frame_timestamp_ms = 0

        logger.info(f"MediaPipe detector initialized (confidence={confidence_threshold}, nms={nms_iou_threshold})")

    def _ensure_model_downloaded(self) -> Path:
        """Download MediaPipe model file if not present.

        Returns:
            Path to model file
        """
        # Store in user cache directory
        cache_dir = Path.home() / ".cache" / "anonator" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / self.MODEL_FILENAME

        if not model_path.exists():
            logger.info(f"Downloading MediaPipe model to {model_path}")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                logger.info("MediaPipe model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download MediaPipe model: {e}")
                raise

        return model_path

    def _convert_detections_to_boxes(self, detection_result) -> Optional[np.ndarray]:
        """Convert MediaPipe detections to bounding box format.

        Args:
            detection_result: MediaPipe detection result

        Returns:
            Array of shape [N, 5] with format [xmin, ymin, xmax, ymax, score]
            or None if no detections
        """
        if not detection_result or not detection_result.detections:
            return None

        boxes = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            score = detection.categories[0].score

            # MediaPipe bbox is in pixel coordinates
            xmin = bbox.origin_x
            ymin = bbox.origin_y
            xmax = xmin + bbox.width
            ymax = ymin + bbox.height

            boxes.append([xmin, ymin, xmax, ymax, score])

        return np.array(boxes, dtype=np.float32) if boxes else None

    def batched_detect(self, frames: np.ndarray) -> List[Optional[np.ndarray]]:
        """Detect faces in a batch of frames.

        Args:
            frames: Batch of frames with shape [B, H, W, C] in RGB format

        Returns:
            List of detection arrays, one per frame. Each array has shape [N, 5]
            with format [xmin, ymin, xmax, ymax, score], or None if no faces detected
        """
        results = []

        for frame in frames:
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Detect faces with incremental timestamp
            detection_result = self.detector.detect_for_video(mp_image, self._frame_timestamp_ms)
            self._frame_timestamp_ms += 1

            # Convert to standard bbox format
            boxes = self._convert_detections_to_boxes(detection_result)
            results.append(boxes)

        return results

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces in a single frame.

        Args:
            frame: Single frame with shape [H, W, C] in RGB format

        Returns:
            Detection array with shape [N, 5] and format [xmin, ymin, xmax, ymax, score]
            or None if no faces detected
        """
        # Use batched_detect with batch size 1
        batch = np.expand_dims(frame, axis=0)
        results = self.batched_detect(batch)
        return results[0] if results else None

    def __del__(self):
        """Clean up detector resources."""
        if hasattr(self, 'detector'):
            self.detector.close()
