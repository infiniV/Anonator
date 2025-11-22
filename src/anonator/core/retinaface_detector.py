import logging
from pathlib import Path
from typing import List, Optional
import urllib.request

import cv2
import numpy as np
import torch
from retinaface.pre_trained_models import get_model

logger = logging.getLogger(__name__)


class RetinaFaceMobileNetDetector:
    """RetinaFace-MobileNet detector using retinaface-pytorch."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize RetinaFace-MobileNet face detector.

        Args:
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            nms_iou_threshold: NMS IoU threshold
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional args ignored for compatibility
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device

        # Load pretrained model (resnet50 is default in the package)
        # The package auto-downloads weights on first use
        self.model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        self.model.eval()

        logger.info(f"RetinaFace-MobileNet detector initialized (confidence={confidence_threshold}, device={device})")

    def _convert_predictions_to_boxes(self, predictions) -> Optional[np.ndarray]:
        """Convert RetinaFace predictions to bounding box format.

        Args:
            predictions: List of predictions from RetinaFace

        Returns:
            Array of shape [N, 5] with format [xmin, ymin, xmax, ymax, score]
            or None if no detections
        """
        if not predictions or len(predictions) == 0:
            return None

        boxes_list = []
        for pred in predictions:
            # pred is a dict with 'bbox' and 'score' keys
            if isinstance(pred, dict):
                bbox = pred.get('bbox', None)
                score = pred.get('score', 0.0)

                if bbox is not None and score >= self.confidence_threshold:
                    # bbox format: [x1, y1, x2, y2]
                    boxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3], float(score)])

        if not boxes_list:
            return None

        return np.array(boxes_list, dtype=np.float32)

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
            # Convert RGB to BGR (RetinaFace expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect faces
            with torch.no_grad():
                predictions = self.model.predict_jsons(
                    frame_bgr,
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_iou_threshold
                )

            # Convert to standard format
            boxes = self._convert_predictions_to_boxes(predictions)
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
