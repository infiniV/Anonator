import logging
from typing import List, Optional

import numpy as np
import torch
from facenet_pytorch import MTCNN as FacenetMTCNN

logger = logging.getLogger(__name__)


class MTCNNDetector:
    """MTCNN face detector adapter using facenet-pytorch implementation."""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize MTCNN face detector.

        Args:
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            nms_iou_threshold: NMS IoU threshold (not used by MTCNN)
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional args ignored for compatibility
        """
        self.confidence_threshold = confidence_threshold
        self.device = device

        # Initialize MTCNN
        torch_device = torch.device(device)
        self.detector = FacenetMTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, confidence_threshold],
            device=torch_device,
            keep_all=True,
            post_process=False
        )

        logger.info(f"MTCNN detector initialized (confidence={confidence_threshold}, device={device})")

    def _convert_boxes_to_standard_format(
        self,
        boxes,
        probs
    ) -> Optional[np.ndarray]:
        """Convert MTCNN boxes to standard format.

        Args:
            boxes: Tensor or array of shape [N, 4] with [x1, y1, x2, y2]
            probs: Tensor or array of shape [N] with detection probabilities

        Returns:
            Array of shape [N, 5] with format [xmin, ymin, xmax, ymax, score]
            or None if no detections
        """
        if boxes is None or probs is None:
            return None

        # Convert to numpy if needed
        if torch.is_tensor(boxes):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = np.array(boxes)

        if torch.is_tensor(probs):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)

        # Combine boxes and scores
        result = np.concatenate([boxes_np, probs_np[:, np.newaxis]], axis=1)
        return result.astype(np.float32)

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
            # MTCNN expects RGB numpy array
            boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)

            # Convert to standard format
            detections = self._convert_boxes_to_standard_format(boxes, probs)
            results.append(detections)

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
