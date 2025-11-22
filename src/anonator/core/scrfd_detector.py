import logging
from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class SCRFDDetector:
    """SCRFD face detector adapter using InsightFace.

    Supports multiple SCRFD variants:
    - SCRFD-10GF: Best balance (buffalo_l pack)
    - SCRFD-2.5GF: Fastest (buffalo_m pack)
    - SCRFD-34GF: Most accurate (buffalo_sc pack)
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize SCRFD face detector.

        Args:
            model_name: Model pack name ('buffalo_l', 'buffalo_m', 'buffalo_s', 'buffalo_sc')
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            nms_iou_threshold: NMS IoU threshold (not directly used by InsightFace)
            device: Device to use ('cpu' or 'cuda')
            **kwargs: Additional args ignored for compatibility
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.device = device

        # Map device to context ID
        ctx_id = 0 if device == 'cuda' else -1

        # Determine providers based on device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Initialize InsightFace app
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers,
            allowed_modules=['detection']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=confidence_threshold)

        variant_name = self._get_variant_name(model_name)
        logger.info(f"SCRFD {variant_name} detector initialized (confidence={confidence_threshold}, device={device})")

    def _get_variant_name(self, model_name: str) -> str:
        """Get human-readable variant name."""
        variant_map = {
            'buffalo_l': '10GF',
            'buffalo_m': '2.5GF',
            'buffalo_s': '500MF',
            'buffalo_sc': '34GF'
        }
        return variant_map.get(model_name, model_name)

    def _convert_faces_to_boxes(self, faces) -> Optional[np.ndarray]:
        """Convert InsightFace face objects to bounding box format.

        Args:
            faces: List of Face objects from InsightFace

        Returns:
            Array of shape [N, 5] with format [xmin, ymin, xmax, ymax, score]
            or None if no detections
        """
        if not faces:
            return None

        boxes = []
        for face in faces:
            bbox = face.bbox.astype(np.float32)
            score = float(face.det_score)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], score])

        return np.array(boxes, dtype=np.float32)

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
            # Convert RGB to BGR (InsightFace expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect faces
            faces = self.app.get(frame_bgr)

            # Convert to standard format
            boxes = self._convert_faces_to_boxes(faces)
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


class SCRFD10GFDetector(SCRFDDetector):
    """SCRFD-10GF detector (best balance)."""

    def __init__(self, confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.3, device: str = 'cpu', **kwargs):
        super().__init__(
            model_name='buffalo_l',
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            **kwargs
        )


class SCRFD2_5GFDetector(SCRFDDetector):
    """SCRFD-2.5GF detector (fastest)."""

    def __init__(self, confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.3, device: str = 'cpu', **kwargs):
        super().__init__(
            model_name='buffalo_m',
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            **kwargs
        )


class SCRFD34GFDetector(SCRFDDetector):
    """SCRFD-34GF detector (most accurate)."""

    def __init__(self, confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.3, device: str = 'cpu', **kwargs):
        super().__init__(
            model_name='buffalo_sc',
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            **kwargs
        )
