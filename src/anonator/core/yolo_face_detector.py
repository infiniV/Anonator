import logging
from pathlib import Path
from typing import List, Optional
import urllib.request

import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class YOLOFaceDetector:
    """Base YOLO face detector adapter using Ultralytics."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize YOLO face detector.

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum detection confidence (0.0 to 1.0)
            nms_iou_threshold: NMS IoU threshold
            device: Device to use ('cpu' or 'cuda', '0', '1', etc.)
            **kwargs: Additional args ignored for compatibility
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device if device != 'cuda' else 0

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)

        model_name = Path(model_path).stem
        logger.info(f"YOLO {model_name} detector initialized (confidence={confidence_threshold}, device={device})")

    def _convert_results_to_boxes(self, results) -> Optional[np.ndarray]:
        """Convert YOLO results to bounding box format.

        Args:
            results: YOLO Results object

        Returns:
            Array of shape [N, 5] with format [xmin, ymin, xmax, ymax, score]
            or None if no detections
        """
        if not results or len(results) == 0:
            return None

        boxes_list = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Extract boxes and scores
            boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
            scores = result.boxes.conf.cpu().numpy()  # [N]

            # Combine into [N, 5]
            for box, score in zip(boxes, scores):
                boxes_list.append([box[0], box[1], box[2], box[3], float(score)])

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
        # Convert frames list to list for YOLO
        frame_list = [frame for frame in frames]

        # Run batch inference
        results = self.model.predict(
            frame_list,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
            device=self.device
        )

        # Convert each result
        detections = []
        for result in results:
            boxes = self._convert_results_to_boxes([result])
            detections.append(boxes)

        return detections

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


class YOLOv8FaceDetector(YOLOFaceDetector):
    """YOLOv8-Face detector using pretrained weights."""

    MODEL_URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    MODEL_FILENAME = "yolov8n-face.pt"

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize YOLOv8-Face detector."""
        model_path = self._ensure_model_downloaded()

        super().__init__(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            **kwargs
        )

    def _ensure_model_downloaded(self) -> Path:
        """Download YOLOv8-Face model if not present."""
        cache_dir = Path.home() / ".cache" / "anonator" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / self.MODEL_FILENAME

        if not model_path.exists():
            logger.info(f"Downloading YOLOv8-Face model to {model_path}")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                logger.info("YOLOv8-Face model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download YOLOv8-Face model: {e}")
                raise

        return model_path


class YOLO11FaceDetector(YOLOFaceDetector):
    """YOLO11-Face detector using Hugging Face weights."""

    HF_REPO_ID = "AdamCodd/YOLOv11n-face-detection"
    HF_FILENAME = "model.pt"
    MODEL_FILENAME = "yolo11n-face.pt"

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = 'cpu',
        **kwargs
    ):
        """Initialize YOLO11-Face detector."""
        model_path = self._ensure_model_downloaded()

        super().__init__(
            model_path=str(model_path),
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
            **kwargs
        )

    def _ensure_model_downloaded(self) -> Path:
        """Download YOLO11-Face model from Hugging Face."""
        cache_dir = Path.home() / ".cache" / "anonator" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / self.MODEL_FILENAME

        if not model_path.exists():
            logger.info(f"Downloading YOLO11-Face model from Hugging Face to {model_path}")
            try:
                # Download from Hugging Face
                downloaded_path = hf_hub_download(
                    repo_id=self.HF_REPO_ID,
                    filename=self.HF_FILENAME,
                    cache_dir=cache_dir
                )
                # Copy to our standard location
                import shutil
                shutil.copy(downloaded_path, model_path)
                logger.info("YOLO11-Face model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download YOLO11-Face model: {e}")
                raise

        return model_path
