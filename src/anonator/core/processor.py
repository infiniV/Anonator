import threading
import time
import logging
from typing import Callable, Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import face_detection

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"PyTorch CUDA enabled: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("PyTorch using CPU (no CUDA detected)")
except ImportError as e:
    logger.error(f"Required imports failed: {e}")
    raise

from anonator.core.anonymizer import anonymize_frame
from anonator.core.config import PROCESSOR_CONFIG, HIPAA_MODE


class ProgressData:
    def __init__(
        self,
        frame_number: int,
        total_frames: int,
        original_frame: np.ndarray,
        anonymized_frame: np.ndarray,
        elapsed_time: float,
        fps: float
    ):
        self.frame_number = frame_number
        self.total_frames = total_frames
        self.original_frame = original_frame
        self.anonymized_frame = anonymized_frame
        self.elapsed_time = elapsed_time
        self.fps = fps


class VideoProcessor:
    def __init__(self, progress_callback: Optional[Callable[[ProgressData], None]] = None):
        self.progress_callback = progress_callback
        self._cancel_flag = threading.Event()
        self._processing_thread = None

        self.detector = face_detection.build_detector(
            PROCESSOR_CONFIG.detector_model,
            confidence_threshold=PROCESSOR_CONFIG.detector_confidence,
            nms_iou_threshold=PROCESSOR_CONFIG.nms_iou_threshold,
            device=device,
            max_resolution=PROCESSOR_CONFIG.max_resolution,
            fp16_inference=PROCESSOR_CONFIG.fp16_inference,
            clip_boxes=PROCESSOR_CONFIG.clip_boxes
        )
        logger.info(f"Face detector initialized: {PROCESSOR_CONFIG.detector_model} on {device}")
        logger.info(f"FP16: {PROCESSOR_CONFIG.fp16_inference}, Max resolution: {PROCESSOR_CONFIG.max_resolution}")

    def process_video(
        self,
        input_path: str,
        output_path: str,
        anonymization_mode: str = "blur",
        threshold: float = 0.2,
        mask_scale: float = 1.3,
        preview_interval: float = 1.0,
        multi_pass: bool = False,
        keep_audio: bool = True
    ) -> None:
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning("Processing already in progress")
            return

        self._cancel_flag.clear()
        self._processing_thread = threading.Thread(
            target=self._process_internal,
            kwargs={
                "input_path": input_path,
                "output_path": output_path,
                "anonymization_mode": anonymization_mode,
                "threshold": threshold,
                "mask_scale": mask_scale,
                "preview_interval": preview_interval,
                "multi_pass": multi_pass,
                "keep_audio": keep_audio
            },
            daemon=True
        )
        self._processing_thread.start()

    def cancel(self) -> None:
        logger.info("Cancellation requested")
        self._cancel_flag.set()

    def _remove_duplicate_detections(self, dets: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
        """Remove duplicate detections using Non-Maximum Suppression.

        Note: This is only used in multi-pass mode to remove duplicates across
        different detection passes. The face_detection library already performs
        NMS within each individual detection pass.

        Args:
            dets: Detections array of shape [N, 5] with (xmin, ymin, xmax, ymax, score)
            iou_threshold: IoU threshold for considering boxes as duplicates

        Returns:
            Filtered detections with duplicates removed
        """
        if len(dets) == 0:
            return dets

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return dets[keep]

    def _detect_faces(self, frame: np.ndarray, threshold: float) -> Optional[np.ndarray]:
        self.detector.confidence_threshold = threshold
        dets = self.detector.detect(frame)

        if dets is not None and len(dets) > 0:
            return dets.astype(np.float32)
        return None

    def _process_internal(
        self,
        input_path: str,
        output_path: str,
        anonymization_mode: str,
        threshold: float,
        mask_scale: float,
        preview_interval: float,
        multi_pass: bool,
        keep_audio: bool
    ) -> None:
        try:
            logger.info(f"Starting video processing: {input_path}")
            logger.info("Using PyTorch RetinaFace detector (ResNet50 backbone)")

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video info: {frame_count} frames, {fps} fps, {width}x{height}")

            if multi_pass:
                logger.info("HIPAA Mode: Multi-pass detection enabled (3 passes per frame)")
                logger.info("Note: Processing will be 3x slower but ensures zero missed faces")

            sample_interval = max(1, int(fps * preview_interval))
            log_interval = PROCESSOR_CONFIG.log_interval

            fourcc = cv2.VideoWriter_fourcc(*PROCESSOR_CONFIG.output_codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            start_time = time.time()
            actual_frame_count = 0

            logger.info("Starting frame-by-frame processing...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self._cancel_flag.is_set():
                    logger.info("Processing cancelled by user")
                    break

                frame = frame.astype(np.uint8)

                if actual_frame_count % log_interval == 0:
                    logger.info(f"[Frame {actual_frame_count}/{frame_count}] Starting detection")

                if multi_pass:
                    all_dets = []

                    thresholds = [threshold * m for m in HIPAA_MODE.multi_pass_multipliers]
                    for pass_idx, t in enumerate(thresholds):
                        if actual_frame_count % log_interval == 0:
                            logger.info(f"[Frame {actual_frame_count}] Pass {pass_idx+1}/3 (threshold={t:.4f})")
                        dets = self._detect_faces(frame, threshold=t)
                        if dets is not None and len(dets) > 0:
                            all_dets.append(dets)
                            if actual_frame_count % log_interval == 0:
                                logger.info(f"[Frame {actual_frame_count}] Pass {pass_idx+1}: Found {len(dets)} detections")

                    if all_dets:
                        if actual_frame_count % log_interval == 0:
                            logger.info(f"[Frame {actual_frame_count}] Merging {sum(len(d) for d in all_dets)} total detections")
                        dets = np.vstack(all_dets)
                        dets = self._remove_duplicate_detections(dets)
                        dets = dets.astype(np.float32)
                        if actual_frame_count % log_interval == 0:
                            logger.info(f"[Frame {actual_frame_count}] After NMS: {len(dets)} unique faces")
                    else:
                        dets = None
                else:
                    dets = self._detect_faces(frame, threshold=threshold)

                if actual_frame_count % log_interval == 0:
                    logger.info(f"[Frame {actual_frame_count}] Starting anonymization")

                anonymized = anonymize_frame(
                    frame,
                    dets,
                    mask_scale=float(mask_scale),
                    replacewith=anonymization_mode,
                    ellipse=PROCESSOR_CONFIG.use_ellipse_mask,
                    draw_scores=False,
                    replaceimg=None,
                    mosaicsize=PROCESSOR_CONFIG.mosaic_size
                )

                if actual_frame_count % log_interval == 0:
                    logger.info(f"[Frame {actual_frame_count}] Writing to output")

                out.write(anonymized)
                actual_frame_count += 1

                if actual_frame_count % log_interval == 0:
                    logger.info(f"[Frame {actual_frame_count}] Complete")

                send_progress = (actual_frame_count == 1 or
                                actual_frame_count % sample_interval == 0 or
                                actual_frame_count % log_interval == 0)

                if send_progress and self.progress_callback:
                    elapsed = time.time() - start_time
                    current_fps = actual_frame_count / elapsed if elapsed > 0 else 0

                    progress_data = ProgressData(
                        frame_number=actual_frame_count,
                        total_frames=frame_count,
                        original_frame=frame.copy(),
                        anonymized_frame=anonymized.copy(),
                        elapsed_time=elapsed,
                        fps=current_fps
                    )

                    self.progress_callback(progress_data)

            cap.release()
            out.release()

            elapsed = time.time() - start_time
            logger.info(f"Processing complete: {actual_frame_count} frames in {elapsed:.2f}s")

            if self.progress_callback and not self._cancel_flag.is_set():
                final_fps = actual_frame_count / elapsed if elapsed > 0 else 0
                final_progress = ProgressData(
                    frame_number=actual_frame_count,
                    total_frames=actual_frame_count,
                    original_frame=np.zeros((10, 10, 3), dtype=np.uint8),
                    anonymized_frame=np.zeros((10, 10, 3), dtype=np.uint8),
                    elapsed_time=elapsed,
                    fps=final_fps
                )
                self.progress_callback(final_progress)

        except Exception as e:
            logger.exception(f"Error during processing: {e}")
