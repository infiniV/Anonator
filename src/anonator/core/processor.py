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
from anonator.core.config import PROCESSOR_CONFIG, HIPAA_MODE, PERFORMANCE_CONFIG
from anonator.core.scene_detector import SceneDetector
from anonator.core.mediapipe_detector import MediaPipeDetector


class ProgressData:
    def __init__(
        self,
        frame_number: int,
        total_frames: int,
        original_frame: np.ndarray,
        anonymized_frame: np.ndarray,
        elapsed_time: float,
        fps: float,
        video_fps: float = 0
    ):
        self.frame_number = frame_number
        self.total_frames = total_frames
        self.original_frame = original_frame
        self.anonymized_frame = anonymized_frame
        self.elapsed_time = elapsed_time
        self.fps = fps
        self.video_fps = video_fps


class VideoProcessor:
    def __init__(self, progress_callback: Optional[Callable[[ProgressData], None]] = None):
        self.progress_callback = progress_callback
        self._cancel_flag = threading.Event()
        self._processing_thread = None

        # Initialize detector based on model selection
        if PROCESSOR_CONFIG.detector_model == "MediaPipe":
            self.detector = MediaPipeDetector(
                confidence_threshold=PROCESSOR_CONFIG.detector_confidence,
                nms_iou_threshold=PROCESSOR_CONFIG.nms_iou_threshold,
                device=device
            )
            logger.info(f"Face detector initialized: MediaPipe on CPU")
        else:
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

    def _calculate_optimal_batch_size(self, frame_width: int, frame_height: int, configured_batch_size: int) -> int:
        """Calculate optimal batch size based on available GPU memory.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            configured_batch_size: User-configured batch size from performance mode

        Returns:
            Optimal batch size (capped by available memory and configured size)
        """
        if device == 'cpu':
            # CPU: use configured batch size (typically smaller)
            return configured_batch_size

        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available_memory = total_memory - allocated_memory

            # Estimate memory per frame (empirical formula)
            # Base: frame size * 4 bytes (RGBA) * ~3x overhead for detection model
            bytes_per_pixel = 4 if PROCESSOR_CONFIG.fp16_inference else 8
            est_memory_per_frame = frame_width * frame_height * bytes_per_pixel * 3

            # Calculate max batch size that fits in 70% of available memory (safety margin)
            max_batch_size = int((available_memory * 0.7) / est_memory_per_frame)
            max_batch_size = max(1, max_batch_size)  # At least 1

            # Use minimum of configured and memory-constrained batch size
            optimal_batch_size = min(configured_batch_size, max_batch_size)

            if optimal_batch_size < configured_batch_size:
                logger.warning(
                    f"Reducing batch size from {configured_batch_size} to {optimal_batch_size} "
                    f"due to GPU memory constraints ({available_memory / 1e9:.2f}GB available)"
                )

            return optimal_batch_size

        except Exception as e:
            logger.warning(f"Could not calculate optimal batch size: {e}. Using configured size.")
            return configured_batch_size

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
        """Detect faces in a single frame.

        Uses batched detection with batch size 1 for consistent device handling.
        """
        self.detector.confidence_threshold = threshold

        try:
            # Use batched_detect for consistent device handling
            batch = np.expand_dims(frame, axis=0)  # [1, H, W, C]
            batch_dets = self.detector.batched_detect(batch)

            if batch_dets is not None and len(batch_dets) > 0:
                dets = batch_dets[0]  # Get first (and only) result
                if dets is not None and len(dets) > 0:
                    return dets.astype(np.float32)
            return None

        except (RuntimeError, AttributeError, TypeError) as e:
            # On any error, return None instead of crashing
            logger.warning(f"Face detection failed: {e}")
            return None

    def _detect_faces_batch(self, frames: list, threshold: float) -> list:
        """Detect faces in a batch of frames.

        Args:
            frames: List of frames to process
            threshold: Detection confidence threshold

        Returns:
            List of detections (one per frame), each can be None or ndarray
        """
        if not frames:
            return []

        self.detector.confidence_threshold = threshold

        # Try batched detection first (GPU optimized)
        try:
            # Stack frames into batch: [B, H, W, C]
            batch = np.stack(frames, axis=0)
            batch_dets = self.detector.batched_detect(batch)

            # Convert to list of individual detections
            results = []
            for dets in batch_dets:
                if dets is not None and len(dets) > 0:
                    results.append(dets.astype(np.float32))
                else:
                    results.append(None)
            return results

        except (AttributeError, NotImplementedError, RuntimeError) as e:
            # Fallback: batched_detect not available or failed (CPU, old library, or device errors)
            # Process frames individually using _detect_faces which has its own device handling
            logger.warning(f"Batched detection failed, using sequential fallback: {e}")
            results = []
            for frame in frames:
                dets = self._detect_faces(frame, threshold)
                results.append(dets)
            return results

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
            logger.info(f"Using {PROCESSOR_CONFIG.detector_model} detector on {device}")

            # Performance configuration
            batch_size = PERFORMANCE_CONFIG.batch_size
            frame_skip = PERFORMANCE_CONFIG.frame_skip_interval
            use_batching = batch_size > 1
            use_skipping = frame_skip > 1

            if use_batching:
                logger.info(f"Batch processing enabled: batch_size={batch_size}")
            if use_skipping:
                logger.info(f"Frame skipping enabled: detect every {frame_skip} frames")
                scene_detector = SceneDetector(threshold=PERFORMANCE_CONFIG.scene_change_threshold)
            else:
                scene_detector = None

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video info: {frame_count} frames, {fps} fps, {width}x{height}")

            if multi_pass:
                logger.info("Multi-pass detection enabled (3 passes per frame)")

            # Calculate optimal batch size based on GPU memory and video resolution
            optimal_batch_size = self._calculate_optimal_batch_size(width, height, batch_size)
            if optimal_batch_size != batch_size:
                logger.info(f"Adjusted batch size: {batch_size} -> {optimal_batch_size}")
                batch_size = optimal_batch_size

            sample_interval = max(1, int(fps * preview_interval))
            log_interval = PROCESSOR_CONFIG.log_interval

            fourcc = cv2.VideoWriter_fourcc(*PROCESSOR_CONFIG.output_codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            start_time = time.time()
            actual_frame_count = 0
            cached_detections = None  # For frame skipping

            logger.info("Starting video processing...")

            # Batch processing loop
            if use_batching:
                frame_buffer = []
                frame_indices = []

                while True:
                    # Fill batch
                    for _ in range(batch_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if self._cancel_flag.is_set():
                            logger.info("Processing cancelled by user")
                            break

                        frame = frame.astype(np.uint8)
                        frame_buffer.append(frame)
                        frame_indices.append(actual_frame_count)
                        actual_frame_count += 1

                    if not frame_buffer or self._cancel_flag.is_set():
                        break

                    # Pre-compute scene changes for the entire batch (optimization)
                    scene_changes = []
                    if use_skipping and scene_detector:
                        for frame in frame_buffer:
                            scene_changes.append(scene_detector.is_scene_change(frame))

                    # Determine which frames need detection
                    frames_to_detect = []
                    detect_indices = []

                    for idx, (frame, frame_idx) in enumerate(zip(frame_buffer, frame_indices)):
                        needs_detection = (frame_idx % frame_skip == 0)

                        if use_skipping and scene_detector and scene_changes[idx]:
                            needs_detection = True
                            if frame_idx % log_interval == 0:
                                logger.info(f"[Frame {frame_idx}] Scene change detected, forcing detection")

                        if needs_detection:
                            frames_to_detect.append(frame)
                            detect_indices.append(idx)

                    # Batch detection
                    if frames_to_detect:
                        if multi_pass:
                            # Multi-pass batch detection
                            all_batch_dets = []
                            thresholds = [threshold * m for m in HIPAA_MODE.multi_pass_multipliers]

                            for pass_idx, t in enumerate(thresholds):
                                batch_dets = self._detect_faces_batch(frames_to_detect, threshold=t)
                                all_batch_dets.append(batch_dets)

                            # Merge multi-pass results for each frame
                            merged_dets = []
                            for frame_dets_across_passes in zip(*all_batch_dets):
                                valid_dets = [d for d in frame_dets_across_passes if d is not None]
                                if valid_dets:
                                    combined = np.vstack(valid_dets)
                                    combined = self._remove_duplicate_detections(combined)
                                    merged_dets.append(combined.astype(np.float32))
                                else:
                                    merged_dets.append(None)

                            batch_detections = merged_dets
                        else:
                            # Single-pass batch detection
                            batch_detections = self._detect_faces_batch(frames_to_detect, threshold=threshold)

                        # Map batch detections back to frame buffer
                        detection_map = {}
                        for det_idx, buffer_idx in enumerate(detect_indices):
                            detection_map[buffer_idx] = batch_detections[det_idx]
                            # Only update cache if we have valid detections
                            if batch_detections[det_idx] is not None:
                                cached_detections = batch_detections[det_idx]

                    # Process all frames in batch with their detections
                    for idx, (frame, frame_idx) in enumerate(zip(frame_buffer, frame_indices)):
                        # Use frame-specific detection if detected, otherwise use cached
                        dets = detection_map.get(idx, cached_detections)

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

                        out.write(anonymized)

                        # Progress reporting
                        send_progress = (frame_idx == 1 or
                                        frame_idx % sample_interval == 0 or
                                        frame_idx % log_interval == 0)

                        if send_progress and self.progress_callback:
                            elapsed = time.time() - start_time
                            current_fps = frame_idx / elapsed if elapsed > 0 else 0

                            progress_data = ProgressData(
                                frame_number=frame_idx,
                                total_frames=frame_count,
                                original_frame=frame.copy(),
                                anonymized_frame=anonymized.copy(),
                                elapsed_time=elapsed,
                                fps=current_fps,
                                video_fps=fps
                            )
                            self.progress_callback(progress_data)

                    frame_buffer.clear()
                    frame_indices.clear()

            else:
                # Original sequential processing (no batching)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if self._cancel_flag.is_set():
                        logger.info("Processing cancelled by user")
                        break

                    frame = frame.astype(np.uint8)

                    # Frame skipping logic
                    needs_detection = (actual_frame_count % frame_skip == 0)
                    if use_skipping and scene_detector:
                        if scene_detector.is_scene_change(frame):
                            needs_detection = True

                    if needs_detection:
                        if multi_pass:
                            all_dets = []
                            thresholds = [threshold * m for m in HIPAA_MODE.multi_pass_multipliers]
                            for t in thresholds:
                                dets = self._detect_faces(frame, threshold=t)
                                if dets is not None:
                                    all_dets.append(dets)

                            if all_dets:
                                dets = np.vstack(all_dets)
                                dets = self._remove_duplicate_detections(dets)
                                cached_detections = dets.astype(np.float32)
                            # else: keep previous cached_detections
                        else:
                            dets = self._detect_faces(frame, threshold=threshold)
                            # Only update cache if we found faces
                            if dets is not None:
                                cached_detections = dets
                    else:
                        # Use cached detections
                        pass

                    anonymized = anonymize_frame(
                        frame,
                        cached_detections,
                        mask_scale=float(mask_scale),
                        replacewith=anonymization_mode,
                        ellipse=PROCESSOR_CONFIG.use_ellipse_mask,
                        draw_scores=False,
                        replaceimg=None,
                        mosaicsize=PROCESSOR_CONFIG.mosaic_size
                    )

                    out.write(anonymized)
                    actual_frame_count += 1

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
                            fps=current_fps,
                            video_fps=fps
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
                    fps=final_fps,
                    video_fps=fps
                )
                self.progress_callback(final_progress)

        except Exception as e:
            logger.exception(f"Error during processing: {e}")
