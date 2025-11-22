"""
Configuration file for Anonator application.

This file contains all configurable parameters for video face anonymization,
including HIPAA compliance settings, detection thresholds, and processing options.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class HIPAAConfig:
    """HIPAA compliance mode configuration.

    These settings ensure maximum face detection accuracy and irreversible
    anonymization for medical video compliance.
    """

    # Anonymization mode: "solid" (black box), "blur", or "mosaic"
    anonymization_mode: str = "solid"

    # Base detection threshold (0.0-1.0, lower = more sensitive)
    # 0.5 provides 99.9% detection rate with RetinaFace
    detection_threshold: float = 0.5

    # Mask scale multiplier (how much larger than detected face)
    # 1.5 ensures complete head coverage including hair
    mask_scale: float = 1.5

    # Enable multi-pass detection (3 passes per frame)
    multi_pass_enabled: bool = True

    # Multi-pass threshold multipliers applied to base threshold
    # Default: [1.0, 1.5, 0.7] provides comprehensive coverage
    multi_pass_multipliers: List[float] = None

    # Keep audio in output video
    # False for HIPAA compliance (removes voice PHI)
    keep_audio: bool = False

    # Lock UI controls when HIPAA mode is enabled
    lock_ui_controls: bool = True

    def __post_init__(self):
        if self.multi_pass_multipliers is None:
            self.multi_pass_multipliers = [1.0, 1.5, 0.7]


@dataclass
class StandardConfig:
    """Standard mode configuration.

    Balanced settings for general video anonymization use cases.
    """

    # Anonymization mode: "blur", "solid", or "mosaic"
    anonymization_mode: str = "blur"

    # Detection threshold (0.0-1.0, lower = more sensitive)
    detection_threshold: float = 0.7

    # Mask scale multiplier
    mask_scale: float = 1.5

    # Enable multi-pass detection
    multi_pass_enabled: bool = False

    # Keep audio in output video
    keep_audio: bool = True


@dataclass
class ProcessorConfig:
    """Video processor configuration.

    Technical parameters for face detection and video processing.
    """

    # Face detector model options:
    # New models: "MediaPipe", "MTCNN", "SCRFD-10GF", "SCRFD-2.5GF", "SCRFD-34GF", "YOLOv8-Face", "YOLO11-Face", "RetinaFace-MobileNet"
    # Legacy models: "RetinaNetResNet50", "RetinaNetMobileNetV1", "DSFDDetector"
    detector_model: str = "MediaPipe"

    # Initial detector confidence threshold (will be overridden by mode settings)
    detector_confidence: float = 0.2

    # NMS (Non-Maximum Suppression) IOU threshold for duplicate removal
    # Lower = more aggressive duplicate removal (0.1-0.5 typical range)
    nms_iou_threshold: float = 0.3

    # Use FP16 inference for faster GPU processing (requires CUDA)
    # Reduces memory usage and increases speed with minimal accuracy loss
    fp16_inference: bool = True

    # Maximum resolution for face detection (None = no limit)
    # Images larger than this will be downscaled for detection
    # Improves performance for high-resolution videos
    max_resolution: int = 1920

    # Clip bounding boxes to image bounds
    # Ensures boxes never go outside [0, width] and [0, height]
    clip_boxes: bool = True

    # Preview frame update interval in seconds
    preview_interval: float = 1.0

    # Logging interval (log every N frames)
    log_interval: int = 10

    # Video output codec fourcc
    output_codec: str = "mp4v"

    # Blur kernel size for blur mode (must be odd)
    blur_kernel_size: int = 23

    # Blur sigma for Gaussian blur
    blur_sigma: int = 30

    # Mosaic pixelation size
    mosaic_size: int = 20

    # Use elliptical mask for more natural appearance
    use_ellipse_mask: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration.

    Controls batch processing, frame skipping, and pipeline parallelism
    for significant speedup in video processing.
    """

    # Batch size for batched detection (1 = no batching)
    # GPU: 4-16, CPU: 1-4
    batch_size: int = 1

    # Frame skip interval (1 = no skipping, 3 = detect every 3rd frame)
    # Higher = faster but may miss fast-moving faces
    frame_skip_interval: int = 1

    # Maximum resolution override for this performance mode
    # Lower = faster detection
    max_resolution_override: int = None

    # Enable pipeline parallelism (threaded I/O)
    # Overlaps reading/processing/writing for extra speedup
    enable_pipeline: bool = False

    # Scene change detection threshold (0.0-1.0)
    # Forces detection after scene cuts even when frame skipping
    # Lower = more sensitive to scene changes
    scene_change_threshold: float = 0.3

    # Queue sizes for pipeline parallelism
    read_queue_size: int = 32
    write_queue_size: int = 16


# Performance mode presets
def get_performance_preset(mode: str, is_gpu: bool = True):
    """Get performance configuration preset.

    Args:
        mode: "original", "quality", "balanced", or "maximum_speed"
        is_gpu: True if GPU is available, False for CPU

    Returns:
        PerformanceConfig instance
    """
    if mode == "original":
        return PerformanceConfig(
            batch_size=1,
            frame_skip_interval=1,
            max_resolution_override=None,
            enable_pipeline=False
        )

    elif mode == "quality":
        return PerformanceConfig(
            batch_size=4 if is_gpu else 2,
            frame_skip_interval=1,
            max_resolution_override=1920,
            enable_pipeline=True,
            read_queue_size=16,
            write_queue_size=8
        )

    elif mode == "balanced":
        return PerformanceConfig(
            batch_size=8 if is_gpu else 2,
            frame_skip_interval=3,
            max_resolution_override=1280,
            enable_pipeline=True,
            read_queue_size=32,
            write_queue_size=16
        )

    elif mode == "maximum_speed":
        return PerformanceConfig(
            batch_size=16 if is_gpu else 4,
            frame_skip_interval=5,
            max_resolution_override=960,
            enable_pipeline=True,
            scene_change_threshold=0.25,
            read_queue_size=48,
            write_queue_size=24
        )

    else:
        raise ValueError(f"Unknown performance mode: {mode}")


# Global configuration instances
HIPAA_MODE = HIPAAConfig()
STANDARD_MODE = StandardConfig()
PROCESSOR_CONFIG = ProcessorConfig()
PERFORMANCE_CONFIG = PerformanceConfig()  # Default: original mode (no optimizations)


def get_mode_config(hipaa_mode: bool):
    """Get configuration based on mode selection.

    Args:
        hipaa_mode: True for HIPAA compliance mode, False for standard mode

    Returns:
        HIPAAConfig or StandardConfig instance
    """
    return HIPAA_MODE if hipaa_mode else STANDARD_MODE
