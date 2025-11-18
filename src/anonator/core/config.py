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

    # Face detector model: "RetinaNetResNet50", "RetinaNetMobileNetV1", "DSFDDetector"
    detector_model: str = "RetinaNetResNet50"

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


# Global configuration instances
HIPAA_MODE = HIPAAConfig()
STANDARD_MODE = StandardConfig()
PROCESSOR_CONFIG = ProcessorConfig()


def get_mode_config(hipaa_mode: bool):
    """Get configuration based on mode selection.

    Args:
        hipaa_mode: True for HIPAA compliance mode, False for standard mode

    Returns:
        HIPAAConfig or StandardConfig instance
    """
    return HIPAA_MODE if hipaa_mode else STANDARD_MODE
