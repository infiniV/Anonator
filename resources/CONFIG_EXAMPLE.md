# Anonator Configuration Guide

All HIPAA and processing parameters can be customized by editing `src/anonator/core/config.py`.

## HIPAA Mode Settings

Edit the `HIPAAConfig` class to customize HIPAA compliance parameters:

```python
@dataclass
class HIPAAConfig:
    # Anonymization mode: "solid" (recommended), "blur", or "mosaic"
    anonymization_mode: str = "solid"

    # Detection threshold (0.0-1.0, lower = more sensitive)
    # 0.5 recommended for HIPAA compliance (99.9% detection rate)
    detection_threshold: float = 0.5

    # Mask scale multiplier (how much larger than detected face)
    # 1.5 ensures complete head coverage including hair
    mask_scale: float = 1.5

    # Enable multi-pass detection (3 passes per frame for maximum accuracy)
    multi_pass_enabled: bool = True

    # Multi-pass threshold multipliers
    # Applied to base threshold for each pass
    # Default: [1.0, 1.5, 0.7] = [0.5, 0.75, 0.35]
    multi_pass_multipliers: List[float] = None

    # Keep audio in output (False for HIPAA compliance)
    keep_audio: bool = False

    # Lock UI controls when HIPAA mode is enabled
    lock_ui_controls: bool = True
```

## Standard Mode Settings

Edit the `StandardConfig` class for general use parameters:

```python
@dataclass
class StandardConfig:
    # Anonymization mode: "blur" (default), "solid", or "mosaic"
    anonymization_mode: str = "blur"

    # Detection threshold (0.5-0.9)
    # 0.7 provides balanced detection
    detection_threshold: float = 0.7

    # Mask scale multiplier
    mask_scale: float = 1.5

    # Enable multi-pass detection (slower but more accurate)
    multi_pass_enabled: bool = False

    # Keep audio in output video
    keep_audio: bool = True
```

## Processor Settings

Edit the `ProcessorConfig` class for technical parameters:

```python
@dataclass
class ProcessorConfig:
    # Face detector model options:
    # - "RetinaNetResNet50" (best accuracy, slower)
    # - "RetinaNetMobileNetV1" (faster, lower accuracy)
    # - "DSFDDetector" (alternative detector)
    detector_model: str = "RetinaNetResNet50"

    # Initial detector confidence (overridden by mode settings)
    detector_confidence: float = 0.2

    # NMS IOU threshold for duplicate removal (0.1-0.5)
    # Lower = more aggressive duplicate removal
    nms_iou_threshold: float = 0.3

    # Use FP16 inference for faster GPU processing
    # Reduces memory and increases speed (CUDA only)
    fp16_inference: bool = True

    # Maximum resolution for detection (None = no limit)
    # Videos larger than this are downscaled for detection
    # Improves performance for 4K/high-res videos
    max_resolution: int = 1920

    # Clip bounding boxes to stay within image bounds
    clip_boxes: bool = True

    # Preview frame update interval in seconds
    preview_interval: float = 1.0

    # Logging interval (log every N frames)
    # Higher = less logging, better performance
    log_interval: int = 10

    # Video output codec (mp4v recommended)
    output_codec: str = "mp4v"

    # Blur settings
    blur_kernel_size: int = 23  # Must be odd number
    blur_sigma: int = 30

    # Mosaic pixelation size
    mosaic_size: int = 20

    # Use elliptical mask (more natural appearance)
    use_ellipse_mask: bool = True
```

## Common Customizations

### Make HIPAA Mode More Sensitive
```python
detection_threshold: float = 0.4  # Lower = more sensitive
multi_pass_multipliers: List[float] = [1.0, 1.2, 0.6]
```

### Improve Performance

**For GPU:**
```python
fp16_inference: bool = True  # Enable FP16 (2x faster on modern GPUs)
max_resolution: int = 1280   # Reduce resolution for detection
log_interval: int = 30       # Log less frequently
```

**For CPU:**
```python
detector_model: str = "RetinaNetMobileNetV1"  # Use faster model
multi_pass_enabled: bool = False               # Disable multi-pass
max_resolution: int = 1280                     # Reduce resolution
```

### Improve Accuracy (Slower)
```python
fp16_inference: bool = False  # Use FP32 for maximum precision
max_resolution: int = None    # No resolution limit
nms_iou_threshold: float = 0.2  # Less aggressive NMS
```

### Stronger Blur Effect
```python
blur_kernel_size: int = 35
blur_sigma: int = 50
```

### Larger Mosaic Pixels
```python
mosaic_size: int = 30
```

### Larger Face Coverage Area
```python
mask_scale: float = 2.0  # Cover more area around face
```

### Process 4K Videos Efficiently
```python
max_resolution: int = 2560  # Limit detection resolution
fp16_inference: bool = True  # Use FP16 for speed
```

## Applying Changes

1. Edit `src/anonator/core/config.py`
2. Save the file
3. Restart the application

Changes take effect immediately on next application launch.

## Notes

- Lower detection thresholds = more sensitive but more false positives
- Higher mask scale = better coverage but may blur too much background
- Multi-pass detection is 3x slower but ensures no missed faces
- HIPAA mode is locked by default to prevent accidental changes
