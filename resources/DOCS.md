# Anonator Documentation

## What is Anonator

Anonator is a Windows desktop application for video face anonymization with GPU acceleration and medical-grade HIPAA compliance.

## Key Features

### GPU Accelerated Processing
- PyTorch CUDA support for NVIDIA GPUs
- FP16 inference for 2x faster processing
- RTX 3060 tested: 15-18 fps (HIPAA), 30-35 fps (standard)
- CPU fallback available
- 10-12x speedup with GPU acceleration

### HIPAA Compliant Mode
- Multi-pass detection (3 passes per frame)
- Detection threshold 0.5
- Multi-pass thresholds: 0.5, 0.75, 0.35
- Solid blackout anonymization
- Audio removal for voice PHI protection
- Mask scale 1.5 for complete coverage
- RetinaFace MobileNetV1 (default, fastest) or selectable models

### Three Anonymization Modes
- Blur: Gaussian blur effect
- Solid: Complete black box coverage
- Mosaic: Pixelation effect

### User Interface
- Black background with white text
- Drag and drop video support
- Real-time progress tracking
- Side-by-side frame preview
- Elapsed time and speed display

## System Requirements

### Minimum Requirements
- Windows 10 or Windows 11 64-bit
- 4GB RAM
- 2GB free disk space
- Python 3.9 or higher

### Recommended Configuration
- Windows 10 or Windows 11 64-bit
- 8GB RAM
- NVIDIA GPU with CUDA 12.6 support
- RTX 2060, RTX 3060, RTX 4060 or higher
- 2GB free disk space

## Installation

### From Source
```bash
uv sync
```

### Run Application
```bash
uv run python -m anonator.main
```

### Run Tests
```bash
uv run pytest
```

## Technical Architecture

### Video Processor
- Selectable face detection models (MobileNetV1, ResNet50, DSFD)
- PyTorch CUDA GPU acceleration
- Multi-pass detection with NMS
- Frame-by-frame streaming
- Audio preservation option
- Cancellation support
- Progress callbacks

### Face Detection Models
- **RetinaNetMobileNetV1** (default): Fastest, optimized for speed (~20 FPS on GTX 1060)
- **RetinaNetResNet50**: Balanced accuracy and speed (~13 FPS, 91.8% WiderFace hard)
- **DSFDDetector**: Highest accuracy, slower processing (~3 FPS)
- PyTorch CUDA FP16 inference support
- Configurable threshold sensitivity (0.5-0.9)
- Non-Maximum Suppression (IoU threshold 0.3)
- Max resolution downscaling for performance
- Bounding box clipping

### Multi-Pass Detection (HIPAA Mode)
- Pass 1: threshold × 1.0
- Pass 2: threshold × 1.5
- Pass 3: threshold × 0.7
- Default base threshold: 0.5
- Actual thresholds: 0.5, 0.75, 0.35
- Combines all detections
- Removes duplicates with NMS

### Video Processing
- OpenCV for video input and output
- Frame streaming for memory efficiency
- H.264 encoding for output
- FFmpeg backend

## Performance Modes and Metrics

### Performance Mode Overview

Anonator offers four performance modes with different speed/quality tradeoffs:

| Mode | Batch Size | Frame Skip | Max Res | GPU FPS | CPU FPS | Use Case |
|------|-----------|------------|---------|---------|---------|----------|
| **Original** | 1 | 1 (none) | 1920 | 30 | 3 | Maximum quality, no optimization |
| **Quality** | 4-8 | 1 (none) | 1920 | 90-120 | 9-12 | Better performance, same quality |
| **Balanced** | 8 | 3 | 1280 | 150-240 | 15-24 | Good for static scenes, medical videos |
| **Maximum Speed** | 16 | 5 | 960 | 300-450 | 30-45 | Fast motion OK, scene detection enabled |

### Optimization Techniques

**1. Batch Processing (3-4x speedup)**
- Processes multiple frames simultaneously on GPU
- Better GPU utilization (30% → 80-90%)
- Automatically falls back to sequential on CPU if batching unavailable
- Adaptive batch sizes: GPU (4-16 frames), CPU (2-4 frames)

**2. Frame Skipping with Tracking (2-3x additional speedup)**
- Detects faces every N frames, reuses bounding boxes for intermediate frames
- Scene change detection forces re-detection after cuts
- Histogram correlation threshold: 0.3 (configurable)
- Works well for: talking heads, medical procedures, static camera angles

**3. Adaptive Resolution (1.5x additional speedup)**
- Lower detection resolution for faster modes
- Face detection quality maintained (faces typically 64x64 or larger)
- Anonymization always at original resolution

### GPU vs CPU Performance

**GPU (RTX 3060, FP16):**
- Original: 30 FPS
- Quality: 90-120 FPS (3-4x)
- Balanced: 150-240 FPS (5-8x)
- Maximum: 300-450 FPS (10-15x)

**CPU (Modern 8-core):**
- Original: 3 FPS
- Quality: 9-12 FPS (3-4x)
- Balanced: 15-24 FPS (5-8x)
- Maximum: 30-45 FPS (10-15x)

### HIPAA Mode Performance

HIPAA mode can be combined with performance modes:
- Multi-pass detection (3x per frame) reduces throughput by ~40%
- Quality mode + HIPAA: ~60-80 FPS (GPU)
- Balanced mode + HIPAA: ~100-150 FPS (GPU)
- Maximum Speed mode not recommended for HIPAA (frame skipping may miss faces)

## HIPAA Compliance

### Default Settings
- Detection threshold: 0.5
- Multi-pass thresholds: 0.5, 0.75, 0.35
- Mask scale: 1.5
- Mode: solid (irreversible blackout)
- Multi-pass: enabled
- Keep audio: disabled
- UI controls: locked

### Compliance Features
- Selectable detection models with MobileNetV1 default
- Multi-pass detection (3 passes per frame)
- Audio removal for voice PHI
- Irreversible solid anonymization
- 1.5x mask scale for complete coverage
- Locked settings prevent accidental changes

### Configuration
All HIPAA parameters configurable in `src/anonator/core/config.py`:
- Detection thresholds
- Multi-pass multipliers
- Mask scale
- Anonymization mode
- Audio handling

## Supported Video Formats

### Input Formats
- MP4
- AVI
- MOV
- MKV
- WEBM

### Output Format
- MP4 with H.264 encoding

## Usage Guide

### Basic Workflow
1. Launch application
2. Drag and drop video file
3. Configure settings or use HIPAA mode
4. Click start processing
5. Monitor progress with preview
6. Output saved with anonymized suffix

### HIPAA Mode
- Enabled by default
- All settings locked
- Ultra-sensitive detection
- Maximum coverage area
- Audio removal enforced
- Solid blackout mode

### Manual Settings
- Uncheck HIPAA mode to customize
- **Select performance mode** based on your needs:
  - Original: Maximum quality, slow
  - Quality: 3-4x faster, no quality loss
  - Balanced: 5-8x faster, best for medical/interview videos
  - Maximum Speed: 10-15x faster, scene detection prevents missed faces
- Select detection model (MobileNetV1, ResNet50, or DSFD)
- Adjust threshold sensitivity
- Choose anonymization mode
- Set mask scale coverage
- Toggle multi-pass detection
- Enable audio preservation

### Choosing Performance Mode

**For Medical/HIPAA Videos:**
- Use Quality or Balanced mode
- Enable multi-pass detection
- Talking heads and procedures have minimal motion

**For General Videos:**
- Quality mode: Interviews, webcam recordings
- Balanced mode: Presentations, lectures
- Maximum Speed: Action videos (scene detection handles cuts)

**For CPU Processing:**
- Quality mode gives best CPU speedup
- Balanced mode acceptable for short videos
- Maximum Speed may skip faces on very slow CPUs

## Troubleshooting

### GPU Not Detected
- Update NVIDIA drivers to latest version
- Install CUDA Toolkit 12.6
- Verify PyTorch CUDA installation
- Application runs on CPU if GPU unavailable

### Slow Processing
- Enable FP16 inference in config (GPU only)
- Set max_resolution=1280 for faster processing
- Use RetinaNetMobileNetV1 model for speed
- Disable multi-pass if not needed

### Application Won't Start
- Requires Python 3.9+
- Run: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
- Run: uv pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
- Run: uv sync

### Out of Memory
- Reduce max_resolution in config
- Enable FP16 inference
- Process shorter video segments
- Close other GPU applications

## Configuration

Edit `src/anonator/core/config.py` to customize:
- HIPAA mode parameters
- Detection thresholds
- Multi-pass settings
- Performance options (FP16, max_resolution)
- Blur/mosaic effects

See `CONFIG_EXAMPLE.md` for detailed configuration guide.

## Dependencies

### Core Libraries
- PyTorch 2.9+ with CUDA 12.6
- RetinaFace (hukkelas/DSFD-Pytorch-Inference)
- opencv-python-headless 4.8+
- Pillow 10.0+
- tkinterdnd2 0.3+
- imageio 2.25+
- imageio-ffmpeg 0.4+
- numpy <2.0

### Build Tools
- PyInstaller 6.16+
- uv package manager

### Development Tools
- pytest 8.3+

## Project Structure

```
anonator/
├── src/anonator/
│   ├── core/
│   │   ├── processor.py       # Video processing engine
│   │   ├── anonymizer.py      # Face anonymization
│   │   └── config.py          # HIPAA and processor configuration
│   ├── ui/
│   │   ├── main_window.py     # Main GUI window
│   │   └── frame_viewer.py    # Frame preview widget
│   └── main.py                # Application entry point
├── tests/
│   ├── test_processor.py      # Processor tests
│   ├── test_hipaa_mode.py     # HIPAA compliance tests
│   └── test_gpu.py            # GPU detection tests
├── pyinstaller_hooks/         # PyInstaller runtime hooks
├── build.py                   # Build script
├── anonator.spec              # PyInstaller spec
├── pyproject.toml             # Dependencies
├── CONFIG_EXAMPLE.md          # Configuration guide
└── README.md                  # Overview
```

## License

MIT License
