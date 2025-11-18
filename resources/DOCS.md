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
- RetinaFace ResNet50 (91.8% accuracy on hard cases)

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
- RetinaFace ResNet50 for face detection
- PyTorch CUDA GPU acceleration
- Multi-pass detection with NMS
- Frame-by-frame streaming
- Audio preservation option
- Cancellation support
- Progress callbacks

### Face Detection
- RetinaFace ResNet50 (91.8% accuracy on WiderFace hard cases)
- PyTorch CUDA FP16 inference
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

## Performance Metrics

### GPU Processing (RTX 3060 with FP16)
- Standard mode: 30-35 fps
- HIPAA mode: 15-18 fps
- 10-12x faster than CPU

### CPU Processing
- Standard mode: 3-4 fps
- HIPAA mode: 1-2 fps

### HIPAA Mode Trade-offs
- 2x slower than standard mode
- Multi-pass detection ensures no missed faces
- Required for medical compliance

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
- RetinaFace ResNet50 (91.8% accuracy)
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
- Adjust threshold sensitivity
- Choose anonymization mode
- Set mask scale coverage
- Toggle multi-pass detection
- Enable audio preservation

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
