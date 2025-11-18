# Anonator - Video Face Anonymization Software

Professional video face blur and anonymization tool for Windows with GPU acceleration and HIPAA compliance.

## What is Anonator

Anonator is a desktop application for automatic face detection and anonymization in video files. It uses GPU-accelerated face detection to blur faces, pixelate faces, or apply solid black boxes over faces in videos. Perfect for privacy protection, medical video anonymization, and HIPAA compliant video processing.

## Key Features

### Medical Grade HIPAA Compliance
- Multi-pass detection (3 passes: 0.5, 0.75, 0.35 thresholds)
- RetinaFace ResNet50 (91.8% accuracy on hard cases)
- Automatic audio removal to protect voice PHI
- Irreversible solid blackout anonymization
- 1.5x mask scale for complete coverage
- Configurable via config.py

### GPU Accelerated Face Detection
- NVIDIA CUDA GPU acceleration with PyTorch
- FP16 inference for 2x faster processing on modern GPUs
- 10-12x faster than CPU processing
- RTX 3060 tested: 15-18 fps (HIPAA mode), 30-35 fps (standard mode)
- CPU fallback mode available
- RetinaFace ResNet50 model (91.8% accuracy on WiderFace hard cases)

### Multiple Anonymization Methods
- Blur mode: Gaussian blur effect for face privacy
- Solid mode: Complete black box coverage
- Mosaic mode: Pixelation effect for faces
- Configurable mask scale for coverage area
- Adjustable detection sensitivity threshold

### Easy to Use Interface
- Simple drag and drop video loading
- Black and white minimalist user interface
- Real-time progress tracking with FPS counter
- Side-by-side original and anonymized frame preview
- Elapsed time and remaining time estimates
- One-click HIPAA mode for medical videos

### Broad Format Support
- Input: MP4, AVI, MOV, MKV, WEBM video formats
- Output: MP4 with H.264 encoding
- Audio preservation option
- Maintains original video quality and resolution

## Use Cases

- Healthcare video anonymization for patient privacy
- Medical procedure recording compliance
- Research video data anonymization
- Security camera footage privacy protection
- Social media video privacy
- Training video face removal
- Public space video anonymization
- Legal compliance for video sharing
- Educational video privacy protection
- Documentary face blurring

## System Requirements

### Minimum Requirements
- Windows 10 or Windows 11 64-bit operating system
- 4GB RAM memory
- 2GB free disk space
- Python 3.9 or higher (required for PyTorch 2.16+)

### Recommended for Best Performance
- Windows 10 or Windows 11 64-bit
- 8GB RAM or more
- NVIDIA GPU with CUDA support (RTX 2060, RTX 3060, RTX 4060, GTX 1660 or better)
- 2GB free disk space

## Quick Start Installation

Install dependencies using uv package manager:

```bash
# Install PyTorch with CUDA 12.6 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install face detection package
uv pip install --no-build-isolation git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git

# Install remaining dependencies
uv sync

# Install additional packages
uv pip install opencv-python Pillow imageio imageio-ffmpeg tkinterdnd2 pyinstaller "numpy<2"
```

Or use the Makefile:

```bash
make install-dev    # Install all dependencies including testing tools
make dev           # Complete dev setup (install + prepare test data)
```

## Running the Application

### Using Makefile (Recommended)
```bash
make run           # Start the UI application
```

### Using Python Directly
```bash
python -m anonator.main
# OR
.venv/Scripts/python.exe -m anonator.main
```

## How to Use

### Basic Video Anonymization
1. Launch the Anonator application
2. Drag and drop your video file into the window
3. Select anonymization mode: blur, solid, or mosaic
4. Adjust detection threshold for sensitivity
5. Click Start Processing button
6. Wait for processing to complete
7. Find output video with "_anonymized" suffix in same folder

### HIPAA Compliant Medical Video Processing
1. Launch Anonator (HIPAA mode enabled by default)
2. Drag and drop medical video file
3. Verify HIPAA Mode checkbox is checked
4. Settings automatically locked to medical compliance standards
5. Click Start Processing
6. Output video will have all faces blacked out and audio removed

## Video Examples

See side-by-side comparison of original videos and anonymized results:

### Example 1: 134436-759734820_small.mp4

| Original | Anonymized |
|----------|------------|
| [testData/134436-759734820_small.mp4](testData/134436-759734820_small.mp4) | [testData/134436-759734820_small_anonymized.mp4](testData/134436-759734820_small_anonymized.mp4) |
| Before face anonymization | After processing with Anonator |

### Example 2: 41315-429396382_small.mp4

| Original | Anonymized |
|----------|------------|
| [testData/41315-429396382_small.mp4](testData/41315-429396382_small.mp4) | [testData/41315-429396382_small_anonymized.mp4](testData/41315-429396382_small_anonymized.mp4) |
| Before face anonymization | After processing with Anonator |

**Demo videos showcase:**
- Automatic face detection in various lighting conditions
- Real-time GPU-accelerated processing
- Complete face coverage with anonymization
- Maintained video quality and resolution

## Performance Benchmarks

### GPU Processing (RTX 3060 with FP16)
- Standard mode: 30-35 fps
- HIPAA mode: 15-18 fps
- 10-12x faster than CPU

### CPU Processing
- Standard mode: 3-4 fps
- HIPAA mode: 1-2 fps

## Technical Specifications

### Face Detection Technology
- RetinaFace ResNet50 deep learning model
- PyTorch CUDA GPU with FP16 inference
- WiderFace benchmark: 91.8% accuracy on hard cases
- Multi-pass detection for HIPAA compliance
- Non-Maximum Suppression (IoU threshold 0.3)
- Configurable threshold 0.5-0.9
- Max resolution downscaling for performance
- Bounding box clipping

### Video Processing Engine
- OpenCV video input and output
- Frame-by-frame streaming processing
- Memory efficient for large video files
- FFmpeg backend for encoding
- H.264 video compression

### HIPAA Compliance Features
- Detection threshold 0.5 (RetinaFace optimized)
- Multi-pass thresholds: 0.5, 0.75, 0.35
- Mask scale 1.5 for complete coverage
- Solid blackout irreversible anonymization
- Automatic audio removal for voice PHI
- Locked UI controls prevent accidental changes
- All settings configurable in config.py

## Supported Video Formats

Input formats: MP4, AVI, MOV, MKV, WEBM
Output format: MP4 with H.264 encoding

## Project Structure

```
anonator/
├── src/anonator/
│   ├── core/
│   │   ├── processor.py       # Video processing engine
│   │   ├── anonymizer.py      # Face anonymization
│   │   └── config.py          # HIPAA and processor config
│   ├── ui/
│   │   ├── main_window.py     # Main application window
│   │   └── frame_viewer.py    # Frame preview widget
│   └── main.py                # Application entry point
├── tests/                     # Comprehensive test suite (130+ tests)
│   ├── unit/                  # Unit tests (fast, isolated)
│   ├── integration/           # Integration tests (real detector)
│   ├── benchmarks/            # Performance benchmarks
│   ├── conftest.py            # Shared test fixtures
│   └── README.md              # Testing documentation
├── Makefile                   # Build system
├── Makefile.md                # Makefile quick reference
├── pyproject.toml             # Python project configuration
└── README.md                  # This file
```

## Development and Testing

This project includes a comprehensive test suite with 130+ tests covering all custom code, model validation, and performance benchmarking.

### Using Makefile (Recommended)

```bash
# Setup development environment
make dev                # Install deps + prepare test data

# Run tests
make test              # All tests (unit + integration)
make test-unit         # Unit tests only (fast, ~10s)
make test-integration  # Integration tests (~40s)
make test-cov          # Tests with coverage report
make test-fast         # Tests in parallel

# Other commands
make run               # Run the application
make build             # Build executable
make clean             # Clean all artifacts
make help              # Show all available commands
```

See `Makefile.md` for complete command reference.

### Running Tests Directly

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/unit/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/benchmarks/ --benchmark-only  # Benchmarks

# With coverage
pytest tests/ --cov=src/anonator --cov-report=html
```

### Test Coverage

- **130 tests passing** (100% of runnable tests)
- **80%+ code coverage** for custom code
- Tests cover: anonymization, configuration, video processing, face detection, multi-pass detection, edge cases, and performance benchmarks

See `tests/README.md` for detailed testing documentation.

## Building Standalone Executable

### Using Makefile (Recommended)

```bash
make build             # Build directory-based executable
make build-onefile     # Build single .exe file
make build-debug       # Build with console window
```

### Using PyInstaller Directly

```bash
# Make sure all dependencies are installed first
pip install pyinstaller

# Build the executable
pyinstaller anonator.spec

# The executable will be in dist/Anonator/Anonator.exe
```

The spec file includes:
- PyTorch CUDA DLLs for GPU acceleration
- RetinaFace pre-trained model weights
- face_detection package with all configs
- Runtime hooks for proper model loading

## Configuration

Edit `src/anonator/core/config.py` to customize:
- HIPAA mode parameters (thresholds, mask scale, multi-pass)
- Standard mode defaults
- Performance settings (FP16, max_resolution, NMS)
- Blur and mosaic effects

See `CONFIG_EXAMPLE.md` for detailed configuration examples.

## Documentation

See `resources/DOCS.md` for complete technical documentation:
- Installation and setup
- HIPAA compliance details
- Performance optimization
- Troubleshooting
- Configuration guide

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install production dependencies |
| `make install-dev` | Install dev dependencies + testing tools |
| `make test` | Run all tests (unit + integration) |
| `make test-unit` | Run unit tests only (fast) |
| `make test-cov` | Run tests with coverage report |
| `make run` | Start the application UI |
| `make build` | Build executable with PyInstaller |
| `make clean` | Clean all artifacts |
| `make dev` | Setup complete dev environment |
| `make info` | Show project information |

See `Makefile.md` for complete documentation.

## Keywords

video face anonymization, face blur software, blur faces in video, anonymize video, face detection, privacy protection software, HIPAA compliant video, medical video anonymization, GPU accelerated face blur, Windows face anonymizer, automatic face detection, video privacy tool, face pixelation, face blackout, healthcare video compliance, patient privacy video, AI face detection, deep learning face blur, real-time face anonymization, batch video anonymization

## License

MIT License - Free and open source software

## Support

For issues, feature requests, or questions, please open an issue on GitHub.

## Citations

### Face Detection Model

This project uses RetinaFace for face detection:

**RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild**
```
Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020).
RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
```
- Paper: https://arxiv.org/abs/1905.00641
- Original Implementation: https://github.com/deepinsight/insightface/tree/master/detection/retinaface

**Implementation Used:**
- Repository: https://github.com/hukkelas/DSFD-Pytorch-Inference
- Author: Håkon Hukkelås
- License: Apache License 2.0

### Core Dependencies

**PyTorch**
- Repository: https://github.com/pytorch/pytorch
- License: BSD-3-Clause
- Citation: Paszke, A., Gross, S., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

**OpenCV**
- Repository: https://github.com/opencv/opencv
- License: Apache License 2.0
- Website: https://opencv.org

**TkinterDnD2**
- Repository: https://github.com/pmgagne/tkinterdnd2
- License: BSD-like license

### Benchmark Dataset

**WiderFace**
```
Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016).
WIDER FACE: A Face Detection Benchmark.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```
- Paper: https://arxiv.org/abs/1511.06523
- Dataset: http://shuoyang1213.me/WIDERFACE/
- Used for accuracy benchmarking (91.8% on hard subset)

## Acknowledgments

This project builds upon the excellent work of:
- **Håkon Hukkelås** for the DSFD-Pytorch-Inference library providing easy-to-use RetinaFace implementation
- **Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou** for the RetinaFace face detection algorithm
- **Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang** for the WiderFace benchmark dataset
- **PyTorch team** for the deep learning framework and CUDA support
- **OpenCV contributors** for computer vision and video processing tools
