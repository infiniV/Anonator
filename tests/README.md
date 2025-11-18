# Anonator Test Suite

Comprehensive Python tests for the Anonator face detection and anonymization system.

## Overview

This test suite provides extensive coverage of custom code, model validation, performance benchmarking, and edge case testing. The tests are organized into three categories:

- **Unit Tests** (tests/unit/): Fast, isolated tests with mocked dependencies
- **Integration Tests** (tests/integration/): Real model validation and end-to-end pipeline tests
- **Benchmark Tests** (tests/benchmarks/): Performance and latency measurements

## Test Statistics

- **Total Tests**: 70+ comprehensive tests
- **Unit Tests**: 43 tests (anonymizer, config, processor)
- **Integration Tests**: 26 tests (detection, video processing)
- **Benchmark Tests**: 18 tests (latency, throughput)

## Requirements

### Dependencies

Install test dependencies:

```bash
pip install -e ".[dev]"
```

Or using dependency groups:

```bash
pip install --dependency-groups dev
```

Required packages:
- pytest>=8.3.5
- pytest-cov>=4.1.0 (coverage reporting)
- pytest-mock>=3.12.0 (mocking)
- pytest-benchmark>=4.0.0 (performance)
- pytest-timeout>=2.2.0 (timeouts)
- pytest-xdist>=3.5.0 (parallel execution)
- hypothesis>=6.92.0 (property-based testing)

### Hardware

- **CPU**: All tests run on CPU
- **GPU (optional)**: CUDA-capable GPU for GPU-specific tests (marked with `@pytest.mark.gpu`)

## Setup

### 1. Prepare Test Data

Extract sample frames and create synthetic test images:

```bash
python tests/prepare_test_data.py
```

This creates:
- `tests/data/sample_frames/`: Frames extracted from testData videos
- `tests/data/synthetic/`: Generated test images
- `tests/data/ground_truth.json`: Face annotations

### 2. Verify Installation

```bash
pytest --version
pytest --markers
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run by Category

**Unit tests only (fast, ~10 seconds):**
```bash
pytest tests/unit/ -v
```

**Integration tests (slower, ~60 seconds):**
```bash
pytest tests/integration/ -v
```

**Benchmarks only:**
```bash
pytest tests/benchmarks/ --benchmark-only
```

### Run by Markers

**Skip GPU tests (CPU only):**
```bash
pytest -m "not gpu"
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

**Run only benchmark tests:**
```bash
pytest -m benchmark
```

### Coverage Reports

**Generate HTML coverage report:**
```bash
pytest --cov=src/anonator --cov-report=html
open htmlcov/index.html
```

**Terminal coverage report:**
```bash
pytest --cov=src/anonator --cov-report=term-missing
```

### Parallel Execution

Run tests in parallel using all CPU cores:

```bash
pytest -n auto
```

### Specific Test Files

```bash
pytest tests/unit/test_anonymizer.py -v
pytest tests/integration/test_detection_integration.py -v
pytest tests/benchmarks/test_model_latency.py --benchmark-only
```

### Specific Test Functions

```bash
pytest tests/unit/test_anonymizer.py::TestScaleBB::test_scale_bb_1_5x_scaling -v
```

## Test Organization

### Unit Tests (tests/unit/)

#### test_anonymizer.py (15 tests)
Tests for face anonymization functions:
- `scale_bb()`: Bounding box scaling with various mask scales
- `anonymize_frame()`: All 4 modes (blur, solid, mosaic, img)
- Edge cases: empty detections, out-of-bounds, small boxes
- Score drawing, ellipse masking

#### test_config.py (10 tests)
Tests for configuration management:
- HIPAAConfig, StandardConfig, ProcessorConfig dataclasses
- `get_mode_config()` mode switching
- Parameter validation, defaults
- HIPAA vs Standard comparison

#### test_processor.py (18 tests)
Tests for VideoProcessor with mocked detector:
- Initialization, progress callbacks
- `_detect_faces()` with mocked results
- `_remove_duplicate_detections()` custom NMS algorithm
- Cancellation, threading, error handling

### Integration Tests (tests/integration/)

#### test_detection_integration.py (14 tests)
Real face detector validation:
- Model loading (CPU/GPU)
- Single/multiple face detection
- Threshold sensitivity
- Multi-pass detection (HIPAA mode)
- FP16 vs FP32 accuracy
- Bounding box format validation
- Edge cases (boundaries, resolutions)

#### test_video_processing.py (12 tests)
End-to-end video processing:
- All anonymization modes (blur, solid, mosaic)
- Video properties preservation (resolution, FPS, frame count)
- Progress callbacks
- Cancellation handling
- Multi-pass mode
- Error handling (invalid paths)
- Real test video processing

### Benchmark Tests (tests/benchmarks/)

#### test_model_latency.py (10 tests)
Model inference latency:
- Single frame: CPU vs GPU
- FP16 vs FP32 comparison
- Resolution impact (480p to 4K)
- Multi-pass overhead
- NMS performance (10, 50, 100 boxes)
- Anonymization operations
- End-to-end frame pipeline
- Model loading time

#### test_processing_throughput.py (8 tests)
Video processing throughput:
- CPU vs GPU processing FPS
- Anonymization mode comparison
- I/O overhead analysis
- Multi-pass performance impact
- Real-world video performance
- Performance regression detection
- Scalability analysis

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.slow`: Tests that take longer (skip with `-m "not slow"`)
- `@pytest.mark.gpu`: Tests requiring CUDA GPU (auto-skipped if GPU unavailable)
- `@pytest.mark.benchmark`: Performance benchmark tests (run with `--benchmark-only`)

## Fixtures (conftest.py)

### Device Fixtures
- `device`: Auto-detect CUDA/CPU
- `has_gpu`: Boolean GPU availability
- `face_detector`: Real RetinaFace detector (session-scoped)

### Frame Fixtures
- `sample_frame`: Random 960x540 RGB frame
- `sample_frame_with_faces`: Frame with synthetic faces
- `sample_frame_no_faces`: Landscape frame
- `synthetic_image_generator`: Factory for generating test images

### Detection Fixtures
- `mock_detections_single`: Single face [N, 5] array
- `mock_detections_multiple`: 3 faces
- `mock_detections_empty`: No detections
- `mock_detections_overlapping`: For NMS testing
- `mock_detections_edge_cases`: Boundary cases

### Config Fixtures
- `processor_config_standard`: Standard mode config
- `processor_config_hipaa`: HIPAA mode config
- `standard_config`: Standard anonymization config
- `hipaa_config`: HIPAA anonymization config

### Video Fixtures
- `short_test_video`: Generated 10-frame test video
- `testdata_videos_dir`: Path to testData directory

### Directory Fixtures
- `temp_output_dir`: Temporary directory for test outputs
- `test_data_dir`: Path to tests/data/
- `sample_frames_dir`: Extracted frames
- `synthetic_dir`: Synthetic images

### Other Fixtures
- `replacement_image`: Image for img mode anonymization
- `ground_truth_data`: Face annotations JSON

## Expected Coverage

Target coverage for custom code:

- **anonymizer.py**: 95%+ (all functions, modes, edge cases)
- **config.py**: 90%+ (all dataclasses, mode switching)
- **processor.py**: 85%+ (main logic, NMS, detection)

Overall target: **80%+ code coverage**

## Benchmark Interpretation

### Latency Benchmarks

Typical single frame inference times:
- **CPU**: 100-500ms (depending on processor)
- **GPU (FP32)**: 20-50ms
- **GPU (FP16)**: 10-30ms (2x speedup over FP32)

Resolution impact:
- **480p**: Baseline
- **720p**: ~2x slower
- **1080p**: ~4x slower
- **4K**: ~8x slower (with downscaling to 1920)

### Throughput Benchmarks

Video processing FPS (frames per second):
- **CPU**: 2-5 FPS
- **GPU**: 10-30 FPS

Multi-pass overhead:
- **Single pass**: Baseline
- **3-pass (HIPAA)**: ~3x slower

Anonymization modes (relative speed):
1. **solid**: Fastest (simple rectangle fill)
2. **mosaic**: Fast (resize operations)
3. **blur**: Medium (Gaussian blur computation)
4. **img**: Medium (resize + copy)

## Common Issues

### GPU Tests Skipped

If GPU tests are skipped but you have a GPU:
1. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check PyTorch CUDA version matches your CUDA installation
3. GPU tests are auto-skipped on CPU-only systems

### Test Data Missing

If tests fail due to missing data:
```bash
python tests/prepare_test_data.py
```

### Slow Tests

Integration and benchmark tests can be slow. Skip them during development:
```bash
pytest tests/unit/ -m "not slow"
```

### Memory Issues

If tests fail with OOM errors:
1. Close other applications
2. Run tests sequentially (don't use `-n auto`)
3. Skip benchmark tests

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Prepare test data
        run: python tests/prepare_test_data.py
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/anonator
      - name: Run integration tests (CPU only)
        run: pytest tests/integration/ -v -m "not gpu"
```

## Contributing

When adding new tests:

1. Follow existing test structure and naming conventions
2. Use appropriate markers (`@pytest.mark.slow`, `@pytest.mark.gpu`)
3. Add docstrings explaining what is tested
4. Use fixtures from conftest.py
5. Group related tests in classes
6. Test both success and failure cases
7. Include edge cases

## Performance Regression

To detect performance regressions:

```bash
# Save baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline
```

Benchmark results are stored in `.benchmarks/` directory.

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
