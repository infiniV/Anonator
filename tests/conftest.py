import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch

from anonator.core.config import (
    HIPAAConfig,
    ProcessorConfig,
    StandardConfig,
    get_mode_config,
)


@pytest.fixture(scope="session")
def device():
    """Auto-detect CUDA/CPU device and skip GPU tests if unavailable."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def has_gpu(device):
    """Check if GPU is available."""
    return device == "cuda"


@pytest.fixture
def sample_frame():
    """Generate a single test frame (960x540 RGB)."""
    frame = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frame_with_faces():
    """Generate a test frame with synthetic face regions."""
    frame = np.random.randint(100, 150, (540, 960, 3), dtype=np.uint8)

    # Draw synthetic "faces" as colored rectangles
    cv2.rectangle(frame, (100, 150), (200, 250), (255, 200, 150), -1)
    cv2.rectangle(frame, (300, 100), (380, 180), (255, 190, 140), -1)
    cv2.rectangle(frame, (500, 200), (600, 320), (250, 195, 145), -1)

    return frame


@pytest.fixture
def sample_frame_no_faces():
    """Generate a frame without faces (landscape image)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw simple landscape elements
    cv2.rectangle(frame, (0, 300), (640, 480), (100, 200, 100), -1)  # Ground
    cv2.rectangle(frame, (0, 0), (640, 300), (150, 200, 255), -1)    # Sky

    return frame


@pytest.fixture
def mock_detections_single():
    """Single face detection with high confidence."""
    return np.array([
        [100, 150, 200, 250, 0.95]  # xmin, ymin, xmax, ymax, score
    ], dtype=np.float32)


@pytest.fixture
def mock_detections_multiple():
    """Multiple face detections."""
    return np.array([
        [100, 150, 200, 250, 0.95],
        [300, 100, 380, 180, 0.87],
        [500, 200, 600, 320, 0.92],
    ], dtype=np.float32)


@pytest.fixture
def mock_detections_empty():
    """Empty detection array (no faces found)."""
    return np.array([], dtype=np.float32).reshape(0, 5)


@pytest.fixture
def mock_detections_overlapping():
    """Overlapping detections for NMS testing."""
    return np.array([
        [100, 100, 200, 200, 0.95],  # High confidence
        [105, 105, 205, 205, 0.80],  # Overlaps with first (should be removed)
        [300, 300, 400, 400, 0.85],  # Different location
        [302, 298, 398, 402, 0.90],  # Overlaps with third (should keep this one)
    ], dtype=np.float32)


@pytest.fixture
def mock_detections_edge_cases():
    """Detections with edge cases (boundaries, small boxes)."""
    return np.array([
        [0, 0, 50, 50, 0.75],           # Top-left corner
        [910, 490, 960, 540, 0.82],     # Bottom-right corner
        [450, 250, 460, 260, 0.65],     # Very small box (10x10)
        [-10, -10, 50, 50, 0.70],       # Extends beyond boundary
    ], dtype=np.float32)


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_frames_dir(test_data_dir):
    """Path to extracted sample frames."""
    return test_data_dir / "sample_frames"


@pytest.fixture
def synthetic_dir(test_data_dir):
    """Path to synthetic test images."""
    return test_data_dir / "synthetic"


@pytest.fixture
def testdata_videos_dir():
    """Path to original test videos in testData."""
    return Path(__file__).parent.parent / "testData"


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def processor_config_standard():
    """Standard mode processor configuration."""
    return ProcessorConfig(
        fp16_inference=True,
        max_resolution=1920,
        log_interval=10,
        nms_iou_threshold=0.3,
    )


@pytest.fixture
def processor_config_hipaa():
    """HIPAA mode processor configuration."""
    return ProcessorConfig(
        fp16_inference=True,
        max_resolution=1920,
        log_interval=10,
        nms_iou_threshold=0.3,
    )


@pytest.fixture
def standard_config():
    """Standard anonymization config."""
    return get_mode_config(hipaa_mode=False)


@pytest.fixture
def hipaa_config():
    """HIPAA anonymization config."""
    return get_mode_config(hipaa_mode=True)


@pytest.fixture(scope="session")
def face_detector(device):
    """
    Real face detector (RetinaFace).
    Session-scoped to avoid reloading model multiple times.
    """
    try:
        import face_detection
        detector = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            nms_iou_threshold=0.3,
            device=device,
            max_resolution=1920,
            fp16_inference=(device == "cuda"),
            clip_boxes=True,
        )
        return detector
    except Exception as e:
        pytest.skip(f"Failed to load face detector: {e}")


@pytest.fixture
def short_test_video(temp_output_dir):
    """
    Generate a short test video (10 frames) for quick integration tests.
    """
    video_path = temp_output_dir / "test_video_10frames.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        25.0,  # FPS
        (640, 480),
    )

    # Write 10 frames with varying content
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add frame number as text
        cv2.putText(
            frame,
            f"Frame {i}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def synthetic_image_generator():
    """
    Factory fixture to generate synthetic test images.
    """
    def generate(width=640, height=480, num_faces=0, seed=None):
        """
        Generate synthetic image with optional face regions.

        Args:
            width: Image width
            height: Image height
            num_faces: Number of synthetic face regions to draw
            seed: Random seed for reproducibility

        Returns:
            Tuple of (image, face_bboxes)
        """
        if seed is not None:
            np.random.seed(seed)

        # Create base image
        image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

        face_bboxes = []
        for i in range(num_faces):
            # Random face position and size
            face_w = np.random.randint(60, 150)
            face_h = int(face_w * 1.2)  # Faces are typically taller

            x1 = np.random.randint(0, max(1, width - face_w))
            y1 = np.random.randint(0, max(1, height - face_h))
            x2 = x1 + face_w
            y2 = y1 + face_h

            # Draw face-like rectangle
            color = (255, int(200 + np.random.randint(-30, 30)), int(150 + np.random.randint(-30, 30)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

            # Add eyes
            eye_y = y1 + face_h // 3
            eye1_x = x1 + face_w // 3
            eye2_x = x1 + 2 * face_w // 3
            cv2.circle(image, (eye1_x, eye_y), 5, (0, 0, 0), -1)
            cv2.circle(image, (eye2_x, eye_y), 5, (0, 0, 0), -1)

            face_bboxes.append([x1, y1, x2, y2])

        return image, np.array(face_bboxes, dtype=np.float32)

    return generate


@pytest.fixture
def ground_truth_data(test_data_dir):
    """
    Load ground truth face annotations if available.
    """
    gt_file = test_data_dir / "ground_truth.json"
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            return json.load(f)
    return {}


@pytest.fixture
def replacement_image():
    """Generate a small replacement image for img mode anonymization."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray square
    cv2.circle(img, (50, 50), 40, (255, 255, 0), -1)    # Yellow circle
    return img


def pytest_configure(config):
    """
    Pytest configuration hook for session setup.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to automatically skip GPU tests if CUDA not available.
    Also skip benchmark tests if pytest-benchmark not installed.
    """
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip benchmark tests if pytest-benchmark not installed
    try:
        import pytest_benchmark
    except ImportError:
        skip_benchmark = pytest.mark.skip(reason="pytest-benchmark not installed")
        for item in items:
            if "benchmark" in item.keywords or "benchmark" in str(item.fspath):
                item.add_marker(skip_benchmark)
