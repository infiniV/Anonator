import pytest
from anonator.core.processor import VideoProcessor, ProgressData


def test_processor_initialization():
    processor = VideoProcessor()
    assert processor is not None
    assert processor.progress_callback is None


def test_processor_initialization_with_callback():
    def callback(data):
        pass

    processor = VideoProcessor(progress_callback=callback)
    assert processor.progress_callback == callback


def test_progress_data_creation():
    import numpy as np

    data = ProgressData(
        frame_number=10,
        total_frames=100,
        original_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        anonymized_frame=np.zeros((10, 10, 3), dtype=np.uint8),
        elapsed_time=5.0,
        fps=2.0
    )

    assert data.frame_number == 10
    assert data.total_frames == 100
    assert data.elapsed_time == 5.0
    assert data.fps == 2.0
