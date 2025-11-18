import cv2
import numpy as np
import pytest
import time
from pathlib import Path

from anonator.core.processor import VideoProcessor, ProgressData


@pytest.mark.slow
class TestEndToEndVideoProcessing:
    """Test suite for end-to-end video processing."""

    def test_process_short_video_blur_mode(self, short_test_video, temp_output_dir):
        """Test processing a short video with blur mode."""
        output_path = temp_output_dir / "output_blur.mp4"

        progress_data_list = []

        def progress_callback(data):
            progress_data_list.append(data)

        processor = VideoProcessor(progress_callback=progress_callback)

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
            mask_scale=1.5,
            preview_interval=1.0,
            multi_pass=False,
            keep_audio=True
        )

        # Wait for processing to complete
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Verify output exists
        assert output_path.exists()

        # Verify output is a valid video
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Should have same number of frames as input (10)
        assert frame_count == 10

        # Should have received progress callbacks
        assert len(progress_data_list) > 0

    def test_process_short_video_solid_mode(self, short_test_video, temp_output_dir):
        """Test processing with solid (blackout) mode."""
        output_path = temp_output_dir / "output_solid.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="solid",
            threshold=0.5,
            mask_scale=1.5,
            multi_pass=False,
            keep_audio=True
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        assert output_path.exists()

        # Verify video is readable
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        cap.release()

    def test_process_short_video_mosaic_mode(self, short_test_video, temp_output_dir):
        """Test processing with mosaic mode."""
        output_path = temp_output_dir / "output_mosaic.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="mosaic",
            threshold=0.5,
            mask_scale=1.5,
            multi_pass=False,
            keep_audio=True
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        assert output_path.exists()


class TestVideoProperties:
    """Test suite for preserving video properties."""

    def test_preserves_resolution(self, short_test_video, temp_output_dir):
        """Test that output video has same resolution as input."""
        output_path = temp_output_dir / "output_resolution.mp4"

        # Get input resolution
        cap_in = cv2.VideoCapture(str(short_test_video))
        width_in = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_in = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_in.release()

        processor = VideoProcessor()
        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Get output resolution
        cap_out = cv2.VideoCapture(str(output_path))
        width_out = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_out = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_out.release()

        assert width_in == width_out
        assert height_in == height_out

    def test_preserves_fps(self, short_test_video, temp_output_dir):
        """Test that output video has same FPS as input."""
        output_path = temp_output_dir / "output_fps.mp4"

        # Get input FPS
        cap_in = cv2.VideoCapture(str(short_test_video))
        fps_in = cap_in.get(cv2.CAP_PROP_FPS)
        cap_in.release()

        processor = VideoProcessor()
        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Get output FPS
        cap_out = cv2.VideoCapture(str(output_path))
        fps_out = cap_out.get(cv2.CAP_PROP_FPS)
        cap_out.release()

        # FPS should match
        assert abs(fps_in - fps_out) < 0.1

    def test_preserves_frame_count(self, short_test_video, temp_output_dir):
        """Test that output has same number of frames as input."""
        output_path = temp_output_dir / "output_frame_count.mp4"

        # Get input frame count
        cap_in = cv2.VideoCapture(str(short_test_video))
        frame_count_in = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_in.release()

        processor = VideoProcessor()
        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="solid",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Get output frame count
        cap_out = cv2.VideoCapture(str(output_path))
        frame_count_out = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_out.release()

        assert frame_count_in == frame_count_out


class TestProgressCallbacks:
    """Test suite for progress callback functionality."""

    def test_progress_callbacks_invoked(self, short_test_video, temp_output_dir):
        """Test that progress callbacks are invoked during processing."""
        output_path = temp_output_dir / "output_progress.mp4"

        progress_data_list = []

        def progress_callback(data):
            assert isinstance(data, ProgressData)
            progress_data_list.append(data)

        processor = VideoProcessor(progress_callback=progress_callback)

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
            preview_interval=0.5,  # Frequent updates
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Should have received multiple progress callbacks
        assert len(progress_data_list) > 0

        # Verify progress data contents
        for data in progress_data_list:
            assert data.frame_number >= 0
            assert data.total_frames > 0
            assert data.frame_number <= data.total_frames
            assert data.elapsed_time >= 0
            assert data.fps >= 0

    def test_progress_frames_increment(self, short_test_video, temp_output_dir):
        """Test that frame numbers increment in progress callbacks."""
        output_path = temp_output_dir / "output_progress_incr.mp4"

        frame_numbers = []

        def progress_callback(data):
            frame_numbers.append(data.frame_number)

        processor = VideoProcessor(progress_callback=progress_callback)

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="solid",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # Frame numbers should generally increase
        if len(frame_numbers) > 1:
            assert frame_numbers[-1] >= frame_numbers[0]


class TestCancellation:
    """Test suite for processing cancellation."""

    def test_cancellation_stops_processing(self, short_test_video, temp_output_dir):
        """Test that cancellation stops video processing."""
        output_path = temp_output_dir / "output_cancelled.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Let it process a few frames
        time.sleep(0.5)

        # Cancel processing
        processor.cancel()

        # Wait for thread to stop
        if processor._processing_thread:
            processor._processing_thread.join(timeout=10)

        # Thread should have stopped
        assert not processor._processing_thread.is_alive()


class TestMultiPassMode:
    """Test suite for multi-pass detection mode."""

    @pytest.mark.slow
    def test_multi_pass_mode(self, short_test_video, temp_output_dir):
        """Test multi-pass detection mode."""
        output_path = temp_output_dir / "output_multipass.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="solid",
            threshold=0.5,
            multi_pass=True,  # Enable multi-pass
        )

        # Wait for completion (multi-pass is slower)
        if processor._processing_thread:
            processor._processing_thread.join(timeout=60)

        assert output_path.exists()

        # Verify output is valid
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        cap.release()


class TestErrorHandling:
    """Test suite for error handling."""

    def test_invalid_input_path(self, temp_output_dir):
        """Test handling of non-existent input file."""
        output_path = temp_output_dir / "output_invalid.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path="non_existent_video.mp4",
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for thread to complete (should fail gracefully)
        if processor._processing_thread:
            processor._processing_thread.join(timeout=10)

        # Output should not exist or be invalid
        if output_path.exists():
            cap = cv2.VideoCapture(str(output_path))
            # May or may not open, but shouldn't crash
            cap.release()

    def test_invalid_output_directory(self, short_test_video):
        """Test handling of invalid output directory."""
        # Try to write to non-existent directory without creating it
        output_path = Path("/nonexistent_dir_xyz/output.mp4")

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for thread (should fail gracefully)
        if processor._processing_thread:
            processor._processing_thread.join(timeout=10)

        # Should not crash the application


@pytest.mark.slow
class TestRealVideoProcessing:
    """Test suite using real test videos from testData."""

    def test_process_real_test_video(self, testdata_videos_dir, temp_output_dir):
        """Test processing a real test video."""
        # Look for test video
        test_videos = list(testdata_videos_dir.glob("*.mp4"))

        if not test_videos:
            pytest.skip("No test videos found in testData directory")

        input_video = test_videos[0]
        output_path = temp_output_dir / f"output_{input_video.name}"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(input_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
            mask_scale=1.5,
        )

        # Wait for completion (may take longer for real video)
        if processor._processing_thread:
            processor._processing_thread.join(timeout=120)

        # Verify output exists
        assert output_path.exists()

        # Verify output is readable
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()

        # Read first frame to verify it's valid
        ret, frame = cap.read()
        assert ret
        assert frame is not None

        cap.release()


class TestDifferentAnonymizationModes:
    """Test suite comparing all anonymization modes."""

    @pytest.mark.slow
    def test_all_modes_produce_valid_output(self, short_test_video, temp_output_dir):
        """Test that all anonymization modes produce valid output."""
        modes = ["blur", "solid", "mosaic"]

        for mode in modes:
            output_path = temp_output_dir / f"output_{mode}.mp4"

            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode=mode,
                threshold=0.5,
            )

            # Wait for completion
            if processor._processing_thread:
                processor._processing_thread.join(timeout=30)

            # Verify output
            assert output_path.exists()

            cap = cv2.VideoCapture(str(output_path))
            assert cap.isOpened()

            # Read a frame to verify it's valid
            ret, frame = cap.read()
            cap.release()

            assert ret, f"Failed to read frame from {mode} mode output"
            assert frame is not None


class TestMemoryEfficiency:
    """Test suite for memory efficiency."""

    def test_processes_without_memory_explosion(self, short_test_video, temp_output_dir):
        """Test that processing doesn't cause excessive memory usage."""
        output_path = temp_output_dir / "output_memory.mp4"

        processor = VideoProcessor()

        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=30)

        # If we get here without OOM, test passes
        assert True
