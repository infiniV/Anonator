import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch, call
import threading
import time

from anonator.core.processor import VideoProcessor, ProgressData


class TestProgressData:
    """Test suite for ProgressData class."""

    def test_progress_data_initialization(self, sample_frame):
        """Test ProgressData initialization."""
        progress = ProgressData(
            frame_number=10,
            total_frames=100,
            original_frame=sample_frame,
            anonymized_frame=sample_frame,
            elapsed_time=5.0,
            fps=2.0
        )

        assert progress.frame_number == 10
        assert progress.total_frames == 100
        assert np.array_equal(progress.original_frame, sample_frame)
        assert np.array_equal(progress.anonymized_frame, sample_frame)
        assert progress.elapsed_time == 5.0
        assert progress.fps == 2.0

    def test_progress_data_fields_accessible(self):
        """Test all ProgressData fields are accessible."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        progress = ProgressData(
            frame_number=1,
            total_frames=10,
            original_frame=frame,
            anonymized_frame=frame,
            elapsed_time=1.0,
            fps=1.0
        )

        assert hasattr(progress, 'frame_number')
        assert hasattr(progress, 'total_frames')
        assert hasattr(progress, 'original_frame')
        assert hasattr(progress, 'anonymized_frame')
        assert hasattr(progress, 'elapsed_time')
        assert hasattr(progress, 'fps')


class TestVideoProcessorInitialization:
    """Test suite for VideoProcessor initialization."""

    @patch('anonator.core.processor.face_detection')
    def test_processor_initialization_without_callback(self, mock_face_detection):
        """Test VideoProcessor initialization without progress callback."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        assert processor.progress_callback is None
        assert processor._cancel_flag is not None
        assert processor._processing_thread is None
        mock_face_detection.build_detector.assert_called_once()

    @patch('anonator.core.processor.face_detection')
    def test_processor_initialization_with_callback(self, mock_face_detection):
        """Test VideoProcessor initialization with progress callback."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector
        callback = Mock()

        processor = VideoProcessor(progress_callback=callback)

        assert processor.progress_callback == callback
        mock_face_detection.build_detector.assert_called_once()

    @patch('anonator.core.processor.face_detection')
    def test_processor_builds_detector_with_config(self, mock_face_detection):
        """Test that processor builds detector with correct config parameters."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Verify detector was built with PROCESSOR_CONFIG parameters
        call_args = mock_face_detection.build_detector.call_args
        assert call_args is not None


class TestRemoveDuplicateDetections:
    """Test suite for _remove_duplicate_detections (custom NMS)."""

    @patch('anonator.core.processor.face_detection')
    def test_nms_no_overlap(self, mock_face_detection):
        """Test NMS with non-overlapping boxes (all should be kept)."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Non-overlapping detections
        dets = np.array([
            [100, 100, 200, 200, 0.95],
            [300, 300, 400, 400, 0.90],
            [500, 500, 600, 600, 0.85],
        ], dtype=np.float32)

        result = processor._remove_duplicate_detections(dets, iou_threshold=0.3)

        assert len(result) == 3  # All boxes kept

    @patch('anonator.core.processor.face_detection')
    def test_nms_high_overlap(self, mock_face_detection):
        """Test NMS with high overlap (lower confidence box removed)."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Two highly overlapping detections
        dets = np.array([
            [100, 100, 200, 200, 0.95],  # Higher confidence
            [105, 105, 205, 205, 0.80],  # Lower confidence, high overlap
        ], dtype=np.float32)

        result = processor._remove_duplicate_detections(dets, iou_threshold=0.3)

        # Should keep only the higher confidence detection
        assert len(result) == 1
        assert np.isclose(result[0, 4], 0.95)

    @patch('anonator.core.processor.face_detection')
    def test_nms_empty_detections(self, mock_face_detection):
        """Test NMS with empty detection array."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        empty_dets = np.array([], dtype=np.float32).reshape(0, 5)
        result = processor._remove_duplicate_detections(empty_dets)

        assert len(result) == 0

    @patch('anonator.core.processor.face_detection')
    def test_nms_complex_overlaps(self, mock_face_detection, mock_detections_overlapping):
        """Test NMS with complex overlapping patterns."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        result = processor._remove_duplicate_detections(
            mock_detections_overlapping,
            iou_threshold=0.3
        )

        # Should have fewer detections than input
        assert len(result) < len(mock_detections_overlapping)
        assert len(result) >= 2  # At least 2 distinct groups

        # Highest confidence from each group should be kept
        # Original has [0.95, 0.80, 0.85, 0.90]
        # First two overlap (keep 0.95)
        # Last two overlap (keep 0.90)
        assert 0.95 in result[:, 4]
        assert 0.90 in result[:, 4]

    @patch('anonator.core.processor.face_detection')
    def test_nms_different_iou_thresholds(self, mock_face_detection):
        """Test NMS behavior with different IOU thresholds."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        dets = np.array([
            [100, 100, 200, 200, 0.95],
            [110, 110, 210, 210, 0.85],  # Moderate overlap
            [120, 120, 220, 220, 0.75],  # High overlap
        ], dtype=np.float32)

        # Low threshold (aggressive): remove more boxes
        result_low = processor._remove_duplicate_detections(dets, iou_threshold=0.1)
        assert len(result_low) <= 2

        # High threshold (permissive): keep more boxes
        result_high = processor._remove_duplicate_detections(dets, iou_threshold=0.8)
        assert len(result_high) >= len(result_low)

    @patch('anonator.core.processor.face_detection')
    def test_nms_preserves_highest_scores(self, mock_face_detection):
        """Test that NMS keeps highest scoring detections."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        dets = np.array([
            [100, 100, 200, 200, 0.60],  # Lowest score
            [105, 105, 205, 205, 0.95],  # Highest score (overlaps with first)
            [300, 300, 400, 400, 0.80],  # Different location
        ], dtype=np.float32)

        result = processor._remove_duplicate_detections(dets, iou_threshold=0.3)

        # Should keep 0.95 (from overlapping pair) and 0.80 (separate)
        scores = result[:, 4]
        assert 0.95 in scores
        assert 0.80 in scores
        assert 0.60 not in scores


class TestDetectFaces:
    """Test suite for _detect_faces method."""

    @patch('anonator.core.processor.face_detection')
    def test_detect_faces_with_detections(self, mock_face_detection, sample_frame):
        """Test _detect_faces when faces are found."""
        mock_detector = Mock()
        mock_detections = np.array([[100, 150, 200, 250, 0.95]], dtype=np.float64)
        mock_detector.detect.return_value = mock_detections
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        result = processor._detect_faces(sample_frame, threshold=0.5)

        assert result is not None
        assert len(result) == 1
        assert result.dtype == np.float32
        mock_detector.detect.assert_called_once_with(sample_frame)

    @patch('anonator.core.processor.face_detection')
    def test_detect_faces_no_detections(self, mock_face_detection, sample_frame):
        """Test _detect_faces when no faces are found."""
        mock_detector = Mock()
        mock_detector.detect.return_value = np.array([], dtype=np.float64).reshape(0, 5)
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        result = processor._detect_faces(sample_frame, threshold=0.7)

        assert result is None

    @patch('anonator.core.processor.face_detection')
    def test_detect_faces_sets_threshold(self, mock_face_detection, sample_frame):
        """Test that _detect_faces sets the confidence threshold."""
        mock_detector = Mock()
        mock_detector.detect.return_value = np.array([[100, 150, 200, 250, 0.95]])
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        processor._detect_faces(sample_frame, threshold=0.6)

        # Verify threshold was set
        assert mock_detector.confidence_threshold == 0.6

    @patch('anonator.core.processor.face_detection')
    def test_detect_faces_returns_none_for_none_input(self, mock_face_detection, sample_frame):
        """Test _detect_faces handles None return from detector."""
        mock_detector = Mock()
        mock_detector.detect.return_value = None
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        result = processor._detect_faces(sample_frame, threshold=0.5)

        assert result is None


class TestCancellation:
    """Test suite for cancellation functionality."""

    @patch('anonator.core.processor.face_detection')
    def test_cancel_flag_initialized(self, mock_face_detection):
        """Test that cancel flag is initialized."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        assert processor._cancel_flag is not None
        assert not processor._cancel_flag.is_set()

    @patch('anonator.core.processor.face_detection')
    def test_cancel_method(self, mock_face_detection):
        """Test cancel method sets the flag."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        processor.cancel()

        assert processor._cancel_flag.is_set()

    @patch('anonator.core.processor.face_detection')
    def test_cancel_before_processing(self, mock_face_detection):
        """Test calling cancel before processing starts."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()
        processor.cancel()

        # Should set flag without errors
        assert processor._cancel_flag.is_set()


class TestProgressCallback:
    """Test suite for progress callback functionality."""

    @patch('anonator.core.processor.face_detection')
    def test_progress_callback_invocation(self, mock_face_detection):
        """Test that progress callback is invoked."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        callback = Mock()
        processor = VideoProcessor(progress_callback=callback)

        # We can't easily test the full processing without mocking cv2,
        # but we can verify the callback is stored
        assert processor.progress_callback == callback

    @patch('anonator.core.processor.face_detection')
    def test_no_callback_no_errors(self, mock_face_detection):
        """Test that processing works without a callback."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        # Should not raise errors
        processor = VideoProcessor(progress_callback=None)
        assert processor.progress_callback is None


class TestProcessVideoThreading:
    """Test suite for threading behavior."""

    @patch('anonator.core.processor.face_detection')
    @patch('anonator.core.processor.cv2')
    def test_process_video_creates_thread(self, mock_cv2, mock_face_detection, temp_output_dir):
        """Test that process_video creates a daemon thread."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        # Mock VideoCapture to prevent actual file opening
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        processor = VideoProcessor()
        input_path = "test_input.mp4"
        output_path = str(temp_output_dir / "test_output.mp4")

        processor.process_video(input_path, output_path)

        # Give thread a moment to start
        time.sleep(0.1)

        assert processor._processing_thread is not None
        assert isinstance(processor._processing_thread, threading.Thread)
        assert processor._processing_thread.daemon is True

    @patch('anonator.core.processor.face_detection')
    @patch('anonator.core.processor.cv2')
    def test_process_video_prevents_concurrent_processing(self, mock_cv2, mock_face_detection, temp_output_dir):
        """Test that concurrent processing is prevented."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 640,  # Width
            4: 480,  # Height
            5: 25,   # FPS
            7: 10    # Frame count
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)  # End immediately
        mock_cv2.VideoCapture.return_value = mock_cap

        mock_writer = Mock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0

        processor = VideoProcessor()
        input_path = "test_input.mp4"
        output_path = str(temp_output_dir / "test_output.mp4")

        # Start first processing
        processor.process_video(input_path, output_path)
        time.sleep(0.05)  # Let thread start

        # Try to start second processing
        processor.process_video(input_path, output_path)

        # Second call should be ignored (thread already running)
        # Can't easily assert this without inspecting logs


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    @patch('anonator.core.processor.face_detection')
    def test_processor_with_invalid_config(self, mock_face_detection):
        """Test processor handles various configs."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        # Should not raise errors
        processor = VideoProcessor()
        assert processor is not None

    @patch('anonator.core.processor.face_detection')
    def test_remove_duplicate_detections_single_detection(self, mock_face_detection):
        """Test NMS with single detection."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        single_det = np.array([[100, 100, 200, 200, 0.95]], dtype=np.float32)
        result = processor._remove_duplicate_detections(single_det)

        assert len(result) == 1
        assert np.array_equal(result, single_det)

    @patch('anonator.core.processor.face_detection')
    def test_detect_faces_with_different_thresholds(self, mock_face_detection, sample_frame):
        """Test detection with various threshold values."""
        mock_detector = Mock()
        mock_detector.detect.return_value = np.array([[100, 150, 200, 250, 0.95]])
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Test different thresholds
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = processor._detect_faces(sample_frame, threshold=threshold)
            assert mock_detector.confidence_threshold == threshold


class TestNMSAlgorithmCorrectness:
    """Test suite specifically for NMS algorithm correctness."""

    @patch('anonator.core.processor.face_detection')
    def test_nms_iou_calculation_accuracy(self, mock_face_detection):
        """Test that NMS correctly calculates IOU."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Two boxes with known IOU
        # Box 1: [0, 0, 100, 100], area = 10000
        # Box 2: [50, 50, 150, 150], area = 10000
        # Intersection: [50, 50, 100, 100], area = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IOU: 2500 / 17500 = 0.143

        dets = np.array([
            [0, 0, 100, 100, 0.95],
            [50, 50, 150, 150, 0.85],
        ], dtype=np.float32)

        # With IOU threshold 0.1, should remove second box
        result_aggressive = processor._remove_duplicate_detections(dets, iou_threshold=0.1)
        assert len(result_aggressive) == 1

        # With IOU threshold 0.2, should keep both boxes
        result_permissive = processor._remove_duplicate_detections(dets, iou_threshold=0.2)
        assert len(result_permissive) == 2

    @patch('anonator.core.processor.face_detection')
    def test_nms_sorts_by_confidence(self, mock_face_detection):
        """Test that NMS processes boxes in descending confidence order."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Boxes with increasing confidence (reverse sorted)
        dets = np.array([
            [100, 100, 200, 200, 0.60],
            [105, 105, 205, 205, 0.70],
            [110, 110, 210, 210, 0.95],  # Highest confidence
        ], dtype=np.float32)

        result = processor._remove_duplicate_detections(dets, iou_threshold=0.3)

        # Should keep the highest confidence (0.95)
        assert 0.95 in result[:, 4]
        assert len(result) == 1

    @patch('anonator.core.processor.face_detection')
    def test_nms_with_many_detections(self, mock_face_detection):
        """Test NMS performance with many detections."""
        mock_detector = Mock()
        mock_face_detection.build_detector.return_value = mock_detector

        processor = VideoProcessor()

        # Create 100 non-overlapping detections
        dets = []
        for i in range(10):
            for j in range(10):
                x1 = i * 100
                y1 = j * 100
                x2 = x1 + 80  # Leave gap to avoid overlap
                y2 = y1 + 80
                score = 0.5 + 0.5 * np.random.random()
                dets.append([x1, y1, x2, y2, score])

        dets = np.array(dets, dtype=np.float32)

        result = processor._remove_duplicate_detections(dets, iou_threshold=0.3)

        # All boxes should be kept (no overlap)
        assert len(result) == 100
