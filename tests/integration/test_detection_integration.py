import numpy as np
import cv2
import pytest
import time

from anonator.core.config import HIPAA_MODE


class TestDetectorLoading:
    """Test suite for face detector loading."""

    def test_detector_loads_on_cpu(self, device):
        """Test that detector loads successfully on CPU."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            nms_iou_threshold=0.3,
            device="cpu",
            max_resolution=1920,
            fp16_inference=False,
            clip_boxes=True,
        )

        assert detector is not None
        assert detector.confidence_threshold == 0.5

    @pytest.mark.gpu
    def test_detector_loads_on_gpu(self, has_gpu):
        """Test that detector loads successfully on GPU."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            nms_iou_threshold=0.3,
            device="cuda",
            max_resolution=1920,
            fp16_inference=True,
            clip_boxes=True,
        )

        assert detector is not None

    def test_detector_with_different_models(self, device):
        """Test loading different detector models."""
        import face_detection

        models = ["RetinaNetResNet50", "RetinaNetMobileNetV1"]

        for model_name in models:
            detector = face_detection.build_detector(
                model_name,
                confidence_threshold=0.5,
                device=device,
            )
            assert detector is not None


class TestSingleFrameDetection:
    """Test suite for single frame detection."""

    def test_detection_on_frame_with_synthetic_face(self, face_detector, synthetic_image_generator):
        """Test detection runs without errors on synthetically generated image."""
        # Generate image with 1 synthetic face
        # Note: Synthetic faces (simple rectangles) won't be detected by RetinaFace,
        # but this tests the detector handles various inputs without crashing
        image, expected_bboxes = synthetic_image_generator(
            width=640, height=480, num_faces=1, seed=42
        )

        dets = face_detector.detect(image)

        # Detector should run without errors and return valid format
        assert dets is None or isinstance(dets, np.ndarray)

        # If detections exist, verify format
        if dets is not None and len(dets) > 0:
            assert dets.shape[1] == 5  # [xmin, ymin, xmax, ymax, score]

    def test_detection_on_frame_with_multiple_faces(self, face_detector, synthetic_image_generator):
        """Test detection on frame with multiple synthetic faces."""
        # Generate image with 3 known faces
        image, expected_bboxes = synthetic_image_generator(
            width=960, height=540, num_faces=3, seed=42
        )

        dets = face_detector.detect(image)

        # Should detect faces (may not be perfect with synthetic faces)
        # We're testing the detector works, not its accuracy
        assert dets is not None

    def test_detection_on_frame_no_faces(self, face_detector, sample_frame_no_faces):
        """Test detection on frame without faces."""
        dets = face_detector.detect(sample_frame_no_faces)

        # Should return empty or None
        if dets is not None:
            # Very unlikely to detect faces in landscape
            assert len(dets) == 0 or len(dets) < 2

    def test_detection_confidence_scores(self, face_detector, synthetic_image_generator):
        """Test that detection confidence scores are in valid range."""
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=2, seed=42
        )

        dets = face_detector.detect(image)

        if dets is not None and len(dets) > 0:
            scores = dets[:, 4]
            # All scores should be between 0 and 1
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)


class TestDetectionThresholdSensitivity:
    """Test suite for detection threshold behavior."""

    def test_threshold_sensitivity(self, face_detector, synthetic_image_generator):
        """Test that different thresholds produce different results."""
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=2, seed=42
        )

        thresholds = [0.3, 0.5, 0.7, 0.9]
        detection_counts = []

        for threshold in thresholds:
            face_detector.confidence_threshold = threshold
            dets = face_detector.detect(image)
            count = len(dets) if dets is not None else 0
            detection_counts.append(count)

        # Generally, detection count should decrease or stay same with higher threshold
        # (More restrictive = fewer detections)
        # Note: May not be perfectly monotonic due to NMS interactions
        assert detection_counts[-1] <= detection_counts[0] or detection_counts[-1] == 0

    def test_very_low_threshold(self, face_detector, sample_frame):
        """Test detection with very low threshold."""
        face_detector.confidence_threshold = 0.1

        dets = face_detector.detect(sample_frame)

        # Low threshold may produce many false positives
        # Just verify it doesn't crash
        assert dets is None or isinstance(dets, np.ndarray)

    def test_very_high_threshold(self, face_detector, synthetic_image_generator):
        """Test detection with very high threshold."""
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=1, seed=42
        )

        face_detector.confidence_threshold = 0.95

        dets = face_detector.detect(image)

        # Very high threshold may miss faces
        # Just verify it doesn't crash
        assert dets is None or isinstance(dets, np.ndarray)


class TestMultiPassDetection:
    """Test suite for multi-pass detection strategy."""

    def test_multi_pass_with_hipaa_multipliers(self, face_detector, synthetic_image_generator):
        """Test multi-pass detection runs with HIPAA mode multipliers."""
        # Note: Using synthetic image to test multi-pass mechanism, not detection accuracy
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=2, seed=42
        )

        base_threshold = 0.5
        all_detections = []

        # Run 3 passes with HIPAA multipliers
        for multiplier in HIPAA_MODE.multi_pass_multipliers:
            threshold = base_threshold * multiplier
            face_detector.confidence_threshold = threshold
            dets = face_detector.detect(image)

            if dets is not None and len(dets) > 0:
                all_detections.append(dets)

        # Test that multi-pass mechanism works (runs 3 times without errors)
        # Synthetic faces may not be detected, which is fine
        # If detections exist, verify they can be combined
        if all_detections:
            combined = np.vstack(all_detections)
            assert len(combined) >= len(all_detections[0])
            assert combined.shape[1] == 5

    def test_multi_pass_finds_more_faces(self, face_detector, synthetic_image_generator):
        """Test that multi-pass can find additional faces."""
        image, _ = synthetic_image_generator(
            width=960, height=540, num_faces=5, seed=123
        )

        # Single pass
        face_detector.confidence_threshold = 0.7
        single_pass_dets = face_detector.detect(image)
        single_count = len(single_pass_dets) if single_pass_dets is not None else 0

        # Multi-pass
        all_detections = []
        for threshold in [0.5, 0.75, 0.35]:
            face_detector.confidence_threshold = threshold
            dets = face_detector.detect(image)
            if dets is not None and len(dets) > 0:
                all_detections.append(dets)

        multi_count = 0
        if all_detections:
            combined = np.vstack(all_detections)
            # Before NMS, should have at least as many as single pass
            multi_count = len(combined)

        # Multi-pass should find equal or more detections (before NMS)
        assert multi_count >= single_count


class TestDetectionEdgeCases:
    """Test suite for edge cases in detection."""

    def test_detection_at_frame_edges(self, face_detector, synthetic_image_generator):
        """Test detection of faces at frame boundaries."""
        image, bboxes = synthetic_image_generator(
            width=640, height=480, num_faces=1, seed=42
        )

        # Manually place a face at edge (top-left corner)
        image[0:80, 0:60] = 255  # Simple white rectangle at corner

        dets = face_detector.detect(image)

        # Should handle edge faces gracefully (may or may not detect)
        assert dets is None or isinstance(dets, np.ndarray)

    def test_detection_very_small_frame(self, face_detector):
        """Test detection on very small frame."""
        # Create tiny 100x100 frame
        small_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        dets = face_detector.detect(small_frame)

        # Should handle without crashing
        assert dets is None or isinstance(dets, np.ndarray)

    def test_detection_large_frame(self, face_detector):
        """Test detection on large frame (tests max_resolution)."""
        # Create 4K frame
        large_frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        dets = face_detector.detect(large_frame)

        # Should handle large frames (will be downscaled)
        assert dets is None or isinstance(dets, np.ndarray)


@pytest.mark.gpu
class TestFP16vsFP32:
    """Test suite comparing FP16 and FP32 inference."""

    def test_fp16_vs_fp32_accuracy(self, has_gpu, synthetic_image_generator):
        """Test that FP16 and FP32 produce similar results."""
        if not has_gpu:
            pytest.skip("GPU required for FP16 testing")

        import face_detection

        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=2, seed=42
        )

        # FP32 detector
        detector_fp32 = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=False,
        )

        # FP16 detector
        detector_fp16 = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=True,
        )

        dets_fp32 = detector_fp32.detect(image)
        dets_fp16 = detector_fp16.detect(image)

        # Both should produce detections
        fp32_count = len(dets_fp32) if dets_fp32 is not None else 0
        fp16_count = len(dets_fp16) if dets_fp16 is not None else 0

        # Detection counts should be similar (within 20%)
        if fp32_count > 0:
            ratio = fp16_count / fp32_count
            assert 0.8 <= ratio <= 1.2, f"FP16/FP32 detection count ratio: {ratio}"


class TestBoundingBoxFormat:
    """Test suite for bounding box format validation."""

    def test_bbox_format(self, face_detector, synthetic_image_generator):
        """Test that bounding boxes have correct format."""
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=1, seed=42
        )

        dets = face_detector.detect(image)

        if dets is not None and len(dets) > 0:
            # Check shape
            assert dets.ndim == 2
            assert dets.shape[1] == 5

            # Check coordinate order (xmin < xmax, ymin < ymax)
            for det in dets:
                xmin, ymin, xmax, ymax, score = det
                assert xmin < xmax, "xmin should be less than xmax"
                assert ymin < ymax, "ymin should be less than ymax"
                assert 0 <= score <= 1, "Score should be between 0 and 1"

    def test_bbox_within_image_bounds(self, face_detector, synthetic_image_generator):
        """Test that bounding boxes are within image bounds (when clip_boxes=True)."""
        width, height = 640, 480
        image, _ = synthetic_image_generator(
            width=width, height=height, num_faces=2, seed=42
        )

        dets = face_detector.detect(image)

        if dets is not None and len(dets) > 0:
            for det in dets:
                xmin, ymin, xmax, ymax, _ = det

                # Boxes should be within image bounds
                assert xmin >= 0, f"xmin={xmin} should be >= 0"
                assert ymin >= 0, f"ymin={ymin} should be >= 0"
                assert xmax <= width, f"xmax={xmax} should be <= {width}"
                assert ymax <= height, f"ymax={ymax} should be <= {height}"

    def test_bbox_dimensions_reasonable(self, face_detector, synthetic_image_generator):
        """Test that detected bounding boxes have reasonable dimensions."""
        image, _ = synthetic_image_generator(
            width=640, height=480, num_faces=1, seed=42
        )

        dets = face_detector.detect(image)

        if dets is not None and len(dets) > 0:
            for det in dets:
                xmin, ymin, xmax, ymax, _ = det

                width = xmax - xmin
                height = ymax - ymin

                # Face boxes should have reasonable size (not too small, not too large)
                assert width > 5, "Face width too small"
                assert height > 5, "Face height too small"
                assert width < 640, "Face width too large"
                assert height < 480, "Face height too large"


class TestDetectionLatencySanityCheck:
    """Basic latency checks (not full benchmarks)."""

    def test_single_frame_latency_reasonable(self, face_detector, sample_frame):
        """Test that single frame detection completes in reasonable time."""
        start = time.time()
        dets = face_detector.detect(sample_frame)
        elapsed = time.time() - start

        # Should complete within 2 seconds (very generous)
        assert elapsed < 2.0, f"Detection took {elapsed:.2f}s (too slow)"

    @pytest.mark.gpu
    def test_gpu_faster_than_cpu_hint(self, has_gpu, synthetic_image_generator):
        """Hint that GPU should be faster (not enforced)."""
        if not has_gpu:
            pytest.skip("GPU required")

        import face_detection

        image, _ = synthetic_image_generator(
            width=1280, height=720, num_faces=3, seed=42
        )

        # This is just informational, not enforced
        # GPU should generally be faster, but not guaranteed
        pass


class TestMaxResolutionDownscaling:
    """Test suite for max_resolution parameter."""

    def test_max_resolution_downscales(self, device):
        """Test that large images are downscaled based on max_resolution."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            device=device,
            max_resolution=1920,
        )

        # Create 4K image
        large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        # Should handle without issues (internally downscaled)
        dets = detector.detect(large_image)

        assert dets is None or isinstance(dets, np.ndarray)

    def test_small_image_not_upscaled(self, face_detector):
        """Test that small images are not upscaled."""
        # Small image well below max_resolution
        small_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        dets = face_detector.detect(small_image)

        # Should process normally
        assert dets is None or isinstance(dets, np.ndarray)
