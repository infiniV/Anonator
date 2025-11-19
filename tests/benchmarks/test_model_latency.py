import numpy as np
import pytest
import time

from anonator.core.anonymizer import anonymize_frame, scale_bb
from anonator.core.config import HIPAA_MODE


@pytest.mark.benchmark
class TestSingleFrameInferenceLatency:
    """Benchmark suite for single frame inference latency."""

    def test_benchmark_single_frame_cpu(self, benchmark, device, sample_frame):
        """Benchmark single frame inference on CPU."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cpu",
            fp16_inference=False,
        )

        # Benchmark the detection
        result = benchmark(detector.detect, sample_frame)

        # Store result for verification
        assert result is None or isinstance(result, np.ndarray)

    @pytest.mark.gpu
    def test_benchmark_single_frame_gpu(self, benchmark, has_gpu, sample_frame):
        """Benchmark single frame inference on GPU."""
        if not has_gpu:
            pytest.skip("GPU not available")

        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=False,
        )

        # Warm-up run
        detector.detect(sample_frame)

        # Benchmark the detection
        result = benchmark(detector.detect, sample_frame)

        assert result is None or isinstance(result, np.ndarray)

    @pytest.mark.gpu
    def test_benchmark_fp16_latency(self, benchmark, has_gpu, sample_frame):
        """Benchmark FP16 inference latency."""
        if not has_gpu:
            pytest.skip("GPU required for FP16")

        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=True,
        )

        # Warm-up
        detector.detect(sample_frame)

        # Benchmark
        result = benchmark(detector.detect, sample_frame)

        assert result is None or isinstance(result, np.ndarray)

    @pytest.mark.gpu
    def test_benchmark_fp32_latency(self, benchmark, has_gpu, sample_frame):
        """Benchmark FP32 inference latency."""
        if not has_gpu:
            pytest.skip("GPU required for comparison")

        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=False,
        )

        # Warm-up
        detector.detect(sample_frame)

        # Benchmark
        result = benchmark(detector.detect, sample_frame)

        assert result is None or isinstance(result, np.ndarray)


@pytest.mark.benchmark
class TestResolutionImpact:
    """Benchmark suite for resolution impact on latency."""

    def test_benchmark_480p_detection(self, benchmark, device):
        """Benchmark detection on 480p frame."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device=device,
        )

        frame_480p = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = benchmark(detector.detect, frame_480p)
        assert result is None or isinstance(result, np.ndarray)

    def test_benchmark_720p_detection(self, benchmark, device):
        """Benchmark detection on 720p frame."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device=device,
        )

        frame_720p = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        result = benchmark(detector.detect, frame_720p)
        assert result is None or isinstance(result, np.ndarray)

    def test_benchmark_1080p_detection(self, benchmark, device):
        """Benchmark detection on 1080p frame."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device=device,
        )

        frame_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        result = benchmark(detector.detect, frame_1080p)
        assert result is None or isinstance(result, np.ndarray)

    @pytest.mark.slow
    def test_benchmark_4k_detection(self, benchmark, device):
        """Benchmark detection on 4K frame (with max_resolution)."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device=device,
            max_resolution=1920,  # Will downscale
        )

        frame_4k = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        result = benchmark(detector.detect, frame_4k)
        assert result is None or isinstance(result, np.ndarray)


@pytest.mark.benchmark
class TestMultiPassOverhead:
    """Benchmark suite for multi-pass detection overhead."""

    def test_benchmark_single_pass_detection(self, benchmark, face_detector, sample_frame):
        """Benchmark single-pass detection."""
        face_detector.confidence_threshold = 0.5

        result = benchmark(face_detector.detect, sample_frame)
        assert result is None or isinstance(result, np.ndarray)

    def test_benchmark_three_pass_detection(self, benchmark, face_detector, sample_frame):
        """Benchmark three-pass detection (HIPAA mode)."""
        def multi_pass_detect(frame):
            all_dets = []
            base_threshold = 0.5

            for multiplier in HIPAA_MODE.multi_pass_multipliers:
                threshold = base_threshold * multiplier
                face_detector.confidence_threshold = threshold
                dets = face_detector.detect(frame)
                if dets is not None and len(dets) > 0:
                    all_dets.append(dets)

            if all_dets:
                return np.vstack(all_dets)
            return None

        result = benchmark(multi_pass_detect, sample_frame)
        assert result is None or isinstance(result, np.ndarray)


@pytest.mark.benchmark
class TestNMSPerformance:
    """Benchmark suite for NMS performance."""

    def test_benchmark_nms_10_boxes(self, benchmark):
        """Benchmark NMS with 10 detections."""
        from anonator.core.processor import VideoProcessor

        processor = VideoProcessor.__new__(VideoProcessor)

        # 10 detections with some overlaps
        dets = np.array([
            [i * 80, j * 80, i * 80 + 100, j * 80 + 100, 0.5 + 0.5 * np.random.random()]
            for i in range(5) for j in range(2)
        ], dtype=np.float32)

        result = benchmark(processor._remove_duplicate_detections, dets, 0.3)
        assert len(result) <= len(dets)

    def test_benchmark_nms_50_boxes(self, benchmark):
        """Benchmark NMS with 50 detections."""
        from anonator.core.processor import VideoProcessor

        processor = VideoProcessor.__new__(VideoProcessor)

        # 50 detections
        dets = np.array([
            [i * 50, j * 50, i * 50 + 80, j * 50 + 80, 0.5 + 0.5 * np.random.random()]
            for i in range(10) for j in range(5)
        ], dtype=np.float32)

        result = benchmark(processor._remove_duplicate_detections, dets, 0.3)
        assert len(result) <= len(dets)

    def test_benchmark_nms_100_boxes(self, benchmark):
        """Benchmark NMS with 100 detections."""
        from anonator.core.processor import VideoProcessor

        processor = VideoProcessor.__new__(VideoProcessor)

        # 100 detections
        dets = np.array([
            [i * 40, j * 40, i * 40 + 60, j * 40 + 60, 0.5 + 0.5 * np.random.random()]
            for i in range(10) for j in range(10)
        ], dtype=np.float32)

        result = benchmark(processor._remove_duplicate_detections, dets, 0.3)
        assert len(result) <= len(dets)


@pytest.mark.benchmark
class TestAnonymizationLatency:
    """Benchmark suite for anonymization operations."""

    def test_benchmark_blur_anonymization(self, benchmark, sample_frame, mock_detections_multiple):
        """Benchmark blur mode anonymization."""
        frame_copy = sample_frame.copy()

        result = benchmark(
            anonymize_frame,
            frame_copy,
            mock_detections_multiple,
            mask_scale=1.5,
            replacewith="blur",
            ellipse=True
        )

        assert result is not None

    def test_benchmark_solid_anonymization(self, benchmark, sample_frame, mock_detections_multiple):
        """Benchmark solid mode anonymization."""
        frame_copy = sample_frame.copy()

        result = benchmark(
            anonymize_frame,
            frame_copy,
            mock_detections_multiple,
            mask_scale=1.5,
            replacewith="solid"
        )

        assert result is not None

    def test_benchmark_mosaic_anonymization(self, benchmark, sample_frame, mock_detections_multiple):
        """Benchmark mosaic mode anonymization."""
        frame_copy = sample_frame.copy()

        result = benchmark(
            anonymize_frame,
            frame_copy,
            mock_detections_multiple,
            mask_scale=1.5,
            replacewith="mosaic",
            mosaicsize=20
        )

        assert result is not None

    def test_benchmark_scale_bb(self, benchmark):
        """Benchmark bounding box scaling operation."""
        result = benchmark(scale_bb, 100, 100, 200, 200, 1.5)

        assert len(result) == 4


@pytest.mark.benchmark
class TestEndToEndFrameLatency:
    """Benchmark suite for complete frame processing pipeline."""

    def test_benchmark_complete_frame_pipeline(self, benchmark, face_detector, sample_frame):
        """Benchmark detection + anonymization pipeline."""
        def complete_pipeline(frame):
            # Detection
            face_detector.confidence_threshold = 0.5
            dets = face_detector.detect(frame)

            # Anonymization
            result = anonymize_frame(
                frame,
                dets,
                mask_scale=1.5,
                replacewith="blur",
                ellipse=True
            )

            return result

        result = benchmark(complete_pipeline, sample_frame.copy())
        assert result is not None


@pytest.mark.benchmark
class TestModelLoadingTime:
    """Benchmark suite for model loading time."""

    @pytest.mark.slow
    def test_benchmark_model_loading_cpu(self, benchmark):
        """Benchmark time to load RetinaFace model on CPU."""
        import face_detection

        def load_model():
            return face_detection.build_detector(
                "RetinaNetMobileNetV1",
                confidence_threshold=0.5,
                device="cpu",
            )

        detector = benchmark(load_model)
        assert detector is not None

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_benchmark_model_loading_gpu(self, benchmark, has_gpu):
        """Benchmark time to load RetinaFace model on GPU."""
        if not has_gpu:
            pytest.skip("GPU not available")

        import face_detection

        def load_model():
            return face_detection.build_detector(
                "RetinaNetMobileNetV1",
                confidence_threshold=0.5,
                device="cuda",
            )

        detector = benchmark(load_model)
        assert detector is not None


@pytest.mark.benchmark
class TestComparisonBenchmarks:
    """Benchmark suite for performance comparisons."""

    @pytest.mark.gpu
    def test_cpu_vs_gpu_speedup(self, has_gpu, sample_frame):
        """Compare CPU vs GPU inference speed."""
        if not has_gpu:
            pytest.skip("GPU required for comparison")

        import face_detection

        # CPU detector
        detector_cpu = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cpu",
        )

        # GPU detector
        detector_gpu = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=False,
        )

        # Warm-up GPU
        detector_gpu.detect(sample_frame)

        # Time CPU
        start = time.time()
        for _ in range(5):
            detector_cpu.detect(sample_frame)
        cpu_time = (time.time() - start) / 5

        # Time GPU
        start = time.time()
        for _ in range(5):
            detector_gpu.detect(sample_frame)
        gpu_time = (time.time() - start) / 5

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        # GPU should generally be faster (speedup > 1)
        # This is informational, not enforced
        assert speedup >= 0  # Just ensure calculation doesn't fail

    @pytest.mark.gpu
    def test_fp16_vs_fp32_speedup(self, has_gpu, sample_frame):
        """Compare FP16 vs FP32 inference speed."""
        if not has_gpu:
            pytest.skip("GPU required for FP16")

        import face_detection

        # FP32 detector
        detector_fp32 = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=False,
        )

        # FP16 detector
        detector_fp16 = face_detection.build_detector(
            "RetinaNetMobileNetV1",
            confidence_threshold=0.5,
            device="cuda",
            fp16_inference=True,
        )

        # Warm-up
        detector_fp32.detect(sample_frame)
        detector_fp16.detect(sample_frame)

        # Time FP32
        start = time.time()
        for _ in range(5):
            detector_fp32.detect(sample_frame)
        fp32_time = (time.time() - start) / 5

        # Time FP16
        start = time.time()
        for _ in range(5):
            detector_fp16.detect(sample_frame)
        fp16_time = (time.time() - start) / 5

        speedup = fp32_time / fp16_time if fp16_time > 0 else 0

        # FP16 should generally be faster (speedup > 1)
        # This is informational
        assert speedup >= 0
