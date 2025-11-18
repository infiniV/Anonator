import cv2
import numpy as np
import pytest
import time

from anonator.core.processor import VideoProcessor


@pytest.mark.benchmark
@pytest.mark.slow
class TestVideoProcessingThroughput:
    """Benchmark suite for video processing throughput."""

    def test_benchmark_video_processing_cpu(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark video processing FPS on CPU."""
        output_path = temp_output_dir / "benchmark_cpu.mp4"

        def process_video():
            # Force CPU processing
            import torch
            original_device = torch.cuda.is_available
            torch.cuda.is_available = lambda: False

            try:
                processor = VideoProcessor()

                processor.process_video(
                    input_path=str(short_test_video),
                    output_path=str(output_path),
                    anonymization_mode="blur",
                    threshold=0.5,
                    multi_pass=False,
                )

                if processor._processing_thread:
                    processor._processing_thread.join(timeout=60)

            finally:
                torch.cuda.is_available = original_device

        benchmark(process_video)

        assert output_path.exists()

    @pytest.mark.gpu
    def test_benchmark_video_processing_gpu(self, benchmark, has_gpu, short_test_video, temp_output_dir):
        """Benchmark video processing FPS on GPU."""
        if not has_gpu:
            pytest.skip("GPU not available")

        output_path = temp_output_dir / "benchmark_gpu.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="blur",
                threshold=0.5,
                multi_pass=False,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=60)

        benchmark(process_video)

        assert output_path.exists()


@pytest.mark.benchmark
@pytest.mark.slow
class TestAnonymizationModePerformance:
    """Benchmark suite comparing anonymization mode performance."""

    def test_benchmark_blur_mode_throughput(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark blur mode processing throughput."""
        output_path = temp_output_dir / "benchmark_blur.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="blur",
                threshold=0.5,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=60)

        benchmark(process_video)

    def test_benchmark_solid_mode_throughput(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark solid mode processing throughput."""
        output_path = temp_output_dir / "benchmark_solid.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="solid",
                threshold=0.5,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=60)

        benchmark(process_video)

    def test_benchmark_mosaic_mode_throughput(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark mosaic mode processing throughput."""
        output_path = temp_output_dir / "benchmark_mosaic.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="mosaic",
                threshold=0.5,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=60)

        benchmark(process_video)


@pytest.mark.benchmark
@pytest.mark.slow
class TestIOOverheadAnalysis:
    """Benchmark suite for I/O overhead analysis."""

    def test_benchmark_detection_only(self, benchmark, face_detector, sample_frame):
        """Benchmark pure detection time (no I/O)."""
        def detect_only():
            face_detector.confidence_threshold = 0.5
            return face_detector.detect(sample_frame)

        result = benchmark(detect_only)

    def test_benchmark_anonymization_only(self, benchmark, sample_frame, mock_detections_multiple):
        """Benchmark pure anonymization time (no I/O)."""
        from anonator.core.anonymizer import anonymize_frame

        def anonymize_only():
            frame_copy = sample_frame.copy()
            return anonymize_frame(
                frame_copy,
                mock_detections_multiple,
                mask_scale=1.5,
                replacewith="blur"
            )

        result = benchmark(anonymize_only)


@pytest.mark.benchmark
@pytest.mark.slow
class TestMultiPassPerformanceImpact:
    """Benchmark suite for multi-pass performance impact."""

    def test_benchmark_single_pass_throughput(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark single-pass processing throughput."""
        output_path = temp_output_dir / "benchmark_single_pass.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="solid",
                threshold=0.5,
                multi_pass=False,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=60)

        benchmark(process_video)

    def test_benchmark_multi_pass_throughput(self, benchmark, short_test_video, temp_output_dir):
        """Benchmark multi-pass processing throughput (3x slower expected)."""
        output_path = temp_output_dir / "benchmark_multi_pass.mp4"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(short_test_video),
                output_path=str(output_path),
                anonymization_mode="solid",
                threshold=0.5,
                multi_pass=True,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=120)

        benchmark(process_video)


@pytest.mark.benchmark
@pytest.mark.slow
class TestRealWorldPerformance:
    """Benchmark suite for real-world video performance."""

    def test_benchmark_real_video_processing(self, benchmark, testdata_videos_dir, temp_output_dir):
        """Benchmark processing on real test video."""
        test_videos = list(testdata_videos_dir.glob("*.mp4"))

        if not test_videos:
            pytest.skip("No test videos found")

        input_video = test_videos[0]
        output_path = temp_output_dir / f"benchmark_{input_video.name}"

        def process_video():
            processor = VideoProcessor()

            processor.process_video(
                input_path=str(input_video),
                output_path=str(output_path),
                anonymization_mode="blur",
                threshold=0.5,
            )

            if processor._processing_thread:
                processor._processing_thread.join(timeout=300)

        benchmark.pedantic(process_video, iterations=1, rounds=1)

        # Calculate actual FPS
        if output_path.exists():
            cap = cv2.VideoCapture(str(input_video))
            input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # FPS metrics would be in benchmark results


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Benchmark suite for performance regression detection."""

    def test_baseline_frame_processing_time(self, benchmark, face_detector, sample_frame):
        """Establish baseline for single frame processing time."""
        from anonator.core.anonymizer import anonymize_frame

        def process_frame():
            face_detector.confidence_threshold = 0.5
            dets = face_detector.detect(sample_frame)
            frame_copy = sample_frame.copy()
            return anonymize_frame(frame_copy, dets, mask_scale=1.5, replacewith="blur")

        result = benchmark(process_frame)

        # This establishes a baseline that can be compared in future runs
        # pytest-benchmark will warn if performance degrades significantly


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityAnalysis:
    """Benchmark suite for scalability analysis."""

    def test_throughput_vs_resolution(self, benchmark, device):
        """Analyze throughput at different resolutions."""
        import face_detection

        detector = face_detection.build_detector(
            "RetinaNetResNet50",
            confidence_threshold=0.5,
            device=device,
        )

        resolutions = [
            (480, 640),   # 480p
            (720, 1280),  # 720p
            (1080, 1920), # 1080p
        ]

        results = {}

        for height, width in resolutions:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            start = time.time()
            for _ in range(3):
                detector.detect(frame)
            elapsed = time.time() - start

            fps = 3 / elapsed
            results[f"{height}p"] = fps

        # Results stored for analysis
        # Lower resolution should have higher FPS
        assert results["480p"] >= results["1080p"] * 0.5  # At least 50% of speed


class TestMemoryUsageProfile:
    """Memory usage profiling tests."""

    @pytest.mark.slow
    def test_memory_usage_during_processing(self, short_test_video, temp_output_dir):
        """Profile memory usage during video processing."""
        output_path = temp_output_dir / "memory_test.mp4"

        processor = VideoProcessor()

        # Start processing
        processor.process_video(
            input_path=str(short_test_video),
            output_path=str(output_path),
            anonymization_mode="blur",
            threshold=0.5,
        )

        # Wait for completion
        if processor._processing_thread:
            processor._processing_thread.join(timeout=60)

        # Memory profiling would require external tools
        # This test verifies no memory leaks cause failures
        assert output_path.exists()
