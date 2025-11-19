import pytest

from anonator.core.config import (
    HIPAAConfig,
    StandardConfig,
    ProcessorConfig,
    get_mode_config,
    HIPAA_MODE,
    STANDARD_MODE,
    PROCESSOR_CONFIG,
)


class TestHIPAAConfig:
    """Test suite for HIPAAConfig dataclass."""

    def test_hipaa_config_defaults(self):
        """Test HIPAA config has correct default values."""
        config = HIPAAConfig()

        assert config.anonymization_mode == "solid"
        assert config.detection_threshold == 0.5
        assert config.mask_scale == 1.5
        assert config.multi_pass_enabled is True
        assert config.keep_audio is False
        assert config.lock_ui_controls is True

    def test_hipaa_config_multi_pass_multipliers(self):
        """Test multi-pass multipliers are set correctly in post_init."""
        config = HIPAAConfig()

        assert config.multi_pass_multipliers is not None
        assert config.multi_pass_multipliers == [1.0, 1.5, 0.7]
        assert len(config.multi_pass_multipliers) == 3

    def test_hipaa_config_custom_multipliers(self):
        """Test custom multi-pass multipliers."""
        custom_multipliers = [0.5, 0.75, 0.35]
        config = HIPAAConfig(multi_pass_multipliers=custom_multipliers)

        assert config.multi_pass_multipliers == custom_multipliers

    def test_hipaa_config_custom_values(self):
        """Test HIPAA config with custom values."""
        config = HIPAAConfig(
            anonymization_mode="mosaic",
            detection_threshold=0.3,
            mask_scale=2.0,
            multi_pass_enabled=False,
            keep_audio=True,
            lock_ui_controls=False
        )

        assert config.anonymization_mode == "mosaic"
        assert config.detection_threshold == 0.3
        assert config.mask_scale == 2.0
        assert config.multi_pass_enabled is False
        assert config.keep_audio is True
        assert config.lock_ui_controls is False

    def test_hipaa_config_threshold_range(self):
        """Test HIPAA detection threshold is within expected range."""
        config = HIPAAConfig()

        assert 0.0 <= config.detection_threshold <= 1.0

    def test_hipaa_config_mask_scale_positive(self):
        """Test HIPAA mask scale is positive."""
        config = HIPAAConfig()

        assert config.mask_scale > 0.0


class TestStandardConfig:
    """Test suite for StandardConfig dataclass."""

    def test_standard_config_defaults(self):
        """Test standard config has correct default values."""
        config = StandardConfig()

        assert config.anonymization_mode == "blur"
        assert config.detection_threshold == 0.7
        assert config.mask_scale == 1.5
        assert config.multi_pass_enabled is False
        assert config.keep_audio is True

    def test_standard_config_custom_values(self):
        """Test standard config with custom values."""
        config = StandardConfig(
            anonymization_mode="solid",
            detection_threshold=0.5,
            mask_scale=2.0,
            multi_pass_enabled=True,
            keep_audio=False
        )

        assert config.anonymization_mode == "solid"
        assert config.detection_threshold == 0.5
        assert config.mask_scale == 2.0
        assert config.multi_pass_enabled is True
        assert config.keep_audio is False

    def test_standard_config_threshold_range(self):
        """Test standard detection threshold is within expected range."""
        config = StandardConfig()

        assert 0.0 <= config.detection_threshold <= 1.0

    def test_standard_config_higher_threshold_than_hipaa(self):
        """Test standard mode uses higher threshold than HIPAA (less sensitive)."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        # Standard should be less sensitive (higher threshold)
        assert standard.detection_threshold > hipaa.detection_threshold


class TestProcessorConfig:
    """Test suite for ProcessorConfig dataclass."""

    def test_processor_config_defaults(self):
        """Test processor config has correct default values."""
        config = ProcessorConfig()

        assert config.detector_model == "RetinaNetMobileNetV1"
        assert config.detector_confidence == 0.2
        assert config.nms_iou_threshold == 0.3
        assert config.fp16_inference is True
        assert config.max_resolution == 1920
        assert config.clip_boxes is True
        assert config.preview_interval == 1.0
        assert config.log_interval == 10
        assert config.output_codec == "mp4v"
        assert config.blur_kernel_size == 23
        assert config.blur_sigma == 30
        assert config.mosaic_size == 20
        assert config.use_ellipse_mask is True

    def test_processor_config_custom_values(self):
        """Test processor config with custom values."""
        config = ProcessorConfig(
            detector_model="RetinaNetMobileNetV1",
            detector_confidence=0.5,
            nms_iou_threshold=0.5,
            fp16_inference=False,
            max_resolution=1080,
            clip_boxes=False,
            log_interval=5
        )

        assert config.detector_model == "RetinaNetMobileNetV1"
        assert config.detector_confidence == 0.5
        assert config.nms_iou_threshold == 0.5
        assert config.fp16_inference is False
        assert config.max_resolution == 1080
        assert config.clip_boxes is False
        assert config.log_interval == 5

    def test_processor_config_nms_threshold_range(self):
        """Test NMS IOU threshold is in typical range."""
        config = ProcessorConfig()

        # Typical NMS IOU range is 0.1-0.5
        assert 0.1 <= config.nms_iou_threshold <= 0.5

    def test_processor_config_blur_kernel_odd(self):
        """Test blur kernel size is odd (required for Gaussian blur)."""
        config = ProcessorConfig()

        # Gaussian blur requires odd kernel size
        assert config.blur_kernel_size % 2 == 1

    def test_processor_config_mosaic_size_positive(self):
        """Test mosaic size is positive."""
        config = ProcessorConfig()

        assert config.mosaic_size > 0

    def test_processor_config_max_resolution_reasonable(self):
        """Test max resolution is a reasonable value."""
        config = ProcessorConfig()

        # Should be between 480p and 4K
        assert 480 <= config.max_resolution <= 3840


class TestGetModeConfig:
    """Test suite for get_mode_config function."""

    def test_get_mode_config_hipaa(self):
        """Test get_mode_config returns HIPAAConfig when hipaa_mode=True."""
        config = get_mode_config(hipaa_mode=True)

        assert isinstance(config, HIPAAConfig)
        assert config.anonymization_mode == "solid"
        assert config.detection_threshold == 0.5
        assert config.multi_pass_enabled is True

    def test_get_mode_config_standard(self):
        """Test get_mode_config returns StandardConfig when hipaa_mode=False."""
        config = get_mode_config(hipaa_mode=False)

        assert isinstance(config, StandardConfig)
        assert config.anonymization_mode == "blur"
        assert config.detection_threshold == 0.7
        assert config.multi_pass_enabled is False

    def test_get_mode_config_returns_global_instances(self):
        """Test get_mode_config returns global singleton instances."""
        hipaa1 = get_mode_config(hipaa_mode=True)
        hipaa2 = get_mode_config(hipaa_mode=True)

        standard1 = get_mode_config(hipaa_mode=False)
        standard2 = get_mode_config(hipaa_mode=False)

        # Should return same instance
        assert hipaa1 is hipaa2
        assert standard1 is standard2

        # Different modes should return different instances
        assert hipaa1 is not standard1


class TestGlobalConfigInstances:
    """Test suite for global configuration instances."""

    def test_global_hipaa_mode_exists(self):
        """Test HIPAA_MODE global variable exists."""
        assert HIPAA_MODE is not None
        assert isinstance(HIPAA_MODE, HIPAAConfig)

    def test_global_standard_mode_exists(self):
        """Test STANDARD_MODE global variable exists."""
        assert STANDARD_MODE is not None
        assert isinstance(STANDARD_MODE, StandardConfig)

    def test_global_processor_config_exists(self):
        """Test PROCESSOR_CONFIG global variable exists."""
        assert PROCESSOR_CONFIG is not None
        assert isinstance(PROCESSOR_CONFIG, ProcessorConfig)

    def test_global_configs_have_correct_defaults(self):
        """Test global config instances have correct default values."""
        assert HIPAA_MODE.anonymization_mode == "solid"
        assert STANDARD_MODE.anonymization_mode == "blur"
        assert PROCESSOR_CONFIG.detector_model == "RetinaNetMobileNetV1"


class TestConfigComparison:
    """Test suite comparing HIPAA vs Standard configurations."""

    def test_hipaa_vs_standard_anonymization_mode(self):
        """Test HIPAA uses solid mode while standard uses blur."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        # HIPAA should use irreversible anonymization
        assert hipaa.anonymization_mode == "solid"
        # Standard can use reversible blur
        assert standard.anonymization_mode == "blur"

    def test_hipaa_vs_standard_detection_sensitivity(self):
        """Test HIPAA is more sensitive (lower threshold) than standard."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        # HIPAA should have lower threshold (more sensitive)
        assert hipaa.detection_threshold < standard.detection_threshold

    def test_hipaa_vs_standard_multi_pass(self):
        """Test HIPAA enables multi-pass while standard does not."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        assert hipaa.multi_pass_enabled is True
        assert standard.multi_pass_enabled is False

    def test_hipaa_vs_standard_audio_handling(self):
        """Test HIPAA removes audio while standard keeps it."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        # HIPAA removes audio for PHI compliance
        assert hipaa.keep_audio is False
        # Standard preserves audio
        assert standard.keep_audio is True

    def test_hipaa_vs_standard_mask_scale(self):
        """Test both modes use same mask scale."""
        hipaa = HIPAAConfig()
        standard = StandardConfig()

        # Both should use 1.5x mask scale
        assert hipaa.mask_scale == standard.mask_scale == 1.5


class TestConfigValidation:
    """Test suite for configuration validation (edge cases)."""

    def test_detection_threshold_boundary_values(self):
        """Test detection threshold at boundary values."""
        # Minimum threshold
        config_min = HIPAAConfig(detection_threshold=0.0)
        assert config_min.detection_threshold == 0.0

        # Maximum threshold
        config_max = HIPAAConfig(detection_threshold=1.0)
        assert config_max.detection_threshold == 1.0

    def test_mask_scale_boundary_values(self):
        """Test mask scale at boundary values."""
        # No scaling
        config_no_scale = StandardConfig(mask_scale=1.0)
        assert config_no_scale.mask_scale == 1.0

        # Large scaling
        config_large_scale = StandardConfig(mask_scale=3.0)
        assert config_large_scale.mask_scale == 3.0

    def test_anonymization_mode_values(self):
        """Test different anonymization modes."""
        valid_modes = ["blur", "solid", "mosaic", "img"]

        for mode in valid_modes:
            config = StandardConfig(anonymization_mode=mode)
            assert config.anonymization_mode == mode

    def test_processor_config_log_interval_values(self):
        """Test different log interval values."""
        # Log every frame
        config_every = ProcessorConfig(log_interval=1)
        assert config_every.log_interval == 1

        # Log every 100 frames
        config_sparse = ProcessorConfig(log_interval=100)
        assert config_sparse.log_interval == 100

    def test_multi_pass_multipliers_different_lengths(self):
        """Test multi-pass with different number of passes."""
        # 2 passes
        config_2pass = HIPAAConfig(multi_pass_multipliers=[1.0, 0.5])
        assert len(config_2pass.multi_pass_multipliers) == 2

        # 5 passes
        config_5pass = HIPAAConfig(multi_pass_multipliers=[1.0, 0.8, 0.6, 0.4, 0.2])
        assert len(config_5pass.multi_pass_multipliers) == 5
