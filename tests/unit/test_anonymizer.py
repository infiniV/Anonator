import numpy as np
import cv2
import pytest

from anonator.core.anonymizer import scale_bb, anonymize_frame


class TestScaleBB:
    """Test suite for scale_bb function."""

    def test_scale_bb_no_scaling(self):
        """Test with mask_scale=1.0 (no scaling)."""
        x1, y1, x2, y2 = scale_bb(100, 100, 200, 200, mask_scale=1.0)
        assert x1 == 100
        assert y1 == 100
        assert x2 == 200
        assert y2 == 200

    def test_scale_bb_1_5x_scaling(self):
        """Test with mask_scale=1.5 (50% expansion)."""
        x1, y1, x2, y2 = scale_bb(100, 100, 200, 200, mask_scale=1.5)

        # Original box: 100x100, expansion: 50 pixels on each side
        assert x1 == 50
        assert y1 == 50
        assert x2 == 250
        assert y2 == 250

    def test_scale_bb_2x_scaling(self):
        """Test with mask_scale=2.0 (100% expansion)."""
        x1, y1, x2, y2 = scale_bb(100, 150, 200, 250, mask_scale=2.0)

        # Width: 100, height: 100
        # Expansion: 100 pixels on each side
        assert x1 == 0
        assert y1 == 50
        assert x2 == 300
        assert y2 == 350

    def test_scale_bb_negative_coordinates(self):
        """Test behavior with negative input coordinates."""
        x1, y1, x2, y2 = scale_bb(50, 50, 100, 100, mask_scale=2.0)

        # Should allow negative results (clipping happens in anonymize_frame)
        assert x1 == 0
        assert y1 == 0

    def test_scale_bb_small_box(self):
        """Test with very small bounding box."""
        x1, y1, x2, y2 = scale_bb(100, 100, 110, 110, mask_scale=1.5)

        # 10x10 box, 50% expansion = 5 pixels on each side
        assert x1 == 95
        assert y1 == 95
        assert x2 == 115
        assert y2 == 115

    def test_scale_bb_rectangular_box(self):
        """Test with non-square (rectangular) bounding box."""
        x1, y1, x2, y2 = scale_bb(100, 100, 300, 200, mask_scale=1.3)

        # Width: 200, height: 100
        # s = 0.3, expand width by 60 pixels, height by 30 pixels
        # Note: Floating point precision causes int(100 - 200*0.3) = 39, not 40
        assert x1 == 39
        assert y1 == 70
        assert x2 == 360
        assert y2 == 230


class TestAnonymizeFrameBlurMode:
    """Test suite for anonymize_frame with blur mode."""

    def test_blur_mode_single_face(self, sample_frame, mock_detections_single):
        """Test blur anonymization with single face."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="blur",
            ellipse=False
        )

        # Frame should be modified
        assert not np.array_equal(result, original)

        # Check that blurred region exists
        x1, y1, x2, y2 = 100, 150, 200, 250
        blurred_region = result[y1:y2, x1:x2]
        original_region = original[y1:y2, x1:x2]

        # Blurred region should be different from original
        assert not np.array_equal(blurred_region, original_region)

    def test_blur_mode_with_ellipse(self, sample_frame, mock_detections_single):
        """Test blur with ellipse masking."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="blur",
            ellipse=True
        )

        # Should successfully apply ellipse mask
        assert result is not None
        assert result.shape == sample_frame.shape

    def test_blur_mode_without_ellipse(self, sample_frame, mock_detections_single):
        """Test blur without ellipse masking."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="blur",
            ellipse=False
        )

        assert result is not None
        assert result.shape == sample_frame.shape

    def test_blur_mode_multiple_faces(self, sample_frame, mock_detections_multiple):
        """Test blur with multiple faces."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_multiple,
            mask_scale=1.5,
            replacewith="blur",
            ellipse=True
        )

        # All face regions should be blurred
        assert not np.array_equal(result, original)

        # Check each detection region is modified
        for det in mock_detections_multiple:
            x1, y1, x2, y2 = map(int, det[:4])
            # Apply same scaling as function
            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, 1.5)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(sample_frame.shape[1], x2), min(sample_frame.shape[0], y2)

            region = result[y1:y2, x1:x2]
            original_region = original[y1:y2, x1:x2]

            if region.size > 0:
                assert not np.array_equal(region, original_region)


class TestAnonymizeFrameSolidMode:
    """Test suite for anonymize_frame with solid mode."""

    def test_solid_mode_single_face(self, sample_frame, mock_detections_single):
        """Test solid (blackout) anonymization."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="solid"
        )

        # Check that region is blacked out
        x1, y1, x2, y2 = 100, 150, 200, 250
        blacked_region = result[y1:y2, x1:x2]

        # All pixels should be zero (black)
        assert np.all(blacked_region == 0)

    def test_solid_mode_multiple_faces(self, sample_frame, mock_detections_multiple):
        """Test solid mode with multiple faces."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_multiple,
            mask_scale=1.0,
            replacewith="solid"
        )

        # Check each detection is blacked out
        for det in mock_detections_multiple:
            x1, y1, x2, y2 = map(int, det[:4])
            blacked_region = result[y1:y2, x1:x2]
            assert np.all(blacked_region == 0)

    def test_solid_mode_with_scaling(self, sample_frame, mock_detections_single):
        """Test solid mode with mask scaling."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.5,
            replacewith="solid"
        )

        # Scaled bounding box
        x1, y1, x2, y2 = scale_bb(100, 150, 200, 250, 1.5)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sample_frame.shape[1], x2), min(sample_frame.shape[0], y2)

        blacked_region = result[y1:y2, x1:x2]
        assert np.all(blacked_region == 0)


class TestAnonymizeFrameMosaicMode:
    """Test suite for anonymize_frame with mosaic mode."""

    def test_mosaic_mode_default_size(self, sample_frame, mock_detections_single):
        """Test mosaic anonymization with default size."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="mosaic",
            mosaicsize=20
        )

        # Check that region is pixelated (different from original)
        x1, y1, x2, y2 = 100, 150, 200, 250
        mosaic_region = result[y1:y2, x1:x2]
        original_region = original[y1:y2, x1:x2]

        assert not np.array_equal(mosaic_region, original_region)

    def test_mosaic_mode_small_size(self, sample_frame, mock_detections_single):
        """Test mosaic with small mosaic size (more pixelation)."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="mosaic",
            mosaicsize=5
        )

        assert result is not None
        assert result.shape == sample_frame.shape

    def test_mosaic_mode_large_size(self, sample_frame, mock_detections_single):
        """Test mosaic with large mosaic size (less pixelation)."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="mosaic",
            mosaicsize=50
        )

        assert result is not None
        assert result.shape == sample_frame.shape

    def test_mosaic_mode_multiple_faces(self, sample_frame, mock_detections_multiple):
        """Test mosaic with multiple faces."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_multiple,
            mask_scale=1.0,
            replacewith="mosaic",
            mosaicsize=10
        )

        # All regions should be pixelated
        assert not np.array_equal(result, original)


class TestAnonymizeFrameImgMode:
    """Test suite for anonymize_frame with image replacement mode."""

    def test_img_mode_with_replacement(self, sample_frame, mock_detections_single, replacement_image):
        """Test image replacement mode."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="img",
            replaceimg=replacement_image
        )

        # Region should be replaced
        x1, y1, x2, y2 = 100, 150, 200, 250
        replaced_region = result[y1:y2, x1:x2]
        original_region = original[y1:y2, x1:x2]

        assert not np.array_equal(replaced_region, original_region)

    def test_img_mode_without_replacement_image(self, sample_frame, mock_detections_single):
        """Test image replacement mode without providing replacement image."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="img",
            replaceimg=None
        )

        # Frame should remain unchanged if no replacement image
        assert np.array_equal(result, original)

    def test_img_mode_multiple_faces(self, sample_frame, mock_detections_multiple, replacement_image):
        """Test image replacement with multiple faces."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_multiple,
            mask_scale=1.0,
            replacewith="img",
            replaceimg=replacement_image
        )

        # All regions should be replaced
        assert not np.array_equal(result, original)


class TestAnonymizeFrameEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_detections(self, sample_frame, mock_detections_empty):
        """Test with no detections."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_empty,
            mask_scale=1.0,
            replacewith="blur"
        )

        # Frame should be unchanged
        assert np.array_equal(result, original)

    def test_none_detections(self, sample_frame):
        """Test with None detections."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            None,
            mask_scale=1.0,
            replacewith="blur"
        )

        # Frame should be unchanged
        assert np.array_equal(result, original)

    def test_out_of_bounds_detection(self, sample_frame):
        """Test detection extending beyond frame boundaries."""
        # Detection that extends beyond 960x540 frame
        out_of_bounds_det = np.array([
            [900, 500, 1000, 600, 0.9]  # Extends beyond frame
        ], dtype=np.float32)

        result = anonymize_frame(
            sample_frame,
            out_of_bounds_det,
            mask_scale=1.0,
            replacewith="solid"
        )

        # Should clip to frame boundaries
        assert result is not None
        assert result.shape == sample_frame.shape

        # Verify clipping worked (only in-bounds region is blacked)
        y1, y2 = 500, 540  # Clipped to frame height
        x1, x2 = 900, 960  # Clipped to frame width
        assert np.all(result[y1:y2, x1:x2] == 0)

    def test_zero_width_or_height_box(self, sample_frame):
        """Test detection with zero width or height."""
        zero_size_det = np.array([
            [100, 100, 100, 100, 0.9]  # Zero width and height
        ], dtype=np.float32)

        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            zero_size_det,
            mask_scale=1.0,
            replacewith="blur"
        )

        # Should skip this detection (ROI is empty)
        # Frame should remain largely unchanged
        assert result.shape == sample_frame.shape

    def test_very_small_detection(self, sample_frame):
        """Test very small face detection (< 10 pixels)."""
        small_det = np.array([
            [100, 100, 105, 105, 0.9]  # 5x5 pixels
        ], dtype=np.float32)

        result = anonymize_frame(
            sample_frame,
            small_det,
            mask_scale=1.0,
            replacewith="solid"
        )

        # Should handle gracefully
        assert result is not None
        assert result.shape == sample_frame.shape

    def test_detections_at_frame_edges(self, sample_frame, mock_detections_edge_cases):
        """Test detections at frame boundaries."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_edge_cases,
            mask_scale=1.0,
            replacewith="solid"
        )

        # Should handle edge cases without errors
        assert result is not None
        assert result.shape == sample_frame.shape


class TestAnonymizeFrameWithScores:
    """Test suite for score drawing functionality."""

    def test_draw_scores_enabled(self, sample_frame, mock_detections_single):
        """Test drawing confidence scores on detections."""
        original = sample_frame.copy()
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="blur",
            draw_scores=True
        )

        # Frame should have score text drawn
        assert not np.array_equal(result, original)

    def test_draw_scores_disabled(self, sample_frame, mock_detections_single):
        """Test without drawing scores."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_single,
            mask_scale=1.0,
            replacewith="blur",
            draw_scores=False
        )

        assert result is not None
        assert result.shape == sample_frame.shape

    def test_draw_scores_multiple_faces(self, sample_frame, mock_detections_multiple):
        """Test score drawing with multiple faces."""
        result = anonymize_frame(
            sample_frame,
            mock_detections_multiple,
            mask_scale=1.0,
            replacewith="solid",
            draw_scores=True
        )

        # All scores should be drawn
        assert result is not None
        assert result.shape == sample_frame.shape


class TestAnonymizeFrameAllModes:
    """Integration tests comparing all anonymization modes."""

    def test_all_modes_produce_different_results(self, sample_frame, mock_detections_single):
        """Test that different modes produce different anonymization results."""
        results = {}

        modes = ["blur", "solid", "mosaic", "img"]
        for mode in modes:
            frame_copy = sample_frame.copy()
            if mode == "img":
                replaceimg = np.ones((100, 100, 3), dtype=np.uint8) * 128
                result = anonymize_frame(
                    frame_copy,
                    mock_detections_single,
                    mask_scale=1.0,
                    replacewith=mode,
                    replaceimg=replaceimg
                )
            else:
                result = anonymize_frame(
                    frame_copy,
                    mock_detections_single,
                    mask_scale=1.0,
                    replacewith=mode
                )
            results[mode] = result

        # All modes should produce different results
        assert not np.array_equal(results["blur"], results["solid"])
        assert not np.array_equal(results["blur"], results["mosaic"])
        assert not np.array_equal(results["solid"], results["mosaic"])

    def test_all_modes_preserve_frame_shape(self, sample_frame, mock_detections_multiple):
        """Test that all modes preserve frame dimensions."""
        modes = ["blur", "solid", "mosaic"]

        for mode in modes:
            frame_copy = sample_frame.copy()
            result = anonymize_frame(
                frame_copy,
                mock_detections_multiple,
                mask_scale=1.5,
                replacewith=mode
            )

            assert result.shape == sample_frame.shape
            assert result.dtype == sample_frame.dtype
