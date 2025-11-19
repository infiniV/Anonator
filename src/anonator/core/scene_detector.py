"""
Scene change detection for video processing.

Detects scene cuts and transitions to force re-detection when frame skipping is enabled.
"""

import cv2
import numpy as np
from typing import Optional


class SceneDetector:
    """Detects scene changes using histogram comparison.

    Used to force face re-detection after scene cuts when frame skipping
    is enabled, preventing missed faces in new scenes.
    """

    def __init__(self, threshold: float = 0.3):
        """Initialize scene detector.

        Args:
            threshold: Scene change threshold (0.0-1.0)
                      Lower = more sensitive to changes
                      Typical range: 0.2-0.4
        """
        self.threshold = threshold
        self.previous_hist = None
        self.previous_frame_gray = None

    def is_scene_change(self, frame: np.ndarray) -> bool:
        """Detect if current frame is a scene change.

        Args:
            frame: Current frame (BGR format)

        Returns:
            True if scene change detected, False otherwise
        """
        # Convert to grayscale for histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # First frame is always a "scene change" (force detection)
        if self.previous_hist is None:
            self.previous_hist = hist
            self.previous_frame_gray = gray
            return True

        # Calculate histogram correlation
        correlation = cv2.compareHist(
            self.previous_hist,
            hist,
            cv2.HISTCMP_CORREL
        )

        # Low correlation = scene change
        # Correlation ranges from -1 to 1, where 1 = identical
        is_change = correlation < (1.0 - self.threshold)

        # Update previous histogram
        if is_change:
            self.previous_hist = hist
            self.previous_frame_gray = gray

        return is_change

    def reset(self):
        """Reset detector state (e.g., for new video)."""
        self.previous_hist = None
        self.previous_frame_gray = None

    def get_frame_difference(self, frame: np.ndarray) -> float:
        """Calculate frame difference metric.

        Useful for debugging and threshold tuning.

        Args:
            frame: Current frame (BGR format)

        Returns:
            Difference metric (0.0-1.0)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.previous_frame_gray is None:
            self.previous_frame_gray = gray
            return 1.0

        # Calculate mean absolute difference
        diff = cv2.absdiff(self.previous_frame_gray, gray)
        mean_diff = np.mean(diff) / 255.0

        return float(mean_diff)
