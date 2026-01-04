"""Tests for OffsetVolatilityTracker - generic offset volatility detection.

Ensures large offset reversals are blocked within the compressor min cycle time (45 min).
This is applied at coordinator level and catches jumpy offsets from ANY layer.
"""

import time
from unittest.mock import patch

import pytest

from custom_components.effektguard.const import VOLATILE_MIN_DURATION_MINUTES
from custom_components.effektguard.utils.volatile_helpers import (
    OffsetVolatilityTracker,
    SECONDS_PER_MINUTE,
)


class TestOffsetVolatilityTrackerBasics:
    """Test basic tracker functionality."""

    def test_init_default_duration(self):
        """Test default min duration uses VOLATILE_MIN_DURATION_MINUTES."""
        tracker = OffsetVolatilityTracker()
        assert tracker._min_duration_minutes == VOLATILE_MIN_DURATION_MINUTES
        assert tracker._min_duration_minutes == 45  # Sanity check

    def test_init_custom_duration(self):
        """Test custom min duration."""
        tracker = OffsetVolatilityTracker(min_duration_minutes=30)
        assert tracker._min_duration_minutes == 30

    def test_last_offset_none_initially(self):
        """Test last_offset is None before any changes."""
        tracker = OffsetVolatilityTracker()
        assert tracker.last_offset is None

    def test_record_change_updates_last_offset(self):
        """Test recording a change updates last_offset."""
        tracker = OffsetVolatilityTracker()
        tracker.record_change(offset=-3.0, reason="test")
        assert tracker.last_offset == -3.0


class TestOffsetVolatilityTrackerReversal:
    """Test reversal detection logic."""

    def test_first_change_not_volatile(self):
        """First change is never volatile (nothing to reverse)."""
        tracker = OffsetVolatilityTracker()
        assert tracker.is_reversal_volatile(-5.0) is False

    def test_small_change_not_volatile(self):
        """Changes < 2°C are not considered volatile."""
        tracker = OffsetVolatilityTracker()
        tracker.record_change(offset=-3.0, reason="initial")

        # 1.5°C change - below threshold
        assert tracker.is_reversal_volatile(-1.5) is False

    def test_large_change_same_direction_not_volatile(self):
        """Large changes in same direction are not reversals."""
        tracker = OffsetVolatilityTracker()
        tracker.record_change(offset=-2.0, reason="initial")

        # Going more negative is same direction
        assert tracker.is_reversal_volatile(-5.0) is False

    def test_large_reversal_within_window_is_volatile(self):
        """Large reversal within min duration is volatile."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            # Record initial change at t=0
            mock_time.return_value = 0
            tracker.record_change(offset=-5.0, reason="initial boost")

            # Check reversal at t=30min (within 45 min window)
            mock_time.return_value = 30 * SECONDS_PER_MINUTE
            # -5.0 → +1.0 is a reversal crossing zero, >2°C change
            assert tracker.is_reversal_volatile(1.0) is True

    def test_large_reversal_after_window_not_volatile(self):
        """Large reversal after min duration is allowed."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            # Record initial change at t=0
            mock_time.return_value = 0
            tracker.record_change(offset=-5.0, reason="initial boost")

            # Check reversal at t=50min (after 45 min window)
            mock_time.return_value = 50 * SECONDS_PER_MINUTE
            # -5.0 → +1.0 is a reversal, but enough time has passed
            assert tracker.is_reversal_volatile(1.0) is False

    def test_reversal_negative_to_positive(self):
        """Test reversal from negative to positive offset."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            tracker.record_change(offset=-4.0, reason="cooling")

            mock_time.return_value = 20 * SECONDS_PER_MINUTE
            # -4.0 → +1.0 crosses zero = reversal
            assert tracker.is_reversal_volatile(1.0) is True

    def test_reversal_positive_to_negative(self):
        """Test reversal from positive to negative offset."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            tracker.record_change(offset=3.0, reason="heating")

            mock_time.return_value = 20 * SECONDS_PER_MINUTE
            # +3.0 → -2.0 crosses zero = reversal
            assert tracker.is_reversal_volatile(-2.0) is True


class TestOffsetVolatilityTrackerReasonString:
    """Test reason string generation for logging."""

    def test_get_volatile_reason_no_previous(self):
        """Test reason is empty when no previous change."""
        tracker = OffsetVolatilityTracker()
        assert tracker.get_volatile_reason(1.0) == ""

    def test_get_volatile_reason_format(self):
        """Test reason string format."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            tracker.record_change(offset=-5.0, reason="initial")

            mock_time.return_value = 20 * SECONDS_PER_MINUTE
            reason = tracker.get_volatile_reason(1.0)

            assert "Offset volatile" in reason
            assert "-5.0→1.0" in reason
            assert "20min < 45min" in reason


class TestOffsetVolatilityTrackerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exactly_at_threshold_not_volatile(self):
        """Change exactly at 2°C threshold is not volatile."""
        tracker = OffsetVolatilityTracker()
        tracker.record_change(offset=0.0, reason="initial")

        # Exactly 2°C change
        assert tracker.is_reversal_volatile(2.0) is False

    def test_just_above_threshold_is_volatile(self):
        """Change just above 2°C threshold can be volatile."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            tracker.record_change(offset=-2.1, reason="initial")

            mock_time.return_value = 10 * SECONDS_PER_MINUTE
            # -2.1 → +0.1 = 2.2°C change, crosses zero
            assert tracker.is_reversal_volatile(0.1) is True

    def test_exactly_at_duration_boundary(self):
        """Change exactly at min duration boundary is not volatile."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            mock_time.return_value = 0
            tracker.record_change(offset=-5.0, reason="initial")

            # Exactly at 45 min boundary - should NOT be volatile
            mock_time.return_value = 45 * SECONDS_PER_MINUTE
            assert tracker.is_reversal_volatile(1.0) is False

    def test_multiple_changes_resets_window(self):
        """Each recorded change resets the time window."""
        tracker = OffsetVolatilityTracker()

        with patch("time.time") as mock_time:
            # First change at t=0
            mock_time.return_value = 0
            tracker.record_change(offset=-5.0, reason="first")

            # Second change at t=30min (allowed, same direction)
            mock_time.return_value = 30 * SECONDS_PER_MINUTE
            assert tracker.is_reversal_volatile(-6.0) is False
            tracker.record_change(offset=-6.0, reason="second")

            # Check reversal at t=50min (20min after second change)
            # Window was reset at second change, so this should be volatile
            mock_time.return_value = 50 * SECONDS_PER_MINUTE
            assert tracker.is_reversal_volatile(1.0) is True

            # Check reversal at t=80min (50min after second change)
            # Now it's outside the window
            mock_time.return_value = 80 * SECONDS_PER_MINUTE
            assert tracker.is_reversal_volatile(1.0) is False
