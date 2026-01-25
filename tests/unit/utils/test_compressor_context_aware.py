"""Tests for context-aware compressor Hz monitoring (DHW vs space heating).

Validates that compressor monitoring differentiates between:
- DHW heating (50°C target): 80 Hz normal, 95 Hz elevated
- Space heating (25-35°C): 80 Hz elevated, 95 Hz high

Test Categories:
1. DHW mode logging (80 Hz = DEBUG, 95 Hz = INFO)
2. Space mode logging (80 Hz = INFO, 95 Hz = WARNING)
3. Critical threshold (100 Hz = WARNING for both modes)
4. Sustained duration warnings (context-aware)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from custom_components.effektguard.utils.compressor_monitor import (
    CompressorHealthMonitor,
)


@pytest.fixture
def monitor():
    """Create a compressor health monitor."""
    return CompressorHealthMonitor(max_history_hours=24)


class TestDHWModeLogging:
    """Test DHW mode logging (80 Hz normal, 95+ Hz elevated)."""

    def test_dhw_80hz_debug_log(self, monitor):
        """DHW at 80 Hz should log at DEBUG level (normal operation)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=80, heating_mode="dhw")

            # Should call debug, not info/warning
            mock_logger.debug.assert_called_once()
            assert "DHW heating" in str(mock_logger.debug.call_args)
            assert "normal operation" in str(mock_logger.debug.call_args)

            # Should NOT call info or warning
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()

    def test_dhw_85hz_debug_log(self, monitor):
        """DHW at 85 Hz should log at DEBUG level (still normal)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=85, heating_mode="dhw")

            mock_logger.debug.assert_called_once()
            assert "DHW heating" in str(mock_logger.debug.call_args)

            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()

    def test_dhw_95hz_debug_log(self, monitor):
        """DHW at 95 Hz should log at DEBUG level (elevated but acceptable)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=95, heating_mode="dhw")

            mock_logger.debug.assert_called_once()
            assert "DHW heating" in str(mock_logger.debug.call_args)

            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()

    def test_dhw_99hz_debug_log(self, monitor):
        """DHW at 99 Hz should log at DEBUG level (high but below critical)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=99, heating_mode="dhw")

            mock_logger.debug.assert_called_once()
            mock_logger.warning.assert_not_called()


class TestSpaceHeatingModeLogging:
    """Test space heating mode logging (80+ Hz elevated)."""

    def test_space_80hz_info_log(self, monitor):
        """Space heating at 80 Hz should log at INFO level (elevated)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=80, heating_mode="space")

            # Should call info (not debug)
            mock_logger.info.assert_called_once()
            assert "space heating" in str(mock_logger.info.call_args)

            # Should NOT call debug (would be for DHW)
            mock_logger.debug.assert_not_called()

    def test_space_90hz_info_log(self, monitor):
        """Space heating at 90 Hz should log at INFO level (elevated)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=90, heating_mode="space")

            mock_logger.info.assert_called_once()
            assert "space heating" in str(mock_logger.info.call_args)

    def test_space_default_mode(self, monitor):
        """No heating_mode specified should default to space heating behavior."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            # Don't pass heating_mode (should default to "space")
            monitor.update(hz=80)

            # Should behave like space heating (INFO log)
            mock_logger.info.assert_called_once()
            assert "space heating" in str(mock_logger.info.call_args)


class TestCriticalThreshold:
    """Test critical threshold (100 Hz = WARNING for both modes)."""

    def test_dhw_100hz_warning(self, monitor):
        """DHW at 100 Hz should trigger WARNING (critical for any mode)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=100, heating_mode="dhw")

            # Should call warning (critical threshold)
            mock_logger.warning.assert_called_once()
            call_str = str(mock_logger.warning.call_args)
            assert "100" in call_str  # Hz value
            assert "dhw" in call_str  # Mode

    def test_space_100hz_warning(self, monitor):
        """Space heating at 100 Hz should trigger WARNING (critical)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=100, heating_mode="space")

            mock_logger.warning.assert_called_once()
            call_str = str(mock_logger.warning.call_args)
            assert "100" in call_str  # Hz value
            assert "space" in call_str  # Mode

    def test_both_modes_110hz_warning(self, monitor):
        """110 Hz should warn for both DHW and space heating."""
        for mode in ["dhw", "space"]:
            # Create a fresh monitor for each mode to reset state
            from custom_components.effektguard.utils.compressor_monitor import (
                CompressorHealthMonitor,
            )

            fresh_monitor = CompressorHealthMonitor()

            with patch(
                "custom_components.effektguard.utils.compressor_monitor._LOGGER"
            ) as mock_logger:
                fresh_monitor.update(hz=110, heating_mode=mode)

                mock_logger.warning.assert_called()
                call_str = str(mock_logger.warning.call_args)
                assert "110" in call_str  # Hz value


class TestSustainedOperationWarnings:
    """Test sustained operation warnings (context-aware durations)."""

    def test_dhw_30min_above_95hz_warns(self, monitor):
        """DHW >95 Hz for >30 minutes should log at INFO (elevated for DHW cycle)."""
        base_time = datetime(2025, 10, 23, 12, 0)

        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            # Simulate 31 minutes at 95 Hz
            for i in range(32):  # 0-31 minutes
                timestamp = base_time + timedelta(minutes=i)
                monitor.update(hz=95, timestamp=timestamp, heating_mode="dhw")

            # Should have info about sustained DHW operation (changed from warning to info)
            info_calls = [
                str(call) for call in mock_logger.info.call_args_list if "DHW" in str(call)
            ]
            assert len(info_calls) > 0

    def test_space_2hours_above_80hz_warns(self, monitor):
        """Space heating >80 Hz for >2 hours should log at INFO (sustained operation)."""
        base_time = datetime(2025, 10, 23, 12, 0)

        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            # Simulate 121 minutes (2 hours 1 minute) at 80 Hz
            for i in range(0, 122, 5):  # Every 5 minutes
                timestamp = base_time + timedelta(minutes=i)
                monitor.update(hz=80, timestamp=timestamp, heating_mode="space")

            # Should have info about sustained space heating operation (changed from warning to info)
            info_calls = [
                str(call)
                for call in mock_logger.info.call_args_list
                if "space" in str(call).lower()
            ]
            assert len(info_calls) > 0

    def test_dhw_25min_no_warning(self, monitor):
        """DHW at 95 Hz for 25 minutes (under 30min) should not warn."""
        base_time = datetime(2025, 10, 23, 12, 0)

        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            # Simulate 25 minutes at 95 Hz
            for i in range(26):
                timestamp = base_time + timedelta(minutes=i)
                monitor.update(hz=95, timestamp=timestamp, heating_mode="dhw")

            # Should NOT have sustained operation warning
            warning_calls = [
                str(call) for call in mock_logger.warning.call_args_list if "elevated" in str(call)
            ]
            assert len(warning_calls) == 0  # Under 30 minute threshold


class TestModeComparison:
    """Test direct comparison between DHW and space heating modes."""

    def test_same_hz_different_logging(self, monitor):
        """Same Hz (80) should log differently for DHW vs space."""
        # DHW at 80 Hz
        dhw_monitor = CompressorHealthMonitor()
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            dhw_monitor.update(hz=80, heating_mode="dhw")
            dhw_debug_called = mock_logger.debug.called
            dhw_info_called = mock_logger.info.called

        # Space at 80 Hz
        space_monitor = CompressorHealthMonitor()
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            space_monitor.update(hz=80, heating_mode="space")
            space_debug_called = mock_logger.debug.called
            space_info_called = mock_logger.info.called

        # DHW should call debug, space should call info
        assert dhw_debug_called
        assert not dhw_info_called
        assert not space_debug_called
        assert space_info_called

    def test_critical_threshold_same_for_both(self, monitor):
        """100 Hz should warn for both DHW and space (no favoritism)."""
        dhw_warned = False
        space_warned = False

        # DHW at 100 Hz
        dhw_monitor = CompressorHealthMonitor()
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            dhw_monitor.update(hz=100, heating_mode="dhw")
            dhw_warned = mock_logger.warning.called

        # Space at 100 Hz
        space_monitor = CompressorHealthMonitor()
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            space_monitor.update(hz=100, heating_mode="space")
            space_warned = mock_logger.warning.called

        # Both should warn
        assert dhw_warned
        assert space_warned


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_mode_defaults_to_space(self, monitor):
        """Empty string heating_mode should default to space heating."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=80, heating_mode="")

            # Should behave like space heating (INFO log)
            mock_logger.info.assert_called_once()

    def test_unknown_mode_defaults_to_space(self, monitor):
        """Unknown heating_mode should default to space heating behavior."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=80, heating_mode="unknown_mode")

            # Should behave like space heating (INFO log)
            mock_logger.info.assert_called_once()

    def test_case_insensitive_mode(self, monitor):
        """Heating mode should handle different cases (DHW, Dhw, dhw)."""
        for mode_variant in ["dhw", "DHW", "Dhw"]:
            test_monitor = CompressorHealthMonitor()
            with patch(
                "custom_components.effektguard.utils.compressor_monitor._LOGGER"
            ) as mock_logger:
                test_monitor.update(hz=80, heating_mode=mode_variant)

                # Should behave as DHW (debug log)
                if mode_variant.lower() == "dhw":
                    mock_logger.debug.assert_called_once()

    def test_79hz_no_logging(self, monitor):
        """79 Hz should not trigger any elevated logging (both modes)."""
        for mode in ["dhw", "space"]:
            test_monitor = CompressorHealthMonitor()
            with patch(
                "custom_components.effektguard.utils.compressor_monitor._LOGGER"
            ) as mock_logger:
                test_monitor.update(hz=79, heating_mode=mode)

                # Should not call info or warning (below 80 Hz threshold)
                mock_logger.info.assert_not_called()
                mock_logger.warning.assert_not_called()

    def test_101hz_still_warns(self, monitor):
        """101 Hz should warn (just above critical threshold)."""
        with patch("custom_components.effektguard.utils.compressor_monitor._LOGGER") as mock_logger:
            monitor.update(hz=101, heating_mode="space")

            mock_logger.warning.assert_called_once()
            assert "101" in str(mock_logger.warning.call_args)  # Hz value
