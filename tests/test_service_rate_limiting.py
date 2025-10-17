"""Tests for service rate limiting functionality.

Verifies that boost_heating and force_offset services respect cooldown periods
and prevent excessive calls that could harm heat pump operation.
"""

from datetime import datetime, timedelta

import pytest
from freezegun import freeze_time

# Import the module to access private functions for testing
import custom_components.effektguard as effektguard_module
from custom_components.effektguard.const import (
    BOOST_COOLDOWN_MINUTES,
    DOMAIN,
    SERVICE_RATE_LIMIT_MINUTES,
)


@pytest.fixture
def clear_service_timestamps():
    """Clear service call timestamps before each test."""
    effektguard_module._service_last_called.clear()
    yield
    effektguard_module._service_last_called.clear()


class TestServiceCooldownHelpers:
    """Test the helper functions for service cooldown management."""

    def test_check_service_cooldown_first_call(self, clear_service_timestamps):
        """Test that first service call is always allowed."""
        is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

        assert is_allowed is True
        assert remaining == 0

    def test_check_service_cooldown_within_period(self, clear_service_timestamps):
        """Test that service is blocked during cooldown period."""
        # First call - updates timestamp
        effektguard_module._update_service_timestamp("test_service")

        # Immediate second call - should be blocked
        is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

        assert is_allowed is False
        assert 0 < remaining <= 300  # Should be close to 5 minutes (300 seconds)

    def test_check_service_cooldown_after_period(self, clear_service_timestamps):
        """Test that service is allowed after cooldown expires."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            # First call
            effektguard_module._update_service_timestamp("test_service")

            # Advance time past cooldown
            frozen_time.move_to(base_time + timedelta(minutes=6))

            # Should be allowed now
            is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

            assert is_allowed is True
            assert remaining == 0

    def test_check_service_cooldown_exact_expiry(self, clear_service_timestamps):
        """Test cooldown exactly at expiry time."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            effektguard_module._update_service_timestamp("test_service")

            # Advance exactly to cooldown expiry
            frozen_time.move_to(base_time + timedelta(minutes=5))

            is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

            assert is_allowed is True
            assert remaining == 0

    def test_check_service_cooldown_multiple_services(self, clear_service_timestamps):
        """Test that different services have independent cooldowns."""
        effektguard_module._update_service_timestamp("service_a")
        effektguard_module._update_service_timestamp("service_b")

        # Both should be blocked
        is_allowed_a, _ = effektguard_module._check_service_cooldown("service_a", 5)
        is_allowed_b, _ = effektguard_module._check_service_cooldown("service_b", 5)

        assert is_allowed_a is False
        assert is_allowed_b is False

    def test_update_service_timestamp_creates_entry(self, clear_service_timestamps):
        """Test that updating timestamp creates correct entry."""
        service_name = "test_service"

        effektguard_module._update_service_timestamp(service_name)

        service_key = f"{DOMAIN}_{service_name}"
        assert service_key in effektguard_module._service_last_called
        assert isinstance(effektguard_module._service_last_called[service_key], datetime)

    def test_remaining_time_calculation(self, clear_service_timestamps):
        """Test that remaining time is calculated correctly."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            effektguard_module._update_service_timestamp("test_service")

            # Move forward 2 minutes
            frozen_time.move_to(base_time + timedelta(minutes=2))

            # With 5-minute cooldown, should have 3 minutes (180 seconds) remaining
            is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

            assert is_allowed is False
            assert 179 <= remaining <= 181  # Allow 1 second tolerance


class TestBoostHeatingCooldown:
    """Test boost_heating specific cooldown behavior."""

    def test_boost_heating_cooldown_duration(self, clear_service_timestamps):
        """Test that boost_heating uses correct cooldown duration."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time):
            effektguard_module._update_service_timestamp("boost_heating")

            # Check remaining time immediately
            is_allowed, remaining = effektguard_module._check_service_cooldown(
                "boost_heating", BOOST_COOLDOWN_MINUTES
            )

            assert is_allowed is False
            # Should be approximately 45 minutes (2700 seconds)
            assert 2695 <= remaining <= 2705

    def test_boost_heating_remaining_minutes(self, clear_service_timestamps):
        """Test boost_heating remaining time calculation in minutes."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            effektguard_module._update_service_timestamp("boost_heating")

            # Move forward 10 minutes
            frozen_time.move_to(base_time + timedelta(minutes=10))

            is_allowed, remaining = effektguard_module._check_service_cooldown(
                "boost_heating", BOOST_COOLDOWN_MINUTES
            )

            # Should have ~35 minutes remaining
            assert is_allowed is False
            remaining_minutes = remaining / 60
            assert 34 <= remaining_minutes <= 36


class TestForceOffsetCooldown:
    """Test force_offset specific cooldown behavior."""

    def test_force_offset_cooldown_duration(self, clear_service_timestamps):
        """Test that force_offset uses correct cooldown duration."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time):
            effektguard_module._update_service_timestamp("force_offset")

            is_allowed, remaining = effektguard_module._check_service_cooldown(
                "force_offset", SERVICE_RATE_LIMIT_MINUTES
            )

            assert is_allowed is False
            # Should be approximately 5 minutes (300 seconds)
            assert 295 <= remaining <= 305

    def test_force_offset_remaining_seconds(self, clear_service_timestamps):
        """Test force_offset remaining time in seconds."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            effektguard_module._update_service_timestamp("force_offset")

            # Move forward 2 minutes
            frozen_time.move_to(base_time + timedelta(minutes=2))

            is_allowed, remaining = effektguard_module._check_service_cooldown(
                "force_offset", SERVICE_RATE_LIMIT_MINUTES
            )

            # Should have ~180 seconds remaining
            assert is_allowed is False
            assert 179 <= remaining <= 181


class TestServiceIndependence:
    """Test that different services have independent cooldowns."""

    def test_different_services_independent_cooldowns(self, clear_service_timestamps):
        """Test that boost_heating and force_offset have independent cooldowns."""
        # Update timestamps for both services
        effektguard_module._update_service_timestamp("boost_heating")
        effektguard_module._update_service_timestamp("force_offset")

        # Both should have timestamps
        assert f"{DOMAIN}_boost_heating" in effektguard_module._service_last_called
        assert f"{DOMAIN}_force_offset" in effektguard_module._service_last_called

        # Both should be in cooldown
        boost_allowed, _ = effektguard_module._check_service_cooldown(
            "boost_heating", BOOST_COOLDOWN_MINUTES
        )
        force_allowed, _ = effektguard_module._check_service_cooldown(
            "force_offset", SERVICE_RATE_LIMIT_MINUTES
        )

        assert boost_allowed is False
        assert force_allowed is False

    def test_one_service_cooldown_doesnt_affect_another(self, clear_service_timestamps):
        """Test that one service's cooldown doesn't interfere with another."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            # Update boost_heating only
            effektguard_module._update_service_timestamp("boost_heating")

            # Move forward 10 minutes (past force_offset cooldown, within boost cooldown)
            frozen_time.move_to(base_time + timedelta(minutes=10))

            # boost_heating should still be blocked
            boost_allowed, _ = effektguard_module._check_service_cooldown(
                "boost_heating", BOOST_COOLDOWN_MINUTES
            )

            # force_offset should be allowed (never called)
            force_allowed, _ = effektguard_module._check_service_cooldown(
                "force_offset", SERVICE_RATE_LIMIT_MINUTES
            )

            assert boost_allowed is False
            assert force_allowed is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_check_cooldown_zero_minutes(self, clear_service_timestamps):
        """Test cooldown with zero minutes (immediate expiry)."""
        effektguard_module._update_service_timestamp("test_service")

        # Zero cooldown should always allow
        is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 0)

        assert is_allowed is True
        assert remaining == 0

    def test_check_cooldown_very_long_period(self, clear_service_timestamps):
        """Test cooldown with very long period (24 hours)."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time):
            effektguard_module._update_service_timestamp("test_service")

            is_allowed, remaining = effektguard_module._check_service_cooldown(
                "test_service", 1440
            )  # 24 hours

            assert is_allowed is False
            # Should be approximately 24 hours (86400 seconds)
            assert 86395 <= remaining <= 86405

    def test_multiple_timestamp_updates(self, clear_service_timestamps):
        """Test that updating timestamp multiple times uses latest."""
        base_time = datetime(2025, 10, 14, 12, 0, 0)

        with freeze_time(base_time) as frozen_time:
            # First update
            effektguard_module._update_service_timestamp("test_service")

            # Advance time
            frozen_time.move_to(base_time + timedelta(minutes=3))

            # Second update (resets cooldown)
            effektguard_module._update_service_timestamp("test_service")

            # Check cooldown from second update
            is_allowed, remaining = effektguard_module._check_service_cooldown("test_service", 5)

            assert is_allowed is False
            # Should have ~5 minutes remaining from second update
            assert 295 <= remaining <= 305

    def test_service_name_sanitization(self, clear_service_timestamps):
        """Test that service names are properly prefixed."""
        service_name = "my_service"
        effektguard_module._update_service_timestamp(service_name)

        # Verify the key uses domain prefix
        expected_key = f"{DOMAIN}_{service_name}"
        assert expected_key in effektguard_module._service_last_called


@pytest.mark.parametrize(
    "service_name,cooldown_minutes,expected_cooldown",
    [
        ("boost_heating", BOOST_COOLDOWN_MINUTES, 45),
        ("force_offset", SERVICE_RATE_LIMIT_MINUTES, 5),
    ],
)
def test_correct_cooldown_constants(
    service_name, cooldown_minutes, expected_cooldown, clear_service_timestamps
):
    """Test that services use correct cooldown constants."""
    assert cooldown_minutes == expected_cooldown

    base_time = datetime(2025, 10, 14, 12, 0, 0)

    with freeze_time(base_time):
        effektguard_module._update_service_timestamp(service_name)

        is_allowed, remaining = effektguard_module._check_service_cooldown(
            service_name, cooldown_minutes
        )

        assert is_allowed is False
        expected_seconds = expected_cooldown * 60
        assert expected_seconds - 5 <= remaining <= expected_seconds + 5
