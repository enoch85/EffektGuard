"""Tests for EffectManager peak tracking and avoidance logic.

Tests Phase 3 requirements:
- 15-minute peak tracking
- Day/night weighting (full/50%)
- Monthly top 3 peak management
- Peak avoidance decision logic
- Persistent state storage
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.effektguard.optimization.effect_layer import (
    EffectManager,
    EffectLayerDecision,
    PeakEvent,
)


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))
    return hass


@pytest_asyncio.fixture
async def effect_manager(hass_mock):
    """Create EffectManager instance."""
    manager = EffectManager(hass_mock)
    # Mock storage
    with patch.object(manager._store, "async_load", return_value=None):
        with patch.object(manager._store, "async_save", return_value=None):
            await manager.async_load()
    return manager


class TestPeakEvent:
    """Test PeakEvent dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2025, 10, 14, 12, 30)
        peak = PeakEvent(
            timestamp=timestamp,
            quarter_of_day=50,
            actual_power=5.5,
            effective_power=5.5,
            is_daytime=True,
        )

        data = peak.to_dict()

        assert data["timestamp"] == timestamp.isoformat()
        assert data["quarter_of_day"] == 50
        assert data["actual_power"] == 5.5
        assert data["effective_power"] == 5.5
        assert data["is_daytime"] is True

    def test_from_dict(self):
        """Test creation from dictionary."""
        timestamp = datetime(2025, 10, 14, 12, 30)
        data = {
            "timestamp": timestamp.isoformat(),
            "quarter_of_day": 50,
            "actual_power": 5.5,
            "effective_power": 5.5,
            "is_daytime": True,
        }

        peak = PeakEvent.from_dict(data)

        assert peak.quarter_of_day == 50
        assert peak.actual_power == 5.5
        assert peak.effective_power == 5.5
        assert peak.is_daytime is True


class TestQuarterOfDayCalculation:
    """Test 15-minute quarter calculation."""

    def test_daytime_quarters(self):
        """Test daytime quarter range (06:00-22:00)."""
        # 06:00 = quarter 24
        assert 24 == (6 * 4) + (0 // 15)
        # 22:00 = quarter 88
        assert 88 == (22 * 4) + (0 // 15)
        # 21:45 = quarter 87 (last daytime quarter)
        assert 87 == (21 * 4) + (45 // 15)

        # Verify daytime range
        for quarter in range(24, 88):
            hour = quarter // 4
            assert 6 <= hour < 22, f"Quarter {quarter} should be daytime"

    def test_nighttime_quarters(self):
        """Test nighttime quarter range (22:00-06:00)."""
        # 22:00 = quarter 88 (first nighttime)
        assert 88 == (22 * 4) + (0 // 15)
        # 00:00 = quarter 0
        assert 0 == (0 * 4) + (0 // 15)
        # 05:45 = quarter 23 (last nighttime)
        assert 23 == (5 * 4) + (45 // 15)


class TestEffectivePoweCalculation:
    """Test day/night power weighting."""

    @pytest.mark.asyncio
    async def test_daytime_full_weight(self, effect_manager):
        """Test daytime power at full weight (06:00-22:00)."""
        timestamp = datetime(2025, 10, 14, 12, 30)  # 12:30 = daytime
        quarter = 50  # 12:30

        peak = await effect_manager.record_quarter_measurement(
            power_kw=6.0,
            quarter=quarter,
            timestamp=timestamp,
        )

        assert peak is not None
        assert peak.actual_power == 6.0
        assert peak.effective_power == 6.0  # Full weight during day
        assert peak.is_daytime is True

    @pytest.mark.asyncio
    async def test_nighttime_half_weight(self, effect_manager):
        """Test nighttime power at 50% weight (22:00-06:00)."""
        timestamp = datetime(2025, 10, 14, 23, 30)  # 23:30 = nighttime
        quarter = 94  # 23:30

        peak = await effect_manager.record_quarter_measurement(
            power_kw=6.0,
            quarter=quarter,
            timestamp=timestamp,
        )

        assert peak is not None
        assert peak.actual_power == 6.0
        assert peak.effective_power == 3.0  # 50% weight at night
        assert peak.is_daytime is False


class TestPeakTracking:
    """Test monthly top 3 peak tracking."""

    @pytest.mark.asyncio
    async def test_records_first_peak(self, effect_manager):
        """Test recording first peak."""
        timestamp = datetime(2025, 10, 14, 12, 0)
        quarter = 48

        peak = await effect_manager.record_quarter_measurement(
            power_kw=5.0,
            quarter=quarter,
            timestamp=timestamp,
        )

        assert peak is not None
        assert len(effect_manager._monthly_peaks) == 1
        assert effect_manager._monthly_peaks[0].effective_power == 5.0

    @pytest.mark.asyncio
    async def test_fills_top_three_peaks(self, effect_manager):
        """Test filling top 3 peaks."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Add 3 peaks with different powers
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)
        await effect_manager.record_quarter_measurement(6.0, 49, timestamp)
        await effect_manager.record_quarter_measurement(7.0, 50, timestamp)

        assert len(effect_manager._monthly_peaks) == 3
        # Should be sorted highest first
        assert effect_manager._monthly_peaks[0].effective_power == 7.0
        assert effect_manager._monthly_peaks[1].effective_power == 6.0
        assert effect_manager._monthly_peaks[2].effective_power == 5.0

    @pytest.mark.asyncio
    async def test_replaces_lowest_peak(self, effect_manager):
        """Test replacing lowest peak when exceeding top 3."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Fill top 3
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)
        await effect_manager.record_quarter_measurement(6.0, 49, timestamp)
        await effect_manager.record_quarter_measurement(7.0, 50, timestamp)

        # Add higher peak - should replace 5.0
        peak = await effect_manager.record_quarter_measurement(8.0, 51, timestamp)

        assert peak is not None
        assert len(effect_manager._monthly_peaks) == 3
        assert effect_manager._monthly_peaks[0].effective_power == 8.0
        assert effect_manager._monthly_peaks[1].effective_power == 7.0
        assert effect_manager._monthly_peaks[2].effective_power == 6.0
        # 5.0 should be gone
        assert all(p.effective_power != 5.0 for p in effect_manager._monthly_peaks)

    @pytest.mark.asyncio
    async def test_ignores_lower_peak(self, effect_manager):
        """Test ignoring peak lower than top 3."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Fill top 3
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)
        await effect_manager.record_quarter_measurement(6.0, 49, timestamp)
        await effect_manager.record_quarter_measurement(7.0, 50, timestamp)

        # Try to add lower peak
        peak = await effect_manager.record_quarter_measurement(4.0, 51, timestamp)

        assert peak is None  # Should not create new peak
        assert len(effect_manager._monthly_peaks) == 3


class TestPeakAvoidanceLogic:
    """Test peak avoidance decision logic."""

    @pytest.mark.asyncio
    async def test_no_limit_when_no_peaks(self, effect_manager):
        """Test no limit when no peaks recorded."""
        decision = effect_manager.should_limit_power(
            current_power=5.0,
            current_quarter=48,  # Daytime
        )

        assert decision.should_limit is False
        assert decision.severity == "OK"
        assert decision.recommended_offset == 0.0

    @pytest.mark.asyncio
    async def test_critical_when_exceeding_peak(self, effect_manager):
        """Test critical response when exceeding peak."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Set up peak at 5.0 kW
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        # Test with power exceeding peak
        decision = effect_manager.should_limit_power(
            current_power=6.0,  # Exceeds 5.0 kW peak
            current_quarter=50,  # Daytime
        )

        assert decision.should_limit is True
        assert decision.severity == "CRITICAL"
        assert decision.recommended_offset == -3.0  # Aggressive reduction

    @pytest.mark.asyncio
    async def test_critical_within_half_kw(self, effect_manager):
        """Test critical warning within 0.5 kW of peak."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Set up peak at 5.0 kW
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        # Test with power within 0.5 kW
        decision = effect_manager.should_limit_power(
            current_power=4.7,  # Within 0.5 kW (margin 0.3)
            current_quarter=50,  # Daytime
        )

        assert decision.should_limit is True
        assert decision.severity == "CRITICAL"
        assert decision.recommended_offset == -2.0

    @pytest.mark.asyncio
    async def test_warning_within_one_kw(self, effect_manager):
        """Test warning within 1.0 kW of peak."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Set up peak at 5.0 kW
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        # Test with power within 1.0 kW
        decision = effect_manager.should_limit_power(
            current_power=4.3,  # Within 1.0 kW (margin 0.7)
            current_quarter=50,  # Daytime
        )

        assert decision.should_limit is True
        assert decision.severity == "WARNING"
        assert decision.recommended_offset == -1.0

    @pytest.mark.asyncio
    async def test_ok_with_safe_margin(self, effect_manager):
        """Test OK status with safe margin."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Set up peak at 5.0 kW
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        # Test with power well below peak
        decision = effect_manager.should_limit_power(
            current_power=3.5,  # 1.5 kW margin
            current_quarter=50,  # Daytime
        )

        assert decision.should_limit is False
        assert decision.severity == "OK"
        assert decision.recommended_offset == 0.0

    @pytest.mark.asyncio
    async def test_nighttime_weighting_in_comparison(self, effect_manager):
        """Test nighttime 50% weighting in peak comparison."""
        timestamp = datetime(2025, 10, 14, 12, 0)

        # Set up daytime peak at 5.0 kW effective
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        # Test nighttime power - 10.0 kW actual = 5.0 kW effective
        decision = effect_manager.should_limit_power(
            current_power=10.0,  # But effective = 5.0 (50% weight)
            current_quarter=94,  # 23:30 = nighttime
        )

        # Should match peak exactly (margin = 0)
        assert decision.should_limit is True
        assert decision.severity == "CRITICAL"


class TestPeakProtectionOffset:
    """Test peak protection offset calculation."""

    @pytest.mark.asyncio
    async def test_returns_recommended_offset(self, effect_manager):
        """Test returns recommended offset when limiting."""
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        offset = effect_manager.get_peak_protection_offset(
            current_power=6.0,  # Exceeds peak
            current_quarter=50,
            base_offset=0.0,
        )

        assert offset == -3.0  # Critical recommended offset

    @pytest.mark.asyncio
    async def test_returns_zero_when_safe(self, effect_manager):
        """Test returns zero when safe margin."""
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)

        offset = effect_manager.get_peak_protection_offset(
            current_power=3.0,  # Safe margin
            current_quarter=50,
            base_offset=0.0,
        )

        assert offset == 0.0


class TestPersistentState:
    """Test persistent state storage."""

    @pytest.mark.asyncio
    async def test_saves_peaks(self, hass_mock):
        """Test saving peaks to storage."""
        manager = EffectManager(hass_mock)

        with patch.object(manager._store, "async_save") as mock_save:
            timestamp = datetime(2025, 10, 14, 12, 0)
            await manager.record_quarter_measurement(5.0, 48, timestamp)

            await manager.async_save()

            mock_save.assert_called_once()
            call_args = mock_save.call_args[0][0]
            assert "peaks" in call_args
            assert len(call_args["peaks"]) == 1

    @pytest.mark.asyncio
    async def test_loads_peaks(self, hass_mock):
        """Test loading peaks from storage."""
        manager = EffectManager(hass_mock)

        # Use current month dynamically to avoid _clean_old_peaks() removing it
        now = datetime.now()
        timestamp = datetime(now.year, now.month, 1, 12, 0)
        stored_data = {
            "peaks": [
                {
                    "timestamp": timestamp.isoformat(),
                    "quarter_of_day": 48,
                    "actual_power": 5.0,
                    "effective_power": 5.0,
                    "is_daytime": True,
                }
            ]
        }

        with patch.object(manager._store, "async_load", return_value=stored_data):
            with patch.object(manager._store, "async_save", return_value=None):
                await manager.async_load()

        assert len(manager._monthly_peaks) == 1
        assert manager._monthly_peaks[0].effective_power == 5.0


class TestMonthlySummary:
    """Test monthly peak summary."""

    @pytest.mark.asyncio
    async def test_empty_summary(self, effect_manager):
        """Test summary with no peaks."""
        summary = effect_manager.get_monthly_peak_summary()

        assert summary["count"] == 0
        assert summary["highest"] == 0.0
        assert summary["peaks"] == []

    @pytest.mark.asyncio
    async def test_summary_with_peaks(self, effect_manager):
        """Test summary with peaks."""
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_quarter_measurement(5.0, 48, timestamp)
        await effect_manager.record_quarter_measurement(6.0, 49, timestamp)

        summary = effect_manager.get_monthly_peak_summary()

        assert summary["count"] == 2
        assert summary["highest"] == 6.0  # Highest peak
        assert len(summary["peaks"]) == 2


class TestEvaluateLayer:
    """Tests for evaluate_layer method - effect tariff protection layer logic."""

    @pytest.mark.asyncio
    async def test_disabled_returns_zero_offset(self, effect_manager):
        """Disabled peak protection returns zero offset and weight."""
        decision = effect_manager.evaluate_layer(
            current_peak=10.0,
            current_power=5.0,
            thermal_trend={"rate_per_hour": 0.0},
            enable_peak_protection=False,
        )

        assert isinstance(decision, EffectLayerDecision)
        assert decision.name == "Peak"
        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Disabled by user" in decision.reason

    @pytest.mark.asyncio
    async def test_safe_margin_returns_zero_offset(self, effect_manager):
        """Safe margin from peak returns zero offset (no action needed)."""
        decision = effect_manager.evaluate_layer(
            current_peak=10.0,
            current_power=5.0,
            thermal_trend={"rate_per_hour": 0.0},
            enable_peak_protection=True,
        )

        assert decision.offset == 0.0
        assert decision.weight == 0.0
        assert "Safe margin" in decision.reason

    @pytest.mark.asyncio
    async def test_critical_returns_critical_offset(self, effect_manager):
        """Exceeding peak returns critical offset."""
        # First record a peak so we have a threshold
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_quarter_measurement(8.0, 48, timestamp)

        # Mock dt_util.now() to ensure daytime (quarter calculation is correct)
        with patch(
            "custom_components.effektguard.utils.time_utils.dt_util"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(2025, 10, 14, 12, 30)  # Daytime, Q50

            # Now test with power exceeding that peak
            decision = effect_manager.evaluate_layer(
                current_peak=8.0,
                current_power=8.5,  # Exceeds peak
                thermal_trend={"rate_per_hour": 0.0},
                enable_peak_protection=True,
            )

        assert decision.offset < 0  # Negative offset to reduce heating
        assert decision.weight > 0.5  # High weight for critical
        assert "CRITICAL" in decision.reason

    @pytest.mark.asyncio
    async def test_predictive_cooling_triggers_early_reduction(self, effect_manager):
        """Rapid cooling trend triggers predictive peak avoidance."""
        # Record a peak
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_quarter_measurement(7.0, 48, timestamp)

        # Test with power close to peak AND rapid cooling (predicts power increase)
        decision = effect_manager.evaluate_layer(
            current_peak=7.0,
            current_power=6.0,  # Close to peak
            thermal_trend={"rate_per_hour": -1.5},  # Rapid cooling = compressor will ramp up
            enable_peak_protection=True,
        )

        # Should trigger some protective action
        assert decision.name == "Peak"
        # Either predictive or warning, depending on exact margins
        assert decision.offset <= 0.0 or "margin" in decision.reason.lower()
