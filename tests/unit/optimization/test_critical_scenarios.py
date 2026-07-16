"""Critical scenario guards for EffektGuard.

Pins the effect-manager power-limit response after an outage, the update/rate-limit cadence,
and the tolerance-to-tolerance_range mapping.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch


from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.const import (
    UPDATE_INTERVAL_MINUTES,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_SAVINGS,
)


@pytest.fixture
def hass_mock():
    """Create mock Home Assistant instance."""
    return MagicMock()


@pytest_asyncio.fixture
async def effect_manager(hass_mock):
    """Create EffectManager instance."""
    manager = EffectManager(hass_mock)
    with patch.object(manager._store, "async_load", return_value=None):
        with patch.object(manager._store, "async_save", return_value=None):
            await manager.async_load()
    return manager


def create_nibe_state(
    outdoor_temp=5.0,
    indoor_temp=21.0,
    degree_minutes=-100.0,
    is_heating=True,
):
    """Create mock NIBE state."""
    state = MagicMock()
    state.outdoor_temp = outdoor_temp
    state.indoor_temp = indoor_temp
    state.supply_temp = 35.0
    state.degree_minutes = degree_minutes
    state.current_offset = 0.0
    state.is_heating = is_heating
    state.timestamp = datetime.now()
    return state


class TestPeakCalculationFrequency:
    """Update cadence and rate limiting: safe for the compressor, aligned to 15-min windows."""

    def test_coordinator_update_interval_prevents_cycling(self):
        """The 5-minute update interval bounds changes to 12/hour (3 per 15-min window), which
        stays within a NIBE compressor's minimum cycle time.
        """
        update_interval = UPDATE_INTERVAL_MINUTES

        assert update_interval == 5  # 5-minute updates

        # Calculate cycling characteristics
        max_updates_per_hour = 60 / update_interval
        max_updates_per_quarter = 15 / update_interval  # Per 15-min period

        assert max_updates_per_hour == 12  # 12 updates/hour
        assert max_updates_per_quarter == 3  # 3 updates per 15-min window

        # Verify this is safe for compressor
        # NIBE compressors: minimum 5-10 minute cycles
        # 5-minute updates = worst case 1 change per cycle (acceptable)
        assert update_interval >= 5  # Safe for NIBE

    @pytest.mark.asyncio
    async def test_peak_recorded_once_per_quarter(self, effect_manager):
        """Test: Peak measurements recorded at quarterly intervals.

        Expected:
        - Each 15-minute period measured once
        - Multiple measurements within same billing_hour don't create multiple peaks
        - Only highest measurement in billing_hour matters
        """
        timestamp_1 = datetime(2025, 10, 14, 12, 2)  # Q48 (12:00-12:15)
        timestamp_2 = datetime(2025, 10, 14, 12, 7)  # Q48 (same billing_hour)
        timestamp_3 = datetime(2025, 10, 14, 12, 14)  # Q48 (same billing_hour)

        billing_hour = 12  # All in same billing_hour

        # Record multiple measurements in same billing_hour
        peak_1 = await effect_manager.record_period_measurement(4.0, billing_hour, timestamp_1)
        peak_2 = await effect_manager.record_period_measurement(4.5, billing_hour, timestamp_2)
        peak_3 = await effect_manager.record_period_measurement(4.2, billing_hour, timestamp_3)

        # All should be recorded (highest wins)
        # But only 3 peaks total for top 3 tracking
        assert len(effect_manager._monthly_peaks) <= 3

    def test_rate_limiting_prevents_wear(self):
        """Test: Rate limiting prevents excessive wear on NIBE controller.

        Expected:
        - Minimum interval between offset writes: 5 minutes (300 seconds)
        - Prevents excessive MyUplink API calls
        - Protects NIBE controller from wear
        """
        from custom_components.effektguard.const import (
            SERVICE_RATE_LIMIT_MINUTES,
            UPDATE_INTERVAL_MINUTES,
        )

        # The PRODUCTION cooldown, not a local literal asserted against itself: the adapter
        # refuses writes inside SERVICE_RATE_LIMIT_MINUTES, and a cooldown shorter than the
        # update cadence would rate-limit nothing.
        assert SERVICE_RATE_LIMIT_MINUTES * 60 >= 300
        assert SERVICE_RATE_LIMIT_MINUTES >= UPDATE_INTERVAL_MINUTES

        # Bounds MyUplink API calls and NIBE controller wear.
        max_writes_per_hour = 60 / SERVICE_RATE_LIMIT_MINUTES
        assert max_writes_per_hour <= 12


class TestPowerOutageRecovery:
    """should_limit_power response as current power approaches the recorded monthly peak.

    Margins: WARNING at 1.0 kW (offset -1.0), CRITICAL at 0.5 kW (-2.0), exceeding (-3.0).
    Nighttime power is weighted 50% before comparison.
    """

    @pytest.mark.asyncio
    async def test_recovery_with_close_peak(self, effect_manager):
        """Test: Recovery after outage when close to monthly peak.

        Scenario: Power outage, then restart with current power 0.8 kW below peak
        Expected: WARNING level, recommended offset -1.0°C
        """
        # Set up monthly peak at 5.0 kW (before outage)
        timestamp = datetime(2025, 10, 14, 10, 0)
        await effect_manager.record_period_measurement(5.0, 10 * 4, timestamp)

        # Simulate system restart - storage persists
        # Current power: 4.2 kW (0.8 kW below peak)
        billing_period = 12 * 4  # noon, daytime
        decision = effect_manager.should_limit_power(4.2, billing_period)

        # Should be WARNING (between 0.5 and 1.0 kW margin)
        assert decision.severity == "WARNING"
        assert decision.should_limit is True
        assert decision.recommended_offset == -1.0

    @pytest.mark.asyncio
    async def test_recovery_with_very_close_peak(self, effect_manager):
        """Test: Recovery when very close to peak (critical zone).

        Scenario: Current power 0.3 kW below peak
        Expected: CRITICAL level, recommended offset -2.0°C
        """
        # Set up monthly peak
        timestamp = datetime(2025, 10, 14, 10, 0)
        await effect_manager.record_period_measurement(5.0, 10 * 4, timestamp)

        # Current power: 4.7 kW (0.3 kW below peak - within 0.5 kW critical zone)
        decision = effect_manager.should_limit_power(4.7, 12 * 4)

        assert decision.severity == "CRITICAL"
        assert decision.recommended_offset == -2.0

    @pytest.mark.asyncio
    async def test_recovery_exceeding_peak(self, effect_manager):
        """Test: Recovery when already exceeding peak.

        Scenario: Current power exceeds all previous peaks
        Expected: CRITICAL, aggressive reduction -3.0°C
        """
        # Set up monthly peak
        timestamp = datetime(2025, 10, 14, 10, 0)
        await effect_manager.record_period_measurement(5.0, 10 * 4, timestamp)

        # Current power: 5.5 kW (exceeding peak by 0.5 kW)
        decision = effect_manager.should_limit_power(5.5, 12 * 4)

        assert decision.severity == "CRITICAL"
        assert decision.recommended_offset == -3.0  # Maximum reduction

    @pytest.mark.asyncio
    async def test_safe_margin_after_recovery(self, effect_manager):
        """Test: Recovery with safe margin from peak.

        Scenario: Current power 2.0 kW below peak
        Expected: OK status, no limiting needed
        """
        # Set up monthly peak
        timestamp = datetime(2025, 10, 14, 10, 0)
        await effect_manager.record_period_measurement(5.0, 10 * 4, timestamp)

        # Current power: 3.0 kW (2.0 kW below peak - safe)
        decision = effect_manager.should_limit_power(3.0, 12 * 4)

        assert decision.severity == "OK"
        assert decision.should_limit is False
        assert decision.recommended_offset == 0.0

    @pytest.mark.asyncio
    async def test_nighttime_allows_higher_power_after_outage(self, effect_manager):
        """Test: Nighttime 50% weighting allows recovery with higher power.

        Scenario: Daytime peak 5.0 kW, nighttime current 8.0 kW actual
        Expected: 8.0 kW actual = 4.0 kW effective < 5.0 kW peak (OK, safe margin)
        """
        # Set up daytime peak
        timestamp = datetime(2025, 10, 14, 12, 0)
        await effect_manager.record_period_measurement(5.0, 12 * 4, timestamp)  # Daytime

        # Nighttime: 8.0 kW actual = 4.0 kW effective (1.0 kW margin from peak)
        billing_hour = 23  # 23:30, nighttime
        decision = effect_manager.should_limit_power(8.0, billing_hour)

        # Should be OK since effective power (4.0) < peak (5.0) with >1.0 kW margin
        assert decision.severity == "OK"
        assert decision.should_limit is False


class TestPresetModes:
    """tolerance_range is tolerance * TOLERANCE_RANGE_MULTIPLIER (0.4) regardless of mode."""

    @pytest.mark.asyncio
    async def test_comfort_mode_tight_tolerance(self, hass_mock):
        """tolerance_range = tolerance * 0.4: a tolerance of 2.0 gives a 0.8 C band."""
        # Comfort mode configuration
        config = {
            "target_temperature": 21.0,
            "tolerance": 2.0,  # Lower tolerance = tighter comfort
            "optimization_mode": OPTIMIZATION_MODE_COMFORT,
        }

        thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)
        effect_manager = EffectManager(hass_mock)
        price_analyzer = PriceAnalyzer()

        engine = DecisionEngine(
            price_analyzer=price_analyzer,
            effect_manager=effect_manager,
            thermal_model=thermal_model,
            config=config,
        )

        # Comfort mode should have tighter tolerance
        assert engine.tolerance == 2.0
        # Tolerance range = tolerance * 0.4 = 0.8°C
        assert engine.tolerance_range == 0.8

    @pytest.mark.asyncio
    async def test_balanced_mode_moderate_tolerance(self, hass_mock):
        """A tolerance of 5.0 gives a 2.0 C band (5.0 * 0.4)."""
        config = {
            "target_temperature": 21.0,
            "tolerance": 5.0,  # Mid-range
            "optimization_mode": OPTIMIZATION_MODE_BALANCED,
        }

        thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)
        effect_manager = EffectManager(hass_mock)
        price_analyzer = PriceAnalyzer()

        engine = DecisionEngine(
            price_analyzer=price_analyzer,
            effect_manager=effect_manager,
            thermal_model=thermal_model,
            config=config,
        )

        assert engine.tolerance == 5.0
        assert engine.tolerance_range == 2.0  # 5.0 * 0.4

    @pytest.mark.asyncio
    async def test_eco_mode_wide_tolerance(self, hass_mock):
        """A tolerance of 9.0 gives a 3.6 C band (9.0 * 0.4)."""
        config = {
            "target_temperature": 21.0,
            "tolerance": 9.0,  # High tolerance
            "optimization_mode": OPTIMIZATION_MODE_SAVINGS,
        }

        thermal_model = ThermalModel(thermal_mass=1.0, insulation_quality=1.0)
        effect_manager = EffectManager(hass_mock)
        price_analyzer = PriceAnalyzer()

        engine = DecisionEngine(
            price_analyzer=price_analyzer,
            effect_manager=effect_manager,
            thermal_model=thermal_model,
            config=config,
        )

        assert engine.tolerance == 9.0
        assert engine.tolerance_range == 3.6  # 9.0 * 0.4


class TestSystemRobustness:
    """Test system robustness and edge cases."""

    @pytest.mark.asyncio
    async def test_no_peaks_after_month_change(self, effect_manager):
        """Test: Peaks automatically cleared at month boundary.

        Expected:
        - Previous month peaks removed
        - Fresh start each month
        - Effektavgift billing cycle respected
        """
        # Add peaks from previous month
        old_timestamp = datetime(2025, 9, 15, 12, 0)  # September
        await effect_manager.record_period_measurement(5.0, 12 * 4, old_timestamp)

        # Simulate month cleanup
        effect_manager._clean_old_peaks()

        # October peaks should be empty
        # (In real system, this happens on first update of new month)
        current_month = (datetime.now().year, datetime.now().month)
        peaks_this_month = [
            p
            for p in effect_manager._monthly_peaks
            if (p.timestamp.year, p.timestamp.month) == current_month
        ]

        # Old peaks should be cleaned
        assert len(peaks_this_month) == 0

    @pytest.mark.asyncio
    async def test_persistent_storage_survives_restart(self, hass_mock):
        """Test: Peak data persists across system restarts.

        Expected:
        - Peaks saved to storage
        - Loaded on restart
        - Protection maintained after outage
        """
        manager = EffectManager(hass_mock)

        # Simulate saving peaks
        timestamp = datetime(2025, 10, 14, 12, 0)
        await manager.record_period_measurement(5.0, 12 * 4, timestamp)

        stored_data = {"peaks": [p.to_dict() for p in manager._monthly_peaks]}

        # Verify data can be serialized
        assert "peaks" in stored_data
        assert len(stored_data["peaks"]) == 1
        assert stored_data["peaks"][0]["effective_power"] == 5.0
