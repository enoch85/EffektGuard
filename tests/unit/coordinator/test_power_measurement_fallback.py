"""Tests for power measurement smart fallback and compressor Hz estimation.

Tests the priority cascade for power measurement:
1. External power meter (whole house) - PRIORITY 1 for peak billing
2. NIBE phase currents (BE1/BE2/BE3) - PRIORITY 2 for NIBE-only measurement
3. Compressor Hz estimation - PRIORITY 3 for display/debugging
4. Fallback estimation - PRIORITY 4 last resort

Power estimation methods are in effect_layer.py and tested in test_shared_layer_methods.py.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.const import (
    POWER_STANDBY_KW,
    COMPRESSOR_POWER_MIN_KW,
)


class TestCompressorHzPowerEstimation:
    """Test estimate_power_from_compressor() via EffectManager."""

    @pytest.fixture
    def effect_manager(self, hass):
        """Create an EffectManager instance."""
        return EffectManager(hass)

    def test_zero_hz_returns_standby_power(self, effect_manager):
        """Test that 0 Hz returns standby power (0.1 kW)."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=0, outdoor_temp=5.0)
        assert power == POWER_STANDBY_KW  # Standby power

    def test_minimum_compressor_20hz(self, effect_manager):
        """Test power estimation at minimum compressor frequency (20 Hz)."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=20, outdoor_temp=5.0)
        # At 20 Hz: base ~1.5-2.0 kW
        assert power >= COMPRESSOR_POWER_MIN_KW
        assert power <= 2.5

    def test_mid_range_compressor_50hz(self, effect_manager):
        """Test power estimation at mid-range frequency (50 Hz)."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=50, outdoor_temp=0.0)
        # At 50 Hz with 0°C (MILD factor 1.0):
        # base = 1.5 + (50-20) * (5.0/100) = 3.0 kW
        assert power >= 2.5
        assert power <= 4.0

    def test_maximum_compressor_80hz(self, effect_manager):
        """Test power estimation at maximum frequency (80 Hz)."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=80, outdoor_temp=-10.0)
        # At 80 Hz with -10°C (COLD factor 1.2):
        # base = 1.5 + (80-20) * (5.0/100) = 4.5 kW
        # result = 4.5 * 1.2 = 5.4 kW
        assert power >= 5.0
        assert power <= 7.0

    def test_cold_weather_increases_power(self, effect_manager):
        """Test that colder outdoor temp increases power for same Hz."""
        power_mild = effect_manager.estimate_power_from_compressor(
            compressor_hz=50, outdoor_temp=5.0
        )
        power_cold = effect_manager.estimate_power_from_compressor(
            compressor_hz=50, outdoor_temp=-15.0
        )
        # Same Hz but colder temp = more power needed
        assert power_cold > power_mild

    def test_extreme_cold_applies_max_factor(self, effect_manager):
        """Test that extreme cold (<-15°C) applies maximum temp factor."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=50, outdoor_temp=-20.0)
        # Extreme cold applies 1.3x factor
        # Base at 50 Hz: 1.5 + (50-20)*0.05 = 3.0 kW
        # With 1.3x = 3.9 kW
        assert power >= 3.5
        assert power <= 5.0

    def test_mild_weather_no_temp_factor(self, effect_manager):
        """Test that mild weather (>0°C) applies no temp factor."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=50, outdoor_temp=10.0)
        # Mild weather: temp_factor = 1.0, so base power only
        # Base at 50 Hz: ~3.5-4.5 kW
        assert power >= 3.0
        assert power <= 5.0

    def test_zero_hz_compressor_returns_standby(self, effect_manager):
        """Test that 0 Hz compressor returns standby power."""
        power = effect_manager.estimate_power_from_compressor(compressor_hz=0, outdoor_temp=5.0)
        assert power == POWER_STANDBY_KW


class TestPowerMeasurementPriority:
    """Test power measurement priority cascade."""

    @pytest.mark.asyncio
    async def test_priority_1_external_meter_used_first(self, coordinator_with_external_meter):
        """Test that external power meter is used first (priority 1)."""
        coordinator = coordinator_with_external_meter

        # Mock power sensor entity
        mock_state = MagicMock()
        mock_state.state = "5500"  # 5500W = 5.5 kW
        mock_state.attributes = {"unit_of_measurement": "W"}
        coordinator.hass.states.get.return_value = mock_state

        # Mock NIBE data
        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-50.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            phase1_current=10.0,  # This would calculate to ~6.9 kW
            phase2_current=8.0,
            phase3_current=12.0,
        )

        await coordinator._update_peak_tracking(nibe_data)

        # Should use external meter (5.5 kW) not NIBE currents (6.9 kW)
        assert coordinator.peak_today == 5.5

    @pytest.mark.asyncio
    async def test_priority_2_nibe_currents_when_no_external_meter(
        self, coordinator_without_external_meter
    ):
        """Test that NIBE phase currents are used when no external meter."""
        coordinator = coordinator_without_external_meter

        # Mock NIBE data with phase currents
        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-50.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            phase1_current=10.0,  # BE1
            phase2_current=8.0,  # BE2
            phase3_current=9.0,  # BE3
            compressor_hz=50,
        )

        await coordinator._update_peak_tracking(nibe_data)

        # Should calculate from currents: ~6.156 kW
        # (240V × 27A × 0.95) / 1000 = 6.156 kW
        assert coordinator.peak_today >= 6.0
        assert coordinator.peak_today <= 6.5

    @pytest.mark.asyncio
    async def test_priority_3_compressor_hz_when_no_currents(self):
        """Test that compressor Hz estimation works with EffectManager.

        Uses EffectManager.estimate_power_from_compressor() which takes
        compressor_hz and outdoor_temp as direct parameters for a clean API.
        """
        effect_manager = EffectManager(MagicMock())

        # Test estimation at 50 Hz, -5°C outdoor
        # Mid-range Hz should give mid-range power
        estimated_power = effect_manager.estimate_power_from_compressor(
            compressor_hz=50,
            outdoor_temp=-5.0,
        )

        # At 50 Hz and -5°C (COOL factor 1.1):
        # base = 1.5 + (50-20)*0.05 = 3.0 kW
        # result = 3.0 * 1.1 = 3.3 kW
        assert estimated_power >= 3.0
        assert estimated_power <= 4.5
        print(f"Priority 3 (Hz): Estimation at 50 Hz, -5°C = {estimated_power:.2f} kW")

    @pytest.mark.asyncio
    async def test_priority_4_fallback_estimation(self, coordinator_without_external_meter):
        """Test fallback estimation when no measurements available."""
        coordinator = coordinator_without_external_meter

        # Mock NIBE data with no sensors
        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-50.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            phase1_current=None,
            phase2_current=None,
            phase3_current=None,
            compressor_hz=None,  # No Hz sensor either
        )

        await coordinator._update_peak_tracking(nibe_data)

        # Should use fallback estimation (based on temps)
        # Fallback typically gives ~4 kW baseline
        assert coordinator.peak_today >= 2.0
        assert coordinator.peak_today <= 6.0


class TestKilowattMeterNotDividedTwice:
    """A kW-unit whole-house meter must not be divided by 1000 again.

    Regression: the coordinator divided every external meter reading by
    1000 ("typically in watts"), so a 6.0 kW meter became 0.006 kW and
    invalidated peak protection and peak records.
    """

    @pytest.mark.asyncio
    async def test_kw_meter_used_verbatim(self, coordinator_with_external_meter):
        coordinator = coordinator_with_external_meter

        mock_state = MagicMock()
        mock_state.state = "6.0"
        mock_state.attributes = {"unit_of_measurement": "kW"}
        coordinator.hass.states.get.return_value = mock_state

        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-50.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
        )

        await coordinator._update_peak_tracking(nibe_data)

        assert coordinator.peak_today == 6.0


class TestAMeterBehindSolarIsStillTheMeter:
    """A grid-import meter reading low behind solar is reporting the truth, and the truth is billed.

    There used to be a "smart fallback" here: a meter reading under 0.5 kW while the compressor ran
    above 20 Hz was assumed to be masked by solar export, so an ESTIMATE of the compressor's draw was
    substituted - and recorded against the effect tariff.

    The grid operator bills grid IMPORT. If solar covers 4.7 kW of a 5.0 kW compressor, the house
    imported 0.3 kW and 0.3 kW is what is charged. Recording ~5.5 kW instead inflated the month's peak
    by an order of magnitude, in the owner's disfavour, and effect tariffs bill the top three quarters
    of the month, so it stood for weeks.

    Owner decision: "Math should be correct. So if solar covers everything but 0.5 kW, count 0.5 kW
    for that period." The meter is the truth. There is nothing to override.
    """

    @pytest.mark.asyncio
    async def test_a_low_reading_with_the_compressor_running_hard_is_taken_at_face_value(
        self, coordinator_with_external_meter
    ):
        coordinator = coordinator_with_external_meter

        mock_state = MagicMock()
        mock_state.state = "300"  # 300 W of grid import; the panels are covering the rest
        mock_state.attributes = {"unit_of_measurement": "W"}
        coordinator.hass.states.get.return_value = mock_state

        nibe_data = NibeState(
            outdoor_temp=-5.0,
            indoor_temp=21.0,
            supply_temp=42.0,
            return_temp=37.0,
            degree_minutes=-150.0,
            current_offset=2.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            compressor_hz=60,  # working hard - this used to trigger the substitution
        )

        await coordinator._update_peak_tracking(nibe_data)

        assert coordinator.peak_today == pytest.approx(0.3), (
            f"The meter reported 0.3 kW of grid import and {coordinator.peak_today:.2f} kW was "
            f"recorded. What the compressor draws is not what the grid delivered, and it is not "
            f"what will be billed."
        )

    @pytest.mark.asyncio
    async def test_a_low_reading_with_the_compressor_idle_is_also_taken_at_face_value(
        self, coordinator_with_external_meter
    ):
        """The same rule, with nothing there to tempt it."""
        coordinator = coordinator_with_external_meter

        mock_state = MagicMock()
        mock_state.state = "300"
        mock_state.attributes = {"unit_of_measurement": "W"}
        coordinator.hass.states.get.return_value = mock_state

        nibe_data = NibeState(
            outdoor_temp=10.0,
            indoor_temp=21.0,
            supply_temp=30.0,
            return_temp=28.0,
            degree_minutes=-20.0,
            current_offset=0.0,
            is_heating=False,
            is_hot_water=False,
            timestamp=datetime.now(),
            compressor_hz=0,
        )

        await coordinator._update_peak_tracking(nibe_data)

        assert coordinator.peak_today == pytest.approx(0.3)


class TestPeakTrackingOnlyWithRealMeasurements:
    """Test that monthly peaks are only recorded with real measurements."""

    @pytest.mark.asyncio
    async def test_no_monthly_peak_with_estimates_only(self, coordinator_without_external_meter):
        """Test that monthly peak is NOT recorded with estimates only."""
        coordinator = coordinator_without_external_meter

        # Mock NIBE data with only estimates (no real measurements)
        nibe_data = NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-50.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            phase1_current=None,  # No current sensors
            phase2_current=None,
            phase3_current=None,
            compressor_hz=None,  # No Hz sensor
        )

        initial_monthly_peak = coordinator.peak_this_month
        await coordinator._update_peak_tracking(nibe_data)

        # Monthly peak should NOT be updated (only daily peak for display)
        # Code should log: "Skipping monthly peak recording: No real power measurement available"
        assert coordinator.peak_this_month == initial_monthly_peak

    @pytest.mark.asyncio
    async def test_monthly_peak_logic_with_external_meter(self, coordinator_with_external_meter):
        """Test that external meter provides real measurements for peak tracking."""
        coordinator = coordinator_with_external_meter

        # Test that external meter is detected
        has_meter = coordinator.nibe.power_sensor_entity is not None

        # Should have external meter configured
        assert has_meter, "External meter should be configured"

        # Verify power sensor entity is accessible
        power_entity_id = coordinator.nibe.power_sensor_entity
        assert power_entity_id is not None
        print(f"External power meter entity: {power_entity_id}")

    @pytest.mark.asyncio
    async def test_monthly_peak_logic_with_nibe_currents(self, coordinator_without_external_meter):
        """Test that NIBE current sensors provide real measurements."""
        coordinator = coordinator_without_external_meter

        # Mock NIBE data with phase currents
        nibe_data = NibeState(
            outdoor_temp=-10.0,
            indoor_temp=21.0,
            supply_temp=45.0,
            return_temp=40.0,
            degree_minutes=-200.0,
            current_offset=2.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime.now(),
            phase1_current=15.0,  # Real measurement
            phase2_current=13.0,
            phase3_current=14.0,
        )

        # Test that phase currents give real power calculation
        power_from_currents = coordinator.nibe.calculate_power_from_currents(
            nibe_data.phase1_current,
            nibe_data.phase2_current,
            nibe_data.phase3_current,
        )

        # Should calculate real power from currents (~9-10 kW)
        assert power_from_currents is not None
        assert power_from_currents >= 8.0
        assert power_from_currents <= 11.0
        print(
            f"NIBE power from currents: {power_from_currents:.2f} kW "
            f"(L1={nibe_data.phase1_current:.1f}, "
            f"L2={nibe_data.phase2_current:.1f}, "
            f"L3={nibe_data.phase3_current:.1f})"
        )


@pytest.fixture
def coordinator_with_external_meter():
    """Create coordinator with external power meter configured."""
    from unittest.mock import MagicMock
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    from custom_components.effektguard.optimization.effect_layer import EffectManager

    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm latitude for climate zone detection
    hass.config.longitude = 18.07  # Stockholm longitude
    nibe_adapter = MagicMock()
    nibe_adapter._power_sensor_entity = "sensor.house_power"  # External meter configured
    nibe_adapter.power_sensor_entity = "sensor.house_power"
    gespot_adapter = MagicMock()
    weather_adapter = MagicMock()
    decision_engine = MagicMock()

    # Use real EffectManager for power estimation tests
    effect_manager = EffectManager(hass)

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass,
        nibe_adapter,
        gespot_adapter,
        weather_adapter,
        decision_engine,
        effect_manager,
        entry,
    )

    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0

    return coordinator


@pytest.fixture
def coordinator_without_external_meter():
    """Create coordinator without external power meter."""
    from unittest.mock import MagicMock
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    from custom_components.effektguard.optimization.effect_layer import EffectManager

    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm latitude for climate zone detection
    hass.config.longitude = 18.07  # Stockholm longitude
    nibe_adapter = MagicMock()
    nibe_adapter._power_sensor_entity = None  # No external meter
    nibe_adapter.power_sensor_entity = None
    # Mock calculate_power_from_currents method
    nibe_adapter.calculate_power_from_currents.side_effect = lambda p1, p2, p3: (
        (240 * (p1 + (p2 or 0) + (p3 or 0)) * 0.95 / 1000) if p1 is not None else None
    )
    gespot_adapter = MagicMock()
    weather_adapter = MagicMock()
    decision_engine = MagicMock()

    # Use real EffectManager for power estimation tests
    effect_manager = EffectManager(hass)

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass,
        nibe_adapter,
        gespot_adapter,
        weather_adapter,
        decision_engine,
        effect_manager,
        entry,
    )

    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0

    return coordinator


@pytest.fixture
def coordinator():
    """Create basic coordinator for testing."""
    from unittest.mock import MagicMock
    from custom_components.effektguard.coordinator import EffektGuardCoordinator
    from custom_components.effektguard.optimization.effect_layer import EffectManager

    hass = MagicMock()
    hass.config.latitude = 59.33  # Stockholm latitude for climate zone detection
    hass.config.longitude = 18.07  # Stockholm longitude
    nibe_adapter = MagicMock()
    gespot_adapter = MagicMock()
    weather_adapter = MagicMock()
    decision_engine = MagicMock()

    # Use real EffectManager for power estimation tests
    effect_manager = EffectManager(hass)

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass,
        nibe_adapter,
        gespot_adapter,
        weather_adapter,
        decision_engine,
        effect_manager,
        entry,
    )

    return coordinator


class TestTheBillingPeriodMeanIsAnHour:
    """This class used to be TestQuarterMeanRecording, and the quantity it pinned is not billed.

    The Swedish effect tariff bills the mean power over an HOUR. Ellevio: "the measurement uses
    hourly averages". Energimarknadsinspektionen: "elnatsforetagen mater din elanvandning per
    timme". The coordinator accumulated quarter-hours, so a 15-minute hot-water cycle at 9 kW inside
    an otherwise idle hour was recorded as a 9 kW billing peak where the meter bills 3.

    Every property these tests pinned is still worth pinning - the mean rather than the spike, the
    time-weighting, the discarded partial period at startup. Only the window changed.
    """

    @pytest.mark.asyncio
    async def test_a_spike_is_averaged_over_the_whole_hour(
        self, coordinator_with_external_meter, monkeypatch
    ):
        """THE BUG, in one test. A hot-water cycle is not a billing peak."""
        from datetime import datetime, timezone

        from homeassistant.util import dt as dt_util

        coordinator = coordinator_with_external_meter
        coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
        nibe_data = NibeState(5.0, 21.0, 35.0, 30.0, -50.0, 0.0, True, False, datetime.now())

        # 9 kW for the first quarter of the hour, then the house idles at 1 kW.
        for minute in range(0, 60, 5):
            state = MagicMock()
            state.state = "9000" if minute < 15 else "1000"
            state.attributes = {"unit_of_measurement": "W"}
            coordinator.hass.states.get.return_value = state
            monkeypatch.setattr(
                dt_util,
                "now",
                lambda tz=None, minute=minute: datetime(
                    2026, 1, 15, 10, minute, tzinfo=timezone.utc
                ),
            )
            await coordinator._update_peak_tracking(nibe_data)

        coordinator.effect.record_period_measurement.assert_not_awaited()

        # The next hour completes it.
        monkeypatch.setattr(
            dt_util, "now", lambda tz=None: datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        )
        await coordinator._update_peak_tracking(nibe_data)

        coordinator.effect.record_period_measurement.assert_awaited_once()
        recorded = coordinator.effect.record_period_measurement.await_args.kwargs

        assert recorded["period"] == 10, "the billing period is the HOUR, and this is hour 10"
        # 9 kW for 15 minutes, 1 kW for 45: (9*15 + 1*45)/60 = 3.0 kW
        assert recorded["power_kw"] == pytest.approx(3.0), (
            f"The hour's mean power is 3.00 kW and that is what Ellevio bills. This recorded "
            f"{recorded['power_kw']:.2f}. The 9 kW quarter is a hot-water cycle; the tariff "
            f"averages it with the quiet 45 minutes around it."
        )

    @pytest.mark.asyncio
    async def test_recording_starts_from_any_update_phase(
        self, coordinator_with_external_meter, monkeypatch
    ):
        """Seeding must not require an update landing on the hour boundary."""
        from datetime import datetime, timezone

        from homeassistant.util import dt as dt_util

        coordinator = coordinator_with_external_meter
        coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
        state = MagicMock()
        state.state = "2000"
        state.attributes = {"unit_of_measurement": "W"}
        coordinator.hass.states.get.return_value = state
        nibe_data = NibeState(5.0, 21.0, 35.0, 30.0, -50.0, 0.0, True, False, datetime.now())

        # First update lands at 10:07 - mid-hour. Hour 10 is partial and must be discarded; hour 11
        # is observed from its start and must be recorded.
        times = [(10, m) for m in range(7, 60, 5)] + [(11, m) for m in range(0, 60, 5)] + [(12, 0)]
        for hour, minute in times:
            monkeypatch.setattr(
                dt_util,
                "now",
                lambda tz=None, hour=hour, minute=minute: datetime(
                    2026, 1, 15, hour, minute, tzinfo=timezone.utc
                ),
            )
            await coordinator._update_peak_tracking(nibe_data)

        coordinator.effect.record_period_measurement.assert_awaited_once()
        recorded = coordinator.effect.record_period_measurement.await_args.kwargs

        assert recorded["period"] == 11, "hour 10 began before observation did, so it is discarded"
        assert recorded["power_kw"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_the_partial_startup_hour_is_discarded(
        self, coordinator_with_external_meter, monkeypatch
    ):
        """An hour that began before the meter was watched is not an hour anyone measured."""
        from datetime import datetime, timezone

        from homeassistant.util import dt as dt_util

        coordinator = coordinator_with_external_meter
        coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
        state = MagicMock()
        state.state = "9000"
        state.attributes = {"unit_of_measurement": "W"}
        coordinator.hass.states.get.return_value = state
        nibe_data = NibeState(5.0, 21.0, 35.0, 30.0, -50.0, 0.0, True, False, datetime.now())

        for hour, minute in ((10, 40), (10, 45), (11, 0)):
            monkeypatch.setattr(
                dt_util,
                "now",
                lambda tz=None, hour=hour, minute=minute: datetime(
                    2026, 1, 15, hour, minute, tzinfo=timezone.utc
                ),
            )
            await coordinator._update_peak_tracking(nibe_data)

        coordinator.effect.record_period_measurement.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_irregular_samples_use_a_time_weighted_mean(
        self, coordinator_with_external_meter, monkeypatch
    ):
        """A sample that stands for 15 minutes must not weigh the same as one standing for 5.

        The claim - the hour's mean is time-weighted, not sample-counted - is unchanged. The SCENARIO
        had to change. It used to read 1 kW at :00, 9 kW at :01 and 1 kW at :59, which is a
        FIFTY-EIGHT MINUTE gap between two readings. That is not an irregular sample, it is a meter
        that stopped answering: the coordinator now refuses to bill an hour containing a silence
        longer than MAX_BILLING_OBSERVATION_GAP_MINUTES, because stretching one reading across most
        of an hour invents a peak rather than measuring one (see
        test_an_hour_the_meter_slept_through_is_not_a_bill.py).

        So the arithmetic is demonstrated on an hour that was actually OBSERVED. Every gap below is
        within the limit, and the two formulas still disagree by 40%:

            time-weighted:   (1*45 + 9*15) / 60  = 3.0 kW   <- what the grid bills
            sample-counted:  (1+1+1+9+9) / 5     = 4.2 kW
        """
        from datetime import datetime, timezone

        from homeassistant.util import dt as dt_util

        coordinator = coordinator_with_external_meter
        coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
        nibe_data = NibeState(5.0, 21.0, 35.0, 30.0, -50.0, 0.0, True, False, datetime.now())

        # 1 kW standing for 45 minutes, then 9 kW for the last 15.
        for watts, minute in (("1000", 0), ("1000", 15), ("1000", 30), ("9000", 45), ("9000", 55)):
            state = MagicMock()
            state.state = watts
            state.attributes = {"unit_of_measurement": "W"}
            coordinator.hass.states.get.return_value = state
            monkeypatch.setattr(
                dt_util,
                "now",
                lambda tz=None, minute=minute: datetime(
                    2026, 1, 15, 10, minute, tzinfo=timezone.utc
                ),
            )
            await coordinator._update_peak_tracking(nibe_data)

        monkeypatch.setattr(
            dt_util, "now", lambda tz=None: datetime(2026, 1, 15, 11, 0, tzinfo=timezone.utc)
        )
        await coordinator._update_peak_tracking(nibe_data)

        recorded = coordinator.effect.record_period_measurement.await_args.kwargs
        assert recorded["power_kw"] == pytest.approx((1 * 45 + 9 * 15) / 60), (
            f"billed {recorded['power_kw']:.2f} kW. 1 kW stood for 45 minutes and 9 kW for fifteen: "
            f"the hour's mean power is 3.0 kW. Counting the samples instead gives 4.2."
        )
