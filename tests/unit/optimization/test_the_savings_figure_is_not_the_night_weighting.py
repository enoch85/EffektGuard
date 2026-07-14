"""I fixed the fabricated-savings bug and fabricated the savings again, the same afternoon.

The original defect was that the effect-tariff saving was computed from the peak itself:

    baseline_peak_kw = current_peak_kw * 1.176      # nothing ever set a real baseline

so `effect_savings` reduced to `0.176 * current_peak * tariff` - a higher peak reported MORE
"savings", and the sensor could never read zero. Unfalsifiable. The fix was to measure the baseline
from the quarters recorded while the optimisation switch is OFF, and to report zero until then.

AND THE MEASUREMENT COMPARED TWO DIFFERENT QUANTITIES.

The Swedish effect tariff weights night quarters at half, so the effect layer carries both numbers:

    PeakEvent.actual_power       6.0 kW     what the house actually drew
    PeakEvent.effective_power    3.0 kW     what the tariff will bill it as, at 02:00

`peak_this_month` - the "current" side of the comparison - is `get_monthly_peak_summary()["highest"]`,
which is `effective_power`. But the coordinator fed the baseline `peak_event.actual_power`. So the two
sides of

    peak_reduction = baseline - current

were THE SAME QUARTER, once un-weighted and once weighted. One 6.0 kW quarter at 02:00, with the
optimiser doing nothing whatsoever:

    reported effect saving:   150 SEK/month
    effect_baseline_measured: True          <- and flagged as MEASURED, not assumed

Every krona of it is the night weighting compared against itself. Same class of bug as the one it
replaced - a savings figure computed from the peak rather than from any saving - and worse, because
this one is stamped "measured".

AND THE BASELINE HAD NO SOURCE GATE. Peak RECORDING accepts nibe_currents (the pump's own current
sensors), because the pump is the dominant controllable load and a NIBE-only history compared against
NIBE-only quarters is a coherent basis for throttling. But the effect tariff bills WHOLE-HOUSE grid
import, and the savings figure is MONEY. A baseline built from a sensor that cannot see the oven, the
EV or the water heater produces a SEK figure from a quantity nobody is billed for. Money comes from
BILLABLE_POWER_SOURCES - the external meter, and nothing else.

THESE TESTS DRIVE THE COORDINATOR, not the savings calculator. The first draft of this file called
`update_baseline_peak(event.effective_power)` in the test body and asserted the result was zero -
which is a test of my own arithmetic, and passes with the production bug fully intact. The bug is in
what the COORDINATOR passes. So that is what is exercised.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager

# 02:00: the night weighting halves this quarter. The whole bug lives in that halving.
NIGHT_HOUR = 2
DAY_HOUR = 10
SIX_KW_OF_CURRENT = 8.7  # amps per phase, 3-phase 230 V -> ~6.0 kW


@pytest.fixture
def coordinator():
    """The owner has a whole-house meter, and optimisation is switched OFF.

    That is the state in which the baseline is measured: the coordinator holds the curve offset at
    0.0, so the quarters recorded now are what this house does WITHOUT EffektGuard.
    """
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    nibe = MagicMock()
    nibe._power_sensor_entity = "sensor.house_power"
    nibe.power_sensor_entity = "sensor.house_power"

    entry = MagicMock()
    entry.data = {"enable_optimization": False}
    entry.options = {}

    coord = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coord.peak_today = 0.0
    coord.peak_this_month = 0.0
    coord.effect._store = MagicMock()
    coord.effect._store.async_save = AsyncMock()
    coord.effect._monthly_peaks = []
    return coord


def _metered_house(hour: int, power_kw: float) -> NibeState:
    """A NibeState timestamped in the given hour. Power comes from the external meter."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, hour, 0, tzinfo=timezone.utc),
        phase1_current=None,
        phase2_current=None,
        phase3_current=None,
        compressor_hz=60,
    )


async def _observe_a_whole_quarter(coord, monkeypatch, hour: int, power_kw: float) -> None:
    """Four samples across one quarter, so it completes and is recorded as a tariff peak.

    The meter has to actually READ. A bare MagicMock state is refused by `power_kw_from_state` -
    correctly, since its unit is a MagicMock and this integration will not guess a power unit - so
    the first draft of this helper recorded no peak at all, set no baseline, reported zero savings,
    and passed with the bug fully intact. Vacuous green is the failure mode this whole audit keeps
    finding, so the callers assert a precondition that the peak was really recorded.
    """
    state = MagicMock()
    state.entity_id = "sensor.house_power"
    state.state = str(power_kw)
    state.attributes = {"unit_of_measurement": "kW"}
    coord.hass.states.get = MagicMock(return_value=state)

    nibe_data = _metered_house(hour, power_kw)

    for minute in (0, 5, 10, 15):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, m=minute, h=hour: datetime(2026, 1, 15, h, m, tzinfo=timezone.utc),
        )
        await coord._update_peak_tracking(nibe_data)

    assert coord.effect._monthly_peaks, (
        "PRECONDITION FAILED: no quarter was recorded, so nothing downstream of here means "
        "anything. The meter did not read."
    )


def _savings(coord):
    return coord.savings_calculator.estimate_monthly_savings(
        current_peak_kw=coord.peak_this_month,
        baseline_peak_kw=coord.savings_calculator._baseline_monthly_peak,
        average_spot_savings_per_day=0.0,
    )


@pytest.mark.asyncio
async def test_a_single_night_quarter_is_not_a_saving(coordinator, monkeypatch):
    """The bug, through the coordinator. The optimiser is OFF and does nothing at all."""
    await _observe_a_whole_quarter(coordinator, monkeypatch, NIGHT_HOUR, 6.0)

    baseline = coordinator.savings_calculator._baseline_monthly_peak
    estimate = _savings(coordinator)

    assert estimate.effect_savings == 0.0, (
        f"One 6.0 kW quarter at 02:00, with optimisation switched OFF, reports "
        f"{estimate.effect_savings:.0f} SEK/month of effect-tariff savings. The baseline was fed "
        f"{baseline:.2f} kW (actual_power) while peak_this_month is "
        f"{coordinator.peak_this_month:.2f} kW (effective_power, halved by the night tariff). It "
        f"is the same quarter compared against itself, and the difference IS the weighting."
    )


@pytest.mark.asyncio
async def test_the_baseline_is_the_same_quantity_peak_this_month_is(coordinator, monkeypatch):
    """The invariant that would have stopped this being written: compare like with like."""
    await _observe_a_whole_quarter(coordinator, monkeypatch, NIGHT_HOUR, 6.0)

    assert coordinator.peak_this_month == pytest.approx(3.0), (
        "precondition: peak_this_month must be the EFFECTIVE peak, halved at 02:00. If the night "
        "weighting did not bite, this test proves nothing."
    )
    assert coordinator.savings_calculator._baseline_monthly_peak == pytest.approx(
        coordinator.peak_this_month
    ), (
        f"The baseline is {coordinator.savings_calculator._baseline_monthly_peak:.2f} kW and "
        f"peak_this_month is {coordinator.peak_this_month:.2f} kW - the same quarter, expressed "
        f"two different ways. Whatever feeds the baseline must be weighted exactly as "
        f"peak_this_month is, or their difference is an artefact of the weighting."
    )


@pytest.mark.asyncio
async def test_the_hour_of_the_day_is_not_a_saving(coordinator, monkeypatch):
    """An unchanged 6 kW peak reports nothing, whether it happened at 02:00 or at 10:00."""
    for hour in (NIGHT_HOUR, DAY_HOUR):
        coordinator.effect._monthly_peaks = []
        coordinator.peak_this_month = 0.0
        coordinator.savings_calculator._baseline_monthly_peak = None
        coordinator._quarter_power_start = None

        await _observe_a_whole_quarter(coordinator, monkeypatch, hour, 6.0)

        assert _savings(coordinator).effect_savings == 0.0, (
            f"An unchanged 6.0 kW peak at {hour:02d}:00 reports "
            f"{_savings(coordinator).effect_savings:.0f} SEK/month of savings. The hour of the day "
            f"is not a saving."
        )


@pytest.mark.asyncio
async def test_a_real_reduction_is_still_reported(coordinator, monkeypatch):
    """The regression guard. Killing the fabrication must not silence a genuine saving."""
    await _observe_a_whole_quarter(coordinator, monkeypatch, DAY_HOUR, 8.0)
    baseline = coordinator.savings_calculator._baseline_monthly_peak

    # Now the optimiser is on, and it holds the house to 5 kW in the same daytime quarter.
    optimised = EffectManager(MagicMock())
    optimised._store = MagicMock()
    optimised._store.async_save = AsyncMock()
    optimised._monthly_peaks = []
    await optimised.record_quarter_measurement(
        power_kw=5.0,
        quarter=DAY_HOUR * 4,
        timestamp=datetime(2026, 1, 20, DAY_HOUR, 0, tzinfo=timezone.utc),
        source="external_meter",
    )

    estimate = coordinator.savings_calculator.estimate_monthly_savings(
        current_peak_kw=optimised.get_monthly_peak_summary()["highest"],
        baseline_peak_kw=baseline,
        average_spot_savings_per_day=0.0,
    )

    assert estimate.effect_savings > 0, (
        f"The house drew 8.0 kW unoptimised and 5.0 kW optimised, both in DAYTIME quarters where "
        f"the weighting is 1.0 on each side. That is a real 3 kW cut in the billed peak, and it "
        f"reported {estimate.effect_savings:.0f} SEK."
    )


@pytest.mark.asyncio
async def test_the_heat_pumps_own_current_sensors_are_not_a_billing_baseline(monkeypatch):
    """A NIBE-only peak may throttle the pump. It may not become a figure in kronor.

    Peak RECORDING deliberately accepts nibe_currents: the pump is the dominant controllable load,
    and this month's NIBE quarters compared against this month's NIBE peaks is a coherent basis for
    deciding whether to back off. `PEAK_CONTROL_POWER_SOURCES` says exactly that.

    But the effect tariff bills WHOLE-HOUSE grid import, and `BILLABLE_POWER_SOURCES` is the
    external meter alone. A baseline built from a sensor that cannot see the oven, the EV or the
    water heater is not a baseline for anything the owner is charged - and the number it feeds is
    denominated in SEK on a dashboard.
    """
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07
    hass.states.get = MagicMock(return_value=None)  # no external meter at all

    nibe = MagicMock()
    nibe._power_sensor_entity = None
    nibe.power_sensor_entity = None
    # 3 x 8.7 A at 230 V is about 6 kW - of HEAT PUMP, not of house.
    nibe.calculate_power_from_currents = MagicMock(return_value=6.0)

    entry = MagicMock()
    entry.data = {"enable_optimization": False}
    entry.options = {}

    coord = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coord.peak_today = 0.0
    coord.peak_this_month = 0.0
    coord.effect._store = MagicMock()
    coord.effect._store.async_save = AsyncMock()
    coord.effect._monthly_peaks = []

    pump_only = _metered_house(DAY_HOUR, 6.0)
    pump_only.phase1_current = SIX_KW_OF_CURRENT
    pump_only.phase2_current = SIX_KW_OF_CURRENT
    pump_only.phase3_current = SIX_KW_OF_CURRENT

    for minute in (0, 5, 10, 15):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, m=minute: datetime(2026, 1, 15, DAY_HOUR, m, tzinfo=timezone.utc),
        )
        await coord._update_peak_tracking(pump_only)

    assert coord.effect._monthly_peaks, (
        "PRECONDITION: the NIBE-currents quarter must still be RECORDED - peak control depends on "
        "it, and refusing to record it would break throttling. The point is what it must not FEED."
    )
    assert coord.savings_calculator._baseline_monthly_peak is None, (
        f"A peak measured from the heat pump's own current sensors "
        f"({coord.savings_calculator._baseline_monthly_peak} kW) became the baseline for a savings "
        f"figure in SEK. That sensor cannot see the oven, the EV or the water heater, and the "
        f"effect tariff bills whole-house grid import. Money comes from the meter, or not at all."
    )


class TestWhatTheOwnerIsTold:
    """The same mismatch, on the dashboard. Both sides must be weighted the way the tariff is."""

    def _peak_today_sensor(self, coord):
        from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor

        description = next(d for d in SENSORS if d.key == "peak_today")
        entry = MagicMock()
        entry.entry_id = "test"
        entry.data = {}
        return EffektGuardSensor(coord, entry, description)

    def _coordinator(self, peak_today, quarter, peak_this_month):
        coord = MagicMock()
        # `extra_state_attributes` returns early on a falsy `data`, so an empty dict here would
        # make every assertion below a KeyError rather than a judgement about the attribute.
        coord.data = {"nibe": MagicMock()}
        coord.peak_today = peak_today
        coord.peak_today_quarter = quarter
        coord.peak_today_source = "external_meter"
        coord.peak_today_time = None
        coord.peak_this_month = peak_this_month
        coord.yesterday_peak = 0.0
        return coord

    def test_a_night_blip_is_not_announced_as_a_new_monthly_peak(self):
        """3.1 kW at 02:00 is billed as 1.55 kW. It cannot beat a 3.0 kW effective monthly peak."""
        coord = self._coordinator(peak_today=3.1, quarter=NIGHT_HOUR * 4, peak_this_month=3.0)

        attrs = self._peak_today_sensor(coord).extra_state_attributes

        assert attrs["will_affect_billing"] is False, (
            "The house drew 3.1 kW at 02:00 and the owner was told it set a new monthly peak "
            "against 3.0 kW. peak_this_month is the EFFECTIVE peak and the night tariff halves "
            "this quarter to 1.55 kW - it is not close. The night weighting is not a peak."
        )

    def test_a_daytime_peak_that_really_does_beat_the_month_is_still_announced(self):
        """The regression guard. Weighting both sides must not silence a genuine new peak."""
        coord = self._coordinator(peak_today=6.0, quarter=DAY_HOUR * 4, peak_this_month=3.0)

        attrs = self._peak_today_sensor(coord).extra_state_attributes

        assert (
            attrs["will_affect_billing"] is True
        ), "6.0 kW at 10:00 is billed in full and beats a 3.0 kW monthly peak. It IS a new peak."

    def test_a_night_peak_big_enough_to_win_on_its_billed_value_is_announced(self):
        """8.0 kW at 02:00 is billed as 4.0 kW, which does beat 3.0. The weighting cuts both ways."""
        coord = self._coordinator(peak_today=8.0, quarter=NIGHT_HOUR * 4, peak_this_month=3.0)

        attrs = self._peak_today_sensor(coord).extra_state_attributes

        assert attrs["will_affect_billing"] is True
        assert "4.00 kW" in attrs["billing_impact"], (
            f"The owner must be shown what the tariff will BILL - 4.00 kW - not the 8.0 kW the "
            f"meter saw. Got: {attrs['billing_impact']!r}"
        )


class TestZeroSavingsMeansTwoDifferentThings:
    """`effect_baseline_measured` was computed and never surfaced. Counted, and ignored."""

    def _savings_sensor(self, measured: bool):
        from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor
        from custom_components.effektguard.optimization.savings_calculator import SavingsEstimate

        coord = MagicMock()
        coord.data = {
            "savings": SavingsEstimate(
                monthly_estimate=0.0,
                effect_savings=0.0,
                spot_savings=0.0,
                baseline_cost=0.0,
                optimized_cost=0.0,
                effect_baseline_measured=measured,
            )
        }
        description = next(d for d in SENSORS if d.key == "savings_estimate")
        entry = MagicMock()
        entry.entry_id = "test"
        entry.data = {}
        return EffektGuardSensor(coord, entry, description)

    def test_an_unmeasured_baseline_says_so(self):
        """0 SEK because we have never seen this house unoptimised - not because we are failing."""
        attrs = self._savings_sensor(measured=False).extra_state_attributes

        assert attrs["effect_baseline_measured"] is False
        assert "effect_savings_note" in attrs, (
            "The savings sensor reads 0 SEK and the owner has no way to tell whether that means "
            "'we have never measured your unoptimised house' or 'we are saving you nothing'. The "
            "flag that distinguishes them was computed and never shown."
        )

    def test_a_measured_baseline_does_not_apologise(self):
        """Once it IS measured, zero means zero and there is nothing to explain."""
        attrs = self._savings_sensor(measured=True).extra_state_attributes

        assert attrs["effect_baseline_measured"] is True
        assert "effect_savings_note" not in attrs
