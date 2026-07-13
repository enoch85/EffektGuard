"""The tariff bills what the grid delivered. Two things were recorded against it that did not.

**NIBE phase currents.** BE1/BE2/BE3 measure the heat pump, and nothing else. Not the oven, not the
EV charger, not the kettle. They were nevertheless accepted as a whole-house billing measurement:

    has_real_measurement = has_external_power_sensor or nibe_data.phase1_current is not None

The peak sensor knew better and said so, in a comment, right next to a line that contradicted it:

    # Only real measurements from external meter affect effect tariff billing
    # NIBE currents measure only heat pump (missing other house loads)
    will_affect = self.coordinator.peak_today_source == "external_meter" and ...

So the owner was told "Not used for billing" about a peak the coordinator had just recorded against
the tariff. Owner decision: **NIBE currents are not billable.** They remain available to the decision
layers, which want a magnitude, not a bill.

**The solar "smart fallback".** When a grid-import meter reads under 0.5 kW while the compressor runs
above 20 Hz, the code concluded the meter was being masked by solar export and substituted an
ESTIMATED compressor power - then recorded the estimate as a tariff peak:

    if is_heating and compressor_hz > 20:
        estimated_power = self.effect.estimate_power_from_compressor(...)
        if estimated_power > 1.0:
            current_power = estimated_power     # <- and this was billed

But the grid operator bills grid IMPORT, and the import is precisely what the meter saw. If solar
covers 4.7 kW of a 5.0 kW compressor, the house imported 0.3 kW and 0.3 kW is what is charged. The
substitution recorded 5.5 kW - an order of magnitude above the truth, in the owner's disfavour, and it
stood for the rest of the month, because effect tariffs bill the top three quarters.

Owner decision: **"Math should be correct. So if solar covers everything but 0.5 kW, count 0.5 kW for
that period."** The meter is the truth. The fallback is gone.

What remains is a single rule, and it is the whole of it: only a whole-house meter reading can become
a billing peak.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    BILLABLE_POWER_SOURCES,
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NIBE_CURRENTS,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.effect_layer import EffectManager


def _coordinator(power_entity: str | None) -> EffektGuardCoordinator:
    hass = MagicMock()
    hass.config.latitude = 59.33
    hass.config.longitude = 18.07

    nibe = MagicMock()
    nibe._power_sensor_entity = power_entity
    nibe.power_sensor_entity = power_entity
    nibe.calculate_power_from_currents.side_effect = lambda p1, p2, p3: (
        240 * (p1 + (p2 or 0) + (p3 or 0)) * 0.95 / 1000 if p1 is not None else None
    )

    entry = MagicMock()
    entry.data = {}
    entry.options = {}

    coordinator = EffektGuardCoordinator(
        hass, nibe, MagicMock(), MagicMock(), MagicMock(), EffectManager(hass), entry
    )
    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0
    coordinator._power_sensor_available = True
    coordinator.effect.record_quarter_measurement = AsyncMock(return_value=None)
    return coordinator


def _meter(hass, value: str, unit: str = "W") -> None:
    state = MagicMock()
    state.state = value
    state.attributes = {"unit_of_measurement": unit}
    hass.states.get.return_value = state


def _pump(compressor_hz: int = 0, currents: float | None = None) -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=42.0,
        return_temp=37.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        phase1_current=currents,
        phase2_current=currents,
        phase3_current=currents,
        compressor_hz=compressor_hz,
    )


async def _run_a_complete_quarter(coordinator, nibe_data, monkeypatch) -> None:
    for minute in (0, 5, 10, 15):
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, minute=minute: datetime(2026, 1, 15, 10, minute, tzinfo=timezone.utc),
        )
        await coordinator._update_peak_tracking(nibe_data)


def test_only_a_whole_house_meter_is_billable():
    """The rule, stated once, where both the recorder and the reporting read it."""
    assert BILLABLE_POWER_SOURCES == frozenset({POWER_SOURCE_EXTERNAL_METER}), (
        f"BILLABLE_POWER_SOURCES is {sorted(BILLABLE_POWER_SOURCES)}. The Swedish effect tariff bills "
        f"whole-house grid import. Only a whole-house meter measures that."
    )


@pytest.mark.asyncio
async def test_nibe_phase_currents_are_not_a_billing_peak(monkeypatch):
    """They measure the pump. The tariff bills the house."""
    coordinator = _coordinator(power_entity=None)  # no whole-house meter, only NIBE currents

    await _run_a_complete_quarter(coordinator, _pump(compressor_hz=60, currents=10.0), monkeypatch)

    coordinator.effect.record_quarter_measurement.assert_not_awaited()
    assert (
        coordinator.peak_today_source == POWER_SOURCE_NIBE_CURRENTS
    ), "precondition: the currents should still be READ - they are useful to the decision layers"
    assert coordinator.peak_today > 0.0, "precondition: and still shown as today's peak"


@pytest.mark.asyncio
async def test_a_meter_masked_by_solar_bills_what_the_grid_actually_delivered(monkeypatch):
    """The owner's rule: if solar covers everything but 0.5 kW, count 0.5 kW.

    Compressor running hard at 60 Hz, meter reading 500 W because the panels are covering the rest.
    The import was 0.5 kW. The bill will be for 0.5 kW. So the record must say 0.5 kW.
    """
    coordinator = _coordinator(power_entity="sensor.house_power")
    _meter(coordinator.hass, "500")  # 500 W of grid import behind solar

    await _run_a_complete_quarter(coordinator, _pump(compressor_hz=60), monkeypatch)

    coordinator.effect.record_quarter_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_quarter_measurement.await_args.kwargs

    assert recorded["power_kw"] == pytest.approx(0.5), (
        f"The grid delivered 0.5 kW and {recorded['power_kw']:.2f} kW was recorded against the "
        f"tariff. The old 'smart fallback' replaced the meter reading with an ESTIMATE of what the "
        f"compressor was drawing (~5.5 kW) on the theory that solar was masking the meter. But the "
        f"operator bills grid import, and the import is exactly what the meter saw. The substitution "
        f"inflated the month's peak by an order of magnitude, in the owner's disfavour."
    )
    assert coordinator.peak_today == pytest.approx(0.5)
    assert coordinator.peak_today_source == POWER_SOURCE_EXTERNAL_METER


@pytest.mark.asyncio
async def test_a_working_meter_still_bills(monkeypatch):
    """The regression guard. Whole-house meter, ordinary reading, must still be recorded."""
    coordinator = _coordinator(power_entity="sensor.house_power")
    _meter(coordinator.hass, "4200")

    await _run_a_complete_quarter(coordinator, _pump(compressor_hz=60), monkeypatch)

    coordinator.effect.record_quarter_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_quarter_measurement.await_args.kwargs
    assert recorded["power_kw"] == pytest.approx(4.2)
