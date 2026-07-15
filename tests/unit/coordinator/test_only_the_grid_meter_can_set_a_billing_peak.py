"""Only a whole-house meter reading can become a billing peak.

Two things were once recorded against the tariff that the grid did not deliver:

- NIBE phase currents (BE1/BE2/BE3) measure the heat pump only - not the oven, EV charger or kettle
  - yet were accepted as a whole-house billing measurement. They are now control-grade, not billable:
  available to the decision layers (which want a magnitude), never reported as the month's bill.

- A solar "smart fallback" substituted an ESTIMATED compressor power when a grid-import meter read
  under 0.5 kW while the compressor ran hard, then billed the estimate (~5.5 kW where the grid
  imported 0.3 kW). The operator bills grid import, which is exactly what the meter saw. The fallback
  is gone: the meter reading is the truth.
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
    coordinator.effect.record_period_measurement = AsyncMock(return_value=None)
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


async def _run_a_complete_billing_hour(coordinator, nibe_data, monkeypatch) -> None:
    """Samples through a whole HOUR, because that is the tariff's billing period."""
    for hour, minute in [(10, m) for m in range(0, 60, 5)] + [(11, 0)]:
        monkeypatch.setattr(
            dt_util,
            "now",
            lambda tz=None, hour=hour, minute=minute: datetime(
                2026, 1, 15, hour, minute, tzinfo=timezone.utc
            ),
        )
        await coordinator._update_peak_tracking(nibe_data)


def test_only_a_whole_house_meter_is_billable():
    """The rule, stated once, where both the recorder and the reporting read it."""
    assert BILLABLE_POWER_SOURCES == frozenset({POWER_SOURCE_EXTERNAL_METER}), (
        f"BILLABLE_POWER_SOURCES is {sorted(BILLABLE_POWER_SOURCES)}. The Swedish effect tariff bills "
        f"whole-house grid import. Only a whole-house meter measures that."
    )


@pytest.mark.asyncio
async def test_nibe_phase_currents_still_drive_peak_protection(monkeypatch):
    """NOT BILLABLE and NOT RECORDED are different things; conflating them would break the feature.

    A house without a whole-house meter must still record NIBE-currents peaks - gating recording on
    billability would leave `should_limit_power` with an empty history, and peak protection would
    never fire. `should_limit_power` compares this quarter against the month's own recorded peaks, so
    a NIBE-only history against NIBE-only power is self-consistent and still throttles the pump. That
    number must never be reported to the owner as the month's BILLING peak.
    """
    coordinator = _coordinator(power_entity=None)  # no whole-house meter, only NIBE currents

    await _run_a_complete_billing_hour(
        coordinator, _pump(compressor_hz=60, currents=10.0), monkeypatch
    )

    coordinator.effect.record_period_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_period_measurement.await_args.kwargs

    assert recorded["source"] == POWER_SOURCE_NIBE_CURRENTS, (
        f"The peak was recorded as {recorded['source']!r}. It must carry its provenance, because "
        f"that is the only thing standing between a pump-only measurement and a billing figure."
    )
    assert coordinator.peak_today_source == POWER_SOURCE_NIBE_CURRENTS
    assert coordinator.peak_today > 0.0


@pytest.mark.asyncio
async def test_a_nibe_currents_peak_is_never_billable(monkeypatch):
    """It drives control. It is not the bill. The PeakEvent itself has to know the difference."""
    from custom_components.effektguard.optimization.effect_layer import PeakEvent

    from_currents = PeakEvent(
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        period_of_day=40,
        actual_power=6.8,
        effective_power=6.8,
        is_daytime=True,
        source=POWER_SOURCE_NIBE_CURRENTS,
    )
    from_meter = PeakEvent(
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        period_of_day=40,
        actual_power=6.8,
        effective_power=6.8,
        is_daytime=True,
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert not from_currents.is_billable, (
        "A peak measured from the pump's own phase currents was marked billable. BE1/BE2/BE3 "
        "measure the heat pump - not the oven, not the EV charger. The tariff bills the house."
    )
    assert from_meter.is_billable

    # And it must survive a round-trip through storage, or the distinction is lost on the next
    # Home Assistant restart - which is exactly when nobody is watching.
    assert PeakEvent.from_dict(from_currents.to_dict()).is_billable is False
    assert PeakEvent.from_dict(from_meter.to_dict()).is_billable is True


@pytest.mark.asyncio
async def test_an_estimate_drives_nothing_at_all(monkeypatch):
    """Compressor-Hz estimates are excluded from BOTH. A guess must not throttle a house."""
    coordinator = _coordinator(power_entity=None)

    # No meter, no phase currents: PRIORITY 3 falls through to a compressor-Hz estimate.
    await _run_a_complete_billing_hour(
        coordinator, _pump(compressor_hz=60, currents=None), monkeypatch
    )

    coordinator.effect.record_period_measurement.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_meter_masked_by_solar_bills_what_the_grid_actually_delivered(monkeypatch):
    """The owner's rule: if solar covers everything but 0.5 kW, count 0.5 kW.

    Compressor running hard at 60 Hz, meter reading 500 W because the panels are covering the rest.
    The import was 0.5 kW. The bill will be for 0.5 kW. So the record must say 0.5 kW.
    """
    coordinator = _coordinator(power_entity="sensor.house_power")
    _meter(coordinator.hass, "500")  # 500 W of grid import behind solar

    await _run_a_complete_billing_hour(coordinator, _pump(compressor_hz=60), monkeypatch)

    coordinator.effect.record_period_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_period_measurement.await_args.kwargs

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

    await _run_a_complete_billing_hour(coordinator, _pump(compressor_hz=60), monkeypatch)

    coordinator.effect.record_period_measurement.assert_awaited_once()
    recorded = coordinator.effect.record_period_measurement.await_args.kwargs
    assert recorded["power_kw"] == pytest.approx(4.2)
