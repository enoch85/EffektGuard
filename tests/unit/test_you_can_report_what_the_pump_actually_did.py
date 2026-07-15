"""The diagnostics hook must hand over what the DECISION saw, and must be downloadable.

`async_get_config_entry_diagnostics` carries the offset it commanded and every layer's vote, the
NIBE state it read, the degree-minute thresholds actually in force (computed per climate zone AND
thermal mass, so the constants prove nothing), and whether the price and weather sources were live
(a missing price source silently withdraws the whole price layer, F-123). It must NOT carry the
home's latitude - the dump is pasted into public issues - but keeps the climate ZONE, which
identifies nobody. The whole dump must JSON-serialise, or the download button 500s.

Separately: each platform declares PARALLEL_UPDATES (HA defaults a coordinator platform to 0,
unlimited); climate is 1 because `set_hvac_mode` reaches `_drive_the_pump`.
"""

from __future__ import annotations

import importlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_the_integration_can_produce_diagnostics():
    """The hook Home Assistant looks for."""
    diagnostics = pytest.importorskip(
        "custom_components.effektguard.diagnostics",
        reason="custom_components/effektguard/diagnostics.py does not exist",
    )

    assert hasattr(diagnostics, "async_get_config_entry_diagnostics"), (
        "diagnostics.py exists but does not define async_get_config_entry_diagnostics, which is "
        "the entry point Home Assistant calls."
    )


@pytest.mark.asyncio
async def test_the_dump_carries_what_the_decision_actually_saw():
    """A bug report about a heat pump has to contain the pump's state and the layer votes."""
    from custom_components.effektguard.diagnostics import async_get_config_entry_diagnostics

    hass, entry = _hass_and_entry()

    dump = await async_get_config_entry_diagnostics(hass, entry)

    decision = dump.get("decision", {})
    assert decision.get("offset") == 1.5, "the offset it commanded must be in the dump"
    assert decision.get("reasoning"), "the reasoning must be in the dump"
    assert decision.get("layers"), (
        "every layer's vote and weight must be in the dump. An offset without the votes behind it "
        "cannot be argued with."
    )

    nibe = dump.get("nibe", {})
    for field in ("degree_minutes", "indoor_temp", "outdoor_temp", "supply_temp", "current_offset"):
        assert field in nibe, f"the NIBE state the decision was made from is missing {field!r}"

    assert "dm_thresholds" in dump, (
        "the degree-minute thresholds actually in force must be in the dump. They are computed per "
        "climate zone AND per thermal mass, so quoting the constants proves nothing about what "
        "this house was being held to."
    )

    sources = dump.get("sources", {})
    assert "price" in sources and "weather" in sources, (
        "whether the price and weather sources were live must be in the dump: a missing price "
        "source silently withdraws the entire price layer (F-123), and the offset looks "
        "inexplicable without it."
    )


@pytest.mark.asyncio
async def test_the_dump_does_not_leak_the_home_location():
    """A diagnostics file is something the owner pastes into a public issue."""
    from custom_components.effektguard.diagnostics import async_get_config_entry_diagnostics

    hass, entry = _hass_and_entry()
    hass.config.latitude = 59.3293
    hass.config.longitude = 18.0686

    dump = await async_get_config_entry_diagnostics(hass, entry)

    flat = repr(dump)
    assert "59.3293" not in flat and "18.0686" not in flat, (
        "The diagnostics dump contains the home's latitude/longitude. The decision engine holds "
        "the latitude because that is how the climate zone is detected - and this file gets pasted "
        "into public issue trackers."
    )
    assert "climate_zone" in flat, (
        "Redacting the coordinates must not throw away the useful part: the climate ZONE (Cold, "
        "Very Cold...) is what the thresholds derive from, and it identifies nobody."
    )


@pytest.mark.parametrize("platform", ["climate", "sensor", "switch"])
def test_every_platform_declares_how_many_calls_it_will_take_at_once(platform):
    """PARALLEL_UPDATES is unset, and climate.set_hvac_mode drives the heat pump."""
    module = importlib.import_module(f"custom_components.effektguard.{platform}")

    assert hasattr(module, "PARALLEL_UPDATES"), (
        f"{platform}.py does not declare PARALLEL_UPDATES. Home Assistant defaults a "
        f"coordinator-based integration to 0 - unlimited concurrent entity service calls - and "
        f"climate.set_hvac_mode reaches set_optimization_enabled(), which calls "
        f"async_refresh_and_apply() and DRIVES THE PUMP."
    )


def test_the_entity_that_drives_the_pump_takes_one_call_at_a_time():
    """Belt and braces with the control lock, and honest about what the entity does."""
    from custom_components.effektguard import climate

    assert climate.PARALLEL_UPDATES == 1, (
        "climate.set_hvac_mode drives the heat pump (set_optimization_enabled -> "
        "async_refresh_and_apply -> _drive_the_pump). PARALLEL_UPDATES must be 1 so Home Assistant "
        "serialises the service calls, rather than 0 (unlimited) which is the coordinator default."
    )


def _hass_and_entry() -> tuple[MagicMock, MagicMock]:
    """A coordinator that has just made a decision, wired the way the integration wires it."""
    from custom_components.effektguard.adapters.nibe_adapter import NibeState
    from custom_components.effektguard.const import DOMAIN
    from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
    from custom_components.effektguard.optimization.decision_engine import (
        LayerDecision,
        OptimizationDecision,
    )

    nibe = NibeState(
        outdoor_temp=-5.0,
        indoor_temp=20.8,
        supply_temp=38.0,
        return_temp=33.0,
        degree_minutes=-320.0,
        current_offset=1.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0),
        compressor_hz=62,
        power_kw=2.4,
    )

    # REAL objects, not mocks: MagicMocks are unserialisable by construction, so the
    # serialisability check below needs the types production actually hands the hook.
    decision = OptimizationDecision(
        offset=1.5,
        reasoning="[Z2] DM -320, boost recovery speed | [Comfort] within band",
        layers=[LayerDecision(name="Emergency", offset=4.0, weight=0.65, reason="T1 recovery")],
        is_emergency=False,
    )

    coordinator = MagicMock()
    coordinator.data = {
        "nibe": nibe,
        "decision": decision,
        "price": None,  # F-123: no price source -> the whole price layer withdrew
        "weather": MagicMock(current_temp=-5.0),
    }
    coordinator.compressor_risk = "OK"
    # The real detector: Stockholm's latitude, so the zone and the band are the ones a real house
    # would be held to - and so the redaction has something genuine to redact.
    coordinator.engine.climate_detector = ClimateZoneDetector(latitude=59.3293)
    # A real string, as the real EmergencyLayer carries: the dump reports the band AFTER the
    # thermal-mass adjustment, so it reads this. An auto-MagicMock here would be unserialisable.
    coordinator.engine.emergency_layer.heating_type = "radiator"
    coordinator.effect.get_monthly_peak_summary.return_value = {"highest": 4.2}

    hass = MagicMock()
    hass.config = MagicMock(latitude=59.3293, longitude=18.0686)

    entry = MagicMock()
    entry.entry_id = "abc"
    entry.data = {"nibe_entity": "number.nibe_offset", "gespot_entity": None}
    entry.options = {"target_indoor_temp": 21.0}

    hass.data = {DOMAIN: {entry.entry_id: coordinator}}
    return hass, entry


@pytest.mark.asyncio
async def test_the_dump_can_actually_be_downloaded():
    """Home Assistant serialises the dump to JSON. If it cannot, the download button 500s.

    A datetime, an enum or a dataclass - anything json.dumps refuses - breaks the download.
    NibeState carries a `timestamp` and the degree-minute range comes back from a detector,
    so the risk is real.
    """
    import json

    from custom_components.effektguard.diagnostics import async_get_config_entry_diagnostics

    hass, entry = _hass_and_entry()

    dump = await async_get_config_entry_diagnostics(hass, entry)

    json.dumps(dump)  # raises TypeError on anything Home Assistant could not serve
