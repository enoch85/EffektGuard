"""Diagnostics: what the decision actually saw.

This integration commands a curve offset on a real heat pump from nine weighted layers, a
climate-zone degree-minute band that is recomputed per house, a compressor-wear risk and a
96-quarter price curve. When it gets that wrong, "the offset looked odd" is not a bug report.

So the dump carries the DECISION, not just the entity states: the offset it commanded, every
layer's vote and weight behind it, the NIBE state it read it from, the degree-minute thresholds
actually in force, and - the one people forget - whether the price and weather sources were even
live. A missing price source silently withdraws the entire price layer (audit F-123), and without
that fact the offset is inexplicable.

What it must NOT carry is the home's coordinates. The decision engine holds the latitude, because
that is how the climate zone is detected, and a diagnostics file is something the owner pastes into
a public issue tracker. The climate ZONE is what the thresholds derive from, and it identifies
nobody - so that is what goes in.
"""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .models.types import DiagnosticsDict
from .optimization.thermal_layer import apply_thermal_mass_buffer

_LOGGER = logging.getLogger(__name__)


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> DiagnosticsDict:
    """Return everything needed to argue with a decision this integration made."""
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is None:
        return {"error": "coordinator not loaded"}

    data = coordinator.data or {}
    nibe = data.get("nibe")
    decision = data.get("decision")

    return {
        "config": _config(entry),
        "sources": _sources(data),
        "nibe": _nibe(nibe),
        "decision": _decision(decision),
        "dm_thresholds": _dm_thresholds(coordinator, nibe),
        "compressor_risk": getattr(coordinator, "compressor_risk", None),
        "peaks": _peaks(coordinator),
    }


def _config(entry: ConfigEntry) -> dict[str, object]:
    """The user's configuration. Entity ids are not secrets; coordinates are, and are not here."""
    return {
        "data": dict(entry.data),
        "options": dict(entry.options),
    }


def _sources(data: dict[str, object]) -> dict[str, str]:
    """Which inputs were actually available.

    The single most useful line in the file. Price data of None is not a missing field - it means
    the price layer abstained entirely and every price-driven vote is absent from the decision
    below (F-123). Read the offset knowing that, or misread it.
    """
    return {
        "price": "live" if data.get("price") is not None else "ABSENT - price layer abstained",
        "weather": "live" if data.get("weather") is not None else "ABSENT - no forecast layers",
        "nibe": "live" if data.get("nibe") is not None else "ABSENT",
    }


def _nibe(nibe: object) -> dict[str, object]:
    """The state the decision was made from."""
    if nibe is None:
        return {}

    return {
        field: getattr(nibe, field, None)
        for field in (
            "degree_minutes",
            "indoor_temp",
            "outdoor_temp",
            "supply_temp",
            "return_temp",
            "current_offset",
            "compressor_hz",
            "power_kw",
            "is_heating",
            "is_hot_water",
        )
    }


def _decision(decision: object) -> dict[str, object]:
    """The offset, and the votes behind it.

    An offset without its layer votes cannot be argued with - it is just a number someone disagrees
    with. With them, the disagreement is about a specific layer's weight, which is a conversation.
    """
    if decision is None:
        return {}

    return {
        "offset": getattr(decision, "offset", None),
        "reasoning": getattr(decision, "reasoning", None),
        "is_emergency": getattr(decision, "is_emergency", None),
        "is_manual_override": getattr(decision, "is_manual_override", None),
        "anti_windup_active": getattr(decision, "anti_windup_active", None),
        "layers": [
            {
                "name": getattr(layer, "name", None),
                "offset": getattr(layer, "offset", None),
                "weight": getattr(layer, "weight", None),
                "reason": getattr(layer, "reason", None),
            }
            for layer in getattr(decision, "layers", []) or []
        ],
    }


def _dm_thresholds(coordinator: object, nibe: object) -> dict[str, object]:
    """The degree-minute band this house was actually being held to.

    Not the constants. The band is computed from the climate zone AND the outdoor temperature, so
    quoting DM_THRESHOLD_AUX_LIMIT tells you nothing about what governed this decision.

    The zone name goes in; the latitude it was derived from does not.
    """
    try:
        detector = coordinator.engine.climate_detector
        outdoor = getattr(nibe, "outdoor_temp", None)
        if detector is None or outdoor is None:
            return {}

        # The band production ENFORCES is the zone range run through the thermal-mass buffer -
        # a slab is helped ~1.3x sooner. Quoting the raw zone table here once told a slab-house
        # owner "-414" while the code was intervening at -318.
        heating_type = getattr(coordinator.engine.emergency_layer, "heating_type", "radiator")
        zone_range = detector.get_expected_dm_range(float(outdoor))
        return {
            "climate_zone": detector.zone_info.name,
            "outdoor_temp": outdoor,
            "heating_type": heating_type,
            "range": apply_thermal_mass_buffer(zone_range, heating_type),
            "zone_range_before_thermal_mass": zone_range,
        }
    except (AttributeError, TypeError, ValueError) as err:
        _LOGGER.debug("Could not resolve DM thresholds for diagnostics: %s", err)
        return {}


def _peaks(coordinator: object) -> dict[str, object]:
    """Effect-tariff state: what the peak protection was defending."""
    try:
        summary = coordinator.effect.get_monthly_peak_summary()
        return {"month_highest_kw": summary.get("highest")}
    except (AttributeError, TypeError) as err:
        _LOGGER.debug("Could not resolve peaks for diagnostics: %s", err)
        return {}
