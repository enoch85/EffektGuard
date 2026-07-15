"""Diagnostics: what the decision actually saw.

The dump carries the DECISION, not just entity states: the commanded offset, every layer's vote
and weight, the NIBE state behind it, the degree-minute band actually in force, and whether the
price and weather sources were live - a missing price source silently withdraws the whole price
layer (F-123), leaving the offset inexplicable without that fact.

It must NOT carry the home's coordinates: this file gets pasted into public issue trackers. The
climate ZONE identifies nobody and is what the thresholds derive from, so the zone goes in and the
latitude does not.
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

    Price data of None is not a missing field - the price layer abstained entirely and every
    price-driven vote is absent from the decision below (F-123). Read the offset knowing that.
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
    """The offset, and the layer votes behind it - without which the offset cannot be argued with."""
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
    """The degree-minute band this house was actually held to.

    Computed from the climate zone AND outdoor temperature, so the DM constants say nothing about
    what governed this decision. The zone name goes in; the latitude it derived from does not.
    """
    try:
        detector = coordinator.engine.climate_detector
        outdoor = getattr(nibe, "outdoor_temp", None)
        if detector is None or outdoor is None:
            return {}

        # The enforced band is the zone range run through apply_thermal_mass_buffer (a high-mass
        # slab is helped sooner); the raw zone table alone understates where the code intervenes.
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
