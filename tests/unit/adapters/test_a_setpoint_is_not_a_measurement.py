"""A NIBE room-temperature SETPOINT could be discovered as the indoor MEASUREMENT.

Entity discovery bound a temperature key to any entity whose `device_class` is `temperature` or
whose unit is °C. It never looked at the DOMAIN.

A `number.` entity is, by definition, something the OWNER SETS. A NIBE room-temperature setpoint is
a `number.` with `device_class: temperature` and a unit of °C - every attribute that gate checked -
and its entity id can match the `room_temperature` discovery pattern.

Bound as the indoor measurement it is catastrophic and completely silent:

    the TARGET is read as the MEASUREMENT, and `indoor_temp_valid` is set to True,
    so the deviation from target is EXACTLY 0.0 forever, whatever the house is actually doing

The comfort layer therefore never corrects. And the 18 °C safety floor can never fire either,
because the safety layer is reading the same setpoint. A house at 12 °C in January reports itself
perfectly on target, and the flag built to prevent precisely this - `indoor_temp_valid` - is True.

The `offset` key already applied the mirror-image rule (a write target must BE a `number.`), so the
distinction is one this file already understood. And `NIBE_DISCOVERY_EXCLUDE` carries
`control_room_sensor` - which is this same problem, being fought one entity id at a time.

Manual entity overrides seed the cache directly and never reach discovery, so an installation that
really does expose a reading as a `number.` can still say so explicitly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    CONF_NIBE_ENTITY,
    NIBE_DISCOVERY_PATTERNS,
    NIBE_TEMPERATURE_KEYS,
)


def _adapter() -> NibeAdapter:
    return NibeAdapter(MagicMock(), {CONF_NIBE_ENTITY: "number.offset"})


def _consider(adapter: NibeAdapter, entity_id: str) -> dict[str, str]:
    """Run the real discovery candidate check against one entity."""
    adapter._entity_cache = {}
    adapter._consider_candidate(
        entity_id=entity_id,
        device_class="temperature",
        unit="°C",
        rank=0,
        ranks={},
        claimed=set(),
    )
    return adapter._entity_cache


def test_the_pattern_that_makes_this_reachable_is_still_there():
    """The premise. `room_temperature` matches a setpoint's entity id just as well as a sensor's."""
    assert "room_temperature" in NIBE_DISCOVERY_PATTERNS["indoor_temp"]
    assert "indoor_temp" in NIBE_TEMPERATURE_KEYS


@pytest.mark.parametrize(
    "setpoint",
    [
        "number.nibe_room_temperature_setpoint_s1",
        "number.f750_room_temperature_s1_47398",
        "number.heatpump_room_temperature",
    ],
)
def test_a_writable_setpoint_is_never_bound_as_the_indoor_measurement(setpoint):
    cache = _consider(_adapter(), setpoint)

    assert "indoor_temp" not in cache, (
        f"Discovery bound {setpoint} - a WRITABLE setpoint, something the owner sets - as the "
        f"indoor temperature MEASUREMENT. The target is then read as the measurement with "
        f"indoor_temp_valid=True, so the deviation from target is exactly 0.0 forever, the comfort "
        f"layer never corrects, and the 18 C safety floor can never fire because it is reading the "
        f"same setpoint. A house at 12 C in January would report itself perfectly on target."
    )


@pytest.mark.parametrize(
    ("entity_id", "key"),
    [
        ("sensor.nibe_bt50_room_temperature", "indoor_temp"),
        ("sensor.nibe_bt1_outdoor_temperature", "outdoor_temp"),
        ("sensor.nibe_bt25_supply_temperature", "supply_temp"),
    ],
)
def test_a_real_sensor_is_still_discovered(entity_id, key):
    """The regression guard. Do not break discovery while hardening it."""
    cache = _consider(_adapter(), entity_id)

    assert cache.get(key) == entity_id, (
        f"{entity_id} is an ordinary temperature sensor and discovery no longer finds it as "
        f"{key}. The domain rule must reject setpoints, not measurements."
    )


def test_every_temperature_key_is_protected_not_just_the_indoor_one():
    """A setpoint bound as the SUPPLY temperature would drive weather compensation on a target."""
    adapter = _adapter()

    for key in NIBE_TEMPERATURE_KEYS:
        patterns = NIBE_DISCOVERY_PATTERNS.get(key, [])
        if not patterns:
            continue
        entity_id = f"number.nibe{patterns[0]}_setpoint"
        cache = _consider(adapter, entity_id)

        assert key not in cache, (
            f"A `number.` entity matching the {key} pattern was bound as a {key} MEASUREMENT. "
            f"Every temperature key reads a value the pump reports; none of them is something the "
            f"owner sets."
        )


def test_the_write_target_still_has_to_be_a_number():
    """The mirror-image rule, which this file already had. It must survive."""
    adapter = _adapter()
    adapter._entity_cache = {}
    adapter._consider_candidate(
        entity_id="sensor.nibe_heat_offset_s1_47011",
        device_class=None,
        unit=None,
        rank=0,
        ranks={},
        claimed=set(),
    )

    assert "offset" not in adapter._entity_cache, (
        "A `sensor.` was bound as the OFFSET write target. The write path calls number.set_value; "
        "a sensor can never work."
    )
