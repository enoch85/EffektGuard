"""One room sensor moving must move the indoor reading by a fraction of its change, not all of it.

The median follows one sensor at a time. The owner's nine sensors (2026-10-20 02:55) sorted 19.8,
20.7, 20.95, 21.44, 22.0, 22, 22, 22.03, 22.66 with BT50 in the middle; BT50 alone fell 22.0 -> 21.7
and the reading fell with it, 0.3 °C, which the comfort layer turned from -5.17 at weight 0.61 into
-2.27 at 0.37 and the pump went -2 -> -1. In 3.6 days of v0.7.0 the median jumped by 0.2 °C or more
31 times. The interquartile mean of the same sensors moves by at most a fifth of one sensor's change.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import async_migrate_entry
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.config_flow import EffektGuardConfigFlow
from custom_components.effektguard.const import (
    CONF_INDOOR_TEMP_METHOD,
    CONF_NIBE_ENTITY,
    CONFIG_ENTRY_MINOR_VERSION,
    DEFAULT_INDOOR_TEMP_METHOD,
    DOMAIN,
    INDOOR_TEMP_METHOD_AVERAGE,
    INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN,
    INDOOR_TEMP_METHOD_MEDIAN,
    INDOOR_TEMP_METHODS,
)
from custom_components.effektguard.utils.room_temperature import (
    combine_room_temperatures,
    interquartile_mean,
    median,
)

# The owner's nine sensors at 02:50, BT50 first.
OWNER = [22.0, 19.8, 20.7, 20.95, 21.44, 22.0, 22.0, 22.03, 22.66]


def test_the_owners_bt50_moving_moves_the_reading_a_fifth_as_much():
    after = [21.7] + OWNER[1:]

    assert median(OWNER) - median(after) == pytest.approx(0.3), "the precondition: the median"
    moved = interquartile_mean(OWNER) - interquartile_mean(after)
    assert 0 < moved <= 0.3 / 5 + 1e-9, (
        f"BT50 fell 0.3 °C and the reading fell {moved:.3f} °C. Five sensors are kept; one of them "
        f"moving should move their mean by at most a fifth of its change."
    )


def test_a_sensor_crossing_the_middle_does_not_drag_the_reading_with_it():
    """The median's worst case: one sensor moves from above the gap to below it."""
    after = list(OWNER)
    after[5] = 21.0  # one of the 22.0 rooms cools a degree

    assert median(OWNER) - median(after) == pytest.approx(0.56), "the median jumps to 21.44"
    assert interquartile_mean(OWNER) - interquartile_mean(after) <= 1.0 / 5 + 1e-9


def test_no_single_sensor_moves_the_reading_more_than_its_share():
    rng = random.Random(20261022)
    for _ in range(2000):
        count = rng.randint(4, 12)
        temps = [rng.uniform(18.0, 24.0) for _ in range(count)]
        which, change = rng.randrange(count), rng.uniform(-2.0, 2.0)
        moved = list(temps)
        moved[which] += change
        kept = count - 2 * (count // 4)

        shift = abs(interquartile_mean(moved) - interquartile_mean(temps))

        assert shift <= abs(change) / kept + 1e-9, (temps, which, change)


def test_an_outlier_room_is_left_out():
    """What the median was chosen for: a sunlit room or a sensor on a radiator."""
    assert interquartile_mean([21.0, 21.2, 21.4, 29.0]) == pytest.approx(21.3)
    assert interquartile_mean([16.0, 21.0, 21.2, 21.4]) == pytest.approx(21.1)


@pytest.mark.parametrize(
    ("temps", "expected"),
    [([21.0], 21.0), ([20.0, 21.0], 20.5), ([20.0, 21.0, 23.0], (20.0 + 21.0 + 23.0) / 3)],
)
def test_with_fewer_than_four_sensors_it_is_their_mean(temps, expected):
    assert interquartile_mean(temps) == pytest.approx(expected)


def test_median_and_average_stay_what_they_say():
    temps = [20.0, 21.0, 24.0]
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_MEDIAN) == pytest.approx(21.0)
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_AVERAGE) == pytest.approx(65 / 3)
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN) == pytest.approx(
        interquartile_mean(temps)
    )


def _adapter(config: dict) -> NibeAdapter:
    """The owner's pump and its eight extra room sensors, all fresh."""
    readings = {
        "sensor.bt1": "-3.0",
        "sensor.bt2": "35.0",
        "sensor.dm": "-150",
        "sensor.bt50": str(OWNER[0]),
        **{f"sensor.room{i}": str(t) for i, t in enumerate(OWNER[1:])},
    }

    def get(entity_id):
        if entity_id not in readings:
            return None
        state = MagicMock()
        state.state = readings[entity_id]
        state.attributes = {"unit_of_measurement": "°C"}
        state.last_reported = state.last_updated = dt_util.utcnow()
        return state

    hass = MagicMock()
    hass.states.get.side_effect = get
    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset", **config})
    adapter._entity_cache = {
        "outdoor_temp": "sensor.bt1",
        "supply_temp": "sensor.bt2",
        "indoor_temp": "sensor.bt50",
        "degree_minutes": "sensor.dm",
    }
    adapter._additional_indoor_sensors = [f"sensor.room{i}" for i in range(len(OWNER) - 1)]
    return adapter


@pytest.mark.asyncio
async def test_the_adapter_combines_the_room_sensors_by_their_interquartile_mean():
    state = await _adapter({}).get_current_state()

    assert state.indoor_temp == pytest.approx(interquartile_mean(OWNER))


@pytest.mark.asyncio
async def test_a_chosen_median_is_still_honoured():
    state = await _adapter({CONF_INDOOR_TEMP_METHOD: INDOOR_TEMP_METHOD_MEDIAN}).get_current_state()

    assert state.indoor_temp == pytest.approx(median(OWNER))


@pytest.mark.parametrize(
    ("stored", "migrated"),
    [
        (INDOOR_TEMP_METHOD_MEDIAN, INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN),
        (INDOOR_TEMP_METHOD_AVERAGE, INDOOR_TEMP_METHOD_AVERAGE),
        (None, None),
    ],
)
async def test_an_entry_from_an_older_version_moves_off_the_median(hass, stored, migrated):
    data = {CONF_NIBE_ENTITY: "number.offset"}
    if stored is not None:
        data[CONF_INDOOR_TEMP_METHOD] = stored
    entry = MockConfigEntry(domain=DOMAIN, version=1, minor_version=1, data=data)
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.data.get(CONF_INDOOR_TEMP_METHOD) == migrated
    assert entry.data[CONF_NIBE_ENTITY] == "number.offset"
    assert entry.minor_version == CONFIG_ENTRY_MINOR_VERSION


async def test_an_entry_from_a_newer_major_version_is_refused(hass):
    entry = MockConfigEntry(domain=DOMAIN, version=2, minor_version=1, data={})
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry) is False


def test_the_flow_asks_home_assistant_to_migrate():
    """Home Assistant calls async_migrate_entry only when the flow's version is ahead."""
    assert EffektGuardConfigFlow.MINOR_VERSION == CONFIG_ENTRY_MINOR_VERSION > 1
    assert DEFAULT_INDOOR_TEMP_METHOD == INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN


@pytest.mark.asyncio
async def test_reconfigure_offers_and_stores_the_method():
    """Setup was the only place to choose it; a migrated user must be able to choose again."""
    entry = MagicMock()
    entry.data = {
        CONF_NIBE_ENTITY: "number.offset",
        CONF_INDOOR_TEMP_METHOD: INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN,
    }
    flow = EffektGuardConfigFlow()
    flow.hass = MagicMock()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)
    flow.async_update_reload_and_abort = MagicMock(return_value={"type": "abort"})

    form = await flow.async_step_reconfigure()
    field = next(k for k in form["data_schema"].schema if k == CONF_INDOOR_TEMP_METHOD)
    assert field.default() == INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN
    selector = form["data_schema"].schema[field]
    assert list(selector.config["options"]) == list(INDOOR_TEMP_METHODS)

    await flow.async_step_reconfigure({CONF_INDOOR_TEMP_METHOD: INDOOR_TEMP_METHOD_MEDIAN})

    stored = flow.async_update_reload_and_abort.call_args.kwargs["data_updates"]
    assert stored[CONF_INDOOR_TEMP_METHOD] == INDOOR_TEMP_METHOD_MEDIAN
