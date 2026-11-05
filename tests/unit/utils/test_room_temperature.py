"""How the room sensors combine: the interquartile mean, and the median and average still on offer.

Through the adapter and the config-entry lifecycle:
tests/unit/adapters/test_one_room_sensor_barely_moves_the_reading.py.
"""

from __future__ import annotations

import random

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import async_migrate_entry
from custom_components.effektguard.const import (
    DOMAIN,
    INDOOR_TEMP_METHOD_AVERAGE,
    INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN,
    INDOOR_TEMP_METHOD_MEDIAN,
)
from custom_components.effektguard.utils.room_temperature import (
    combine_room_temperatures,
    interquartile_mean,
    median,
)


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
    assert median(temps) == pytest.approx(21.0)
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_MEDIAN) == pytest.approx(21.0)
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_AVERAGE) == pytest.approx(65 / 3)
    assert combine_room_temperatures(temps, INDOOR_TEMP_METHOD_INTERQUARTILE_MEAN) == pytest.approx(
        interquartile_mean(temps)
    )


async def test_an_entry_from_a_newer_major_version_is_refused(hass):
    """Home Assistant 2025.11, the minimum, hands a newer entry to the integration to judge."""
    entry = MockConfigEntry(domain=DOMAIN, version=2, minor_version=1, data={})
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry) is False
