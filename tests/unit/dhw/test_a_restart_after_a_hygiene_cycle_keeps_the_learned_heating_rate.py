"""A restart after a detected hygiene cycle keeps the hot-water heating rate learned so far.

A detected high-temperature cycle is saved at once, so a restart cannot forget it. That save wrote a
hot-water state of the cycle's time alone over the stored one, and the learned heating rate - which
sets how early the scheduled hot-water window opens - was gone until the next hourly save. A restart
in between fell back to the default rate.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.const import DOMAIN

LEARNED_RATE = 9.5  # °C/h - a slow tank, well off DHW_DEFAULT_HEATING_RATE
LEARNED_OBSERVATIONS = 4
CYCLE_SEEN_AT = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
PREVIOUS_CYCLE = CYCLE_SEEN_AT - timedelta(days=7)


async def test_the_learned_heating_rate_survives_a_restart_after_a_hygiene_cycle(
    hass, hass_storage
):
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    before = await _create_coordinator(hass, entry)
    dhw = before.dhw_optimizer
    dhw.learned_heating_rate = LEARNED_RATE
    dhw.heating_rate_observations = LEARNED_OBSERVATIONS
    dhw.last_legionella_boost = PREVIOUS_CYCLE
    await before._save_learned_data(
        before.adaptive_learning, before.thermal_predictor, before.weather_learner
    )  # the hourly save

    dhw.last_legionella_boost = CYCLE_SEEN_AT  # a cycle is detected, and saved at once
    await before._save_dhw_state_immediate()

    after = (await _create_coordinator(hass, entry)).dhw_optimizer  # restart

    restored = (
        after.last_legionella_boost,
        after.learned_heating_rate,
        after.heating_rate_observations,
    )
    assert restored == (
        CYCLE_SEEN_AT,
        LEARNED_RATE,
        LEARNED_OBSERVATIONS,
    ), f"restarted after the cycle was saved: (cycle, rate, observations) = {restored}"
