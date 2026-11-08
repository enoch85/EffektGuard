"""The production save/restart path preserves the evidence behind learned parameters."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from types import SimpleNamespace

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.const import (
    CONF_INSULATION_QUALITY,
    CONF_THERMAL_MASS,
    DOMAIN,
    LEARNING_CONFIDENCE_THRESHOLD,
)

NOW = datetime(2026, 1, 15, tzinfo=timezone.utc)
MASS = 1.8
INSULATION = 0.7


async def test_a_restart_keeps_the_parameters_and_their_confidence(hass, hass_storage, freezer):
    freezer.move_to(NOW)
    options = {CONF_THERMAL_MASS: MASS, CONF_INSULATION_QUALITY: INSULATION}
    entry = MockConfigEntry(domain=DOMAIN, data={}, options=options)
    entry.add_to_hass(hass)
    source = await _create_coordinator(hass, entry)
    from tests.unit.optimization.test_learning_can_actually_learn import _observe_a_real_house

    calibration = _observe_a_real_house(60, start=NOW - timedelta(hours=30 * 24 - 1))
    for observation in calibration.observations:
        source.adaptive_learning.record_observation(
            observation.timestamp,
            observation.indoor_temp,
            observation.outdoor_temp,
            observation.heating_offset,
        )
    before = source.adaptive_learning.get_parameters()
    assert before.confidence >= LEARNING_CONFIDENCE_THRESHOLD
    await source._save_learned_data(
        source.adaptive_learning, source.thermal_predictor, source.weather_learner
    )

    restored_entry = MockConfigEntry(domain=DOMAIN, data={}, options=options)
    restored_entry.add_to_hass(hass)
    restored = await _create_coordinator(hass, restored_entry)
    after = restored.adaptive_learning.get_parameters()

    assert after is not None
    assert asdict(after) == asdict(before)
    assert list(restored.adaptive_learning.observations) == list(
        source.adaptive_learning.observations
    )
    assert restored.engine.thermal is restored.adaptive_learning
    assert restored.engine.thermal.thermal_mass == MASS
    assert restored.engine.thermal.insulation_quality == pytest.approx(INSULATION)
    assert restored._last_learning_observation == NOW
    await restored._record_learning_observations(
        SimpleNamespace(indoor_temp=21.0, outdoor_temp=-5.0, flow_temp=35.0, degree_minutes=-200.0),
        None,
        0.0,
    )
    assert list(restored.adaptive_learning.observations) == list(
        source.adaptive_learning.observations
    )
