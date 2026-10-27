"""What learning works out never replaces the owner's thermal mass, or the look-ahead it sets.

The learned pre-heat calls the thermal model's get_parameters() every cycle. Once 96 hourly
observations existed, that wrote learning's own thermal mass over the owner's setting - the price
layer looks 5 h x thermal mass ahead - and from 192 the horizon of the learned pre-heat followed the
detected response type instead of the setting. A restart restored the type from storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.const import (
    CONF_THERMAL_MASS,
    DOMAIN,
    LEARNING_STORAGE_VERSION,
    STORAGE_KEY_LEARNING,
    UFH_CONCRETE_PREDICTION_HORIZON,
    UFHType,
)

CONCRETE_SLAB_MASS = 1.8
HOURS = 200  # past the 192 observations the response-type detector needs
START = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _a_fast_room(model) -> None:
    """Four hours at +2, four at 0: the room moves 0.8 C in the hour the offset changes.

    Learning reads that as a light house (thermal mass 1.25) answering within the hour (fast
    response, a 6 h horizon) - not what the owner set.
    """
    indoor = 20.0
    for hour in range(HOURS):
        offset = 2.0 if hour % 8 >= 4 else 0.0
        indoor += 0.8 if offset else -0.8
        model.record_observation(
            timestamp=START + timedelta(hours=hour),
            indoor_temp=indoor,
            outdoor_temp=0.0,
            heating_offset=offset,
        )


@pytest.mark.parametrize(
    "stored_type",
    [None, UFHType.FAST_RESPONSE.value],
    ids=["first start", "restart with a fast response type saved"],
)
async def test_learning_does_not_move_the_look_ahead(hass, hass_storage, stored_type):
    if stored_type is not None:
        hass_storage[STORAGE_KEY_LEARNING] = {
            "version": LEARNING_STORAGE_VERSION,
            "minor_version": 1,
            "key": STORAGE_KEY_LEARNING,
            "data": {
                "version": LEARNING_STORAGE_VERSION,
                "thermal_model": {"thermal_mass": 0.5, "ufh_type": stored_type},
            },
        }
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={CONF_THERMAL_MASS: CONCRETE_SLAB_MASS})
    entry.add_to_hass(hass)
    engine = (await _create_coordinator(hass, entry)).engine

    _a_fast_room(engine.thermal)
    learned = engine.thermal.get_parameters()  # as the learned pre-heat calls it, every cycle

    assert (learned.thermal_mass, learned.ufh_type) == (
        pytest.approx(1.25),
        UFHType.FAST_RESPONSE,
    ), "the observations no longer teach learning something other than the owner's setting"
    assert (engine.thermal.thermal_mass, engine.thermal.get_prediction_horizon()) == (
        pytest.approx(CONCRETE_SLAB_MASS),
        UFH_CONCRETE_PREDICTION_HORIZON,
    ), (
        f"the owner set thermal mass {CONCRETE_SLAB_MASS}, a {UFH_CONCRETE_PREDICTION_HORIZON:.0f} h "
        f"horizon; after learning the price layer and the learned pre-heat run on thermal mass "
        f"{engine.thermal.thermal_mass} and {engine.thermal.get_prediction_horizon():.0f} h"
    )
