"""Setting the thermostat to OFF must actually disable the optimiser, not just read OFF.

The coordinator's master gate is `entry.data["enable_optimization"]`. Setting HVACMode.OFF must
write that key (via `set_optimization_enabled`), or the optimiser goes quiet for one cycle and then
resumes driving the pump while the thermostat still displays OFF. There is ONE piece of state -
`entry.data["enable_optimization"]`, which the `enable_optimization` switch writes too - and the
thermostat's `hvac_mode` is a VIEW of it, so the two controls cannot disagree and the mode survives a
restart from the entry (no RestoreEntity shadowing it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components.climate.const import HVACMode

from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.const import CONF_ENABLE_OPTIMIZATION


def _climate(optimization_enabled: bool = True) -> tuple[EffektGuardClimate, MagicMock]:
    entry = MagicMock()
    entry.entry_id = "entry_1"
    entry.data = {CONF_ENABLE_OPTIMIZATION: optimization_enabled}
    entry.options = {}

    coordinator = MagicMock()

    async def _set_optimization_enabled(enabled: bool) -> None:
        new_data = dict(entry.data)
        new_data[CONF_ENABLE_OPTIMIZATION] = enabled
        entry.data = new_data

    coordinator.set_optimization_enabled = AsyncMock(side_effect=_set_optimization_enabled)
    coordinator.data = {}

    climate = EffektGuardClimate(coordinator, entry)
    climate.hass = MagicMock()
    climate.async_write_ha_state = MagicMock()

    # Home Assistant writes the new entry through and the entry object reflects it.
    def _update_entry(target_entry, data=None, options=None, **kwargs):
        if data is not None:
            target_entry.data = data
        if options is not None:
            target_entry.options = options

    climate.hass.config_entries.async_update_entry = MagicMock(side_effect=_update_entry)
    return climate, entry


@pytest.mark.asyncio
async def test_setting_the_thermostat_to_off_disables_the_master_gate():
    """THE BUG. OFF reset the offset once and left the optimiser enabled."""
    climate, entry = _climate(optimization_enabled=True)

    await climate.async_set_hvac_mode(HVACMode.OFF)

    assert entry.data[CONF_ENABLE_OPTIMIZATION] is False, (
        "the thermostat was set to OFF and `enable_optimization` is still "
        f"{entry.data[CONF_ENABLE_OPTIMIZATION]}. That key is the ONLY thing the coordinator's "
        "decision gate consults. So the optimiser goes quiet for a single cycle - the offset is "
        "reset to 0.0 - and then the next aligned refresh, five minutes later, decides an offset "
        "and writes it to the heat pump, while the thermostat still reads OFF."
    )


@pytest.mark.asyncio
async def test_setting_it_back_to_heat_re_enables_the_gate():
    """The other direction, or OFF becomes a trap you cannot leave."""
    climate, entry = _climate(optimization_enabled=False)

    await climate.async_set_hvac_mode(HVACMode.HEAT)

    assert entry.data[CONF_ENABLE_OPTIMIZATION] is True
    climate.coordinator.set_optimization_enabled.assert_awaited_with(True)


def test_the_thermostat_shows_what_the_optimiser_is_actually_doing():
    """The display must be a VIEW of the master gate, not a second copy of it.

    The `enable_optimization` SWITCH writes the same key. With two independent pieces of state, the
    switch could be off and the thermostat could read HEAT - one fact, two answers.
    """
    off_climate, _ = _climate(optimization_enabled=False)
    on_climate, _ = _climate(optimization_enabled=True)

    assert off_climate.hvac_mode == HVACMode.OFF, (
        "the master switch is off - the coordinator is holding a neutral offset and optimising "
        "nothing - and the thermostat says it is HEATing. The switch entity and the thermostat "
        "write the same fact and must read the same fact."
    )
    assert on_climate.hvac_mode == HVACMode.HEAT


def test_the_mode_survives_a_restart_because_it_lives_in_the_entry():
    """And it is the TRUTH that survives, not a display of it.

    The mode used to be restored by RestoreEntity from the entity's own last state - a copy of a copy.
    It restored OFF perfectly while the optimiser, whose gate had never been told anything, resumed
    driving the pump. The entry survives restarts on its own, and it is what the coordinator reads.
    """
    climate, entry = _climate(optimization_enabled=False)

    # A fresh entity, as after a restart: same entry, no restored entity state anywhere.
    reborn = EffektGuardClimate(climate.coordinator, entry)

    assert reborn.hvac_mode == HVACMode.OFF, (
        "after a restart the thermostat does not reflect the optimiser's actual state. It must be "
        "read from the config entry, which is the thing the coordinator's gate reads too."
    )
