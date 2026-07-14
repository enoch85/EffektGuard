"""Turning the thermostat OFF did not turn the optimiser off. It reads OFF while it drives the pump.

The climate entity offers HVACMode.OFF and documents it as:

    OFF: Optimization disabled (safety monitoring only)

What it actually does:

    async def async_set_hvac_mode(self, hvac_mode):
        self._attr_hvac_mode = hvac_mode                       # a private copy
        if hvac_mode == HVACMode.OFF:
            await self.coordinator.set_optimization_enabled(False)

and `set_optimization_enabled(False)` resets the curve offset to 0.0 once. That is all it does. It
writes no flag anywhere.

The coordinator's master gate reads something else entirely - the config entry:

    if not self.entry.data.get("enable_optimization", True):
        decision = OptimizationDecision(offset=0.0, reasoning="Optimization disabled by user", ...)

Nothing sets that key except the `enable_optimization` SWITCH entity, which writes it into
`entry.data` with `async_update_entry`. The thermostat never touches it. So:

    user sets the thermostat to:               off
    entry.data['enable_optimization'] =        True     <- the only thing the gate checks
    next aligned refresh, five minutes later:  optimises, decides an offset, writes it to the pump

The user turned the heating optimiser off, it went quiet for one cycle, and then it went back to
driving their heat pump - with the thermostat still displaying OFF. And because the mode lived in a
private attribute rather than in the entry, RestoreEntity dutifully restored the OFF display across a
Home Assistant restart while the optimiser ran on, so the lie survived a reboot.

The two controls could also simply disagree: switch off, thermostat HEAT. Two pieces of state for one
fact, which is the failure this audit has now found in the billed quantity, in the DST hour, and in
who started a hot-water boost.

There is ONE piece of state now - `entry.data["enable_optimization"]` - and the thermostat is a view
of it, not a second copy.
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
