"""NIBE's Modbus registers hold DECI-degrees. Omit `scale: 0.1` and BT50 reads 213 C.

`get_current_state` already states the principle, about a MISSING sensor:

    Never substitute a plausible constant for a missing one: that makes a broken installation
    indistinguishable from a healthy one and still writes a curve offset to the pump.

A value that cannot be a temperature is the same thing wearing a number. And the mechanism is
mundane, not exotic - the repo's OWN Modbus simulator documents the register:

    40033 BT50 room temp        213   (21.3 C)

A hand-written Modbus YAML that omits `scale: 0.1` reports that as 213.0 C.

THE ADAPTER HAD A PLAUSIBILITY BAND AND APPLIED IT TO THE WRONG SENSORS. It checked the ADDITIONAL
room sensors the user adds - arbitrary entities, so the caution is fair - and did NOT check the one
the HEAT PUMP sends, which is the only one exposed to a scaling typo in the first place.

At 213.0 C indoor:

    comfort layer -> offset -10.00 at weight 1.00, "Overshoot: 192.0 C above target"

Maximum heat reduction, at critical weight, in a Swedish January, forever. And the 18 C safety
floor never fires either, because the safety layer is reading the same 213 C.

AND THE PLACEHOLDER WAS BEING SEEDED INTO THE MEDIAN. `_calculate_multi_sensor_temperature`'s own
docstring forbids it in as many words - "A placeholder must NEVER be passed here - seeding the
median with DEFAULT_INDOOR_TEMP would drag the combined reading toward the target and mask a real
deviation" - and it was being passed anyway. On a sensorless NIBE with one added sensor reading
17.0 C in a house targeting 21.0, the median of [21.0, 17.0] is 19.0: a two-degree mask on a cold
house, biased toward the target.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    DEFAULT_INDOOR_TEMP,
    INDOOR_SENSOR_PLAUSIBLE_MAX,
    INDOOR_SENSOR_PLAUSIBLE_MIN,
    NIBE_OUTDOOR_PLAUSIBLE_MAX,
    NIBE_OUTDOOR_PLAUSIBLE_MIN,
    NIBE_WATER_PLAUSIBLE_MAX,
    NIBE_WATER_PLAUSIBLE_MIN,
)

# The registers, as the repo's own Modbus simulator documents them, and what a missing
# `scale: 0.1` turns each of them into.
DECI_SCALING_TYPO = {
    "BT50 room temp (register 40033)": (213, 21.3, 213.0),
    "BT1 outdoor temp (register 40004)": (-32, -3.2, -32.0),
    "BT2 supply temp (register 40008)": (358, 35.8, 358.0),
}


def _adapter(states: dict[str, str]) -> NibeAdapter:
    hass = MagicMock()

    def get(entity_id):
        if entity_id not in states:
            return None
        state = MagicMock()
        state.state = states[entity_id]
        state.attributes = {"unit_of_measurement": "°C"}
        state.last_reported = dt_util.utcnow()
        state.last_updated = state.last_reported
        return state

    hass.states.get.side_effect = get

    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset"})
    adapter._entity_cache = {
        "outdoor_temp": "sensor.bt1",
        "supply_temp": "sensor.bt2",
        "indoor_temp": "sensor.bt50",
        "degree_minutes": "sensor.dm",
    }
    return adapter


HEALTHY = {
    "sensor.bt1": "-3.2",
    "sensor.bt2": "35.8",
    "sensor.bt50": "21.3",
    "sensor.dm": "-150",
}


def test_the_deci_degree_trap_is_real_and_this_is_what_it_looks_like():
    """The premise, spelled out, so nobody argues the scenario is contrived."""
    for name, (register, correct, unscaled) in DECI_SCALING_TYPO.items():
        assert register / 10.0 == pytest.approx(correct), f"{name}: check the fixture"
        assert unscaled == pytest.approx(float(register)), f"{name}: check the fixture"


class TestTheRoomSensorTheHeatPumpSends:
    """BT50 is the one exposed to the typo, and it was the one not being checked."""

    @pytest.mark.asyncio
    async def test_a_bt50_reading_213_degrees_is_not_a_room_temperature(self):
        adapter = _adapter({**HEALTHY, "sensor.bt50": "213.0"})

        state = await adapter.get_current_state()

        assert state.indoor_temp_valid is False, (
            f"BT50 reported 213.0 C - a missing `scale: 0.1` on a deci-degree register - and it "
            f"was accepted as a room temperature with indoor_temp_valid=True. The comfort layer "
            f"then reads a 192 C overshoot and commands -10.0 C at critical weight, forever, and "
            f"the 18 C safety floor never fires because it is reading the same 213 C."
        )
        assert state.indoor_temp == DEFAULT_INDOOR_TEMP, (
            "An implausible BT50 must degrade to 'no room sensor' - a configuration this "
            "integration already handles, by having the comfort-reasoning layers abstain."
        )

    @pytest.mark.asyncio
    async def test_a_healthy_bt50_is_still_trusted(self):
        state = await _adapter(HEALTHY).get_current_state()

        assert state.indoor_temp_valid is True
        assert state.indoor_temp == pytest.approx(21.3)

    @pytest.mark.parametrize("reading", [15.0, 21.3, 30.0])
    @pytest.mark.asyncio
    async def test_the_whole_habitable_band_is_accepted(self, reading):
        """The band's job is to catch a value that cannot be a temperature, not to second-guess."""
        state = await _adapter({**HEALTHY, "sensor.bt50": str(reading)}).get_current_state()

        assert state.indoor_temp_valid is True
        assert INDOOR_SENSOR_PLAUSIBLE_MIN <= state.indoor_temp <= INDOOR_SENSOR_PLAUSIBLE_MAX


class TestTheRequiredSensors:
    """Outdoor and supply drive every decision. An impossible one must stop the integration."""

    @pytest.mark.asyncio
    async def test_a_bt1_reading_105_below_zero_stops_the_integration(self):
        """-105 C demands a 96.8 C flow and pushes the DM warning to within 50 of the aux limit."""
        adapter = _adapter({**HEALTHY, "sensor.bt1": "-105.0"})

        with pytest.raises(UpdateFailed, match="outdoor temperature"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_a_supply_temperature_of_358_degrees_stops_the_integration(self):
        """A missing scale on BT2: 358 deci-degrees is 35.8 C. Water cannot be at 358 C."""
        adapter = _adapter({**HEALTHY, "sensor.bt2": "358.0"})

        with pytest.raises(UpdateFailed, match="supply"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_a_healthy_pump_still_reads(self):
        state = await _adapter(HEALTHY).get_current_state()

        assert state.outdoor_temp == pytest.approx(-3.2)
        assert state.supply_temp == pytest.approx(35.8)
        assert state.degree_minutes == pytest.approx(-150.0)

    @pytest.mark.parametrize(
        ("outdoor", "ok"),
        [(-45.0, True), (-50.0, True), (-51.0, False), (40.0, True), (60.0, False)],
    )
    @pytest.mark.asyncio
    async def test_the_outdoor_band_reaches_below_kiruna(self, outdoor, ok):
        """Kiruna reaches -40 C. The band must not reject a real Nordic winter."""
        assert NIBE_OUTDOOR_PLAUSIBLE_MIN <= -45.0, "the band must accommodate Kiruna"
        adapter = _adapter({**HEALTHY, "sensor.bt1": str(outdoor)})

        if ok:
            state = await adapter.get_current_state()
            assert state.outdoor_temp == pytest.approx(outdoor)
        else:
            with pytest.raises(UpdateFailed):
                await adapter.get_current_state()

    def test_water_cannot_freeze_or_boil(self):
        assert NIBE_WATER_PLAUSIBLE_MIN == 0.0
        assert NIBE_WATER_PLAUSIBLE_MAX == 100.0
        assert NIBE_OUTDOOR_PLAUSIBLE_MAX < NIBE_WATER_PLAUSIBLE_MAX


class TestThePlaceholderNeverSeedsTheMedian:
    """`_calculate_multi_sensor_temperature`'s own docstring forbids exactly what was happening."""

    @pytest.mark.asyncio
    async def test_a_sensorless_pump_with_one_added_sensor_reports_that_sensor(self):
        """median([21.0 placeholder, 17.0 real]) is 19.0. The house is at 17.0."""
        adapter = _adapter({**HEALTHY, "sensor.hall": "17.0"})
        del adapter._entity_cache["indoor_temp"]  # no BT50
        adapter._additional_indoor_sensors = ["sensor.hall"]

        state = await adapter.get_current_state()

        assert state.indoor_temp == pytest.approx(17.0), (
            f"A sensorless NIBE with one added room sensor reading 17.0 C reported "
            f"{state.indoor_temp:.1f} C. DEFAULT_INDOOR_TEMP ({DEFAULT_INDOOR_TEMP}) was seeded "
            f"into the median, so the combined reading is dragged TOWARD the target and a cold "
            f"house looks two degrees warmer than it is. The function's own docstring forbids it."
        )
        assert state.indoor_temp_valid is True

    @pytest.mark.asyncio
    async def test_the_placeholder_does_not_bias_a_two_sensor_median_either(self):
        adapter = _adapter({**HEALTHY, "sensor.hall": "18.0", "sensor.living": "18.4"})
        del adapter._entity_cache["indoor_temp"]
        adapter._additional_indoor_sensors = ["sensor.hall", "sensor.living"]

        state = await adapter.get_current_state()

        assert state.indoor_temp == pytest.approx(18.2), (
            f"Two sensors at 18.0 and 18.4 have a median of 18.2. Got {state.indoor_temp:.2f} - "
            f"the 21.0 placeholder was seeded in, biasing the reading toward the target."
        )

    @pytest.mark.asyncio
    async def test_a_real_bt50_is_still_combined_with_the_added_sensors(self):
        """The regression guard: a pump WITH a room sensor must still use it."""
        adapter = _adapter({**HEALTHY, "sensor.hall": "20.0", "sensor.living": "22.0"})
        adapter._additional_indoor_sensors = ["sensor.hall", "sensor.living"]

        state = await adapter.get_current_state()

        # median of [21.3 (BT50), 20.0, 22.0]
        assert state.indoor_temp == pytest.approx(21.3)
        assert state.indoor_temp_valid is True


def test_the_helper_returns_none_rather_than_clamping():
    """Clamping would invent a reading. The whole point is that we do not have one."""
    adapter = _adapter(HEALTHY)

    assert adapter._plausible(213.0, 15.0, 30.0, "BT50") is None
    assert adapter._plausible(None, 15.0, 30.0, "BT50") is None
    assert adapter._plausible(21.3, 15.0, 30.0, "BT50") == pytest.approx(21.3)
