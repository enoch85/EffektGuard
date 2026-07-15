"""An implausible temperature reading is not a reading - _plausible must return None.

NIBE's Modbus registers hold deci-degrees, so a hand-written YAML that omits `scale: 0.1`
reports BT50's 21.3 C as 213.0 C, BT1's -3.2 as -32.0, BT2's 35.8 as 358.0. The
plausibility band must cover the sensor the HEAT PUMP sends (BT50), not only the
user-added room sensors originally checked. An implausible required sensor (outdoor,
supply) raises UpdateFailed; an implausible BT50 degrades to "no room sensor" (comfort
layers abstain, 18 C floor unaffected). The placeholder must never seed the multi-sensor
median - DEFAULT_INDOOR_TEMP would drag a cold house toward the target and mask the
deviation, which _calculate_multi_sensor_temperature's own docstring forbids.
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
