"""An implausible temperature reading is not a reading - _plausible must return None.

NIBE's Modbus registers hold deci-degrees, so a hand-written YAML that omits `scale: 0.1`
reports BT50's 21.3 C as 213.0 C, BT1's -3.2 as -32.0, BT2's 35.8 as 358.0. The
plausibility band must cover the sensor the HEAT PUMP sends (BT50), not only the
user-added room sensors originally checked. An implausible required sensor (outdoor,
supply) raises UpdateFailed; an implausible BT50 degrades to "no room sensor" (the comfort
layers and the 18 C safety floor abstain, degree minutes protect the pump). A cold room is not
implausible: it is the room the safety floor exists for. The placeholder must never seed the
multi-sensor median - DEFAULT_INDOOR_TEMP would drag a cold house toward the target and mask the
deviation, which _calculate_multi_sensor_temperature's own docstring forbids.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    DEFAULT_INDOOR_TEMP,
    INDOOR_SENSOR_PLAUSIBLE_MAX,
    INDOOR_SENSOR_PLAUSIBLE_MIN,
    LAYER_WEIGHT_SAFETY,
    NIBE_OUTDOOR_PLAUSIBLE_MAX,
    NIBE_OUTDOOR_PLAUSIBLE_MIN,
    NIBE_WATER_PLAUSIBLE_MAX,
    NIBE_WATER_PLAUSIBLE_MIN,
    SAFETY_EMERGENCY_OFFSET,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import (
    SAFETY_LAYER_NAME,
    DecisionEngine,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


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

    @pytest.mark.parametrize("reading", [8.0, 14.9, 21.3, 30.0])
    @pytest.mark.asyncio
    async def test_every_room_temperature_is_accepted_however_cold(self, reading):
        """The band's job is to catch a value that cannot be a temperature, not to second-guess."""
        state = await _adapter({**HEALTHY, "sensor.bt50": str(reading)}).get_current_state()

        assert state.indoor_temp_valid is True
        assert INDOOR_SENSOR_PLAUSIBLE_MIN <= state.indoor_temp <= INDOOR_SENSOR_PLAUSIBLE_MAX

    @pytest.mark.asyncio
    async def test_a_bt50_scaled_twice_is_not_a_room_temperature(self):
        """21.3 C with the deci-degree scale applied twice reads 2.13 - not a room, however cold."""
        state = await _adapter({**HEALTHY, "sensor.bt50": "2.13"}).get_current_state()

        assert state.indoor_temp_valid is False


class TestAColdRoomGetsTheSafetyFloor:
    """The 18 C floor fires through the adapter, not only on a reading handed to the engine."""

    @pytest.mark.parametrize("reading", ["8.0", "14.9", "17.9"])
    @pytest.mark.asyncio
    async def test_a_cold_room_at_a_price_peak_is_heated_not_cut(self, reading):
        now = datetime(2026, 1, 15, 18, 5, tzinfo=timezone.utc)
        midnight = now.replace(hour=0, minute=0)
        peak = range(72, 76)  # 18:00-19:00, the evening peak this test sits in
        prices = PriceData(
            today=[
                QuarterPeriod(
                    start_time=midnight + timedelta(minutes=15 * q),
                    price=1000.0 if q in peak else 40.0 + q,
                )
                for q in range(96)
            ],
            tomorrow=[],
            has_tomorrow=False,
        )
        engine = DecisionEngine(
            price_analyzer=PriceAnalyzer(),
            effect_manager=EffectManager(MagicMock()),
            thermal_model=ThermalModel(0.7, 1.0),
            config={"target_indoor_temp": 22.0, "tolerance": 0.5, "latitude": 59.33},
            heat_pump_model=NibeF750Profile(),
        )

        with (
            patch.object(dt_util, "now", lambda *args: now),
            patch.object(dt_util, "utcnow", lambda: now),
        ):
            state = await _adapter({**HEALTHY, "sensor.bt50": reading}).get_current_state()
            decision = engine.calculate_decision(state, prices, None, 0.0, 0.0)

        safety = next(layer for layer in decision.layers if layer.name == SAFETY_LAYER_NAME)
        assert (
            safety.weight == LAYER_WEIGHT_SAFETY and decision.offset == SAFETY_EMERGENCY_OFFSET
        ), (
            f"A room at {reading} C during a price peak got {decision.offset:+.1f} C "
            f"(safety layer: {safety.reason!r}). Below 18 C the safety floor must raise the curve "
            f"to {SAFETY_EMERGENCY_OFFSET:+.0f} C whatever the price says."
        )


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

        # 21.3 (BT50), 20.0 and 22.0: the interquartile mean of three sensors is their mean
        assert state.indoor_temp == pytest.approx((21.3 + 20.0 + 22.0) / 3)
        assert state.indoor_temp_valid is True


def test_the_helper_returns_none_rather_than_clamping():
    """Clamping would invent a reading. The whole point is that we do not have one."""
    adapter = _adapter(HEALTHY)

    assert adapter._plausible(213.0, 15.0, 30.0, "BT50") is None
    assert adapter._plausible(None, 15.0, 30.0, "BT50") is None
    assert adapter._plausible(21.3, 15.0, 30.0, "BT50") == pytest.approx(21.3)
