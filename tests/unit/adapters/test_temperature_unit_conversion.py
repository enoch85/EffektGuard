"""NIBE temperature readings must be normalised to °C (`_read_temperature`).

NibeState documents every temperature as °C and the optimization stack assumes it.
Discovery accepts °F entities, and HA presents a temperature sensor in the user's
preferred unit, so on an imperial install BT1 reading 32 (= 0 °C) and BT25 reading 95
(= 35 °C) would be taken as +32 °C and a 95 °C flow if passed through as bare floats -
driving weather compensation to minimum offset in winter. Conversion must happen after
the unknown-value marker check, so a raw -32768 marker is dropped, not converted.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

OUTDOOR = "sensor.nibe_bt1_outdoor"
SUPPLY = "sensor.nibe_bt25_supply"
INDOOR = "sensor.nibe_bt50_room"
DEGREE_MINUTES = "sensor.nibe_degree_minutes"
OFFSET = "number.nibe_heat_offset"

CACHE = {
    "outdoor_temp": OUTDOOR,
    "supply_temp": SUPPLY,
    "indoor_temp": INDOOR,
    "degree_minutes": DEGREE_MINUTES,
    "offset": OFFSET,
}


async def _noop() -> None:
    return None


def build_adapter(readings: dict[str, tuple[str, str | None]]) -> NibeAdapter:
    """readings maps entity_id -> (state_value, unit_of_measurement)."""
    hass = MagicMock()

    def get_state(entity_id: str):
        if entity_id not in readings:
            return None
        value, unit = readings[entity_id]
        state = MagicMock()
        state.state = value
        state.attributes = {"unit_of_measurement": unit} if unit else {}
        return state

    hass.states.get.side_effect = get_state

    adapter = NibeAdapter(hass, {"nibe_entity": OFFSET})
    adapter._entity_cache = dict(CACHE)
    adapter._discover_nibe_entities = _noop
    return adapter


class TestFahrenheitIsConvertedToCelsius:
    @pytest.mark.asyncio
    async def test_fahrenheit_sensors_are_converted(self):
        """A pump reported entirely in °F must arrive as °C."""
        adapter = build_adapter(
            {
                OUTDOOR: ("32", "°F"),  # 0 °C - freezing
                SUPPLY: ("95", "°F"),  # 35 °C - a normal flow temp
                INDOOR: ("68", "°F"),  # 20 °C
                DEGREE_MINUTES: ("-420", None),
                OFFSET: ("0", None),
            }
        )

        state = await adapter.get_current_state()

        assert state.outdoor_temp == pytest.approx(0.0), (
            f"BT1 at 32 °F is FREEZING, but was read as {state.outdoor_temp:.1f} °C. "
            "Weather compensation would think it is a mild day."
        )
        assert state.supply_temp == pytest.approx(35.0), (
            f"BT25 at 95 °F is a normal 35 °C flow, but was read as "
            f"{state.supply_temp:.1f} °C - an impossible flow temperature."
        )
        assert state.indoor_temp == pytest.approx(20.0)
        assert state.indoor_temp_valid is True

    @pytest.mark.asyncio
    async def test_celsius_sensors_pass_through_unchanged(self):
        """Do not over-correct: °C must not be touched."""
        adapter = build_adapter(
            {
                OUTDOOR: ("-8.4", "°C"),
                SUPPLY: ("38.2", "°C"),
                INDOOR: ("20.6", "°C"),
                DEGREE_MINUTES: ("-420", None),
                OFFSET: ("0", None),
            }
        )

        state = await adapter.get_current_state()

        assert state.outdoor_temp == pytest.approx(-8.4)
        assert state.supply_temp == pytest.approx(38.2)
        assert state.indoor_temp == pytest.approx(20.6)

    @pytest.mark.asyncio
    async def test_missing_unit_is_assumed_celsius(self):
        """Modbus/template sensors often carry no unit. Celsius is the right assumption."""
        adapter = build_adapter(
            {
                OUTDOOR: ("-8.4", None),
                SUPPLY: ("38.2", None),
                INDOOR: ("20.6", None),
                DEGREE_MINUTES: ("-420", None),
                OFFSET: ("0", None),
            }
        )

        state = await adapter.get_current_state()

        assert state.outdoor_temp == pytest.approx(-8.4)
        assert state.supply_temp == pytest.approx(38.2)

    @pytest.mark.asyncio
    async def test_unknown_value_marker_is_still_rejected_before_conversion(self):
        """-32768 is a raw s16 'no reading' marker - it must not be converted, it must be dropped.

        Converting it from °F would yield -18204 °C, a plausible-looking float.
        """
        adapter = build_adapter(
            {
                OUTDOOR: ("-8.4", "°C"),
                SUPPLY: ("38.2", "°C"),
                INDOOR: ("-32768", "°F"),  # disconnected sensor, reported in °F
                DEGREE_MINUTES: ("-420", None),
                OFFSET: ("0", None),
            }
        )

        state = await adapter.get_current_state()

        # The marker must be treated as "no reading", not converted into a temperature.
        assert state.indoor_temp_valid is False
        assert state.indoor_temp > 0  # the placeholder, not -18204
