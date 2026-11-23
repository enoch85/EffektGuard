"""An SMO 20/40 (or VVM 225/310/320/325/500) on nibe_heatpump publishes its outdoor unit's
compressor frequency as "EB101-EP14 Actual Cpr Frequency Outdoor Unit" (44701), a name no frequency
pattern matched (issue #158). An SMO 40 can drive up to eight outdoor units; their frequencies carry
the same name, EB102-EB108 first in the coil list, and EffektGuard reads the first unit's.

Entity ids are nibe_heatpump's (slugified nibe 2.25.0 coil names).
"""

from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from conftest import make_mock_async_all
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

TEMP = {"device_class": "temperature", "unit_of_measurement": "°C"}
HZ = {"device_class": "frequency", "unit_of_measurement": "Hz"}
REQUIRED = [
    ("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP),
    ("sensor.bt25_ext_supply_40071", "35.8", TEMP),
    ("number.degree_minutes_32_bit_40940", "-150.0", {"unit_of_measurement": "DM"}),
]
EB101_HZ = "sensor.eb101_ep14_actual_cpr_frequency_outdoor_unit_44701"


def make_hass(states: list[tuple[str, str, dict]], monkeypatch) -> MagicMock:
    """These states, in this order, each a nibe_heatpump entity in the registry."""
    by_id, entries = {}, {}
    for entity_id, value, attributes in states:
        state = MagicMock()
        state.entity_id, state.state, state.attributes = entity_id, value, attributes
        by_id[entity_id] = state
        entry = MagicMock()
        entry.entity_id, entry.platform, entry.original_name = entity_id, "nibe_heatpump", None
        entry.domain, entry.disabled_by = entity_id.split(".")[0], None
        entries[entity_id] = entry
    hass = MagicMock(spec=HomeAssistant)
    hass.states = MagicMock()
    hass.states.async_all = make_mock_async_all(by_id)
    hass.states.get = by_id.get
    registry = MagicMock()
    registry.entities.values.return_value = list(entries.values())
    registry.async_get = entries.get
    monkeypatch.setattr(
        "custom_components.effektguard.adapters.nibe_adapter.er.async_get", lambda h: registry
    )
    return hass


@pytest.mark.parametrize(
    "frequencies",
    [
        pytest.param([(EB101_HZ, "62.0", HZ)], id="one outdoor unit (SMO 20)"),
        pytest.param(
            [
                ("sensor.eb108_ep14_actual_cpr_frequency_outdoor_unit_44505", "0.0", HZ),
                ("sensor.eb102_ep14_actual_cpr_frequency_outdoor_unit_44673", "0.0", HZ),
                (EB101_HZ, "62.0", HZ),
            ],
            id="a cascade, the other units first (SMO 40)",
        ),
    ],
)
async def test_the_first_outdoor_unit_s_frequency_is_read(frequencies, monkeypatch):
    hass = make_hass(REQUIRED + frequencies, monkeypatch)
    adapter = NibeAdapter(hass, {})
    await adapter.discover_entities()

    assert adapter.entity_cache.get("compressor_hz") == EB101_HZ
    assert (await adapter.get_current_state()).compressor_hz == 62
