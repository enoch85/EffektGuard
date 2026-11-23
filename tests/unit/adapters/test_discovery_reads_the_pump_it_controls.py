"""Discovery reads the pump EffektGuard controls on nibe_heatpump - climate system 1 and the first
outdoor unit - never a look-alike: another circuit's sensor, another unit's, a cooling sensor, or a
setting that holds "prio". And a pattern is a whole word or number everywhere: BT10 is not BT1,
140004 not 40004.

The first match wins in the order Home Assistant created the states, and a restart re-creates them
in the coil order, where look-alikes come first (issue #158, seen on an SMO 40). So each row puts
them first. Entity ids are nibe_heatpump's (slugified nibe 2.25.0 coil names) unless a row has no
registry platform.
"""

from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant

from conftest import make_mock_async_all
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

TEMP = {"device_class": "temperature", "unit_of_measurement": "°C"}
DM = {"unit_of_measurement": "DM"}
AMP = {"device_class": "current", "unit_of_measurement": "A"}
NONE: dict = {}
NIBE = "nibe_heatpump"


def make_hass(states: list[tuple[str, dict]], platform: str | None, monkeypatch) -> MagicMock:
    """These states, in this order; each in the entity registry under `platform`, unless None."""
    by_id, entries = {}, {}
    for entity_id, attributes in states:
        state = MagicMock()
        state.entity_id, state.state, state.attributes = entity_id, "1", attributes
        by_id[entity_id] = state
        if platform:
            entry = MagicMock()
            entry.entity_id, entry.platform, entry.original_name = entity_id, platform, None
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
    ("platform", "look_alikes", "own"),
    [
        pytest.param(
            NIBE,
            [
                ("sensor.ep23_bt2_supply_temp_s4_40005", TEMP),
                ("sensor.ep22_bt2_supply_temp_s3_40006", TEMP),
                ("sensor.ep21_bt2_supply_temp_s2_40007", TEMP),
                ("sensor.ep47_bt2_supply_temp_s8_40159", TEMP),
                ("sensor.ep46_bt2_supply_temp_s7_40160", TEMP),
                ("sensor.ep45_bt2_supply_temp_s6_40161", TEMP),
                ("sensor.ep44_bt2_supply_temp_s5_40162", TEMP),
                ("sensor.ep21_bt50_room_temp_s2_40032", TEMP),
                ("sensor.bt70_hw_comfort_supply_temp_40147", TEMP),
            ],
            {
                "supply_temp": ("sensor.bt2_supply_temp_s1_40008", TEMP),
                "indoor_temp": ("sensor.bt50_room_temp_s1_40033", TEMP),
            },
            id="other circuits (F-series, SMO, VVM): S2-S8, hot-water comfort",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.supply_line_ep23_bt2_30003", TEMP),
                ("sensor.room_average_temp_clim_system_8_bt50_30110", TEMP),
                ("sensor.room_average_temp_clim_system_7_bt50_30111", TEMP),
                ("sensor.room_average_temp_clim_system_6_bt50_30112", TEMP),
                ("sensor.room_average_temp_clim_system_5_bt50_30113", TEMP),
                ("sensor.room_average_temp_clim_system_4_bt50_30114", TEMP),
                ("sensor.room_average_temp_clim_system_3_bt50_30115", TEMP),
                ("sensor.room_average_temp_clim_system_2_bt50_30116", TEMP),
            ],
            {
                "supply_temp": ("sensor.supply_line_bt2_30006", TEMP),
                "indoor_temp": ("sensor.room_average_temp_clim_system_1_bt50_30117", TEMP),
            },
            id="other circuits (S-series): S2-S8",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.eq1_bt64_cool_supply_temp_40045", TEMP),
                ("sensor.eq1_bt65_cool_return_temp_40046", TEMP),
                ("sensor.bt25_ext_supply_temp_cooling_40216", TEMP),
                ("number.cool_degree_minutes_44266", DM),
            ],
            {
                "supply_temp": ("sensor.bt25_ext_supply_40071", TEMP),
                "return_temp": ("sensor.bt71_ext_return_temp_40152", TEMP),
                "degree_minutes": ("number.degree_minutes_16_bit_43005", DM),
            },
            id="cooling (F-series, SMO, VVM)",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.external_cooling_supply_temperature_bt25_30120", TEMP),
                ("sensor.supply_temperature_bt64_30338", TEMP),
                ("sensor.return_line_eq1_bt65_30032", TEMP),
                ("sensor.cooling_degree_minutes_40021", DM),
            ],
            {
                "supply_temp": ("sensor.supply_line_bt2_30006", TEMP),
                "return_temp": ("sensor.return_line_bt3_30008", TEMP),
                "degree_minutes": ("number.degree_minutes_40012", DM),
            },
            id="cooling (S-series)",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.eb108_ep14_compressor_state_44415", NONE),
                ("sensor.eb107_ep14_compressor_state_44421", NONE),
                ("sensor.eb106_ep14_compressor_state_44427", NONE),
                ("sensor.eb105_ep14_compressor_state_44433", NONE),
                ("sensor.eb104_ep14_compressor_state_44439", NONE),
                ("sensor.eb103_ep14_compressor_state_44445", NONE),
                ("sensor.eb102_ep14_compressor_state_44451", NONE),
                ("sensor.status_compressor_heater_s135_30349", NONE),
            ],
            {"compressor_status": ("sensor.eb101_ep14_compressor_state_44457", NONE)},
            id="other units: outdoor units 2-8, the S135 module",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.opt_boiler_has_priority_hot_water_41287", NONE),
                ("sensor.smart_energy_source_energy_source_prio_7_start_dm_41393", NONE),
                ("number.max_diff_ses_prio_1_49208", NONE),
                ("number.max_difference_ses_lower_priority_energy_source_41328", NONE),
                ("sensor.prio_hot_water_opt_30605", NONE),
                ("sensor.permit_prioritised_additional_heat_30691", NONE),
                ("number.speed_circ_pump_add_prio_49301", NONE),
                ("sensor.all_sub_units_operating_prio_33826", NONE),
            ],
            {"prio": ("sensor.priority_31029", NONE)},
            id="settings that hold prio",
        ),
        pytest.param(
            NIBE,
            [
                ("sensor.eb100_ep14_bt10_brine_in_temp_40015", TEMP),
                ("sensor.suction_gas_bt17_30017", TEMP),
                ("sensor.eb100_bt61_supply_radiator_temp_40047", TEMP),
                ("number.mode_energy_meter_be11_44128", NONE),
            ],
            {
                "outdoor_temp": ("sensor.bt1_outdoor_temperature_40004", TEMP),
                "dhw_charging_temp": ("sensor.bt6_hw_load_40014", TEMP),
                "phase1_current": ("sensor.eb100_be1_current_40083", AMP),
            },
            id="a number runs on: BT10, BT17, BT61, BE11",
        ),
        pytest.param(
            None,
            [("sensor.other_temperature_140004", TEMP)],
            {"outdoor_temp": ("sensor.bt1_outdoor_40004", TEMP)},
            id="a number inside a longer one (a generic Modbus name)",
        ),
    ],
)
async def test_the_pump_s_own_reading_wins_over_a_look_alike(
    platform, look_alikes, own, monkeypatch
):
    hass = make_hass(look_alikes + list(own.values()), platform, monkeypatch)
    adapter = NibeAdapter(hass, {})
    await adapter.discover_entities()

    bound = {key: adapter.entity_cache.get(key) for key in own}
    assert bound == {key: entity_id for key, (entity_id, _) in own.items()}
