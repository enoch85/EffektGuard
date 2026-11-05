"""Tests for NIBE adapter entity discovery (issue #18: multi-source support).

Covers the three real-world entity sources:
- nibe_heatpump (local NibeGW/Modbus): register-suffixed ids, writable coils
  as NUMBER entities, all entities disabled by default
- myuplink (cloud): name-based ids without register numbers, writable
  parameters as NUMBER entities
- generic modbus YAML: entities WITHOUT unique_id that never enter the entity
  registry (discovery must scan the state machine)
"""

from unittest.mock import MagicMock

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed

from conftest import make_mock_async_all
from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    CONF_DHW_TEMP_ENTITY,
    CONF_NIBE_ENTITY,
    CONF_OUTDOOR_TEMP_ENTITY,
)


def make_state(entity_id: str, state: str, attributes: dict | None = None) -> MagicMock:
    """Create a mock HA state object."""
    mock = MagicMock()
    mock.entity_id = entity_id
    mock.state = state
    mock.attributes = attributes or {}
    return mock


def required_core_states() -> list[MagicMock]:
    """The readings get_current_state() refuses to run without.

    Outdoor temp, supply/flow temp and degree minutes are no longer substituted with
    plausible constants when absent - the adapter raises UpdateFailed instead, so a broken
    installation cannot be mistaken for a healthy one. Tests that exercise unrelated
    behaviour (status parsing, priority, discovery ranking) must therefore supply them.
    """
    return [
        make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS),
        make_state("sensor.bt2_supply_temp_s1_40008", "35.8", TEMP_ATTRS),
        make_state("sensor.degree_minutes_16_bit_43005", "-120", {}),
    ]


def make_registry_entry(
    entity_id: str,
    disabled_by: str | None = None,
    original_device_class: str | None = None,
    unit_of_measurement: str | None = None,
) -> MagicMock:
    """Create a mock entity registry entry."""
    entry = MagicMock()
    entry.entity_id = entity_id
    entry.domain = entity_id.split(".")[0]
    entry.disabled_by = disabled_by
    entry.device_class = None
    entry.original_device_class = original_device_class
    entry.unit_of_measurement = unit_of_measurement
    return entry


def make_hass(states: list[MagicMock], registry_entries: list[MagicMock], monkeypatch):
    """Create a mock hass with a state machine and entity registry."""
    hass = MagicMock(spec=HomeAssistant)
    states_by_id = {s.entity_id: s for s in states}

    states_obj = MagicMock()
    states_obj.async_all = make_mock_async_all(states_by_id)
    states_obj.get = lambda entity_id: states_by_id.get(entity_id)
    hass.states = states_obj

    entries_by_id = {e.entity_id: e for e in registry_entries}
    registry = MagicMock()
    registry.entities.values.return_value = registry_entries
    registry.async_get = lambda entity_id: entries_by_id.get(entity_id)
    monkeypatch.setattr(
        "custom_components.effektguard.adapters.nibe_adapter.er.async_get",
        lambda h: registry,
    )
    return hass


TEMP_ATTRS = {"device_class": "temperature", "unit_of_measurement": "°C"}


class TestNibeHeatpumpNaming:
    """Discovery against official nibe_heatpump entity naming (F1155)."""

    @pytest.fixture
    def hass(self, monkeypatch):
        states = [
            make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS),
            make_state("sensor.bt2_supply_temp_s1_40008", "35.8", TEMP_ATTRS),
            make_state("sensor.eb100_ep14_bt3_return_temp_40012", "31.2", TEMP_ATTRS),
            make_state("sensor.bt7_hw_top_40013", "48.7", TEMP_ATTRS),
            make_state("sensor.bt6_hw_load_40014", "44.2", TEMP_ATTRS),
            make_state("sensor.bt50_room_temp_s1_40033", "21.3", TEMP_ATTRS),
            make_state("number.degree_minutes_16_bit_43005", "-150.0", {}),
            make_state("number.heat_offset_s1_47011", "0", {}),
            make_state("sensor.compressor_frequency_actual_43136", "62.0", {}),
            make_state("sensor.compressor_state_ep14_43427", "Running", {}),
            make_state("sensor.prio_43086", "Hot Water", {}),
        ]
        return make_hass(states, [], monkeypatch)

    async def test_discovers_all_core_entities(self, hass):
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"
        assert cache["supply_temp"] == "sensor.bt2_supply_temp_s1_40008"
        assert cache["return_temp"] == "sensor.eb100_ep14_bt3_return_temp_40012"
        assert cache["dhw_top_temp"] == "sensor.bt7_hw_top_40013"
        assert cache["dhw_charging_temp"] == "sensor.bt6_hw_load_40014"
        assert cache["indoor_temp"] == "sensor.bt50_room_temp_s1_40033"
        assert cache["degree_minutes"] == "number.degree_minutes_16_bit_43005"
        assert cache["offset"] == "number.heat_offset_s1_47011"
        assert cache["compressor_hz"] == "sensor.compressor_frequency_actual_43136"
        assert cache["compressor_status"] == "sensor.compressor_state_ep14_43427"
        assert cache["prio"] == "sensor.prio_43086"

    async def test_prio_not_matched_as_phase_current(self, hass):
        """43086 is the Prio register, NOT a BE current sensor."""
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        for key in ("phase1_current", "phase2_current", "phase3_current"):
            assert adapter.entity_cache.get(key) != "sensor.prio_43086"

    async def test_dhw_priority_run_is_not_heating(self, hass):
        """The fixture's prio is 'Hot Water' while the compressor reports
        'Running': that is a DHW production run, NOT space heating."""
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_hot_water is True
        assert state.is_heating is False

    async def test_heating_priority_run_is_heating(self, hass, monkeypatch):
        """With prio 'Heat' and compressor 'Running', the run counts as
        space heating and not as hot water."""
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()
        hass.states.get("sensor.prio_43086").state = "Heat"

        state = await adapter.get_current_state()
        assert state.is_heating is True
        assert state.is_hot_water is False


class TestMyUplinkNaming:
    """Discovery against official HA core myuplink entity naming."""

    @pytest.fixture
    def hass(self, monkeypatch):
        states = [
            make_state("sensor.gotham_city_current_outd_temp_bt1", "-3.2", TEMP_ATTRS),
            make_state("sensor.gotham_city_supply_line_bt2", "35.8", TEMP_ATTRS),
            make_state("sensor.gotham_city_return_line_bt3", "31.2", TEMP_ATTRS),
            make_state("sensor.gotham_city_hot_water_top_bt7", "48.7", TEMP_ATTRS),
            make_state("sensor.gotham_city_hot_water_charging_bt6", "44.2", TEMP_ATTRS),
            make_state("sensor.gotham_city_room_temperature_bt50", "21.3", TEMP_ATTRS),
            make_state("number.gotham_city_degree_minutes", "-150.0", {}),
            make_state("number.gotham_city_heating_offset_climate_system_1", "0", {}),
            make_state("sensor.gotham_city_status_compressor", "Running", {}),
        ]
        return make_hass(states, [], monkeypatch)

    async def test_discovers_all_core_entities(self, hass):
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["outdoor_temp"] == "sensor.gotham_city_current_outd_temp_bt1"
        assert cache["supply_temp"] == "sensor.gotham_city_supply_line_bt2"
        assert cache["return_temp"] == "sensor.gotham_city_return_line_bt3"
        assert cache["dhw_top_temp"] == "sensor.gotham_city_hot_water_top_bt7"
        assert cache["dhw_charging_temp"] == "sensor.gotham_city_hot_water_charging_bt6"
        assert cache["indoor_temp"] == "sensor.gotham_city_room_temperature_bt50"
        assert cache["degree_minutes"] == "number.gotham_city_degree_minutes"
        assert cache["offset"] == "number.gotham_city_heating_offset_climate_system_1"
        assert cache["compressor_status"] == "sensor.gotham_city_status_compressor"


class TestMyUplinkF750LegacyNaming:
    """MyUplink remains a first-class source: the maintainer's own F750
    system naming (f750_cu_3x400v_*) must keep resolving after the Modbus
    additions. Guards against any regression that 'removes MyUplink'."""

    @pytest.fixture
    def hass(self, monkeypatch):
        states = [
            make_state("sensor.f750_cu_3x400v_outdoor_temperature_bt1", "-3.2", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_room_temperature_bt50", "21.3", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_heating_medium_supply_bt63", "35.8", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_return_line_bt3", "31.2", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_hot_water_top_bt7", "48.7", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_hot_water_charging_bt6", "44.2", TEMP_ATTRS),
            make_state("sensor.f750_cu_3x400v_degree_minutes", "-150", {}),
            make_state("number.f750_cu_3x400v_offset", "0", {}),
            make_state("sensor.f750_cu_3x400v_current_be1", "3.1", {}),
            make_state("sensor.f750_cu_3x400v_current_be2", "2.9", {}),
            make_state("sensor.f750_cu_3x400v_current_be3", "3.0", {}),
            make_state("sensor.f750_cu_3x400v_compressor_status", "on", {}),
            make_state("sensor.f750_cu_3x400v_current_compressor_frequency", "62", {}),
            make_state("switch.f750_cu_3x400v_increased_ventilation", "off", {}),
        ]
        registry_entries = []
        for s in states:
            entry = make_registry_entry(s.entity_id)
            entry.platform = "myuplink"
            registry_entries.append(entry)
        return make_hass(states, registry_entries, monkeypatch)

    async def test_all_f750_myuplink_entities_discovered(self, hass):
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["outdoor_temp"] == "sensor.f750_cu_3x400v_outdoor_temperature_bt1"
        assert cache["indoor_temp"] == "sensor.f750_cu_3x400v_room_temperature_bt50"
        assert cache["supply_temp"] == "sensor.f750_cu_3x400v_heating_medium_supply_bt63"
        assert cache["return_temp"] == "sensor.f750_cu_3x400v_return_line_bt3"
        assert cache["dhw_top_temp"] == "sensor.f750_cu_3x400v_hot_water_top_bt7"
        assert cache["dhw_charging_temp"] == "sensor.f750_cu_3x400v_hot_water_charging_bt6"
        assert cache["degree_minutes"] == "sensor.f750_cu_3x400v_degree_minutes"
        assert cache["offset"] == "number.f750_cu_3x400v_offset"
        assert cache["phase1_current"] == "sensor.f750_cu_3x400v_current_be1"
        assert cache["phase2_current"] == "sensor.f750_cu_3x400v_current_be2"
        assert cache["phase3_current"] == "sensor.f750_cu_3x400v_current_be3"
        assert cache["compressor_status"] == "sensor.f750_cu_3x400v_compressor_status"
        assert cache["compressor_hz"] == ("sensor.f750_cu_3x400v_current_compressor_frequency")
        assert cache["increased_ventilation"] == ("switch.f750_cu_3x400v_increased_ventilation")

    async def test_f750_state_reads_correctly(self, hass):
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.outdoor_temp == -3.2
        assert state.supply_temp == 35.8
        assert state.degree_minutes == -150
        assert state.is_heating is True
        assert state.dhw_top_temp == 48.7

    async def test_hot_water_settings_and_counters_are_not_its_hot_water_state(self, monkeypatch):
        """The F750's hot-water comfort mode, running hours, periodic-increase step and setpoint
        temperature match "hot_water" by name, but say nothing about whether the tank is charging:
        only the priority does."""
        names = {
            "sensor.f750_cu_3x400v_current_hot_water_mode": "3.0",
            "sensor.f750_cu_3x400v_oper_time_hot_water": "2573.0",
            "sensor.f750_cu_3x400v_periodic_hot_water_increase_in_progress": "2.0",
            "sensor.f750_cu_3x400v_hot_water_temp": "56.0",
        }
        states = required_core_states() + [make_state(e, v, {}) for e, v in names.items()]
        adapter = NibeAdapter(make_hass(states, [], monkeypatch), {})
        await adapter.discover_entities()

        assert not set(names) & set(adapter.entity_cache.values())

    async def test_the_pumps_maximum_frequency_is_not_its_frequency(self, monkeypatch):
        """The F750 publishes its current maximum frequency beside its frequency, and both names
        hold "compressor_frequency": the maximum must never be read as the frequency, whichever the
        integration creates first. Its maximum for heating (40322) is not the current maximum."""
        names = [
            "sensor.f750_cu_3x400v_max_compressor_frequency",
            "sensor.f750_cu_3x400v_max_compressor_frequency_heating",
            "sensor.f750_cu_3x400v_current_compressor_frequency",
        ]
        states = required_core_states() + [make_state(e, "86", {}) for e in names]
        adapter = NibeAdapter(make_hass(states, [], monkeypatch), {})
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["compressor_hz"] == names[2]
        assert cache["compressor_hz_max"] == names[0]
        assert names[1] not in cache.values()


class TestGenericModbusStatesOnly:
    """Generic modbus YAML entities have no unique_id and never enter the
    entity registry - discovery must find them in the state machine."""

    @pytest.fixture
    def hass(self, monkeypatch):
        # Reproduces the live devbox setup that reproduced issue #18
        states = [
            make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS),
            make_state("sensor.bt7_hw_top_40013", "48.7", TEMP_ATTRS),
            make_state("sensor.degree_minutes_43005", "-150.0", {}),
            make_state("sensor.heating_offset_s1_47011", "0", {}),
            make_state("number.heating_offset_climate_system_1_47011", "0.0", {}),
        ]
        return make_hass(states, [], monkeypatch)

    async def test_states_without_registry_are_discovered(self, hass):
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"
        assert cache["dhw_top_temp"] == "sensor.bt7_hw_top_40013"
        assert cache["degree_minutes"] == "sensor.degree_minutes_43005"

    async def test_offset_never_binds_to_readonly_sensor(self, hass):
        """The write path calls number.set_value - offset must be a number."""
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["offset"] == "number.heating_offset_climate_system_1_47011"

    async def test_offset_unset_when_only_sensor_exists(self, monkeypatch):
        hass = make_hass([make_state("sensor.heating_offset_s1_47011", "0", {})], [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert "offset" not in adapter.entity_cache


class TestManualOverrides:
    """Manual config-flow overrides beat pattern discovery."""

    async def test_overrides_win_over_discovery(self, monkeypatch):
        states = [
            make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS),
            make_state("sensor.my_own_outdoor", "-5.0", TEMP_ATTRS),
            make_state("sensor.my_own_dhw_top", "47.0", TEMP_ATTRS),
            make_state("number.heat_offset_s1_47011", "0", {}),
            make_state("number.my_offset_control", "0", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(
            hass,
            {
                CONF_NIBE_ENTITY: "number.my_offset_control",
                CONF_OUTDOOR_TEMP_ENTITY: "sensor.my_own_outdoor",
                CONF_DHW_TEMP_ENTITY: "sensor.my_own_dhw_top",
            },
        )
        await adapter.discover_entities()

        cache = adapter.entity_cache
        assert cache["offset"] == "number.my_offset_control"
        assert cache["outdoor_temp"] == "sensor.my_own_outdoor"
        assert cache["dhw_top_temp"] == "sensor.my_own_dhw_top"


class TestRegistryEdgeCases:
    """Disabled and stale registry entries (nibe_heatpump disables all coil
    entities by default; re-added entries leave stale ids behind)."""

    async def test_disabled_registry_entries_skipped(self, monkeypatch):
        registry_entries = [
            make_registry_entry(
                "sensor.bt7_hw_top_40013",
                disabled_by="integration",
                original_device_class="temperature",
            ),
        ]
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert "dhw_top_temp" not in adapter.entity_cache

    async def test_live_entity_beats_stale_registry_entry(self, monkeypatch):
        """Issue #18: sensor.bt1_..._40004_2 is live; the base id is a stale
        registry leftover without a state. The live entity must win."""
        states = [
            make_state("sensor.bt1_outdoor_temperature_40004_2", "-3.2", TEMP_ATTRS),
        ]
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        hass = make_hass(states, registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004_2"

    async def test_registry_only_entry_used_when_no_live_entity(self, monkeypatch):
        """Enabled-but-not-yet-loaded entities are still cached (startup)."""
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"

    async def test_live_entity_displaces_registry_entry_on_later_pass(self, monkeypatch):
        """Ranks persist across passes: an entity that becomes live AFTER the
        first discovery pass (source integration finished starting) must
        replace the registry-only entry cached during startup."""
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()
        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"

        # The integration finishes loading: a different live BT1 appears
        hass2 = make_hass(
            [make_state("sensor.bt1_outdoor_temperature_40004_2", "-3.2", TEMP_ATTRS)],
            registry_entries,
            monkeypatch,
        )
        adapter.hass = hass2
        await adapter.discover_entities()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004_2"

    async def test_own_entities_excluded_by_platform(self, monkeypatch):
        """EffektGuard's own entities must never be discovered as sources."""
        states = [
            make_state("sensor.effektguard_degree_minutes", "-150", {}),
        ]
        registry_entries = [
            make_registry_entry("sensor.effektguard_degree_minutes"),
        ]
        registry_entries[0].platform = "effektguard"
        hass = make_hass(states, registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert "degree_minutes" not in adapter.entity_cache


class TestStartupRaces:
    """Source integrations create entities after EffektGuard's first pass."""

    async def test_core_keys_from_registry_trigger_rediscovery(self, monkeypatch):
        """A core key backed only by a stale registry entry must not stop
        re-discovery: the live entity appearing later has to take over
        (issue #18: re-added nibe_heatpump leaves stale enabled ids)."""
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})

        # First cycle: only the stale registry entry exists, so no reading resolves.
        # Refusing here is correct - the adapter must not invent values - but discovery
        # still runs and caches the registry-backed id.
        with pytest.raises(UpdateFailed):
            await adapter.get_current_state()
        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"

        # Live entities appear; the next cycle must re-discover and displace
        hass2 = make_hass(
            [
                make_state("sensor.bt1_outdoor_temperature_40004_2", "-3.2", TEMP_ATTRS),
                make_state("sensor.bt50_room_temp_s1_40033", "21.3", TEMP_ATTRS),
                make_state("sensor.bt2_supply_temp_s1_40008", "35.8", TEMP_ATTRS),
                make_state("sensor.degree_minutes_16_bit_43005", "-120", {}),
            ],
            registry_entries,
            monkeypatch,
        )
        adapter.hass = hass2
        state = await adapter.get_current_state()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004_2"
        assert state.outdoor_temp == -3.2

    async def test_switch_found_via_registry_before_platform_loads(self, monkeypatch):
        """Ventilation switch registered but without a state yet is found."""
        registry_entries = [
            make_registry_entry("switch.f750_increased_ventilation"),
        ]
        registry_entries[0].platform = "myuplink"
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["increased_ventilation"] == (
            "switch.f750_increased_ventilation"
        )

    async def test_bedroom_temp_not_matched_as_indoor(self, monkeypatch):
        """Unrelated *room_temp* sensors must not claim indoor_temp."""
        states = [
            make_state("sensor.bedroom_temp", "19.5", TEMP_ATTRS),
            make_state("sensor.storeroom_temp_2", "15.0", TEMP_ATTRS),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert "indoor_temp" not in adapter.entity_cache

    async def test_offset_override_available_before_discovery(self, monkeypatch):
        """The coordinator reads entity_cache['offset'] before the first
        discovery pass - manual overrides must be seeded at construction."""
        hass = make_hass([], [], monkeypatch)
        adapter = NibeAdapter(hass, {CONF_NIBE_ENTITY: "number.my_offset"})

        assert adapter.entity_cache.get("offset") == "number.my_offset"

    async def test_registered_entity_beats_unregistered_lookalike(self, monkeypatch):
        """A template/YAML lookalike without a registry entry must not steal
        a key from a real integration entity, regardless of iteration order."""
        lookalike = make_state("sensor.outdoor_temp_smoothed", "-2.9", TEMP_ATTRS)
        real = make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS)
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        registry_entries[0].platform = "nibe_heatpump"
        # Lookalike iterates FIRST
        hass = make_hass([lookalike, real], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"

    async def test_registry_binding_upgrades_to_live_and_resists_theft(self, monkeypatch):
        """A key bound from the registry pass must upgrade to live rank when
        its state appears, so a later lookalike cannot displace it."""
        registry_entries = [
            make_registry_entry(
                "sensor.bt1_outdoor_temperature_40004",
                original_device_class="temperature",
            ),
        ]
        registry_entries[0].platform = "nibe_heatpump"
        hass = make_hass([], registry_entries, monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        # BT1 comes alive; an unregistered lookalike iterates before it
        states = [
            make_state("sensor.outdoor_temp_smoothed", "-2.9", TEMP_ATTRS),
            make_state("sensor.bt1_outdoor_temperature_40004", "-3.2", TEMP_ATTRS),
        ]
        hass2 = make_hass(states, registry_entries, monkeypatch)
        adapter.hass = hass2
        await adapter.discover_entities()

        assert adapter.entity_cache["outdoor_temp"] == "sensor.bt1_outdoor_temperature_40004"


class TestStatusParsing:
    """Status reads across sources: on/off, mapped strings, numeric enums."""

    async def test_numeric_compressor_state_falls_back_to_frequency(self, monkeypatch):
        """Raw Modbus 43427 is a numeric enum the bool parser cannot read;
        is_heating must derive from compressor frequency instead."""
        states = required_core_states() + [
            make_state("sensor.compressor_status_ep14_43427", "60", {}),
            make_state("sensor.compressor_frequency_43136", "62.0", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_heating is True
        assert state.compressor_hz == 62

    async def test_numeric_dhw_priority_is_not_heating(self, monkeypatch):
        """Raw Modbus priority 20 (Hot Water) with the compressor spinning:
        DHW production, not space heating - even though the numeric status
        enum is unreadable and the frequency fallback would fire."""
        states = required_core_states() + [
            make_state("sensor.prio_43086", "20", {}),
            make_state("sensor.compressor_status_ep14_43427", "60", {}),
            make_state("sensor.compressor_frequency_43136", "62.0", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_hot_water is True
        assert state.is_heating is False

    async def test_numeric_heating_priority_is_heating(self, monkeypatch):
        """Raw Modbus priority 30 (Heat) with the compressor spinning."""
        states = required_core_states() + [
            make_state("sensor.prio_43086", "30", {}),
            make_state("sensor.compressor_frequency_43136", "62.0", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_heating is True
        assert state.is_hot_water is False

    async def test_an_off_priority_is_no_hot_water_charge(self, monkeypatch):
        """Priority 10 is Off: a third of the owner's charges end there rather than in Heating,
        and read as "cannot say" they were never seen to end."""
        states = required_core_states() + [
            make_state("sensor.gotham_city_priority", "Off", {}),
            make_state("sensor.gotham_city_status_compressor", "Running", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_hot_water is False
        assert state.is_heating is True  # from compressor status, not prio

    async def test_unknown_priority_leaves_status_reads_untouched(self, monkeypatch):
        """MyUplink priority enums are not fully documented (raw 31 observed): an
        unknown value must not override the compressor-status read, and it says
        nothing about whether the compressor serves the tank."""
        states = required_core_states() + [
            make_state("sensor.gotham_city_priority", "31", {}),
            make_state("sensor.gotham_city_status_compressor", "Running", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        assert adapter.entity_cache["prio"] == "sensor.gotham_city_priority"
        state = await adapter.get_current_state()
        assert state.is_heating is True  # from compressor status, not prio
        assert state.is_hot_water is None

    async def test_zero_frequency_is_not_heating_and_not_none(self, monkeypatch):
        states = required_core_states() + [
            make_state("sensor.compressor_frequency_43136", "0.0", {}),
        ]
        hass = make_hass(states, [], monkeypatch)
        adapter = NibeAdapter(hass, {})
        await adapter.discover_entities()

        state = await adapter.get_current_state()
        assert state.is_heating is False
        # 0 Hz is a valid reading, not "unknown"
        assert state.compressor_hz == 0


@pytest.mark.parametrize(
    "others, frequency",
    [
        pytest.param(
            [
                "sensor.compressor_frequency_request_40321",
                "sensor.compressor_frequency_target_43182",
            ],
            "sensor.compressor_frequency_actual_43136",
            id="nibe_heatpump-f-series",
        ),
        pytest.param(
            [
                "sensor.requested_compressor_frequency_30141",
                "sensor.requested_compressor_frequency_31857",
            ],
            "sensor.compressor_frequency_current_31047",
            id="nibe_heatpump-s-series",
        ),
        pytest.param(
            ["sensor.f750_cu_3x400v_min_compressor_frequency"],
            "sensor.f750_cu_3x400v_current_compressor_frequency",
            id="myuplink",
        ),
    ],
)
async def test_the_compressor_frequency_is_the_one_it_runs_at(monkeypatch, others, frequency):
    """A NIBE publishes several frequencies whose names hold "compressor_frequency": what it was
    asked for, what it aims at, the least it may run at. Only the one it runs at is its frequency,
    whichever the integration creates first (nibe_heatpump creates them in register order)."""
    states = required_core_states() + [make_state(e, "50", {}) for e in others + [frequency]]
    adapter = NibeAdapter(make_hass(states, [], monkeypatch), {})
    await adapter.discover_entities()

    assert adapter.entity_cache["compressor_hz"] == frequency


@pytest.mark.parametrize(
    "heater",
    [
        pytest.param("sensor.f750_cu_3x400v_power_internal_add_heat", id="myuplink"),
        pytest.param("sensor.int_el_add_power_43084", id="nibe_heatpump-f-series"),
        pytest.param("sensor.power_internal_additional_heat_31028", id="nibe_heatpump-s-series"),
    ],
)
async def test_the_heater_power_is_the_heaters_own_and_not_a_setting(monkeypatch, heater):
    """A NIBE publishes its immersion heater's power beside two settings in the same words: the most
    the heater may draw (47212) and what it may draw in emergency mode (43029)."""
    settings = [
        "number.max_int_add_power_47212",
        "sensor.immersion_heater_power_emergency_mode_43029",
    ]
    states = (
        required_core_states()
        + [make_state(e, "6.5", {}) for e in settings]
        + [make_state(heater, "0.5", {})]
    )
    adapter = NibeAdapter(make_hass(states, [], monkeypatch), {})
    await adapter.discover_entities()

    assert adapter.entity_cache["heater_power"] == heater
    assert (await adapter.get_current_state()).heater_power_kw == 0.5
