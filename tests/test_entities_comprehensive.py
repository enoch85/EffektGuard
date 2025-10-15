"""Comprehensive Entity Tests - All Production Scenarios.

Complete test coverage for all EffektGuard entities including:
- Climate entity (temperature control, HVAC modes, presets)
- All 14 diagnostic sensors (with real and missing data scenarios)
- All 5 number configuration entities
- All 2 select configuration entities
- All 5 feature toggle switches
- Entity setup and creation
- Error handling for missing/None data
- Device info and unique ID validation
- Extra state attributes

This test suite ensures all entity production code has proper test coverage
for normal operation, edge cases, and error conditions.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.climate import ClimateEntityFeature, HVACMode
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.sensor import EffektGuardSensor, SENSORS
from custom_components.effektguard.number import EffektGuardNumber, NUMBERS
from custom_components.effektguard.select import EffektGuardSelect, SELECTS
from custom_components.effektguard.switch import EffektGuardSwitch, SWITCHES
from custom_components.effektguard.const import (
    CONF_DEGREE_MINUTES_ENTITY,
    CONF_ENABLE_OPTIMIZATION,
    CONF_INSULATION_QUALITY,
    CONF_OPTIMIZATION_MODE,
    CONF_PEAK_PROTECTION_MARGIN,
    CONF_POWER_SENSOR_ENTITY,
    CONF_TARGET_INDOOR_TEMP,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    CONF_WEATHER_ENTITY,
    DOMAIN,
    OPTIMIZATION_MODE_BALANCED,
)


@pytest.fixture
def full_coordinator():
    """Create coordinator with complete data."""
    coordinator = MagicMock(spec=DataUpdateCoordinator)
    coordinator.last_update_success = True
    coordinator.data = {
        "nibe": MagicMock(
            indoor_temp=21.5,
            outdoor_temp=5.0,
            supply_temp=35.0,
            degree_minutes=-60,
        ),
        "price": MagicMock(
            current_price=1.25,
            current_quarter=42,
            current_classification="normal",
            today_classifications={"0": "cheap", "42": "normal"},
            today_prices=[0.8, 1.0, 1.2],
            tomorrow_prices=[0.9, 1.1],
        ),
        "decision": MagicMock(
            offset=1.5,
            reasoning="Test reasoning",
            peak_status="safe",
            peak_margin=2.5,
            current_power=3.5,
            timestamp=datetime.now(),
            layers=[],
        ),
        "thermal": MagicMock(
            temperature_trend=-0.2,
            prediction_3h=21.0,
        ),
        "weather": MagicMock(
            forecast_hours=[
                MagicMock(datetime="2025-10-14T15:00:00", temperature=4.0),
            ]
        ),
        "savings": MagicMock(
            monthly_estimate=450.0,
            effect_savings=200.0,
            spot_savings=250.0,
            baseline_cost=1200.0,
            optimized_cost=750.0,
        ),
    }
    coordinator.peak_today = 4.5
    coordinator.peak_this_month = 5.2
    coordinator.current_peak = 5.2
    coordinator.async_request_refresh = AsyncMock()
    coordinator.hass = MagicMock(spec=HomeAssistant)
    coordinator.config_entry = MagicMock()
    coordinator.config_entry.data = {
        CONF_DEGREE_MINUTES_ENTITY: "sensor.nibe_degree_minutes",
        CONF_POWER_SENSOR_ENTITY: "sensor.house_power",
        CONF_WEATHER_ENTITY: "weather.home",
    }
    return coordinator


@pytest.fixture
def empty_coordinator():
    """Create coordinator with None/missing data."""
    coordinator = MagicMock(spec=DataUpdateCoordinator)
    coordinator.last_update_success = True
    coordinator.data = None  # Simulate no data yet
    coordinator.peak_today = 0.0
    coordinator.peak_this_month = 0.0
    coordinator.current_peak = 0.0
    coordinator.async_request_refresh = AsyncMock()
    coordinator.hass = MagicMock(spec=HomeAssistant)
    coordinator.config_entry = MagicMock()
    coordinator.config_entry.data = {}
    return coordinator


@pytest.fixture
def mock_entry():
    """Create mock config entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        CONF_TARGET_INDOOR_TEMP: 21.0,
        CONF_TOLERANCE: 0.5,
        CONF_THERMAL_MASS: 1.0,
        CONF_INSULATION_QUALITY: 1.0,
        CONF_PEAK_PROTECTION_MARGIN: 0.5,
        CONF_OPTIMIZATION_MODE: OPTIMIZATION_MODE_BALANCED,
        CONF_ENABLE_OPTIMIZATION: True,
    }
    return entry


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant."""
    import threading
    from homeassistant.const import UnitOfTemperature

    hass = MagicMock(spec=HomeAssistant)
    hass.config_entries = MagicMock()
    hass.config_entries.async_update_entry = MagicMock()
    hass.loop_thread_id = threading.get_ident()
    hass.config = MagicMock()
    hass.config.units = MagicMock()
    hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS

    # Mock states for optional_features_status sensor
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)

    return hass


# ============================================================================
# CLIMATE ENTITY - Edge Cases
# ============================================================================


def test_climate_temperature_limits(full_coordinator, mock_entry):
    """Test climate entity min/max temperature attributes."""
    climate = EffektGuardClimate(full_coordinator, mock_entry)

    assert climate._attr_min_temp == 15.0
    assert climate._attr_max_temp == 25.0
    assert climate._attr_target_temperature_step == 0.5


async def test_climate_set_temperature_clamping_max(mock_hass, full_coordinator, mock_entry):
    """Test temperature clamping at maximum."""
    climate = EffektGuardClimate(full_coordinator, mock_entry)
    climate.hass = mock_hass

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        await climate.async_set_temperature(**{ATTR_TEMPERATURE: 30.0})

        # Should clamp to max (25.0)
        call_args = mock_update.call_args
        updated_data = call_args[1]["data"]
        assert updated_data[CONF_TARGET_INDOOR_TEMP] == 25.0


async def test_climate_set_temperature_clamping_min(mock_hass, full_coordinator, mock_entry):
    """Test temperature clamping at minimum."""
    climate = EffektGuardClimate(full_coordinator, mock_entry)
    climate.hass = mock_hass

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        await climate.async_set_temperature(**{ATTR_TEMPERATURE: 10.0})

        # Should clamp to min (15.0)
        call_args = mock_update.call_args
        updated_data = call_args[1]["data"]
        assert updated_data[CONF_TARGET_INDOOR_TEMP] == 15.0


def test_climate_current_temperature_no_data(empty_coordinator, mock_entry):
    """Test current temperature with no coordinator data."""
    climate = EffektGuardClimate(empty_coordinator, mock_entry)

    assert climate.current_temperature is None


def test_climate_current_temperature_no_nibe(full_coordinator, mock_entry):
    """Test current temperature with no NIBE state."""
    full_coordinator.data = {"price": MagicMock()}  # No nibe key
    climate = EffektGuardClimate(full_coordinator, mock_entry)

    assert climate.current_temperature is None


def test_climate_extra_attributes_no_data(empty_coordinator, mock_entry):
    """Test extra attributes with no coordinator data."""
    climate = EffektGuardClimate(empty_coordinator, mock_entry)

    attrs = climate.extra_state_attributes
    assert attrs == {}


# ============================================================================
# ALL 14 SENSORS - Value Retrieval
# ============================================================================


def test_all_sensors_with_full_data(full_coordinator, mock_entry):
    """Test all 14 sensors can retrieve values with full data."""
    sensor_values = {}

    for sensor_desc in SENSORS:
        sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)
        sensor_values[sensor_desc.key] = sensor.native_value

    # Verify all sensors returned something (not all None)
    assert sensor_values["current_offset"] == 1.5
    assert sensor_values["degree_minutes"] == -60
    assert sensor_values["supply_temperature"] == 35.0
    assert sensor_values["outdoor_temperature"] == 5.0
    assert sensor_values["current_price"] == 1.25
    assert sensor_values["peak_today"] == 4.5
    assert sensor_values["peak_this_month"] == 5.2
    assert sensor_values["optimization_reasoning"] == "Test reasoning"
    assert sensor_values["quarter_of_day"] == 42
    assert sensor_values["hour_classification"] == "normal"
    assert sensor_values["peak_status"] == "safe"
    assert sensor_values["temperature_trend"] == -0.2
    assert sensor_values["savings_estimate"] == 450.0
    assert sensor_values["optional_features_status"] == "active"


def test_all_sensors_with_no_data(empty_coordinator, mock_entry):
    """Test all 14 sensors handle missing data gracefully."""
    for sensor_desc in SENSORS:
        sensor = EffektGuardSensor(empty_coordinator, mock_entry, sensor_desc)

        # Should not raise exception
        value = sensor.native_value

        # Check expected values based on how each sensor handles missing data
        if sensor_desc.key == "optional_features_status":
            assert value == "initializing", f"{sensor_desc.key}: expected 'initializing'"
        elif sensor_desc.key in ["peak_today", "peak_this_month"]:
            assert value == 0.0, f"{sensor_desc.key}: expected 0.0"
        elif sensor_desc.key == "current_offset":
            assert value == 0.0, f"{sensor_desc.key}: expected 0.0"
        elif sensor_desc.key == "optimization_reasoning":
            assert value == "No decision yet", f"{sensor_desc.key}: expected 'No decision yet'"
        elif sensor_desc.key in ["hour_classification", "peak_status"]:
            assert value == "unknown", f"{sensor_desc.key}: expected 'unknown'"
        elif sensor_desc.key == "heat_pump_model":
            assert value == "Unknown", f"{sensor_desc.key}: expected 'Unknown'"
        else:
            # All other sensors should return None
            assert value is None, f"Sensor {sensor_desc.key} returned {value!r}, expected None"


def test_sensor_degree_minutes(full_coordinator, mock_entry):
    """Test degree minutes sensor specifically."""
    sensor_desc = next(s for s in SENSORS if s.key == "degree_minutes")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == -60


def test_sensor_supply_temperature(full_coordinator, mock_entry):
    """Test supply temperature sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "supply_temperature")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 35.0


def test_sensor_outdoor_temperature(full_coordinator, mock_entry):
    """Test outdoor temperature sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "outdoor_temperature")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 5.0


def test_sensor_quarter_of_day(full_coordinator, mock_entry):
    """Test quarter of day sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "quarter_of_day")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 42


def test_sensor_optimization_reasoning(full_coordinator, mock_entry):
    """Test optimization reasoning sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "optimization_reasoning")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == "Test reasoning"


def test_sensor_peak_this_month(full_coordinator, mock_entry):
    """Test peak this month sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "peak_this_month")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 5.2


# ============================================================================
# SENSOR - optional_features_status (Critical!)
# ============================================================================


def test_sensor_optional_features_status_active(full_coordinator, mock_entry, mock_hass):
    """Test optional_features_status sensor with active data."""
    full_coordinator.hass = mock_hass
    sensor_desc = next(s for s in SENSORS if s.key == "optional_features_status")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == "active"


def test_sensor_optional_features_status_attributes(full_coordinator, mock_entry, mock_hass):
    """Test optional_features_status sensor extra attributes."""
    full_coordinator.hass = mock_hass

    # Mock state responses
    mock_dm_state = MagicMock()
    mock_dm_state.state = "-60"

    mock_power_state = MagicMock()
    mock_power_state.state = "3500"
    mock_power_state.attributes = {"unit_of_measurement": "W"}

    mock_weather_state = MagicMock()
    mock_weather_state.attributes = {"forecast": [{"temp": 15}] * 24}

    def mock_get_state(entity_id):
        if entity_id == "sensor.nibe_degree_minutes":
            return mock_dm_state
        elif entity_id == "sensor.house_power":
            return mock_power_state
        elif entity_id == "weather.home":
            return mock_weather_state
        return None

    mock_hass.states.get = mock_get_state

    sensor_desc = next(s for s in SENSORS if s.key == "optional_features_status")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    attrs = sensor.extra_state_attributes

    # Verify all optional feature statuses
    assert "degree_minutes" in attrs
    assert attrs["degree_minutes"]["status"] == "detected"
    assert attrs["degree_minutes"]["entity"] == "sensor.nibe_degree_minutes"
    assert attrs["degree_minutes"]["value"] == "-60"

    assert "power_meter" in attrs
    assert attrs["power_meter"]["status"] == "detected"
    assert attrs["power_meter"]["entity"] == "sensor.house_power"

    assert "tomorrow_prices" in attrs
    assert attrs["tomorrow_prices"]["status"] == "available"
    assert attrs["tomorrow_prices"]["count"] == 2

    assert "weather_forecast" in attrs
    assert attrs["weather_forecast"]["status"] == "available"


def test_sensor_optional_features_status_missing_features(full_coordinator, mock_entry, mock_hass):
    """Test optional_features_status when features not configured."""
    # Remove optional features from config
    full_coordinator.config_entry.data = {}
    full_coordinator.hass = mock_hass

    sensor_desc = next(s for s in SENSORS if s.key == "optional_features_status")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    attrs = sensor.extra_state_attributes

    # Should show estimated/not_configured statuses
    assert attrs["degree_minutes"]["status"] == "estimated"
    assert attrs["power_meter"]["status"] == "estimated"
    assert attrs["weather_forecast"]["status"] == "not_configured"


# ============================================================================
# SENSOR - Extra Attributes Edge Cases
# ============================================================================


def test_sensor_optimization_reasoning_attributes(full_coordinator, mock_entry):
    """Test optimization_reasoning sensor extra attributes."""
    sensor_desc = next(s for s in SENSORS if s.key == "optimization_reasoning")
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)

    attrs = sensor.extra_state_attributes

    assert "decision_timestamp" in attrs
    assert "applied_offset" in attrs
    assert attrs["applied_offset"] == 1.5


# ============================================================================
# ALL 5 NUMBER ENTITIES - Complete Coverage
# ============================================================================


def test_number_tolerance_value(full_coordinator, mock_entry):
    """Test tolerance number entity."""
    number_desc = next(n for n in NUMBERS if n.key == "tolerance")
    number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)

    assert number.native_value == 0.5
    assert number.entity_description.native_min_value == 0.2
    assert number.entity_description.native_max_value == 2.0


def test_number_thermal_mass_value(full_coordinator, mock_entry):
    """Test thermal mass number entity."""
    number_desc = next(n for n in NUMBERS if n.key == "thermal_mass")
    number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)

    assert number.native_value == 1.0
    assert number.entity_description.native_min_value == 0.5
    assert number.entity_description.native_max_value == 2.0


def test_number_insulation_quality_value(full_coordinator, mock_entry):
    """Test insulation quality number entity."""
    number_desc = next(n for n in NUMBERS if n.key == "insulation_quality")
    number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)

    assert number.native_value == 1.0
    assert number.entity_description.native_min_value == 0.5
    assert number.entity_description.native_max_value == 2.0


def test_number_default_fallback(full_coordinator, mock_entry):
    """Test number entity default value fallback."""
    # Remove config value
    del mock_entry.data[CONF_THERMAL_MASS]

    number_desc = next(n for n in NUMBERS if n.key == "thermal_mass")
    number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)

    # Should return default (1.0)
    assert number.native_value == 1.0


async def test_number_set_all_entities(mock_hass, full_coordinator, mock_entry):
    """Test setting values on all 5 number entities."""
    test_values = {
        "target_temperature": 22.0,
        "tolerance": 0.8,
        "thermal_mass": 1.5,
        "insulation_quality": 1.2,
        "peak_protection_margin": 0.75,
    }

    for key, value in test_values.items():
        number_desc = next(n for n in NUMBERS if n.key == key)
        number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)
        number.hass = mock_hass
        number.entity_id = f"number.effektguard_{key}"

        with patch.object(mock_hass.config_entries, "async_update_entry"):
            with patch.object(number, "async_write_ha_state"):
                await number.async_set_native_value(value)
                full_coordinator.async_request_refresh.assert_called()


# ============================================================================
# SELECT ENTITIES - Edge Cases
# ============================================================================


async def test_select_invalid_option(mock_hass, full_coordinator, mock_entry, caplog):
    """Test selecting invalid option logs error."""
    select_desc = next(s for s in SELECTS if s.key == "optimization_mode")
    select = EffektGuardSelect(full_coordinator, mock_entry, select_desc)
    select.hass = mock_hass
    select.entity_id = "select.effektguard_optimization_mode"

    with patch.object(select, "async_write_ha_state"):
        await select.async_select_option("invalid_mode")

        # Should log error
        assert "Invalid option" in caplog.text


def test_select_default_fallback(full_coordinator, mock_entry):
    """Test select entity default value fallback."""
    # Remove config value
    del mock_entry.data[CONF_OPTIMIZATION_MODE]

    select_desc = next(s for s in SELECTS if s.key == "optimization_mode")
    select = EffektGuardSelect(full_coordinator, mock_entry, select_desc)

    # Should return default ("balanced")
    assert select.current_option == "balanced"


# ============================================================================
# ENTITY CREATION TESTS
# ============================================================================


async def test_climate_entity_setup(mock_hass, full_coordinator, mock_entry):
    """Test climate entity async_setup_entry."""
    from custom_components.effektguard.climate import async_setup_entry

    mock_hass.data = {DOMAIN: {mock_entry.entry_id: full_coordinator}}
    async_add_entities = MagicMock()

    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 1
    assert isinstance(entities[0], EffektGuardClimate)


async def test_sensor_entities_setup(mock_hass, full_coordinator, mock_entry):
    """Test sensor entities async_setup_entry."""
    from custom_components.effektguard.sensor import async_setup_entry

    mock_hass.data = {DOMAIN: {mock_entry.entry_id: full_coordinator}}
    async_add_entities = MagicMock()

    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 15


async def test_number_entities_setup(mock_hass, full_coordinator, mock_entry):
    """Test number entities async_setup_entry."""
    from custom_components.effektguard.number import async_setup_entry

    mock_hass.data = {DOMAIN: {mock_entry.entry_id: full_coordinator}}
    async_add_entities = MagicMock()

    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 5


async def test_select_entities_setup(mock_hass, full_coordinator, mock_entry):
    """Test select entities async_setup_entry."""
    from custom_components.effektguard.select import async_setup_entry

    mock_hass.data = {DOMAIN: {mock_entry.entry_id: full_coordinator}}
    async_add_entities = MagicMock()

    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 2


async def test_switch_entities_setup(mock_hass, full_coordinator, mock_entry):
    """Test switch entities async_setup_entry."""
    from custom_components.effektguard.switch import async_setup_entry

    mock_hass.data = {DOMAIN: {mock_entry.entry_id: full_coordinator}}
    async_add_entities = MagicMock()

    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    entities = async_add_entities.call_args[0][0]
    assert len(entities) == 5


# ============================================================================
# DEVICE INFO VALIDATION
# ============================================================================


def test_all_entities_have_device_info(full_coordinator, mock_entry):
    """Test all entities have proper device_info."""
    # Test one entity of each type
    climate = EffektGuardClimate(full_coordinator, mock_entry)
    assert climate._attr_device_info is not None
    assert (DOMAIN, mock_entry.entry_id) in climate._attr_device_info["identifiers"]

    sensor_desc = SENSORS[0]
    sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)
    assert sensor._attr_device_info is not None
    assert (DOMAIN, mock_entry.entry_id) in sensor._attr_device_info["identifiers"]

    number_desc = NUMBERS[0]
    number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)
    assert number._attr_device_info is not None

    select_desc = SELECTS[0]
    select = EffektGuardSelect(full_coordinator, mock_entry, select_desc)
    assert select._attr_device_info is not None

    switch_desc = SWITCHES[0]
    switch = EffektGuardSwitch(full_coordinator, mock_entry, switch_desc)
    assert switch._attr_device_info is not None


def test_all_entities_have_unique_id(full_coordinator, mock_entry):
    """Test all entities have unique_id."""
    climate = EffektGuardClimate(full_coordinator, mock_entry)
    assert climate._attr_unique_id == f"{mock_entry.entry_id}_climate"

    for sensor_desc in SENSORS:
        sensor = EffektGuardSensor(full_coordinator, mock_entry, sensor_desc)
        assert sensor._attr_unique_id == f"{mock_entry.entry_id}_{sensor_desc.key}"

    for number_desc in NUMBERS:
        number = EffektGuardNumber(full_coordinator, mock_entry, number_desc)
        assert number._attr_unique_id == f"{mock_entry.entry_id}_{number_desc.key}"

    for select_desc in SELECTS:
        select = EffektGuardSelect(full_coordinator, mock_entry, select_desc)
        assert select._attr_unique_id == f"{mock_entry.entry_id}_{select_desc.key}"

    for switch_desc in SWITCHES:
        switch = EffektGuardSwitch(full_coordinator, mock_entry, switch_desc)
        assert switch._attr_unique_id == f"{mock_entry.entry_id}_{switch_desc.key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
