"""Tests for EffektGuard entities.

Tests sensor, number, select, switch, and climate entity functionality
including preset modes, state attributes, and value updates.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.components.climate.const import (
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_ECO,
    PRESET_NONE,
)
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.sensor import EffektGuardSensor, SENSORS
from custom_components.effektguard.number import EffektGuardNumber, NUMBERS
from custom_components.effektguard.select import EffektGuardSelect, SELECTS
from custom_components.effektguard.switch import EffektGuardSwitch, SWITCHES
from custom_components.effektguard.const import (
    CONF_CONTROL_PRIORITY,
    CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    CONF_ENABLE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_WEATHER_PREDICTION,
    CONF_INSULATION_QUALITY,
    CONF_OPTIMIZATION_MODE,
    CONF_PEAK_PROTECTION_MARGIN,
    CONF_TARGET_INDOOR_TEMP,
    CONF_THERMAL_MASS,
    CONF_TOLERANCE,
    CONTROL_PRIORITY_BALANCED,
    CONTROL_PRIORITY_COMFORT,
    CONTROL_PRIORITY_SAVINGS,
    DOMAIN,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)


@pytest.fixture
def mock_coordinator():
    """Create mock coordinator."""
    from datetime import datetime, timezone

    coordinator = MagicMock(spec=DataUpdateCoordinator)
    coordinator.last_update_success = True  # Add for availability check
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
            today_classifications={"0": "cheap", "42": "normal", "72": "peak"},
            today_prices=[0.8, 1.0, 1.2, 1.5, 2.0],
        ),
        "decision": MagicMock(
            offset=1.5,
            reasoning="Pre-heating before peak hours",
            peak_status="safe",
            peak_margin=2.5,
            current_power=3.5,
            timestamp=datetime.now(),
            layers=[
                type(
                    "Layer",
                    (),
                    {"name": "Safety", "offset": 0.0, "weight": 1.0, "reason": "Temp OK"},
                )(),
                type(
                    "Layer",
                    (),
                    {"name": "Price", "offset": 2.0, "weight": 0.6, "reason": "Cheap period"},
                )(),
            ],
        ),
        "thermal": MagicMock(temperature_trend=-0.2, prediction_3h=21.0, prediction_6h=20.5),
        "thermal_trend": {
            "rate_per_hour": -0.2,
            "trend": "falling",
            "confidence": 0.8,
            "samples": 12,
        },
        "weather": MagicMock(
            forecast_hours=[
                MagicMock(datetime="2025-10-14T15:00:00", temperature=4.0),
                MagicMock(datetime="2025-10-14T16:00:00", temperature=3.5),
            ]
        ),
        "savings": MagicMock(
            monthly_estimate=450.0,
            effect_savings=200.0,
            spot_savings=250.0,
            baseline_cost=1200.0,
            optimized_cost=750.0,
        ),
        "current_classification": "normal",  # Added for hour_classification sensor
        "current_quarter": 42,  # Added for consistency
        "dhw_status": "idle",  # DHW sensors read from coordinator.data
        "dhw_recommendation": "Run DHW now - cheap period",
    }
    coordinator.peak_today = 4.5
    coordinator.peak_this_month = 5.2
    coordinator.current_peak = 5.2  # Add this for climate entity

    # Peak tracking metadata (use timezone-aware datetime)
    coordinator.peak_today_time = datetime.now(timezone.utc)
    coordinator.peak_today_source = "nibe_currents"
    coordinator.peak_today_quarter = 42
    coordinator.yesterday_peak = 4.2

    # Additional sensor values
    coordinator.nibe_power = None  # Not available if phase currents missing
    coordinator.optimization_reasoning = "Pre-heating before peak hours"

    # Heat pump model (mock as object with model_name attribute)
    coordinator.heat_pump_model = type("HeatPumpModel", (), {"model_name": "NIBE F2040"})()

    coordinator.async_request_refresh = AsyncMock()
    return coordinator


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    import threading
    from homeassistant.const import UnitOfTemperature

    hass = MagicMock(spec=HomeAssistant)
    hass.config_entries = MagicMock()
    hass.config_entries.async_update_entry = MagicMock()
    hass.loop_thread_id = threading.get_ident()  # Add loop_thread_id for async_write_ha_state

    # Mock config for climate entity precision
    hass.config = MagicMock()
    hass.config.units = MagicMock()
    hass.config.units.temperature_unit = UnitOfTemperature.CELSIUS

    # Mock hass.data to prevent AttributeError in async_write_ha_state
    hass.data = {}

    # Mock states for async_write_ha_state
    hass.states = MagicMock()
    hass.states.async_set_internal = MagicMock()

    return hass


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
        CONF_CONTROL_PRIORITY: CONTROL_PRIORITY_BALANCED,
        CONF_ENABLE_OPTIMIZATION: True,
        CONF_ENABLE_PRICE_OPTIMIZATION: True,
        CONF_ENABLE_PEAK_PROTECTION: True,
        CONF_ENABLE_WEATHER_PREDICTION: True,
        CONF_ENABLE_HOT_WATER_OPTIMIZATION: False,
    }
    # Add options dict for entry.options (user-configurable settings)
    # Phase 1 fix: TARGET_INDOOR_TEMP and OPTIMIZATION_MODE moved to options
    entry.options = {
        CONF_TARGET_INDOOR_TEMP: 21.0,
        CONF_OPTIMIZATION_MODE: OPTIMIZATION_MODE_BALANCED,
        CONF_THERMAL_MASS: 1.0,
        CONF_INSULATION_QUALITY: 1.0,
    }
    return entry


# ============================================================================
# SENSOR TESTS
# ============================================================================


def test_sensor_count():
    """Test that all expected sensors are defined."""
    from custom_components.effektguard.sensor import SENSORS

    assert (
        len(SENSORS) == 20
    )  # All sensors including outdoor_temperature_trend, dhw_status, dhw_recommendation


async def test_sensor_entities_created(mock_coordinator, mock_hass, mock_entry):
    """Test sensor entities are created."""
    from custom_components.effektguard.sensor import async_setup_entry

    # Setup mock hass.data structure
    mock_hass.data = {DOMAIN: {mock_entry.entry_id: mock_coordinator}}

    async_add_entities = MagicMock()
    await async_setup_entry(mock_hass, mock_entry, async_add_entities)

    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert (
        len(sensors) == 20
    )  # All sensors including outdoor_temperature_trend, dhw_status, dhw_recommendation


def test_sensor_current_offset(mock_coordinator, mock_entry):
    """Test current offset sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "current_offset")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 1.5
    assert sensor.entity_description.device_class is not None


def test_sensor_hour_classification(mock_coordinator, mock_entry):
    """Test hour classification sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "hour_classification")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == "normal"
    attrs = sensor.extra_state_attributes
    assert "today_classifications" in attrs
    assert "today_min" in attrs
    assert "today_max" in attrs


def test_sensor_peak_status(mock_coordinator, mock_entry):
    """Test peak status sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "peak_status")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # With current_power=3.5, peak_month=5.2, status should be "normal"
    assert sensor.native_value == "normal"
    attrs = sensor.extra_state_attributes
    # These attributes come from the extra_state_attributes logic
    assert attrs["monthly_peak"] == 5.2
    assert attrs["daily_peak"] == 4.5


def test_sensor_temperature_trend(mock_coordinator, mock_entry):
    """Test temperature trend sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "temperature_trend")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == -0.2
    attrs = sensor.extra_state_attributes
    assert "trend_direction" in attrs
    assert attrs["trend_direction"] == "falling"
    assert "confidence" in attrs
    assert attrs["confidence"] == 0.8


def test_sensor_savings_estimate(mock_coordinator, mock_entry):
    """Test savings estimate sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "savings_estimate")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 450.0
    attrs = sensor.extra_state_attributes
    assert attrs["effect_savings"] == 200.0
    assert attrs["spot_savings"] == 250.0


def test_sensor_extra_attributes_current_offset(mock_coordinator, mock_entry):
    """Test current offset sensor attributes with layer votes."""
    sensor_desc = next(s for s in SENSORS if s.key == "current_offset")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    attrs = sensor.extra_state_attributes
    assert "layer_votes" in attrs
    assert len(attrs["layer_votes"]) == 2
    # Check actual fields in layer_votes (offset, weight, reason - not name)
    assert attrs["layer_votes"][0]["offset"] == 0.0
    assert attrs["layer_votes"][0]["reason"] == "Temp OK"
    assert attrs["layer_votes"][1]["offset"] == 2.0
    assert attrs["layer_votes"][1]["reason"] == "Cheap period"


def test_sensor_degree_minutes(mock_coordinator, mock_entry):
    """Test degree minutes sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "degree_minutes")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == -60  # From mock nibe.degree_minutes
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_supply_temperature(mock_coordinator, mock_entry):
    """Test supply temperature sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "supply_temperature")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 35.0
    assert sensor.entity_description.device_class == SensorDeviceClass.TEMPERATURE
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_outdoor_temperature(mock_coordinator, mock_entry):
    """Test outdoor temperature sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "outdoor_temperature")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 5.0
    assert sensor.entity_description.device_class == SensorDeviceClass.TEMPERATURE
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_indoor_temperature(mock_coordinator, mock_entry):
    """Test indoor temperature sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "indoor_temperature")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 21.5  # From mock nibe.indoor_temp
    assert sensor.entity_description.device_class == SensorDeviceClass.TEMPERATURE
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC

    # Check attributes
    attrs = sensor.extra_state_attributes
    assert "nibe_bt50" in attrs
    assert attrs["nibe_bt50"] == 21.5
    assert "calculation_method" in attrs
    assert "sensor_count" in attrs


def test_sensor_current_price(mock_coordinator, mock_entry):
    """Test current electricity price sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "current_price")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 1.25  # From mock price.current_price
    assert sensor.entity_description.device_class == SensorDeviceClass.MONETARY


def test_sensor_peak_today(mock_coordinator, mock_entry):
    """Test peak today sensor with new attributes."""
    sensor_desc = next(s for s in SENSORS if s.key == "peak_today")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 4.5  # From mock coordinator.peak_today
    assert sensor.entity_description.device_class == SensorDeviceClass.POWER

    # Test new peak_today attributes
    attrs = sensor.extra_state_attributes
    assert "peak_time" in attrs
    assert "peak_quarter" in attrs
    assert attrs["peak_quarter"] == 42  # From mock coordinator.peak_today_quarter
    assert "measurement_source" in attrs
    assert attrs["measurement_source"] == "nibe_currents"  # From mock
    assert "will_affect_billing" in attrs
    assert "yesterday_peak" in attrs
    assert attrs["yesterday_peak"] == 4.2  # From mock coordinator.yesterday_peak
    assert "measurement_description" in attrs
    assert "is_real_measurement" in attrs


def test_sensor_peak_this_month(mock_coordinator, mock_entry):
    """Test peak this month sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "peak_this_month")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 5.2
    assert sensor.entity_description.device_class == SensorDeviceClass.POWER


def test_sensor_nibe_power(mock_coordinator, mock_entry):
    """Test NIBE power calculation sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "nibe_power")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # Mock returns None when phase currents not available
    assert sensor.native_value is None  # From mock coordinator.nibe_power
    assert sensor.entity_description.device_class == SensorDeviceClass.POWER
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_optimization_reasoning(mock_coordinator, mock_entry):
    """Test optimization reasoning sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "optimization_reasoning")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value is not None
    assert isinstance(sensor.native_value, str)
    # Full reasoning should be in attributes
    attrs = sensor.extra_state_attributes
    assert "full_reasoning" in attrs or len(sensor.native_value) <= 255


def test_sensor_quarter_of_day(mock_coordinator, mock_entry):
    """Test quarter of day sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "quarter_of_day")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    assert sensor.native_value == 42  # From mock_coordinator
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_optional_features_status(mock_coordinator, mock_entry):
    """Test optional features status sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "optional_features_status")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # Sensor returns "active" if coordinator.data exists (see value_fn lambda)
    assert sensor.native_value == "active"
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_heat_pump_model(mock_coordinator, mock_entry):
    """Test heat pump model sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "heat_pump_model")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # Should return model name from coordinator
    assert sensor.native_value == "NIBE F2040"  # From mock
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_dhw_status(mock_coordinator, mock_entry):
    """Test DHW status sensor."""
    sensor_desc = next(s for s in SENSORS if s.key == "dhw_status")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # Sensor reads from coordinator.data["dhw_status"]
    assert sensor.native_value == "idle"  # From mock data["dhw_status"]
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC


def test_sensor_dhw_recommendation(mock_coordinator, mock_entry):
    """Test DHW recommendation sensor with attributes."""
    sensor_desc = next(s for s in SENSORS if s.key == "dhw_recommendation")
    sensor = EffektGuardSensor(mock_coordinator, mock_entry, sensor_desc)

    # Sensor reads from coordinator.data["dhw_recommendation"]
    assert (
        sensor.native_value == "Run DHW now - cheap period"
    )  # From mock data["dhw_recommendation"]
    assert sensor.entity_description.entity_category == EntityCategory.DIAGNOSTIC

    # Test DHW planning attributes
    attrs = sensor.extra_state_attributes
    # Should have planning attributes if available
    if "planning_summary" in attrs:
        assert isinstance(attrs["planning_summary"], str)


# ============================================================================
# NUMBER ENTITY TESTS
# ============================================================================


def test_number_count():
    """Test that all required number entities are defined."""
    assert len(NUMBERS) == 5  # target_temp, tolerance, thermal_mass, insulation, peak_margin


def test_number_entities_created(mock_coordinator, mock_entry):
    """Test number entity creation."""
    numbers = [EffektGuardNumber(mock_coordinator, mock_entry, desc) for desc in NUMBERS]
    assert len(numbers) == 5
    assert all(number._attr_has_entity_name for number in numbers)


def test_number_target_temperature(mock_coordinator, mock_entry):
    """Test target temperature number entity."""
    number_desc = next(n for n in NUMBERS if n.key == "target_temperature")
    number = EffektGuardNumber(mock_coordinator, mock_entry, number_desc)

    assert number.native_value == 21.0
    assert number.entity_description.native_min_value == 18.0
    assert number.entity_description.native_max_value == 26.0


async def test_number_set_value(mock_hass, mock_coordinator, mock_entry):
    """Test setting number entity value."""
    number_desc = next(n for n in NUMBERS if n.key == "target_temperature")
    number = EffektGuardNumber(mock_coordinator, mock_entry, number_desc)
    number.hass = mock_hass
    number.entity_id = "number.effektguard_target_temperature"  # Set entity_id

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        with patch.object(number, "async_write_ha_state"):  # Skip async_write_ha_state
            await number.async_set_native_value(22.5)
            mock_update.assert_called_once()
            mock_coordinator.async_request_refresh.assert_called_once()


def test_number_peak_protection_margin(mock_coordinator, mock_entry):
    """Test peak protection margin number entity."""
    number_desc = next(n for n in NUMBERS if n.key == "peak_protection_margin")
    number = EffektGuardNumber(mock_coordinator, mock_entry, number_desc)

    assert number.native_value == 0.5
    assert number.entity_description.native_unit_of_measurement == "kW"
    assert number.entity_description.native_min_value == 0.0
    assert number.entity_description.native_max_value == 2.0


# ============================================================================
# SELECT ENTITY TESTS
# ============================================================================


def test_select_count():
    """Test that all required select entities are defined."""
    assert len(SELECTS) == 2  # optimization_mode, control_priority


def test_select_entities_created(mock_coordinator, mock_entry):
    """Test select entity creation."""
    selects = [EffektGuardSelect(mock_coordinator, mock_entry, desc) for desc in SELECTS]
    assert len(selects) == 2
    assert all(select._attr_has_entity_name for select in selects)


def test_select_optimization_mode(mock_coordinator, mock_entry):
    """Test optimization mode select entity."""
    select_desc = next(s for s in SELECTS if s.key == "optimization_mode")
    select = EffektGuardSelect(mock_coordinator, mock_entry, select_desc)

    assert select.current_option == OPTIMIZATION_MODE_BALANCED
    assert OPTIMIZATION_MODE_COMFORT in select.entity_description.options
    assert OPTIMIZATION_MODE_BALANCED in select.entity_description.options
    assert OPTIMIZATION_MODE_SAVINGS in select.entity_description.options


async def test_select_change_option(mock_hass, mock_coordinator, mock_entry):
    """Test changing select entity option."""
    select_desc = next(s for s in SELECTS if s.key == "optimization_mode")
    select = EffektGuardSelect(mock_coordinator, mock_entry, select_desc)
    select.hass = mock_hass
    select.entity_id = "select.effektguard_optimization_mode"  # Set entity_id

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        with patch.object(select, "async_write_ha_state"):  # Skip async_write_ha_state
            await select.async_select_option(OPTIMIZATION_MODE_COMFORT)
            mock_update.assert_called_once()
            mock_coordinator.async_request_refresh.assert_called_once()


def test_select_control_priority(mock_coordinator, mock_entry):
    """Test control priority select entity."""
    select_desc = next(s for s in SELECTS if s.key == "control_priority")
    select = EffektGuardSelect(mock_coordinator, mock_entry, select_desc)

    assert select.current_option == CONTROL_PRIORITY_BALANCED
    assert CONTROL_PRIORITY_COMFORT in select.entity_description.options
    assert CONTROL_PRIORITY_BALANCED in select.entity_description.options
    assert CONTROL_PRIORITY_SAVINGS in select.entity_description.options


# ============================================================================
# SWITCH ENTITY TESTS
# ============================================================================


def test_switch_count():
    """Test that all required switch entities are defined."""
    assert len(SWITCHES) == 5  # enable, price, peak, weather, hot_water


def test_switch_entities_created(mock_coordinator, mock_entry):
    """Test switch entity creation."""
    switches = [EffektGuardSwitch(mock_coordinator, mock_entry, desc) for desc in SWITCHES]
    assert len(switches) == 5
    assert all(switch._attr_has_entity_name for switch in switches)


def test_switch_enable_optimization(mock_coordinator, mock_entry):
    """Test master enable optimization switch."""
    switch_desc = next(s for s in SWITCHES if s.key == "enable_optimization")
    switch = EffektGuardSwitch(mock_coordinator, mock_entry, switch_desc)

    assert switch.is_on is True  # Default on


def test_switch_price_optimization(mock_coordinator, mock_entry):
    """Test price optimization switch."""
    switch_desc = next(s for s in SWITCHES if s.key == "price_optimization")
    switch = EffektGuardSwitch(mock_coordinator, mock_entry, switch_desc)

    assert switch.is_on is True


def test_switch_hot_water_optimization(mock_coordinator, mock_entry):
    """Test hot water optimization switch (experimental)."""
    switch_desc = next(s for s in SWITCHES if s.key == "hot_water_optimization")
    switch = EffektGuardSwitch(mock_coordinator, mock_entry, switch_desc)

    assert switch.is_on is False  # Default off (experimental)


async def test_switch_turn_on(mock_hass, mock_coordinator, mock_entry):
    """Test turning switch on."""
    switch_desc = next(s for s in SWITCHES if s.key == "weather_prediction")
    switch = EffektGuardSwitch(mock_coordinator, mock_entry, switch_desc)
    switch.hass = mock_hass
    switch.entity_id = "switch.effektguard_weather_prediction"  # Set entity_id

    # Set to off first
    mock_entry.data[CONF_ENABLE_WEATHER_PREDICTION] = False

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        with patch.object(switch, "async_write_ha_state"):  # Skip async_write_ha_state
            await switch.async_turn_on()
            mock_update.assert_called_once()
            mock_coordinator.async_request_refresh.assert_called_once()


async def test_switch_turn_off(mock_hass, mock_coordinator, mock_entry):
    """Test turning switch off."""
    switch_desc = next(s for s in SWITCHES if s.key == "peak_protection")
    switch = EffektGuardSwitch(mock_coordinator, mock_entry, switch_desc)
    switch.hass = mock_hass
    switch.entity_id = "switch.effektguard_peak_protection"  # Set entity_id

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        with patch.object(switch, "async_write_ha_state"):  # Skip async_write_ha_state
            await switch.async_turn_off()
            mock_update.assert_called_once()
            mock_coordinator.async_request_refresh.assert_called_once()


# ============================================================================
# CLIMATE ENTITY TESTS
# ============================================================================


def test_climate_entity_creation(mock_coordinator, mock_entry):
    """Test climate entity creation."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    assert climate._attr_has_entity_name
    assert climate._attr_temperature_unit == "°C"
    assert HVACMode.HEAT in climate._attr_hvac_modes
    assert HVACMode.OFF in climate._attr_hvac_modes


def test_climate_supported_features(mock_coordinator, mock_entry):
    """Test climate entity supported features."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    assert climate._attr_supported_features & ClimateEntityFeature.TARGET_TEMPERATURE
    assert climate._attr_supported_features & ClimateEntityFeature.PRESET_MODE


def test_climate_preset_modes(mock_coordinator, mock_entry):
    """Test climate entity has all preset modes."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    assert PRESET_NONE in climate._attr_preset_modes
    assert PRESET_ECO in climate._attr_preset_modes
    assert PRESET_AWAY in climate._attr_preset_modes
    assert PRESET_COMFORT in climate._attr_preset_modes


def test_climate_current_temperature(mock_coordinator, mock_entry):
    """Test climate entity current temperature from NIBE."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    assert climate.current_temperature == 21.5


def test_climate_target_temperature(mock_coordinator, mock_entry):
    """Test climate entity target temperature from config."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    assert climate.target_temperature == 21.0


def test_climate_preset_to_optimization_mode_mapping(mock_coordinator, mock_entry):
    """Test preset mode to optimization mode mapping."""
    # Test COMFORT preset
    mock_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_COMFORT
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    assert climate.preset_mode == PRESET_COMFORT

    # Test BALANCED preset
    mock_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_BALANCED
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    assert climate.preset_mode == PRESET_NONE

    # Test SAVINGS preset
    mock_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_SAVINGS
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    assert climate.preset_mode == PRESET_ECO


async def test_climate_set_temperature(mock_hass, mock_coordinator, mock_entry):
    """Test setting target temperature."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    climate.hass = mock_hass
    climate.entity_id = "climate.effektguard"  # Set entity_id to prevent NoEntitySpecifiedError

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        await climate.async_set_temperature(**{ATTR_TEMPERATURE: 22.0})
        mock_update.assert_called_once()
        mock_coordinator.async_request_refresh.assert_called_once()


async def test_climate_set_preset_mode(mock_hass, mock_coordinator, mock_entry):
    """Test setting preset mode updates optimization mode."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    climate.hass = mock_hass

    with patch.object(mock_hass.config_entries, "async_update_entry") as mock_update:
        await climate.async_set_preset_mode(PRESET_COMFORT)
        mock_update.assert_called_once()

        # Verify it set the correct optimization mode in options (not data)
        call_args = mock_update.call_args
        updated_options = call_args[1]["options"]
        assert updated_options[CONF_OPTIMIZATION_MODE] == OPTIMIZATION_MODE_COMFORT


async def test_climate_set_hvac_mode_heat(mock_hass, mock_coordinator, mock_entry):
    """Test setting HVAC mode to HEAT enables optimization."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    climate.hass = mock_hass
    climate.entity_id = "climate.effektguard"  # Set entity_id
    mock_coordinator.set_optimization_enabled = AsyncMock()

    with patch.object(climate, "async_write_ha_state"):  # Skip async_write_ha_state
        await climate.async_set_hvac_mode(HVACMode.HEAT)
        mock_coordinator.set_optimization_enabled.assert_called_once_with(True)


async def test_climate_set_hvac_mode_off(mock_hass, mock_coordinator, mock_entry):
    """Test setting HVAC mode to OFF disables optimization."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)
    climate.hass = mock_hass
    climate.entity_id = "climate.effektguard"  # Set entity_id
    mock_coordinator.set_optimization_enabled = AsyncMock()

    with patch.object(climate, "async_write_ha_state"):  # Skip async_write_ha_state
        await climate.async_set_hvac_mode(HVACMode.OFF)
        mock_coordinator.set_optimization_enabled.assert_called_once_with(False)


def test_climate_extra_state_attributes(mock_coordinator, mock_entry):
    """Test climate entity extra state attributes."""
    climate = EffektGuardClimate(mock_coordinator, mock_entry)

    attrs = climate.extra_state_attributes

    assert attrs["current_offset"] == 1.5
    assert attrs["optimization_reasoning"] == "Pre-heating before peak hours"
    assert attrs["outdoor_temp"] == 5.0
    assert attrs["supply_temp"] == 35.0
    assert attrs["degree_minutes"] == -60
    assert attrs["current_price"] == 1.25
    assert attrs["monthly_peak"] == 5.2
