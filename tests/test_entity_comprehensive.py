"""Comprehensive tests for all EffektGuard entities, sensors, and attributes.

Tests verify:
- All sensor entities are created and have correct configuration
- All sensor attributes are properly defined
- Climate entity functions correctly with proper attributes
- Switch entities are created with correct configuration
- All entity unique IDs, names, and device info are correct
- Sensor value functions handle None/missing data gracefully
- Extra state attributes are populated correctly

This ensures complete entity coverage for production use.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from homeassistant.components.climate import ClimateEntityFeature, HVACMode
from homeassistant.components.climate.const import PRESET_COMFORT, PRESET_ECO, PRESET_NONE
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import UnitOfPower, UnitOfTemperature, ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.effektguard.climate import EffektGuardClimate
from custom_components.effektguard.sensor import EffektGuardSensor, SENSORS
from custom_components.effektguard.switch import EffektGuardSwitch, SWITCHES
from custom_components.effektguard.const import (
    CONF_ENABLE_OPTIMIZATION,
    CONF_ENABLE_PEAK_PROTECTION,
    CONF_ENABLE_PRICE_OPTIMIZATION,
    CONF_ENABLE_WEATHER_PREDICTION,
    CONF_ENABLE_HOT_WATER_OPTIMIZATION,
    CONF_OPTIMIZATION_MODE,
    CONF_TARGET_INDOOR_TEMP,
    DOMAIN,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
)


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = Mock(spec=HomeAssistant)
    hass.data = {DOMAIN: {}}
    hass.config_entries = Mock()
    hass.states = Mock()
    hass.states.get = Mock(return_value=None)
    return hass


@pytest.fixture
def mock_config_entry():
    """Create mock config entry."""
    entry = Mock()
    entry.entry_id = "test_entry_id"
    entry.domain = DOMAIN
    entry.data = {
        "nibe_entity": "number.nibe_offset_s1",
        "gespot_entity": "sensor.gespot_current_price",
        "weather_entity": "weather.forecast_home",
        # Switch states stored in data
        CONF_ENABLE_OPTIMIZATION: True,
        CONF_ENABLE_PEAK_PROTECTION: True,
        CONF_ENABLE_PRICE_OPTIMIZATION: True,
        CONF_ENABLE_WEATHER_PREDICTION: True,
        CONF_ENABLE_HOT_WATER_OPTIMIZATION: False,
    }
    entry.options = {
        CONF_TARGET_INDOOR_TEMP: 21.0,
        CONF_OPTIMIZATION_MODE: OPTIMIZATION_MODE_BALANCED,
    }
    return entry


@pytest.fixture
def mock_coordinator_with_data(mock_hass, mock_config_entry):
    """Create mock coordinator with complete data."""
    coordinator = Mock(spec=DataUpdateCoordinator)
    coordinator.hass = mock_hass
    coordinator.entry = mock_config_entry
    coordinator.last_update_success = True

    # Mock NIBE data
    nibe_data = Mock()
    nibe_data.indoor_temp = 21.5
    nibe_data.outdoor_temp = 5.0
    nibe_data.supply_temp = 35.0
    nibe_data.degree_minutes = -60
    nibe_data.phase1_current = 10.0
    nibe_data.phase2_current = 9.0
    nibe_data.phase3_current = 11.0

    # Mock decision data
    decision_data = Mock()
    decision_data.offset = 2.0
    decision_data.reasoning = "Optimizing for low price period"
    decision_data.layers = [
        Mock(offset=1.0, weight=0.5, reason="Price optimization"),
        Mock(offset=1.0, weight=0.5, reason="Weather compensation"),
    ]

    # Mock price data
    # Note: PriceData has .today (list of QuarterPeriod), not .today_prices
    from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
    from datetime import datetime
    from homeassistant.util import dt as dt_util
    
    price_data = Mock()
    price_data.current_price = 1.25
    
    # Create mock QuarterPeriod objects with prices
    now = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    price_data.today = [
        QuarterPeriod(start_time=now, price=0.8),
        QuarterPeriod(start_time=now.replace(minute=15), price=1.0),
        QuarterPeriod(start_time=now.replace(minute=30), price=1.2),
        QuarterPeriod(start_time=now.replace(minute=45), price=1.5),
        QuarterPeriod(start_time=now.replace(hour=1, minute=0), price=1.8),
        QuarterPeriod(start_time=now.replace(hour=1, minute=15), price=2.0),
    ]
    # Extend to 96 periods (fill rest with average price)
    avg_price = 1.3
    for i in range(6, 96):
        hour = i // 4
        minute = (i % 4) * 15
        price_data.today.append(
            QuarterPeriod(start_time=now.replace(hour=hour, minute=minute), price=avg_price)
        )

    # Mock thermal trend data
    thermal_trend = {
        "rate_per_hour": 0.05,
        "trend": "rising",
        "confidence": 0.85,
        "samples": 10,
    }

    outdoor_trend = {
        "rate_per_hour": -0.10,
        "trend": "falling",
        "confidence": 0.90,
        "samples": 12,
        "temp_change_2h": -0.20,
    }

    # Mock DHW planning data
    dhw_planning = {
        "should_heat": True,
        "priority_reason": "Low price period",
        "current_temperature": 45.0,
        "target_temperature": 50.0,
        "thermal_debt": -80,
        "thermal_debt_threshold_block": -500,
        "thermal_debt_threshold_abort": -1500,
        "thermal_debt_status": "normal",
        "space_heating_demand_kw": 2.5,
        "current_price_classification": "cheap",
        "outdoor_temperature": 5.0,
        "indoor_temperature": 21.5,
        "climate_zone": "temperate",
        "weather_opportunity": True,
        "optimal_heating_windows": [
            {
                "time_range": "14:00-16:00",
                "price_classification": "cheap",
                "duration_hours": 2.0,
                "thermal_debt_ok": True,
            }
        ],
        "next_optimal_window": {
            "time_range": "14:00-16:00",
            "price_classification": "cheap",
            "duration_hours": 2.0,
        },
    }

    # Mock savings data
    savings_data = Mock()
    savings_data.monthly_estimate = 350.0

    # Mock heat pump model
    heat_pump_model = Mock()
    heat_pump_model.model_name = "NIBE F2040-12"

    coordinator.data = {
        "nibe": nibe_data,
        "decision": decision_data,
        "price": price_data,
        "thermal_trend": thermal_trend,
        "outdoor_trend": outdoor_trend,
        "dhw_planning": dhw_planning,
        "dhw_planning_summary": "Heat now during cheap period",
        "dhw_status": "heating",
        "dhw_recommendation": "Heat DHW now - optimal price window",
        "savings": savings_data,
        "current_quarter": 56,  # 14:00
        "current_classification": "cheap",
    }

    coordinator.peak_today = 8.5
    coordinator.peak_this_month = 12.3
    coordinator.heat_pump_model = heat_pump_model

    # Mock NIBE adapter for power calculation
    nibe_adapter = Mock()
    nibe_adapter.calculate_power_from_currents = Mock(return_value=6.9)
    coordinator.nibe = nibe_adapter

    return coordinator


class TestSensorEntityDefinitions:
    """Test all sensor entity definitions are correct."""

    def test_all_sensors_have_required_fields(self):
        """Verify all sensor descriptions have required fields."""
        for sensor in SENSORS:
            assert sensor.key is not None, f"Sensor missing key"
            assert sensor.name is not None, f"Sensor {sensor.key} missing name"
            assert sensor.icon is not None, f"Sensor {sensor.key} missing icon"
            assert sensor.value_fn is not None, f"Sensor {sensor.key} missing value_fn"

    def test_sensor_count_matches_expected(self):
        """Test that we have the expected number of sensors."""
        assert len(SENSORS) == 22, f"Expected 22 sensors, found {len(SENSORS)}"

    def test_all_sensor_keys_are_unique(self):
        """Verify no duplicate sensor keys."""
        keys = [sensor.key for sensor in SENSORS]
        assert len(keys) == len(set(keys)), "Duplicate sensor keys found"

    def test_temperature_sensors_have_correct_config(self):
        """Verify temperature sensors have proper device class and units."""
        temp_sensors = [
            "current_offset",
            "supply_temperature",
            "outdoor_temperature",
            "indoor_temperature",
        ]

        for sensor in SENSORS:
            if sensor.key in temp_sensors:
                assert sensor.device_class == SensorDeviceClass.TEMPERATURE
                assert sensor.native_unit_of_measurement == UnitOfTemperature.CELSIUS
                assert sensor.state_class == SensorStateClass.MEASUREMENT

    def test_power_sensors_have_correct_config(self):
        """Verify power sensors have proper device class and units."""
        power_sensors = ["peak_today", "peak_this_month", "nibe_power"]

        for sensor in SENSORS:
            if sensor.key in power_sensors:
                assert sensor.device_class == SensorDeviceClass.POWER
                assert sensor.native_unit_of_measurement == UnitOfPower.KILO_WATT
                assert sensor.state_class == SensorStateClass.MEASUREMENT

    def test_diagnostic_sensors_have_category(self):
        """Verify diagnostic sensors have correct entity category."""
        diagnostic_sensors = [
            "degree_minutes",
            "supply_temperature",
            "outdoor_temperature",
            "indoor_temperature",
            "nibe_power",
            "quarter_of_day",
            "temperature_trend",
            "outdoor_temperature_trend",
            "optional_features_status",
            "heat_pump_model",
            "dhw_status",
            "dhw_recommendation",
        ]

        for sensor in SENSORS:
            if sensor.key in diagnostic_sensors:
                assert sensor.entity_category == EntityCategory.DIAGNOSTIC


class TestSensorEntityCreation:
    """Test sensor entity creation and initialization."""

    def test_sensor_entity_creation(self, mock_coordinator_with_data, mock_config_entry):
        """Test creating sensor entities."""
        for description in SENSORS:
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, description)

            assert sensor.coordinator == mock_coordinator_with_data
            assert sensor.entity_description == description
            assert sensor.unique_id == f"{mock_config_entry.entry_id}_{description.key}"

    def test_sensor_device_info(self, mock_coordinator_with_data, mock_config_entry):
        """Test sensor device info is correct."""
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, SENSORS[0])

        assert sensor.device_info is not None
        assert sensor.device_info["identifiers"] == {(DOMAIN, mock_config_entry.entry_id)}
        assert sensor.device_info["name"] == "EffektGuard"
        assert sensor.device_info["manufacturer"] == "EffektGuard"


class TestSensorValueFunctions:
    """Test sensor value functions with real data."""

    def test_current_offset_sensor(self, mock_coordinator_with_data, mock_config_entry):
        """Test current_offset sensor reads decision offset."""
        sensor_desc = next(s for s in SENSORS if s.key == "current_offset")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        assert sensor.native_value == 2.0

    def test_degree_minutes_sensor(self, mock_coordinator_with_data, mock_config_entry):
        """Test degree_minutes sensor reads NIBE data."""
        sensor_desc = next(s for s in SENSORS if s.key == "degree_minutes")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        assert sensor.native_value == -60

    def test_temperature_sensors(self, mock_coordinator_with_data, mock_config_entry):
        """Test all temperature sensors."""
        temp_tests = {
            "supply_temperature": 35.0,
            "outdoor_temperature": 5.0,
            "indoor_temperature": 21.5,
        }

        for sensor_key, expected_value in temp_tests.items():
            sensor_desc = next(s for s in SENSORS if s.key == sensor_key)
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.native_value == expected_value

    def test_power_sensors(self, mock_coordinator_with_data, mock_config_entry):
        """Test all power sensors."""
        power_tests = {
            "peak_today": 8.5,
            "peak_this_month": 12.3,
            "nibe_power": 6.9,
        }

        for sensor_key, expected_value in power_tests.items():
            sensor_desc = next(s for s in SENSORS if s.key == sensor_key)
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.native_value == expected_value

    def test_price_sensor(self, mock_coordinator_with_data, mock_config_entry):
        """Test current_price sensor."""
        sensor_desc = next(s for s in SENSORS if s.key == "current_price")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        assert sensor.native_value == 1.25

    def test_dhw_sensors(self, mock_coordinator_with_data, mock_config_entry):
        """Test DHW sensors."""
        dhw_tests = {
            "dhw_status": "heating",
            "dhw_recommendation": "Heat DHW now - optimal price window",
        }

        for sensor_key, expected_value in dhw_tests.items():
            sensor_desc = next(s for s in SENSORS if s.key == sensor_key)
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.native_value == expected_value

    def test_trend_sensors(self, mock_coordinator_with_data, mock_config_entry):
        """Test temperature trend sensors."""
        trend_tests = {
            "temperature_trend": 0.05,
            "outdoor_temperature_trend": -0.10,
        }

        for sensor_key, expected_value in trend_tests.items():
            sensor_desc = next(s for s in SENSORS if s.key == sensor_key)
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.native_value == expected_value

    def test_savings_sensor(self, mock_coordinator_with_data, mock_config_entry):
        """Test savings estimate sensor."""
        sensor_desc = next(s for s in SENSORS if s.key == "savings_estimate")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        assert sensor.native_value == 350.0

    def test_heat_pump_model_sensor(self, mock_coordinator_with_data, mock_config_entry):
        """Test heat pump model sensor."""
        sensor_desc = next(s for s in SENSORS if s.key == "heat_pump_model")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        assert sensor.native_value == "NIBE F2040-12"


class TestSensorNoneDataHandling:
    """Test sensors handle None/missing data gracefully."""

    def test_sensors_with_no_data(self, mock_hass, mock_config_entry):
        """Test sensors return None when coordinator has no data."""
        coordinator = Mock(spec=DataUpdateCoordinator)
        coordinator.hass = mock_hass
        coordinator.entry = mock_config_entry
        coordinator.data = None
        coordinator.peak_today = 0.0
        coordinator.peak_this_month = 0.0

        for sensor_desc in SENSORS:
            sensor = EffektGuardSensor(coordinator, mock_config_entry, sensor_desc)
            # Should not raise exception
            value = sensor.native_value
            # Most sensors return None with no data
            assert value is None or isinstance(value, (int, float, str))

    def test_sensors_with_missing_nibe_data(self, mock_coordinator_with_data, mock_config_entry):
        """Test sensors handle missing NIBE data."""
        mock_coordinator_with_data.data["nibe"] = None

        nibe_sensors = [
            "degree_minutes",
            "supply_temperature",
            "outdoor_temperature",
            "indoor_temperature",
        ]

        for sensor_key in nibe_sensors:
            sensor_desc = next(s for s in SENSORS if s.key == sensor_key)
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.native_value is None


class TestSensorExtraStateAttributes:
    """Test sensor extra state attributes are populated correctly."""

    def test_current_offset_attributes(self, mock_coordinator_with_data, mock_config_entry):
        """Test current_offset sensor has layer breakdown."""
        sensor_desc = next(s for s in SENSORS if s.key == "current_offset")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        attrs = sensor.extra_state_attributes
        assert "layer_votes" in attrs
        assert len(attrs["layer_votes"]) == 2
        assert attrs["layer_votes"][0]["offset"] == 1.0
        assert attrs["layer_votes"][0]["weight"] == 0.5
        assert attrs["layer_votes"][0]["reason"] == "Price optimization"

    def test_hour_classification_attributes(self, mock_coordinator_with_data, mock_config_entry):
        """Test hour_classification sensor has price data."""
        sensor_desc = next(s for s in SENSORS if s.key == "hour_classification")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        attrs = sensor.extra_state_attributes
        # Note: today_classifications removed - PriceData only has .today (QuarterPeriod list)
        # The sensor extracts min/max/average from period.price
        assert "today_min" in attrs
        assert "today_max" in attrs
        assert "today_average" in attrs
        assert attrs["today_min"] == 0.8
        assert attrs["today_max"] == 2.0

    def test_dhw_recommendation_attributes(self, mock_coordinator_with_data, mock_config_entry):
        """Test DHW recommendation sensor has complete planning data."""
        sensor_desc = next(s for s in SENSORS if s.key == "dhw_recommendation")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        attrs = sensor.extra_state_attributes

        # Core attributes
        assert attrs["should_heat"] is True
        assert attrs["priority_reason"] == "Low price period"
        assert attrs["current_temperature"] == 45.0
        assert attrs["target_temperature"] == 50.0

        # Thermal debt attributes
        assert attrs["thermal_debt"] == -80
        assert attrs["thermal_debt_threshold_block"] == -500
        assert attrs["thermal_debt_status"] == "normal"

        # Planning attributes
        assert attrs["space_heating_demand_kw"] == 2.5
        assert attrs["current_price_classification"] == "cheap"
        assert attrs["climate_zone"] == "temperate"

        # Note: optimal_windows removed in clean solution - DHW optimizer now returns
        # single recommended_start_time directly, not multiple windows

    def test_temperature_trend_attributes(self, mock_coordinator_with_data, mock_config_entry):
        """Test temperature trend sensor has trend details."""
        sensor_desc = next(s for s in SENSORS if s.key == "temperature_trend")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        attrs = sensor.extra_state_attributes
        assert attrs["trend_direction"] == "rising"
        assert attrs["confidence"] == 0.85
        assert attrs["samples"] == 10

    def test_outdoor_temperature_trend_attributes(
        self, mock_coordinator_with_data, mock_config_entry
    ):
        """Test outdoor temperature trend sensor has trend details."""
        sensor_desc = next(s for s in SENSORS if s.key == "outdoor_temperature_trend")
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)

        attrs = sensor.extra_state_attributes
        assert attrs["trend_direction"] == "falling"
        assert attrs["confidence"] == 0.90
        assert attrs["samples"] == 12
        assert attrs["temp_change_2h"] == -0.20
        assert attrs["current_outdoor_temp"] == 5.0


class TestClimateEntity:
    """Test climate entity functionality."""

    def test_climate_entity_creation(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity is created correctly."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert climate.coordinator == mock_coordinator_with_data
        assert climate.unique_id == f"{mock_config_entry.entry_id}_climate"
        assert climate.temperature_unit == UnitOfTemperature.CELSIUS

    def test_climate_supported_features(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity has correct supported features."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert climate.supported_features & ClimateEntityFeature.TARGET_TEMPERATURE
        assert climate.supported_features & ClimateEntityFeature.PRESET_MODE

    def test_climate_hvac_modes(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity has correct HVAC modes."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert HVACMode.HEAT in climate.hvac_modes
        assert HVACMode.OFF in climate.hvac_modes

    def test_climate_preset_modes(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity has correct preset modes."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert PRESET_NONE in climate.preset_modes
        assert PRESET_ECO in climate.preset_modes
        assert PRESET_COMFORT in climate.preset_modes

    def test_climate_current_temperature(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity reads current temperature from NIBE."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert climate.current_temperature == 21.5

    def test_climate_target_temperature(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity reads target temperature from config."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert climate.target_temperature == 21.0

    def test_climate_preset_mode_mapping(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity maps optimization mode to preset correctly."""
        # Test balanced mode
        mock_config_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_BALANCED
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)
        assert climate.preset_mode == PRESET_NONE

        # Test comfort mode
        mock_config_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_COMFORT
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)
        assert climate.preset_mode == PRESET_COMFORT

        # Test savings mode
        mock_config_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_SAVINGS
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)
        assert climate.preset_mode == PRESET_ECO

    def test_climate_device_info(self, mock_coordinator_with_data, mock_config_entry):
        """Test climate entity has correct device info."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        assert climate.device_info is not None
        assert climate.device_info["identifiers"] == {(DOMAIN, mock_config_entry.entry_id)}
        assert climate.device_info["name"] == "EffektGuard"
        assert climate.device_info["manufacturer"] == "EffektGuard"
        assert climate.device_info["model"] == "Heat Pump Optimizer"


class TestSwitchEntities:
    """Test switch entity definitions and creation."""

    def test_all_switches_have_required_fields(self):
        """Verify all switch descriptions have required fields."""
        for switch in SWITCHES:
            assert switch.key is not None, f"Switch missing key"
            assert switch.name is not None, f"Switch {switch.key} missing name"
            assert switch.icon is not None, f"Switch {switch.key} missing icon"
            assert switch.config_key is not None, f"Switch {switch.key} missing config_key"

    def test_switch_count_matches_expected(self):
        """Verify we have all expected switches."""
        # EffektGuard has 5 feature toggle switches
        assert len(SWITCHES) == 5, f"Expected 5 switches, found {len(SWITCHES)}"

    def test_all_switch_keys_are_unique(self):
        """Verify no duplicate switch keys."""
        keys = [switch.key for switch in SWITCHES]
        assert len(keys) == len(set(keys)), "Duplicate switch keys found"

    def test_switch_config_keys_match_constants(self):
        """Verify switch config keys match defined constants."""
        expected_config_keys = {
            CONF_ENABLE_OPTIMIZATION,
            CONF_ENABLE_PRICE_OPTIMIZATION,
            CONF_ENABLE_PEAK_PROTECTION,
            CONF_ENABLE_WEATHER_PREDICTION,
            CONF_ENABLE_HOT_WATER_OPTIMIZATION,
        }

        actual_config_keys = {switch.config_key for switch in SWITCHES}
        assert actual_config_keys == expected_config_keys

    def test_switch_entity_creation(self, mock_coordinator_with_data, mock_config_entry):
        """Test creating switch entities."""
        for description in SWITCHES:
            switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, description)

            assert switch.coordinator == mock_coordinator_with_data
            assert switch.entity_description == description
            assert switch.unique_id == f"{mock_config_entry.entry_id}_{description.key}"

    def test_switch_reads_state_from_options(self, mock_coordinator_with_data, mock_config_entry):
        """Test switches read state from config entry options."""
        # Enable optimization switch
        switch_desc = next(s for s in SWITCHES if s.key == "enable_optimization")
        switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, switch_desc)
        assert switch.is_on is True

        # DHW optimization switch (disabled in fixture)
        switch_desc = next(s for s in SWITCHES if s.key == "hot_water_optimization")
        switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, switch_desc)
        assert switch.is_on is False


class TestEntityIntegration:
    """Integration tests for entity interactions."""

    def test_all_entities_share_same_device(self, mock_coordinator_with_data, mock_config_entry):
        """Verify all entities belong to same device."""
        # Create one of each entity type
        sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, SENSORS[0])
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)
        switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, SWITCHES[0])

        # All should have same device identifier
        device_id = (DOMAIN, mock_config_entry.entry_id)
        assert sensor.device_info["identifiers"] == {device_id}
        assert climate.device_info["identifiers"] == {device_id}
        assert switch.device_info["identifiers"] == {device_id}

    def test_unique_ids_are_unique_across_entities(
        self, mock_coordinator_with_data, mock_config_entry
    ):
        """Verify no duplicate unique IDs across all entities."""
        unique_ids = set()

        # Add sensor unique IDs
        for sensor_desc in SENSORS:
            sensor = EffektGuardSensor(mock_coordinator_with_data, mock_config_entry, sensor_desc)
            assert sensor.unique_id not in unique_ids, f"Duplicate unique_id: {sensor.unique_id}"
            unique_ids.add(sensor.unique_id)

        # Add climate unique ID
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)
        assert climate.unique_id not in unique_ids, f"Duplicate unique_id: {climate.unique_id}"
        unique_ids.add(climate.unique_id)

        # Add switch unique IDs
        for switch_desc in SWITCHES:
            switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, switch_desc)
            assert switch.unique_id not in unique_ids, f"Duplicate unique_id: {switch.unique_id}"
            unique_ids.add(switch.unique_id)

        # Total entities: 22 sensors + 1 climate + 5 switches = 28 entities
        assert len(unique_ids) == 28


class TestConfigReloadIntegration:
    """Test entities properly respond to config reload."""

    @pytest.mark.asyncio
    async def test_climate_target_temp_updates_from_options(
        self, mock_coordinator_with_data, mock_config_entry
    ):
        """Test climate entity reads updated target temperature."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        # Initial value
        assert climate.target_temperature == 21.0

        # Update options
        mock_config_entry.options[CONF_TARGET_INDOOR_TEMP] = 22.5

        # Should read new value
        assert climate.target_temperature == 22.5

    @pytest.mark.asyncio
    async def test_climate_preset_updates_from_options(
        self, mock_coordinator_with_data, mock_config_entry
    ):
        """Test climate preset mode updates with optimization mode."""
        climate = EffektGuardClimate(mock_coordinator_with_data, mock_config_entry)

        # Initial value (balanced = PRESET_NONE)
        assert climate.preset_mode == PRESET_NONE

        # Update to comfort mode
        mock_config_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_COMFORT
        assert climate.preset_mode == PRESET_COMFORT

        # Update to savings mode
        mock_config_entry.options[CONF_OPTIMIZATION_MODE] = OPTIMIZATION_MODE_SAVINGS
        assert climate.preset_mode == PRESET_ECO

    @pytest.mark.asyncio
    async def test_switch_state_updates_from_options(
        self, mock_coordinator_with_data, mock_config_entry
    ):
        """Test switch entities read updated state from data (not options)."""
        switch_desc = next(s for s in SWITCHES if s.key == "enable_optimization")
        switch = EffektGuardSwitch(mock_coordinator_with_data, mock_config_entry, switch_desc)

        # Initial value - switches read from entry.data
        mock_config_entry.data[CONF_ENABLE_OPTIMIZATION] = True
        assert switch.is_on is True

        # Update data (switches use data, not options)
        mock_config_entry.data[CONF_ENABLE_OPTIMIZATION] = False

        # Should read new value
        assert switch.is_on is False
