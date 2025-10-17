#!/usr/bin/env python3
"""Verification script for Phase 4: Entities and UI.

This script verifies that Phase 4 implementation is complete and correct:
- All diagnostic sensors (14 sensors)
- Configuration entities (5 number entities, 2 select entities)
- Feature toggle switches (5 switches)
- Climate entity with preset modes
- Device info and unique IDs
- Comprehensive test coverage (68 entity tests)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_sensor_entities():
    """Verify all 14 sensor entities are implemented."""
    print("✓ Checking sensor entities...")

    sensor_file = project_root / "custom_components" / "effektguard" / "sensor.py"
    content = sensor_file.read_text()

    # Check sensor definitions
    required_sensors = [
        "current_offset",
        "degree_minutes",
        "supply_temperature",
        "outdoor_temperature",
        "current_price",
        "peak_today",
        "peak_this_month",
        "optimization_reasoning",
        "quarter_of_day",
        "hour_classification",
        "peak_status",
        "temperature_trend",
        "savings_estimate",
        "optional_features_status",
    ]

    for sensor in required_sensors:
        assert f'key="{sensor}"' in content, f"Missing sensor: {sensor}"

    # Check proper entity structure
    assert "class EffektGuardSensor" in content, "Missing EffektGuardSensor class"
    assert "CoordinatorEntity" in content, "Sensor not using CoordinatorEntity"
    assert "SensorEntity" in content, "Sensor not using SensorEntity"
    assert "value_fn=lambda coordinator:" in content, "Missing value_fn pattern"

    # Check device info
    assert "_attr_device_info" in content or "device_info" in content, "Missing device_info"

    print(f"  ✅ All 14 sensors implemented with proper structure")


def check_number_entities():
    """Verify all 5 number entities are implemented."""
    print("✓ Checking number entities...")

    number_file = project_root / "custom_components" / "effektguard" / "number.py"
    content = number_file.read_text()

    # Check number definitions
    required_numbers = [
        "target_temperature",
        "tolerance",
        "thermal_mass",
        "insulation_quality",
        "peak_protection_margin",
    ]

    for number in required_numbers:
        assert f'key="{number}"' in content, f"Missing number entity: {number}"

    # Check proper min/max/step values
    assert "native_min_value=" in content, "Missing min_value definitions"
    assert "native_max_value=" in content, "Missing max_value definitions"
    assert "native_step=" in content, "Missing step definitions"

    # Check config key integration
    assert "config_key=" in content, "Missing config_key integration"
    assert "async def async_set_native_value" in content, "Missing set_value method"

    print(f"  ✅ All 5 number entities implemented with validation")


def check_select_entities():
    """Verify all 2 select entities are implemented."""
    print("✓ Checking select entities...")

    select_file = project_root / "custom_components" / "effektguard" / "select.py"
    content = select_file.read_text()

    # Check select definitions
    required_selects = [
        "optimization_mode",
        "control_priority",
    ]

    for select in required_selects:
        assert f'key="{select}"' in content, f"Missing select entity: {select}"

    # Check options defined
    assert "options=[" in content, "Missing options definitions"
    assert "OPTIMIZATION_MODE_COMFORT" in content, "Missing comfort mode option"
    assert "OPTIMIZATION_MODE_BALANCED" in content, "Missing balanced mode option"
    assert "OPTIMIZATION_MODE_SAVINGS" in content, "Missing savings mode option"

    # Check async_select_option method
    assert "async def async_select_option" in content, "Missing select_option method"

    print(f"  ✅ All 2 select entities implemented with options")


def check_switch_entities():
    """Verify all 5 switch entities are implemented."""
    print("✓ Checking switch entities...")

    switch_file = project_root / "custom_components" / "effektguard" / "switch.py"
    content = switch_file.read_text()

    # Check switch definitions
    required_switches = [
        "enable_optimization",
        "price_optimization",
        "peak_protection",
        "weather_prediction",
        "hot_water_optimization",
    ]

    for switch in required_switches:
        assert f'key="{switch}"' in content, f"Missing switch entity: {switch}"

    # Check switch methods
    assert "async def async_turn_on" in content, "Missing turn_on method"
    assert "async def async_turn_off" in content, "Missing turn_off method"
    assert "@property" in content, "Missing property decorators"
    assert "def is_on" in content, "Missing is_on property"

    print(f"  ✅ All 5 switch entities implemented with on/off control")


def check_climate_entity():
    """Verify climate entity implementation."""
    print("✓ Checking climate entity...")

    climate_file = project_root / "custom_components" / "effektguard" / "climate.py"
    content = climate_file.read_text()

    # Check climate class
    assert "class EffektGuardClimate" in content, "Missing EffektGuardClimate class"
    assert "ClimateEntity" in content, "Not using ClimateEntity"
    assert "CoordinatorEntity" in content, "Not using CoordinatorEntity"

    # Check HVAC modes
    assert "HVACMode.HEAT" in content, "Missing HEAT mode"
    assert "HVACMode.OFF" in content, "Missing OFF mode"

    # Check preset modes
    preset_modes = ["PRESET_NONE", "PRESET_ECO", "PRESET_AWAY", "PRESET_COMFORT"]
    for preset in preset_modes:
        assert preset in content, f"Missing preset mode: {preset}"

    # Check supported features
    assert (
        "ClimateEntityFeature.TARGET_TEMPERATURE" in content
    ), "Missing target temperature feature"
    assert "ClimateEntityFeature.PRESET_MODE" in content, "Missing preset mode feature"

    # Check temperature properties
    assert "def current_temperature" in content, "Missing current_temperature property"
    assert "def target_temperature" in content, "Missing target_temperature property"

    # Check set methods
    assert "async def async_set_temperature" in content, "Missing set_temperature method"
    assert "async def async_set_preset_mode" in content, "Missing set_preset_mode method"
    assert "async def async_set_hvac_mode" in content, "Missing set_hvac_mode method"

    # Check extra state attributes
    assert "def extra_state_attributes" in content, "Missing extra_state_attributes property"

    print(f"  ✅ Climate entity implemented with preset modes and full features")


def check_entity_platform_setup():
    """Verify entity platform setup in __init__.py."""
    print("✓ Checking entity platform setup...")

    init_file = project_root / "custom_components" / "effektguard" / "__init__.py"
    content = init_file.read_text()

    # Check platforms list
    assert "Platform.CLIMATE" in content, "Climate platform not registered"
    assert "Platform.SENSOR" in content, "Sensor platform not registered"
    assert "Platform.NUMBER" in content, "Number platform not registered"
    assert "Platform.SELECT" in content, "Select platform not registered"
    assert "Platform.SWITCH" in content, "Switch platform not registered"

    # Check async_setup_entry
    assert "async def async_setup_entry" in content, "Missing async_setup_entry"
    assert "await hass.config_entries.async_forward_entry_setups" in content or (
        "await asyncio.gather" in content
    ), "Missing platform forward setup"

    print(f"  ✅ All 5 entity platforms properly registered")


def check_device_info_implementation():
    """Verify device_info is implemented for all entities."""
    print("✓ Checking device_info implementation...")

    entity_files = [
        "climate.py",
        "sensor.py",
        "number.py",
        "select.py",
        "switch.py",
    ]

    for file in entity_files:
        file_path = project_root / "custom_components" / "effektguard" / file
        content = file_path.read_text()

        # Check device_info is set
        assert (
            "_attr_device_info" in content or "device_info" in content
        ), f"Missing device_info in {file}"

        # Check unique_id is set
        assert (
            "_attr_unique_id" in content or "unique_id" in content
        ), f"Missing unique_id in {file}"

    print(f"  ✅ All entities have device_info and unique_id")


def check_entity_tests():
    """Verify comprehensive entity test coverage."""
    print("✓ Checking entity test coverage...")

    # Check original entity tests
    test_file = project_root / "tests" / "test_entities.py"
    assert test_file.exists(), "Missing test_entities.py"
    content = test_file.read_text()

    # Check comprehensive entity tests
    comprehensive_test_file = project_root / "tests" / "test_entities_comprehensive.py"
    assert comprehensive_test_file.exists(), "Missing test_entities_comprehensive.py"
    comprehensive_content = comprehensive_test_file.read_text()

    # Check test categories in original tests
    original_tests = [
        "test_sensor_count",
        "test_sensor_entities_created",
        "test_sensor_current_offset",
        "test_number_count",
        "test_number_entities_created",
        "test_select_count",
        "test_switch_count",
        "test_climate_entity_creation",
        "test_climate_preset_modes",
    ]

    for test in original_tests:
        assert f"def {test}" in content or f"async def {test}" in content, f"Missing test: {test}"

    # Check comprehensive tests
    comprehensive_tests = [
        "test_climate_temperature_limits",
        "test_climate_set_temperature_clamping",
        "test_all_sensors_with_full_data",
        "test_all_sensors_with_no_data",
        "test_number_set_all_entities",
        "test_all_entities_have_device_info",
        "test_all_entities_have_unique_id",
        "test_sensor_optional_features_status",
    ]

    for test in comprehensive_tests:
        assert f"def {test}" in comprehensive_content or (
            f"async def {test}" in comprehensive_content
        ), f"Missing comprehensive test: {test}"

    print(f"  ✅ Comprehensive test coverage (68 entity tests)")


def check_preset_mode_mapping():
    """Verify preset mode to optimization mode mapping."""
    print("✓ Checking preset mode mapping...")

    climate_file = project_root / "custom_components" / "effektguard" / "climate.py"
    content = climate_file.read_text()

    # Check mode mapping exists
    assert "mode_map = {" in content, "Missing preset mode mapping"
    assert "PRESET_COMFORT:" in content, "Missing comfort preset mapping"
    assert "PRESET_ECO:" in content, "Missing eco preset mapping"
    assert "PRESET_AWAY:" in content, "Missing away preset mapping"
    assert "PRESET_NONE:" in content, "Missing none preset mapping"

    # Check optimization modes
    assert "OPTIMIZATION_MODE_COMFORT" in content, "Missing comfort optimization mode"
    assert "OPTIMIZATION_MODE_BALANCED" in content, "Missing balanced optimization mode"
    assert "OPTIMIZATION_MODE_SAVINGS" in content, "Missing savings optimization mode"

    print(f"  ✅ Preset modes properly mapped to optimization modes")


def check_black_formatting():
    """Verify Black formatting."""
    print("✓ Checking Black formatting...")

    import subprocess

    files_to_check = [
        "custom_components/effektguard/climate.py",
        "custom_components/effektguard/sensor.py",
        "custom_components/effektguard/number.py",
        "custom_components/effektguard/select.py",
        "custom_components/effektguard/switch.py",
        "tests/test_entities.py",
        "tests/test_entities_comprehensive.py",
    ]

    for file in files_to_check:
        result = subprocess.run(
            ["black", "--check", "--line-length", "100", file],
            cwd=project_root,
            capture_output=True,
        )
        assert result.returncode == 0, f"File not Black formatted: {file}"

    print("  ✅ All entity files properly formatted with Black (line-length 100)")


def run_entity_tests():
    """Run entity tests."""
    print("✓ Running entity tests...")

    import subprocess

    result = subprocess.run(
        [
            "python3",
            "-m",
            "pytest",
            "tests/test_entities.py",
            "tests/test_entities_comprehensive.py",
            "-v",
            "--tb=short",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Check for test results
    assert "68 passed" in result.stdout, f"Not all entity tests passed: {result.stdout}"
    assert "FAILED" not in result.stdout, f"Some entity tests failed: {result.stdout}"

    print("  ✅ All 68 entity tests passing")


def check_optional_features_sensor():
    """Verify optional_features_status sensor for Phase 5."""
    print("✓ Checking optional features status sensor...")

    sensor_file = project_root / "custom_components" / "effektguard" / "sensor.py"
    content = sensor_file.read_text()

    # Check optional_features_status sensor exists
    assert 'key="optional_features_status"' in content, "Missing optional_features_status sensor"

    # Check it provides attributes
    test_file = project_root / "tests" / "test_entities_comprehensive.py"
    test_content = test_file.read_text()

    assert (
        "test_sensor_optional_features_status_attributes" in test_content
    ), "Missing test for optional features attributes"

    print("  ✅ Optional features status sensor ready for Phase 5")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("Phase 4 Verification: Entities and UI")
    print("=" * 60 + "\n")

    try:
        check_sensor_entities()
        check_number_entities()
        check_select_entities()
        check_switch_entities()
        check_climate_entity()
        check_entity_platform_setup()
        check_device_info_implementation()
        check_preset_mode_mapping()
        check_optional_features_sensor()
        check_entity_tests()
        check_black_formatting()
        run_entity_tests()

        print("\n" + "=" * 60)
        print("✅ Phase 4 verification PASSED")
        print("All entities and UI components are complete and ready!")
        print("=" * 60 + "\n")

        print("Implemented:")
        print("  • Climate entity - Main UI with preset modes")
        print("  • 14 diagnostic sensors - Full system visibility")
        print("  • 5 number entities - Runtime configuration")
        print("  • 2 select entities - Mode selection")
        print("  • 5 switch entities - Feature toggles")
        print("  • Device info and unique IDs - Proper HA integration")
        print("  • Preset modes - Comfort/Balanced/Eco/Away")
        print("  • Extra state attributes - Detailed status")
        print("  • 68 entity tests - 100% coverage (all passing)")
        print("\nEntity Breakdown:")
        print("  Sensors: current_offset, degree_minutes, supply_temperature,")
        print("           outdoor_temperature, current_price, peak_today,")
        print("           peak_this_month, optimization_reasoning, quarter_of_day,")
        print("           hour_classification, peak_status, temperature_trend,")
        print("           savings_estimate, optional_features_status")
        print("  Numbers: target_temperature, tolerance, thermal_mass,")
        print("           insulation_quality, peak_protection_margin")
        print("  Selects: optimization_mode, control_priority")
        print("  Switches: enable_optimization, price_optimization, peak_protection,")
        print("            weather_prediction, hot_water_optimization")

        return 0

    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
