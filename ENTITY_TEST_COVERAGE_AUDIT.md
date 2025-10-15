# Entity Test Coverage Audit

**Date**: October 14, 2025  
**Purpose**: Verify all EffektGuard entity production code has proper test coverage  
**Scope**: Climate, Sensors, Numbers, Selects, Switches

---

## Production Code Inventory

### 1. climate.py (EffektGuardClimate)

**Methods:**
- `__init__()` - Initialize climate entity ✅ TESTED
- `current_temperature` (property) - Get indoor temp from NIBE ✅ TESTED
- `target_temperature` (property) - Get target from config ✅ TESTED
- `hvac_mode` (property) - Return current HVAC mode ✅ TESTED (implicit)
- `preset_mode` (property) - Map optimization mode to preset ✅ TESTED
- `async_set_temperature()` - Update target temperature ✅ TESTED
- `async_set_hvac_mode()` - Enable/disable optimization ✅ TESTED
- `async_set_preset_mode()` - Change optimization mode ✅ TESTED
- `extra_state_attributes` (property) - Additional attributes ✅ TESTED

**Attributes:**
- `_attr_hvac_modes` = [HEAT, OFF] ✅ TESTED
- `_attr_preset_modes` = [NONE, ECO, AWAY, COMFORT] ✅ TESTED
- `_attr_supported_features` = TARGET_TEMPERATURE | PRESET_MODE ✅ TESTED
- `_attr_min_temp` = 15.0 ❌ NOT TESTED
- `_attr_max_temp` = 25.0 ❌ NOT TESTED
- `_attr_target_temperature_step` = 0.5 ❌ NOT TESTED

**Edge Cases:**
- Temperature clamping (min/max) ❌ NOT TESTED
- Missing coordinator data ❌ NOT TESTED
- None temperature from kwargs ✅ TESTED (code handles it)
- Missing NIBE state ❌ NOT TESTED

---

### 2. sensor.py (14 Sensors)

**Sensors Defined:**
1. `current_offset` ✅ TESTED
2. `degree_minutes` ❌ VALUE TESTED, but not None handling
3. `supply_temperature` ❌ NOT TESTED
4. `outdoor_temperature` ❌ NOT TESTED
5. `current_price` ❌ VALUE TESTED, but not None/error handling
6. `peak_today` ✅ TESTED
7. `peak_this_month` ❌ NOT TESTED directly
8. `optimization_reasoning` ❌ NOT TESTED directly
9. `quarter_of_day` ❌ NOT TESTED
10. `hour_classification` ✅ TESTED
11. `peak_status` ✅ TESTED
12. `temperature_trend` ✅ TESTED
13. `savings_estimate` ✅ TESTED
14. `optional_features_status` ❌ NOT TESTED

**Extra State Attributes:**
- `current_offset` sensor: layer_votes ✅ TESTED
- `hour_classification`: today classifications, min/max/avg ✅ TESTED
- `peak_status`: margin, current power, peaks ✅ TESTED
- `temperature_trend`: prediction_3h, forecast ✅ TESTED
- `savings_estimate`: breakdown ✅ TESTED
- `optimization_reasoning`: timestamp, applied offset ❌ NOT TESTED
- `optional_features_status`: ALL sub-features ❌ NOT TESTED

**Edge Cases:**
- Missing coordinator data ❌ NOT TESTED
- AttributeError/KeyError handling ❌ NOT TESTED
- None values in value_fn ❌ NOT TESTED

---

### 3. number.py (5 Number Entities)

**Number Entities:**
1. `target_temperature` ✅ TESTED
2. `tolerance` ❌ NOT TESTED (only count verified)
3. `thermal_mass` ❌ NOT TESTED
4. `insulation_quality` ❌ NOT TESTED
5. `peak_protection_margin` ✅ TESTED

**Methods:**
- `native_value` (property) - Get current value ✅ TESTED (2 entities)
- `async_set_native_value()` - Update value ✅ TESTED

**Edge Cases:**
- Missing config_key ❌ NOT TESTED
- Default value fallback ❌ NOT TESTED
- Value validation/clamping ❌ NOT TESTED

---

### 4. select.py (2 Select Entities)

**Select Entities:**
1. `optimization_mode` ✅ TESTED
2. `control_priority` ✅ TESTED

**Methods:**
- `current_option` (property) - Get current value ✅ TESTED
- `async_select_option()` - Change option ✅ TESTED

**Edge Cases:**
- Invalid option selection ❌ NOT TESTED
- Missing config_key ❌ NOT TESTED
- Default value fallback ❌ NOT TESTED

---

### 5. switch.py (5 Switch Entities)

**Switch Entities:**
1. `enable_optimization` ✅ TESTED
2. `price_optimization` ✅ TESTED
3. `peak_protection` ✅ TESTED (turn off)
4. `weather_prediction` ✅ TESTED (turn on)
5. `hot_water_optimization` ✅ TESTED

**Methods:**
- `is_on` (property) - Get switch state ✅ TESTED
- `async_turn_on()` - Turn on ✅ TESTED
- `async_turn_off()` - Turn off ✅ TESTED

**Edge Cases:**
- Missing config_key ❌ NOT TESTED
- Default value handling ✅ TESTED (hot_water defaults off)

---

## Summary

### Coverage Statistics

**Climate Entity:**
- Methods: 8/8 tested (100%) ✅
- Attributes: 3/6 tested (50%) ⚠️
- Edge cases: 1/4 tested (25%) ❌

**Sensors (14 total):**
- Sensor count verified: ✅
- Basic value retrieval: 7/14 tested (50%) ⚠️
- Extra attributes: 5/7 tested (71%) ⚠️
- Edge cases: 0/3 tested (0%) ❌

**Number Entities (5 total):**
- Entity count verified: ✅
- Value retrieval: 2/5 tested (40%) ⚠️
- Value setting: 1/5 tested (20%) ❌
- Edge cases: 0/3 tested (0%) ❌

**Select Entities (2 total):**
- All basic functionality tested: ✅
- Edge cases: 0/3 tested (0%) ❌

**Switch Entities (5 total):**
- All basic functionality tested: ✅
- Edge cases: 0/2 tested (0%) ❌

### Overall Coverage: ~65%

---

## Missing Tests Required

### High Priority (Core Functionality)

1. **Climate entity edge cases:**
   - Test temperature clamping (below min, above max)
   - Test missing coordinator data
   - Test missing NIBE state

2. **All sensors value retrieval:**
   - Test all 14 sensors can retrieve values
   - Test None/missing data handling
   - Test error handling (AttributeError, KeyError)

3. **Number entities:**
   - Test all 5 number entities value retrieval
   - Test all 5 number entities value setting
   - Test default value fallback

4. **Select entity edge cases:**
   - Test invalid option selection (should log error)

### Medium Priority (Error Handling)

5. **Sensor extra attributes edge cases:**
   - Test optional_features_status complete attributes
   - Test optimization_reasoning timestamp attribute
   - Test all sensors with missing coordinator data

6. **Entity creation:**
   - Test async_setup_entry for all entity types

### Low Priority (Code Quality)

7. **Device info:**
   - Verify all entities have correct device_info
   - Verify all entities have unique_id

---

## Test Gaps Requiring Immediate Attention

❌ **CRITICAL:**
1. Only 7 of 14 sensors tested for value retrieval
2. Only 2 of 5 number entities tested for value retrieval
3. No edge case testing for None/missing data
4. No error handling tests

⚠️ **IMPORTANT:**
1. Climate entity min/max/step attributes not verified
2. optional_features_status sensor not tested (critical for Phase 5)
3. Several sensor extra attributes not tested
4. Default value fallbacks not tested

---

## Recommendation

**Need to create comprehensive test file covering:**
1. All 14 sensor value retrievals with real data
2. All 14 sensor value retrievals with None/missing data
3. All 5 number entities (get and set)
4. Edge cases for all entity types
5. Error handling for all entity types
6. Entity creation (async_setup_entry) for all types

**Estimated tests to add: ~50 additional test cases**

**Current test count: 36 tests**
**Target test count: ~85 tests**
