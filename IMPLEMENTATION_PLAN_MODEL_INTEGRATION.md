# Model Profile Integration Implementation Plan

**Date**: October 15, 2025  
**Status**: Ready to implement  
**Estimated Time**: 2-3 hours

---

## Problem Statement

Model profiles exist and are tested (F730/F750/F2040/S1155) but are **NOT integrated** into production code.

**Current State**:
- ✅ Model profiles created in `custom_components/effektguard/models/nibe/`
- ✅ Tests passing (52/52)
- ✅ Documentation complete
- ❌ Config flow doesn't offer model selection
- ❌ Coordinator doesn't load model profile
- ❌ Decision engine doesn't use model limits
- ❌ No power validation

**Impact**: Users can't benefit from model-specific optimization. System treats all heat pumps the same.

---

## Implementation Steps

### Step 1: Add Configuration Constant ⏱️ 5 minutes

**File**: `custom_components/effektguard/const.py`

**Action**: Add new configuration key and default

```python
# After line 26 (after CONF_INSULATION_QUALITY)
CONF_HEAT_PUMP_MODEL: Final = "heat_pump_model"

# After line 37 (after DEFAULT_WEATHER_COMPENSATION_WEIGHT)
DEFAULT_HEAT_PUMP_MODEL: Final = "nibe_f750"  # Most common model
```

**Verification**:
```bash
grep "CONF_HEAT_PUMP_MODEL" custom_components/effektguard/const.py
```

---

### Step 2: Update Config Flow ⏱️ 30 minutes

**File**: `custom_components/effektguard/config_flow.py`

**Action 1**: Import new constant
```python
# In imports section (around line 13-30), add:
from .const import (
    # ... existing imports ...
    CONF_HEAT_PUMP_MODEL,  # ADD THIS
    DEFAULT_HEAT_PUMP_MODEL,  # ADD THIS
)
```

**Action 2**: Add model selection step after GE-Spot configuration

Add new method after `async_step_gespot()` (around line 100):

```python
async def async_step_model(self, user_input: dict[str, Any] | None = None) -> FlowResult:
    """Handle heat pump model selection."""
    errors = {}

    if user_input is not None:
        self._data[CONF_HEAT_PUMP_MODEL] = user_input[CONF_HEAT_PUMP_MODEL]
        return await self.async_step_weather()  # Continue to existing weather step

    return self.async_show_form(
        step_id="model",
        data_schema=vol.Schema(
            {
                vol.Required(
                    CONF_HEAT_PUMP_MODEL,
                    default=DEFAULT_HEAT_PUMP_MODEL,
                ): vol.In({
                    "nibe_f730": "NIBE F730 (6kW ASHP)",
                    "nibe_f750": "NIBE F750 (8kW ASHP - Most Common)",
                    "nibe_f2040": "NIBE F2040 (12-16kW ASHP)",
                    "nibe_s1155": "NIBE S1155 (GSHP)",
                }),
            }
        ),
        errors=errors,
        description_placeholders={
            "model_info": "Select your heat pump model for optimized control"
        },
    )
```

**Action 3**: Update flow routing

In `async_step_gespot()`, change the return statement to:
```python
# OLD: return await self.async_step_weather()
# NEW:
return await self.async_step_model()
```

**Verification**:
```bash
grep -n "async_step_model" custom_components/effektguard/config_flow.py
grep -n "CONF_HEAT_PUMP_MODEL" custom_components/effektguard/config_flow.py
```

---

### Step 3: Update Coordinator to Load Model ⏱️ 20 minutes

**File**: `custom_components/effektguard/coordinator.py`

**Action 1**: Import model profiles
```python
# Add to imports at top of file
from .models.nibe import (
    NibeF730Profile,
    NibeF750Profile,
    NibeF2040Profile,
    NibeS1155Profile,
)
```

**Action 2**: Add model registry dictionary (after imports, before class):
```python
# Model registry for quick lookup
HEAT_PUMP_MODELS = {
    "nibe_f730": NibeF730Profile,
    "nibe_f750": NibeF750Profile,
    "nibe_f2040": NibeF2040Profile,
    "nibe_s1155": NibeS1155Profile,
}
```

**Action 3**: Load model in coordinator `__init__`

Find the `__init__` method and add:
```python
def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Initialize coordinator."""
    # ... existing code ...
    
    # Load heat pump model profile
    model_key = entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL)
    model_class = HEAT_PUMP_MODELS.get(model_key, NibeF750Profile)
    self.heat_pump_model = model_class()
    
    _LOGGER.info(
        "Loaded heat pump model: %s (%s)",
        self.heat_pump_model.model_name,
        self.heat_pump_model.model_type,
    )
```

**Action 4**: Make model available to decision engine

Add property to coordinator class:
```python
@property
def model_profile(self):
    """Get heat pump model profile."""
    return self.heat_pump_model
```

**Verification**:
```bash
grep -n "heat_pump_model" custom_components/effektguard/coordinator.py
grep -n "HEAT_PUMP_MODELS" custom_components/effektguard/coordinator.py
python3 -c "from custom_components.effektguard.models.nibe import NibeF750Profile; print('✓ Import works')"
```

---

### Step 4: Use Model in Decision Engine ⏱️ 30 minutes

**File**: `custom_components/effektguard/optimization/decision_engine.py`

**Action 1**: Accept model profile in constructor

Update `__init__` method:
```python
def __init__(
    self,
    # ... existing parameters ...
    heat_pump_model=None,  # ADD THIS
):
    """Initialize decision engine."""
    # ... existing code ...
    self.heat_pump_model = heat_pump_model
```

**Action 2**: Add power validation method

Add new method to class:
```python
def _validate_power_consumption(
    self,
    current_power_kw: float,
    outdoor_temp: float,
) -> dict[str, Any]:
    """Validate current power against model expectations.
    
    Args:
        current_power_kw: Current electrical consumption (kW)
        outdoor_temp: Outdoor temperature (°C)
        
    Returns:
        Dict with validation status and warnings
    """
    if not self.heat_pump_model:
        return {"valid": True, "warning": None}
    
    min_power, max_power = self.heat_pump_model.typical_electrical_range_kw
    
    # Allow 20% margin for startup/defrost
    max_with_margin = max_power * 1.2
    
    if current_power_kw > max_with_margin:
        return {
            "valid": False,
            "warning": f"Power {current_power_kw:.1f}kW exceeds {self.heat_pump_model.model_name} max {max_power:.1f}kW (auxiliary heating active?)",
            "severity": "warning",
        }
    
    # Check if unusually low (possible sensor issue)
    if current_power_kw < min_power * 0.5 and outdoor_temp < 0:
        return {
            "valid": True,
            "warning": f"Power {current_power_kw:.1f}kW below expected for {outdoor_temp:.1f}°C",
            "severity": "info",
        }
    
    return {"valid": True, "warning": None}
```

**Action 3**: Use validation in decision calculation

In `calculate_decision()` method, add power validation:
```python
# After reading current power sensor (if available)
if power_kw:
    validation = self._validate_power_consumption(power_kw, outdoor_temp)
    if validation["warning"]:
        _LOGGER.log(
            logging.WARNING if validation["severity"] == "warning" else logging.INFO,
            validation["warning"],
        )
```

**Action 4**: Use model limits for offset bounds

In offset calculation, use model-specific limits:
```python
# Use model-specific max offset if available
max_offset = MAX_OFFSET  # Default from const.py
if self.heat_pump_model and hasattr(self.heat_pump_model, 'max_safe_offset'):
    max_offset = self.heat_pump_model.max_safe_offset

# Clamp offset
final_offset = max(MIN_OFFSET, min(final_offset, max_offset))
```

**Verification**:
```bash
grep -n "heat_pump_model" custom_components/effektguard/optimization/decision_engine.py
grep -n "_validate_power_consumption" custom_components/effektguard/optimization/decision_engine.py
```

---

### Step 5: Pass Model from Coordinator to Decision Engine ⏱️ 15 minutes

**File**: `custom_components/effektguard/coordinator.py`

**Action**: Update decision engine initialization

Find where `DecisionEngine` is created and add model parameter:
```python
# OLD:
self.decision_engine = DecisionEngine(
    # ... existing parameters ...
)

# NEW:
self.decision_engine = DecisionEngine(
    # ... existing parameters ...
    heat_pump_model=self.heat_pump_model,  # ADD THIS
)
```

**Verification**:
```bash
grep -n "DecisionEngine(" custom_components/effektguard/coordinator.py
```

---

### Step 6: Add Model Info Sensor (Optional) ⏱️ 20 minutes

**File**: `custom_components/effektguard/sensor.py`

**Action**: Add new sensor entity for model information

```python
class EffektGuardModelInfoSensor(CoordinatorEntity, SensorEntity):
    """Sensor showing heat pump model information."""

    _attr_icon = "mdi:heat-pump"
    
    def __init__(self, coordinator, entry):
        """Initialize sensor."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_model_info"
        self._attr_name = "Heat Pump Model"
        
    @property
    def native_value(self):
        """Return the model name."""
        if hasattr(self.coordinator, 'heat_pump_model'):
            return self.coordinator.heat_pump_model.model_name
        return "Unknown"
    
    @property
    def extra_state_attributes(self):
        """Return model attributes."""
        if not hasattr(self.coordinator, 'heat_pump_model'):
            return {}
        
        model = self.coordinator.heat_pump_model
        return {
            "model_type": model.model_type,
            "electrical_range_kw": f"{model.typical_electrical_range_kw[0]}-{model.typical_electrical_range_kw[1]}",
            "heat_output_range_kw": f"{model.typical_heat_output_range_kw[0]}-{model.typical_heat_output_range_kw[1]}",
            "cop_range": f"{model.cop_range[0]}-{model.cop_range[1]}",
            "optimal_flow_delta": model.optimal_flow_delta,
        }
```

Register in `async_setup_entry()`:
```python
entities.append(EffektGuardModelInfoSensor(coordinator, entry))
```

---

## Testing Checklist

### Unit Tests
```bash
# Test model profiles still work
pytest tests/test_heat_pump_models.py -v

# Test config flow
pytest tests/test_config_flow.py -v

# Test coordinator
pytest tests/test_coordinator.py -v

# Test decision engine
pytest tests/test_decision_engine.py -v
```

### Manual Testing

1. **Config Flow**:
   - [ ] Start new integration setup
   - [ ] Verify model selection step appears
   - [ ] All 4 models shown in dropdown
   - [ ] Default is F750
   - [ ] Selection saves correctly

2. **Coordinator**:
   - [ ] Check logs for "Loaded heat pump model: NIBE F750"
   - [ ] Verify no import errors
   - [ ] Model profile accessible via `coordinator.heat_pump_model`

3. **Decision Engine**:
   - [ ] Power validation logs warnings when appropriate
   - [ ] No crashes when model is None (backward compatibility)
   - [ ] Offset bounds respected

4. **Sensor** (if implemented):
   - [ ] Model info sensor appears
   - [ ] Shows correct model name
   - [ ] Attributes populated correctly

---

## Migration Strategy

**For Existing Installations**:

Add migration in `async_migrate_entry()` in `__init__.py`:

```python
async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating configuration from version %s", entry.version)

    if entry.version == 1:
        # Add default heat pump model for existing installations
        new_data = {**entry.data}
        if CONF_HEAT_PUMP_MODEL not in new_data:
            new_data[CONF_HEAT_PUMP_MODEL] = DEFAULT_HEAT_PUMP_MODEL
            _LOGGER.info("Added default heat pump model: %s", DEFAULT_HEAT_PUMP_MODEL)
            
        hass.config_entries.async_update_entry(entry, data=new_data)
        entry.version = 2
        
    _LOGGER.info("Migration to version %s successful", entry.version)
    return True
```

Update VERSION in config_flow to 2:
```python
VERSION = 2
```

---

## Rollback Plan

If issues occur:

1. **Revert config flow changes** - Remove model selection step
2. **Make model parameter optional** - Add `heat_pump_model=None` defaults
3. **Skip validation** - Check `if self.heat_pump_model:` before using
4. **Remove sensor** - Comment out model info sensor registration

All changes are backward compatible - code works with or without model selection.

---

## Success Criteria

- ✅ New installations can select heat pump model
- ✅ Existing installations migrate with default F750
- ✅ Coordinator loads correct model profile
- ✅ Decision engine uses model for validation
- ✅ No crashes or errors in logs
- ✅ All tests passing
- ✅ Power warnings appear when consumption exceeds model specs

---

## Time Estimate

| Step | Time | Cumulative |
|------|------|------------|
| 1. Add constant | 5 min | 5 min |
| 2. Config flow | 30 min | 35 min |
| 3. Coordinator | 20 min | 55 min |
| 4. Decision engine | 30 min | 85 min |
| 5. Wire up | 15 min | 100 min |
| 6. Sensor (optional) | 20 min | 120 min |
| Testing | 30 min | 150 min |

**Total**: ~2.5 hours

---

## Next Steps After Implementation

1. **Update documentation** - Add model selection screenshots
2. **Update README** - Mention model-specific optimization
3. **Add to ROADMAP** - Mark as complete
4. **User announcement** - Highlight in release notes

---

## Notes

- **Backward compatibility**: All changes allow missing model (defaults to F750)
- **Low risk**: Model profiles are read-only, no breaking changes
- **Testable**: Can verify with existing test suite
- **User value**: Immediate benefit from model-specific validation and optimization
