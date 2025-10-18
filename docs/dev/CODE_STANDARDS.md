# Code Standards and Best Practices

Code quality standards for EffektGuard development.

## Core Principles

1. **Safety First** - Heat pump health and comfort over cost savings
2. **Configuration-Driven** - All values from `const.py` or config, never hardcode
3. **Read Before Editing** - Always read entire files to understand context
4. **No Backward Compatibility** - Rename directly, update all callers
5. **Test Everything** - Clear `__pycache__`, run tests, verify behavior
6. **Keep It Simple** - Clean code over complexity
7. **Black Formatting** - Line length 100, format before commit

## Code Formatting

### Black Formatter

All Python code must be formatted with Black (line length 100):

```bash
# Format all code
black custom_components/effektguard/ --line-length 100

# Check formatting (CI/CD)
black custom_components/effektguard/ --check --line-length 100

# Format specific file
black custom_components/effektguard/climate.py --line-length 100
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
```

### Import Organization

Group imports as: stdlib, third-party, Home Assistant, local. Use relative imports within package.

```python
# ✅ Correct
# Stdlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# Third-party
import voluptuous as vol

# Home Assistant
from homeassistant.components.climate import ClimateEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

# Local (relative imports)
from .const import DOMAIN, DM_THRESHOLD_CRITICAL
from .optimization.decision_engine import DecisionEngine

# ❌ Wrong - mixed ordering
from .const import DOMAIN
from datetime import datetime
from homeassistant.core import HomeAssistant
import voluptuous as vol
```

## Type Hints

Required on all public functions. Use Home Assistant types where applicable.

```python
from homeassistant.core import HomeAssistant
from typing import Optional

async def set_curve_offset(
    self,
    hass: HomeAssistant,
    offset: float,
    entity_id: str,
) -> None:
    """Set heating curve offset via NIBE entity."""
    ...
```

## Documentation

### Docstrings

Every public function/class must have a docstring including:
- What it does
- Parameters with types
- Returns
- NIBE-specific behavior notes
- Safety implications if applicable
- Research references

```python
def calculate_preheating_target(
    self,
    current_temp: float,
    desired_temp: float,
    hours_until_peak: int,
    outdoor_temp: float,
    forecast_min_temp: float,
) -> float:
    """Calculate target temperature for pre-heating before expensive/cold period.
    
    Uses thermal decay modeling to account for heat loss during forecast period.
    More conservative than aggressive load-shifting approaches to prevent thermal debt.
    
    Args:
        current_temp: Current indoor temperature (°C)
        desired_temp: Target indoor temperature during expensive hours (°C)
        hours_until_peak: Hours until expensive period begins
        outdoor_temp: Current outdoor temperature (°C)
        forecast_min_temp: Minimum outdoor temperature in forecast period (°C)
        
    Returns:
        Target indoor temperature for pre-heating phase (°C)
        
    Notes:
        Based on research showing load shifting without battery is ineffective.
        Uses moderate pre-heating to prevent thermal debt (DM < -500 catastrophic).
        
    References:
        - Forum_Summary.md: stevedvo's thermal debt case study
        - Enhancement_Proposals.md: Thermal model mathematics
    """
    ...
```

### Comments

Explain WHY, not WHAT. Reference research documents for NIBE-specific decisions.

```python
# ✅ Good - explains reasoning with reference
# Open-loop UFH requires 10-20% pump circulation when compressor off
# to prevent BT25 sensor false readings (glyn.hudson case study)
if system_type == "open_loop":
    min_pump_speed = 10  # ASHP: 10%, GSHP: 20%

# ❌ Bad - obvious what
# Set pump speed to 10
min_pump_speed = 10
```

## Configuration Constants

### Never Hardcode Values

```python
# ✅ Do this
from .const import (
    DM_THRESHOLD_CRITICAL,
    UFH_CONCRETE_PREDICTION_HORIZON,
    OPTIMAL_FLOW_DELTA_SPF_4,
)

if ufh_type == "concrete_slab":
    horizon = UFH_CONCRETE_PREDICTION_HORIZON  # 12.0 hours

# ❌ Never this
if ufh_type == "concrete_slab":
    horizon = 12.0  # Hardcoded!
```

### Safety Thresholds

Safety thresholds are non-negotiable and must use constants:

```python
# ✅ Correct - validate against constants
def validate_offset(offset: float) -> float:
    """Ensure offset within safe range."""
    return max(MIN_OFFSET, min(offset, MAX_OFFSET))

# ❌ Wrong - magic numbers for safety
def validate_offset(offset: float) -> float:
    return max(-10, min(offset, 10))  # Where did these come from?
```

## Error Handling

Use specific exceptions, log with context, graceful fallback.

```python
from homeassistant.exceptions import HomeAssistantError

try:
    nibe_state = await self.nibe_adapter.get_current_state()
except HomeAssistantError as e:
    _LOGGER.error("Failed to read NIBE state: %s", e)
    # Fall back to safe operation
    return await self.get_safe_default_decision()
```

## Async/Await

Almost everything in Home Assistant is async. Use `await`, never blocking calls.

```python
# ✅ Correct
async def calculate_decision(self) -> OptimizationDecision:
    nibe_data = await self.nibe.get_current_state()
    price_data = await self.gespot.get_prices()
    return self.engine.calculate(nibe_data, price_data)

# ❌ Wrong - blocking in async
async def calculate_decision(self) -> OptimizationDecision:
    time.sleep(5)  # BLOCKS EVENT LOOP!
    return result
```

## NIBE-Specific Knowledge

### Climate-Aware Safety Thresholds

```python
# ✅ Correct - climate zone-aware from ClimateZoneDetector
from .optimization.climate_zones import ClimateZoneDetector

climate_detector = ClimateZoneDetector(latitude=59.33)  # Stockholm
outdoor_temp = -10.0
dm_range = climate_detector.get_expected_dm_range(outdoor_temp)

# Thresholds adapt to location:
# Stockholm at -10°C: warning=-700, critical=-1500
# Kiruna at -30°C: warning=-1200, critical=-1500

if degree_minutes < dm_range["critical"]:
    # Emergency recovery
elif degree_minutes < dm_range["warning"]:
    # Warning: Stop optimization

# ❌ Wrong - NEVER hardcode DM thresholds
if degree_minutes < -500:  # DANGEROUS - ignores climate context!
```

### Document NIBE Behavior

Always reference research for NIBE-specific decisions:

```python
# ✅ Good - traceable to research
# Thermal debt threshold based on stevedvo's F2040 real-world failure
# DM -500 caused: 15kW spikes, 10°K overshoot, catastrophic inefficiency
# Swedish term: Gradminuter (GM), NIBE Menu 4.9.3
# Source: Forum_Summary.md, Swedish_NIBE_Forum_Findings.md
DM_THRESHOLD_CRITICAL = -500

# ❌ Bad - no justification
DM_THRESHOLD_CRITICAL = -500  # Don't go below this
```

## Code Quality

### Remove Dead Code

```python
# ❌ Don't leave
UNUSED_THRESHOLD = -1000  # Never used

# ✅ Delete it immediately
```

### Fix Duplicates

```python
# ❌ Same file with different values
DM_CRITICAL = -500  # Line 18
DM_CRITICAL = -400  # Line 26 - CONFLICT!

# ✅ One definition in const.py
DM_THRESHOLD_CRITICAL = -500  # stevedvo case study
```

### No Backward Compatibility

```python
# ❌ Don't create aliases
def calculate_heating_offset(*args, **kwargs):
    return calculate_curve_offset(*args, **kwargs)

# ✅ Just rename and update ALL callers
def calculate_curve_offset(
    indoor_temp: float,
    outdoor_temp: float,
    target_temp: float,
) -> float:
    """Calculate optimal heating curve offset based on thermal model."""
    ...
```

## Verification Checklist

Before committing:

```bash
# Search for old terms
grep -r "old_function_name" custom_components/effektguard/

# Find hardcoded safety values
grep -r "= -500" custom_components/effektguard/

# Find hardcoded thresholds
grep -r "= 6.0" custom_components/effektguard/

# Test imports
python3 -c "from custom_components.effektguard.optimization.thermal_model import ThermalModel"

# Run Black formatting
black custom_components/effektguard/ --check --line-length 100

# Run tests
pytest tests/ -v
```

## Common Mistakes

❌ **Editing without full context** - Always read entire file first
❌ **Hardcoding safety thresholds** - Use constants from `const.py`
❌ **Guessing NIBE behavior** - Verify with research documents
❌ **Incomplete refactoring** - Update ALL callers including tests
❌ **Forgetting safety validation** - All optimization must respect thermal debt
❌ **Skipping Black formatting** - Always format before commit
❌ **No research references** - Link to source document for NIBE decisions
❌ **Blocking event loop** - Use async/await, never `time.sleep()`

## Home Assistant Best Practices

### Use Coordinator Pattern

```python
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

class EffektGuardCoordinator(DataUpdateCoordinator):
    """Coordinate data updates for EffektGuard."""
    
    def __init__(self, hass: HomeAssistant, ...):
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
        )
```

### Entity Naming

```python
from homeassistant.helpers.entity import Entity

class EffektGuardClimate(CoordinatorEntity, ClimateEntity):
    """Climate entity for EffektGuard."""
    
    _attr_has_entity_name = True
    _attr_name = None  # Will use device name
```

## Related Documentation

- [TESTING.md](TESTING.md) - Testing guidelines
- [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) - How to contribute
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
