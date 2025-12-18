# GitHub Copilot Instructions for EffektGuard

Home Assistant custom integration for intelligent NIBE heat pump control, optimizing for Swedish electricity costs (spot prices and effect tariffs) while maintaining comfort. **Production code affecting real homes, heating comfort, and heat pump health.**

## Development Environment Setup

**On new setups, run the development environment setup script first:**
```bash
bash scripts/setup_dev.sh
```

This will:
- Install Python 3.13 (required for Home Assistant compatibility)
- Create a virtual environment in `.venv/`
- Install all test requirements

After setup, activate the environment with:
```bash
source .venv/bin/activate
```

## Core Principles

1. **Read entire files before editing** - Never edit based on grep/partial reads
2. **Configuration-driven, never hardcode** - All values from `const.py` or config
3. **Constants only, no hardcoded values** - All numeric values, thresholds, and tuning parameters MUST be defined in `const.py` and imported where needed
4. **Reuse constants across all files** - If a constant exists in `const.py`, use it everywhere applicable (production code, tests, scripts)
5. **Safety-first approach** - Heat pump health and comfort over cost savings
6. **No backward compatibility** - Rename directly, update all callers, no aliases
7. **Test after every change** - Clear `__pycache__`, run tests, verify heat pump behavior
8. **No verbose summaries** - User sees changes in editor, only respond if asked
9. **Keep it simple** - Clean code over complexity, cleanup as you go
10. **Ask before acting** - When uncertain about NIBE behavior, clarify first
11. **No analysis/summary files in git** - All `*ANALYSIS*.md`, `*SUMMARY*.md` files stay untracked
12. **Follow Home Assistant best practices** - Use official patterns, coordinator, config flow
13. **Read NIBE documentation** - Never guess heat pump behavior, verify with research docs
14. **Use Black formatting** - All Python code must be formatted with Black (line length 100)
15. **No `Any` type imports** - Use specific types (dataclasses, TypedDict, Protocol, etc.)
16. **All imports at file top** - Place all imports at the top of files to avoid circular imports

---

## Architecture

### Safety-First Design

**Climate-aware thresholds control entire system behavior:**
```python
# ✅ Correct - climate zone-aware from ClimateZoneDetector
from .optimization.climate_zones import ClimateZoneDetector

# Initialize climate detector (uses Home Assistant's latitude)
climate_detector = ClimateZoneDetector(latitude=59.33)  # Stockholm example

# Get context-aware DM thresholds for current outdoor temperature
outdoor_temp = -10.0
dm_range = climate_detector.get_expected_dm_range(outdoor_temp)

# Thresholds automatically adapt to location and conditions:
# Stockholm at -10°C: normal_min=-450, normal_max=-700, warning=-700, critical=-1500
# Kiruna at -30°C: normal_min=-800, normal_max=-1200, warning=-1200, critical=-1500
# Paris at 5°C: normal_min=-200, normal_max=-350, warning=-350, critical=-1500

if degree_minutes < dm_range["critical"]:  # Always -1500 (absolute maximum)
    # Emergency recovery mode
elif degree_minutes < dm_range["warning"]:  # Zone-specific warning threshold
    # Warning: Stop cost optimization, gentle recovery
    
# ❌ Wrong - NEVER hardcode DM thresholds (not climate-aware!)
if degree_minutes < -500:  # DANGEROUS - ignores climate context!
```

**Key Safety Constants:**
```python
from .const import (
    DM_THRESHOLD_START,              # -60 (normal compressor start)
    DM_THRESHOLD_EXTENDED,           # -240 (extended runs, acceptable)
    DM_THRESHOLD_AUX_LIMIT,          # -1500 (auxiliary heat limit, avoid exceeding)
)

# Climate zone system provides context-aware thresholds:
# - DM_THRESHOLD_WARNING: Climate/temp specific (e.g., -700 for Stockholm at -10°C)
# - DM_THRESHOLD_AUX_LIMIT: Always -1500 (validated in Swedish forums)
```

### Four-Layer Structure

1. **Integration Layer** (`custom_components/effektguard/`): Home Assistant-specific
   - Config flow for user setup (`config_flow.py`)
   - Entity creation (`climate.py`, `sensor.py`, `number.py`, etc.)
   - Coordinator for data updates (`coordinator.py`)
   - Service registration (`services.yaml`)

2. **Optimization Engine** (`optimization/`): Pure Python logic
   - Energy budget management (`effect_manager.py`)
   - Thermal modeling (`thermal_model.py`)
   - Multi-layer decision engine (`decision_engine.py`)
   - Price analysis (`price_analyzer.py`)
   - All return domain objects, no HA dependencies

3. **Data Adapters** (`adapters/`): External integration interfaces
   - NIBE Myuplink reader (`nibe_adapter.py`)
   - Spot price reader (`gespot_adapter.py`)
   - Weather forecast reader (`weather_adapter.py`)
   - All read from existing HA entities, no direct API calls

4. **Validation Layer** (`utils/validators.py`): Configuration safety
   - Pump configuration detection (open-loop requires Auto mode)
   - System type identification (open-loop vs buffered vs mixed)
   - UFH type detection (concrete slab vs timber vs radiator)
   - Critical warnings that block activation

**Data Flow:**
```
NIBE/Spot Price/Weather Entities → Adapters → Coordinator →
Optimization Engine → Decision → Climate Entity → NIBE Offset Control
```

**Cross-Cutting:**
- **Safety** (`utils/safety.py`): Thermal debt tracking, emergency recovery
- **Configuration** (`config_flow.py`): Validation, warnings, guided setup
- **Constants** (`const.py`): All thresholds, defaults, entity IDs patterns

---

## Critical Rules

### Always Read Full Files Before Editing

Before editing, use `read_file` for entire file. Understand heat pump context, identify all change locations, check for safety implications.

### Use Configuration Constants

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

### Safety Thresholds Are Non-Negotiable

```python
# ✅ Correct - validate against constants
def validate_offset(offset: float) -> float:
    """Ensure offset within safe range."""
    return max(MIN_OFFSET, min(offset, MAX_OFFSET))

# ❌ Wrong - magic numbers for safety
def validate_offset(offset: float) -> float:
    return max(-10, min(offset, 10))  # Where did these come from?
```

### NIBE-Specific Knowledge Required

```python
# ✅ Correct - based on NIBE research
# Open-loop UFH requires Pump Auto mode at 10-20% idle speed
# Source: glyn.hudson F2040 case study (8-hour off periods with Intermittent mode)
if system_type == "open_loop" and pump_mode != "AUTO":
    raise ConfigurationError("Open-loop UFH requires Pump AUTO mode")

# ❌ Wrong - guessing NIBE behavior
if pump_mode == "off":  # NIBE doesn't have "off" mode!
    turn_on_pump()  # This isn't how NIBE works!
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
```

### Verify Your Work

```bash
# Search for old terms
grep -r "old_function_name" custom_components/effektguard/

# Find hardcoded numeric values (CRITICAL CHECK!)
grep -rE "if .* [<>] -?[0-9]+\.[0-9]+" custom_components/effektguard/ | grep -v "const.py"
grep -rE "= -?[0-9]+\.[0-9]+ *#" custom_components/effektguard/ | grep -v "const.py"
grep -rE "\* -?[0-9]+\.[0-9]+" custom_components/effektguard/ | grep -v "const.py"

# Verify all constants imported where used
grep -r "0.4" custom_components/effektguard/  # Example: tolerance multiplier
grep -r "WARNING_DEVIATION_THRESHOLD" custom_components/effektguard/  # Should be imported

# Test imports
python3 -c "from custom_components.effektguard.optimization.thermal_model import ThermalModel"

# Run Black formatting
black custom_components/effektguard/ --check --line-length 100
```

---

## Code Standards

### Constants Usage (Critical Rule)

**ALL numeric values, thresholds, and tuning parameters MUST be constants from `const.py`.**

```python
# ✅ Correct - import and use constants
from .const import (
    TOLERANCE_RANGE_MULTIPLIER,
    WARNING_DEVIATION_THRESHOLD,
    PROACTIVE_ZONE1_OFFSET,
    RAPID_COOLING_THRESHOLD,
)

def calculate_tolerance_range(tolerance: float) -> float:
    return tolerance * TOLERANCE_RANGE_MULTIPLIER

def check_deviation(deviation: float) -> str:
    if deviation > WARNING_DEVIATION_THRESHOLD:
        return "severe"
    return "moderate"

# ❌ Wrong - NEVER hardcode numeric values
def calculate_tolerance_range(tolerance: float) -> float:
    return tolerance * 0.4  # WRONG! Use constant

def check_deviation(deviation: float) -> str:
    if deviation > 200:  # WRONG! Use constant
        return "severe"
```

**Constant Reuse Across Files:**
- Production code (`decision_engine.py`, `thermal_model.py`, etc.) imports from `const.py`
- Test code (`tests/`, `scripts/test_decision_scenarios.py`) imports SAME constants
- NO duplicate definitions - single source of truth
- If constant exists, use it everywhere applicable

**When to Add New Constants:**
1. Any numeric threshold or tuning parameter
2. Any value used in multiple places
3. Any value that might need tuning
4. Add to `const.py` with descriptive comment and research reference
5. Update imports in ALL files that need it

**Naming Conventions:**
- Layer weights: `LAYER_WEIGHT_<LAYER>_<VARIANT>`
- DM thresholds: `DM_<CONTEXT>_<LEVEL>_<PROPERTY>`
- Effect layer: `EFFECT_<PROPERTY>_<LEVEL>`
- Proactive zones: `PROACTIVE_ZONE<N>_<PROPERTY>`
- Rapid cooling: `RAPID_COOLING_<PROPERTY>`
- WARNING layer: `WARNING_<PROPERTY>`

### Imports

Group as: stdlib, third-party, Home Assistant, local. Relative imports within `custom_components/effektguard/`.

```python
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
```

### Error Handling

Specific exceptions, log with context, graceful fallback. **Never let optimization crash the heat pump control.**

```python
from homeassistant.exceptions import HomeAssistantError

try:
    nibe_state = await self.nibe_adapter.get_current_state()
except HomeAssistantError as e:
    _LOGGER.error("Failed to read NIBE state: %s", e)
    # Fall back to safe operation
    return await self.get_safe_default_decision()
```

### Type Hints

Required on all functions. Use Home Assistant types where applicable.

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
```

### Async

Almost everything is async. Use `await`, never blocking calls in event loop.

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

### Docstrings

Every public function/class. Include:
- What it does
- Parameters with types
- Returns
- NIBE-specific behavior notes
- Safety implications if applicable

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
```

### Comments

Explain WHY, not WHAT. Reference research documents for NIBE-specific decisions.

```python
# ✅ Good - explains reasoning with reference
# Open-loop UFH requires 10-20% pump circulation when compressor off
# to prevent BT25 sensor false readings (glyn.hudson case study)
if system_type == "open_loop":
    min_pump_speed = 10  # ASHP

# ❌ Bad - obvious what
# Set pump speed to 10
min_pump_speed = 10
```

### Black Formatting

All code must be formatted with Black (line length 100):

```bash
# Format all code
black custom_components/effektguard/ --line-length 100

# Check formatting
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

---

## Testing

### Priority

Test production code with real NIBE behavior patterns. Mock external APIs but use realistic NIBE values.

```python
# ✅ Test real NIBE behavior
def test_thermal_debt_prevention():
    """Test that DM -500 triggers emergency recovery."""
    tracker = ThermalDebtTracker()
    tracker.degree_minutes = -500
    
    assert tracker.get_severity() == "CRITICAL"
    assert tracker.get_recovery_offset() == 3.0  # Emergency recovery
    assert tracker.should_block_dhw() is True

# ❌ Test implementation details
def test_private_method_returns_correct_type():
    result = tracker._internal_helper()
    assert isinstance(result, float)
```

### Test Safety-Critical Code

All safety thresholds must have tests:

```python
def test_prevents_catastrophic_thermal_debt():
    """Verify DM -500 threshold prevents heat pump damage."""
    # Based on stevedvo's real failure: DM -500 = 15kW spikes, 10°K overshoot
    engine = DecisionEngine()
    state = NibeState(degree_minutes=-500)
    
    decision = engine.calculate_decision(state, prices, weather)
    
    # Must trigger emergency recovery, ignore cost optimization
    assert decision.offset >= 2.0
    assert "EMERGENCY" in decision.reasoning

def test_blocks_activation_wrong_pump_config():
    """Verify open-loop with Intermittent mode blocks activation."""
    # Based on glyn.hudson's 8-hour off period failure
    validator = SystemConfigValidator()
    
    warnings = validator.validate_pump_configuration(
        system_type="open_loop",
        pump_mode="INTERMITTENT"
    )
    
    assert len(warnings) > 0
    assert warnings[0].severity == "CRITICAL"
    assert warnings[0].blocks_activation is True
```

### When to Mock

- External APIs (MyUplink, spot price, Met.no)
- Home Assistant core functions
- Time-dependent operations
- Network calls

### When Real Values

- NIBE thresholds (use climate-aware DM ranges from ClimateZoneDetector)
- Thermal calculations
- Configuration validation
- Decision engine logic

### Run Tests

```bash
# Use the test runner script (RECOMMENDED)
bash scripts/run_all_tests.sh                    # All tests
bash scripts/run_all_tests.sh unit               # Unit tests only
bash scripts/run_all_tests.sh validation         # Validation tests
bash scripts/run_all_tests.sh optimization       # Specific category
bash scripts/run_all_tests.sh -v -c all          # Verbose with coverage

# Direct pytest (when needed)
pytest tests/unit/ -v

# Integration tests (requires NIBE entities)
pytest tests/integration/ -v

# Specific test
pytest tests/unit/test_thermal_debt.py::test_dm_500_emergency -v

# With coverage
pytest tests/ --cov=custom_components/effektguard --cov-report=html
```

**Important:** Always use `bash scripts/run_all_tests.sh` for running tests. It provides organized output and handles all categories properly.

---

## Common Tasks

### Adding New Safety Threshold

1. Add constant to `const.py` with research reference
2. Update `ThermalDebtTracker` or relevant safety class
3. Add unit tests for threshold behavior
4. Update documentation with consequences
5. Add to configuration validation if user-facing

### Adding New Decision Layer

1. Create layer method in `decision_engine.py`
2. Add to layer priority ordering
3. Implement with clear reasoning string
4. Add unit tests with edge cases
5. Document with research references

### Debugging Optimization

1. Check logs: "Decision: offset X.X, layers: ..."
2. Verify each layer vote and weight
3. Check thermal debt tracker state
4. Verify NIBE entity readings
5. Test with `logger.setLevel(logging.DEBUG)`
6. **Create visualization graphs** for complex optimization issues

### Debugging with Visualization Graphs

**Create matplotlib graphs when debugging optimization issues.** Graphs provide:
- Clear visual comparison of current vs expected behavior
- Concrete data to discuss with stakeholders
- Evidence of the bug and proof the fix works

**Graph template:**
```python
import matplotlib.pyplot as plt
import numpy as np

# 3-panel comparison: Prices, Current Behavior, Expected Behavior
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Issue: [Description]", fontsize=14, fontweight="bold")

# Panel 1: Prices (color-coded by classification)
# Panel 2: Current (buggy) behavior - offsets over time
# Panel 3: Expected (fixed) behavior - offsets over time

# Use clear colors:
# - Red/orange for EXPENSIVE/PEAK periods
# - Green for CHEAP periods  
# - Blue for heating (+offset), Red for reducing (-offset)

plt.savefig("docs/dev/debug_issue_description.png", dpi=150)
```

**Graph requirements:**
1. Top panel: Input data (prices, temperatures, DM values)
2. Middle panel: Current system behavior with offsets
3. Bottom panel: Expected behavior after fix
4. Clear color coding and annotations
5. Save to `docs/dev/` for discussion

### Configuration Issues

1. Check setup validator warnings
2. Verify entity IDs exist and update
3. Test pump configuration detection
4. Validate UFH type settings
5. Check MyUplink entity availability

---

## Key Files

**Entry:**
- `custom_components/effektguard/__init__.py` - Integration setup
- `custom_components/effektguard/manifest.json` - Metadata

**Core:**
- `coordinator.py` - DataUpdateCoordinator pattern
- `const.py` - ALL constants and thresholds
- `climate.py` - Main climate entity

**Optimization:**
- `optimization/decision_engine.py` - Multi-layer decision logic
- `optimization/thermal_model.py` - Thermal calculations
- `optimization/effect_manager.py` - Peak tracking (15-minute windows)
- `optimization/price_analyzer.py` - Spot price classification

**Adapters:**
- `adapters/nibe_adapter.py` - Read NIBE MyUplink entities
- `adapters/gespot_adapter.py` - Read spot price entities
- `adapters/weather_adapter.py` - Read weather forecast

**Safety:**
- `utils/validators.py` - Configuration validation
- `utils/safety.py` - Thermal debt tracking

---

## Refactoring

### Phase Order

```
Safety → Configuration → Optimization → Entities → Integration
Constants → Validators → Decision Engine → Climate → Coordinator
Base classes → Adapters → Services → Config Flow → Tests
```

### Before Claiming Phase Complete

- [ ] All files updated (including tests, type hints, docstrings)
- [ ] All safety thresholds validated against research
- [ ] All NIBE-specific behavior documented
- [ ] All imports/callers updated
- [ ] Black formatting applied (`black . --check --line-length 100`)
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] No hardcoded values (grep for magic numbers)
- [ ] Configuration validation updated if user-facing
- [ ] Research references added to docstrings
- [ ] No dead code or duplicates

### Commit After Each Phase

```bash
git commit -m "Phase N: Description

- Change 1 (with safety implications noted)
- Change 2 (NIBE research reference)

Tests: passing
Safety: validated
Format: black applied"
```

---

## Common Mistakes

❌ **Editing without full NIBE context** - Read research docs first
❌ **Hardcoding safety thresholds** - Use constants from `const.py`
❌ **Guessing NIBE behavior** - Verify with Forum_Summary.md or similar
❌ **Incomplete refactoring** - Update ALL callers including tests
❌ **Forgetting safety validation** - All optimization must respect thermal debt
❌ **Skipping Black formatting** - Always format before commit
❌ **No research references** - Link to source document for NIBE decisions
❌ **Blocking event loop** - Use async/await, never `time.sleep()`
❌ **Ignoring Home Assistant patterns** - Use coordinator, config flow, best practices

---

## Code Quality

### Remove Dead Code Immediately

```python
# ❌ Don't leave
UNUSED_THRESHOLD = -1000  # Never used

# ✅ Delete it
```

### Fix Duplicates

```python
# ❌ Same file
DM_CRITICAL = -500  # Line 18
DM_CRITICAL = -400  # Line 26 (different value!)

# ✅ One definition in const.py
DM_THRESHOLD_CRITICAL = -500  # stevedvo case study
```

### Document NIBE-Specific Calculations

```python
# ✅ Show NIBE research basis
# Open-loop UFH prediction horizon (glyn.hudson case study)
# Concrete slab: 6+ hours thermal lag observed
# Requires 12-hour prediction for proper pre-heating
UFH_CONCRETE_PREDICTION_HORIZON = 12.0  # hours

# ❌ No context
HORIZON = 12.0
```

### Reference Research Documents

```python
# ✅ Good - traceable to research
# Thermal debt threshold based on stevedvo's F2040 real-world failure
# DM -500 caused: 15kW spikes, 10°K overshoot, catastrophic inefficiency
# Swedish term: Gradminuter (GM), NIBE Menu 4.9.3
# Swedish auxiliary optimization: -1000 to -1500 (prevents excessive aux heat)
# Source: Forum_Summary.md, Swedish_NIBE_Forum_Findings.md
DM_THRESHOLD_CRITICAL = -500
DM_THRESHOLD_AUX_SWEDISH = -1500

# ❌ Bad - no justification
DM_THRESHOLD_CRITICAL = -500  # Don't go below this
```

### Black Formatting in Docstrings

```python
# ✅ Formatted with Black
def calculate_optimal_flow_temp(
    self,
    indoor_setpoint: float,
    outdoor_temp: float,
    heat_loss_coefficient: float = 180.0,
) -> float:
    """Calculate optimal flow temperature using André Kühne's formula.
    
    Validated across manufacturers: Vaillant, Daikin, Mitsubishi, NIBE.
    
    Args:
        indoor_setpoint: Target indoor temperature (°C)
        outdoor_temp: Current outdoor temperature (°C)
        heat_loss_coefficient: Building heat loss (W/°C), default 180.0
        
    Returns:
        Optimal flow temperature (°C)
        
    References:
        Mathematical_Enhancement_Summary.md: André Kühne's formula
        Formula: TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset
    """
```

---

## NIBE Heat Pump Specifics

### Critical NIBE Knowledge

**Degree Minutes (DM) - Climate Zone Aware:**
- Swedish term: Gradminuter (GM) in NIBE Menu 4.9.3
- Tracks thermal balance: `DM = ∫(BT25 - S1) dt`
- BT25 = actual flow temperature
- S1 = target flow temperature
- Standard compressor start: DM -60
- Extended runs: DM -240 (stevedvo custom setting, acceptable)
- **CLIMATE-AWARE WARNING**: Varies by zone and outdoor temp
  - Stockholm at -10°C: DM -700 warning threshold
  - Kiruna at -30°C: DM -1200 warning threshold
  - Paris at 5°C: DM -350 warning threshold
- **AUXILIARY LIMIT: DM -1500** (validated in Swedish forums, avoid exceeding to prevent expensive aux heat)
- Use `ClimateZoneDetector.get_expected_dm_range(outdoor_temp)` for context-aware thresholds

**Pump Configuration:**
- **Open-loop UFH**: MUST use Auto mode, 10% (ASHP) or 20% (GSHP) idle
- **Buffered systems**: Intermittent mode acceptable
- Wrong setting = 8-hour off periods (glyn.hudson case)

**UFH Types:**
- **Concrete slab**: 6+ hours thermal lag, 12h prediction horizon
- **Timber**: 2-3 hours lag, 6h prediction horizon  
- **Radiators**: <1 hour lag, 2h prediction horizon

**Flow Temperature Targets (OEM Research):**
- SPF 4.0+ systems: Flow = Outdoor + 27°C ±3°C
- SPF 3.5+ systems: Flow = Outdoor + 30°C ±4°C
- Most systems run 5-15°C below optimal (huge opportunity)

**MyUplink API:**
- Update interval: ~60 seconds
- Requires Premium for write access (curve offset control)
- Entity pattern: `number.{device}_offset_s1_47011`

### Always Check Research Before Implementing

When implementing NIBE-specific features, reference:
1. `IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md` - Real F2040 cases
2. `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md` - F750 optimizations
3. `IMPLEMENTATION_PLAN/01_Algorithm/Setpoint_Optimizing_Algorithm.md` - Algorithm spec
4. `IMPLEMENTATION_PLAN/03_API/MyUplink_Complete_Guide.md` - API details

---

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

### Config Flow for Setup

```python
from homeassistant import config_entries
import voluptuous as vol

class EffektGuardConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for EffektGuard."""
    
    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        # Validation logic here
```

### Entity Naming

```python
from homeassistant.helpers.entity import Entity

class EffektGuardClimate(CoordinatorEntity, ClimateEntity):
    """Climate entity for EffektGuard."""
    
    _attr_has_entity_name = True
    _attr_name = None  # Will use device name
```

### Services

Define in `services.yaml`, implement in `__init__.py`:
```yaml
set_optimization_mode:
  name: Set optimization mode
  description: Change optimization behavior
  fields:
    mode:
      description: Optimization mode (comfort/balanced/savings)
      example: "balanced"
      selector:
        select:
          options:
            - "comfort"
            - "balanced"
            - "savings"
```

---

## GitHub Release Notes Format

When creating release notes via `gh release create`, use this exact format:

```markdown
## Key Changes

### Bug Fixes
• Fixed [description]

### Enhancements  
• **Feature name**: Brief description

### Removed
• `CONSTANT_NAME` - reason for removal

**Full Changelog**: https://github.com/enoch85/EffektGuard/compare/vX.Y.Z...vX.Y.W
```

**Rules:**
- **Source of Truth**: Always perform a full `git diff` between tags (e.g., `git diff v0.4.11..v0.4.12`) to identify actual code changes. NEVER guess or assume changes.
- **Pushing Changes**: Use `gh release edit` to update existing ones with the generated notes (e.g., `gh release edit v0.4.12 --notes "## Key Changes..."`).
- **Header**: Use `## Key Changes` as main header.
- **Grouping**: Group changes under `### Bug Fixes`, `### Enhancements`, `### Removed`, `### Renamed` as needed.
- **Bullets**: Use bullet points with `•` character.
- **Features**: Bold feature names with `**Feature**:`
- **Code Symbols**: Use backticks for constant/function names.
- **Conciseness**: Keep descriptions concise (one line each).
- **Changelog**: Always include Full Changelog link at bottom.
- **Cleanup**: Omit empty sections.

---

## Project Context

**Purpose:** Intelligent NIBE heat pump control for Swedish effect tariff optimization

**Critical:**
- Safety over savings (thermal debt prevention)
- Comfort over cost (moderate optimization, not aggressive)
- Real homes depend on this (heat pump health matters)

**Research-Based:**
- All thresholds from real NIBE failures and Swedish forum validation
- Climate zone system: Adapts DM thresholds from Arctic (-30°C) to Mediterranean (5°C)
- Mathematical formulas from OEM research (André Kühne, Timbones)

**Swedish-Specific:**
- 15-minute effect tariff windows (quarterly measurement)
- Spot price integration for native 15-minute prices
- F750/F2040 focus with S-series support

**Known Critical Issues:**
- Climate-aware thermal debt thresholds (DM -1500 auxiliary limit, validated in Swedish forums)
- Open-loop pump Intermittent = 8-hour off periods
- BT50 indoor sensor + UFH = instability (not recommended)
- DHW during heating demand = thermal debt accumulation

**Integration:** Home Assistant DataUpdateCoordinator pattern, config flow, HACS compatible

**Quality:** Production code for real homes. When uncertain about NIBE behavior, **ask and verify** before implementing. Reference research documents. Use Black formatting. Test safety-critical code thoroughly.

Quality, safety, and correctness over speed. When uncertain, ask before implementing.
