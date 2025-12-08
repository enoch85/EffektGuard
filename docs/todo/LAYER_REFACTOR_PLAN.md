# Layer Refactor Implementation Plan

## Goal

Extract layer logic FROM `decision_engine.py` (2860 lines) INTO existing optimization files to:
1. Make decision_engine.py lighter (~1000 lines, just orchestration)
2. Enable DHW optimizer to reuse the same layer logic
3. Improve testability and maintainability
4. Rename files to consistent `*_layer.py` naming convention

---

## Current State

### Decision Engine Layer Sizes (calculated from line numbers)

| Layer | Lines | Size | Current Target File | Rename To |
|-------|-------|------|---------------------|-----------|
| `_safety_layer` | 816-853 | 38 | Keep inline (too small) | - |
| `_emergency_layer` | 854-1259 | **406** | `thermal_model.py` | `thermal_layer.py` |
| `_proactive_debt_prevention_layer` | 1260-1458 | **199** | `thermal_model.py` | `thermal_layer.py` |
| `_effect_layer` | 1459-1572 | 114 | `effect_manager.py` | `effect_layer.py` |
| `_prediction_layer` | 1573-1679 | 107 | `thermal_predictor.py` | `prediction_layer.py` |
| `_weather_layer` | 1680-1783 | 104 | `weather_compensation.py` | `weather_layer.py` |
| `_weather_compensation_layer` | 1784-1939 | 156 | `weather_compensation.py` | `weather_layer.py` |
| `_price_layer` | 1940-2419 | **480** | `price_analyzer.py` | `price_layer.py` |
| `_comfort_layer` | 2420-2681 | **262** | NEW | `comfort_layer.py` |

**Total extractable: ~1,866 lines → decision_engine.py: 2860 → ~1000 lines**

---

## Phase 0: Rename Existing Files to `*_layer.py` Convention

### Files to Rename

| Current Name | New Name | Lines |
|--------------|----------|-------|
| `effect_manager.py` | `effect_layer.py` | 345 |
| `thermal_model.py` | `thermal_layer.py` | 44 |
| `thermal_predictor.py` | `prediction_layer.py` | 492 |
| `weather_compensation.py` | `weather_layer.py` | 541 |
| `price_analyzer.py` | `price_layer.py` | 262 |

### Steps
1. `git mv effect_manager.py effect_layer.py`
2. `git mv thermal_model.py thermal_layer.py`
3. `git mv thermal_predictor.py prediction_layer.py`
4. `git mv weather_compensation.py weather_layer.py`
5. `git mv price_analyzer.py price_layer.py`
6. Update ALL imports in:
   - `decision_engine.py`
   - `dhw_optimizer.py`
   - `coordinator.py`
   - `climate.py`
   - `__init__.py`
   - All test files
7. Run full test suite

**Effort:** 2h | **Risk:** Medium (many import updates)

---

## Phase 1: Extract Effect Layer

### Current: `decision_engine.py` lines 1459-1572 (114 lines)

### Target: `effect_layer.py` (formerly `effect_manager.py`)

### Steps
1. Add `LayerDecision` dataclass to `effect_layer.py`
2. Add `evaluate_layer(nibe_state, current_peak, current_power) -> LayerDecision` method
3. Move logic from `_effect_layer` into new method
4. Update `decision_engine._effect_layer` to call `self.effect_layer.evaluate_layer(...)`
5. Run tests, verify behavior unchanged

**Effort:** 2h | **Risk:** Low | **Lines moved:** 114

---

## Phase 2: Extract Price Layer

### Current: `decision_engine.py` lines 1940-2419 (480 lines)

### Target: `price_layer.py` (formerly `price_analyzer.py`)

### Steps
1. Add `evaluate_layer(nibe_state, price_data) -> LayerDecision` method
2. Move proactive zone logic, price classification logic
3. Add `should_heat_dhw(adequate: bool, classification: str) -> bool` for DHW
4. Update decision_engine to delegate
5. Run tests

**Effort:** 4h | **Risk:** Medium | **Lines moved:** 480

---

## Phase 3: Extract Prediction Layer

### Current: `decision_engine.py` lines 1573-1679 (107 lines)

### Target: `prediction_layer.py` (formerly `thermal_predictor.py`)

### Steps
1. Add `evaluate_layer(nibe_state, weather_data) -> LayerDecision` method
2. Move prediction logic
3. Update decision_engine to delegate
4. Run tests

**Effort:** 2h | **Risk:** Low | **Lines moved:** 107

---

## Phase 4: Extract Weather Layer

### Current: `decision_engine.py` lines 1680-1783 (104 lines)

### Target: `weather_layer.py` (formerly `weather_compensation.py`)

### Steps
1. Add `evaluate_weather_layer(nibe_state, weather_data) -> LayerDecision` method
2. Move weather layer logic
3. Update decision_engine to delegate
4. Run tests

**Effort:** 2h | **Risk:** Low | **Lines moved:** 104

---

## Phase 5: Extract Weather Compensation Layer

### Current: `decision_engine.py` lines 1784-1939 (156 lines)

### Target: `weather_layer.py` (same file as Phase 4)

### Steps
1. Add `evaluate_compensation_layer(nibe_state, weather_data) -> LayerDecision` method
2. Move weather compensation logic
3. Update decision_engine to delegate
4. Run tests

**Effort:** 2h | **Risk:** Low | **Lines moved:** 156

---

## Phase 6: Extract Emergency Layer (Thermal Debt)

### Current: `decision_engine.py` lines 854-1259 (406 lines)

### Target: `thermal_layer.py` (formerly `thermal_model.py`)

### Complexity
This is the hardest extraction because:
- Complex tiered response logic
- Depends on climate zones
- Used by both space heating AND DHW

### Steps
1. Expand `thermal_layer.py` significantly
2. Add `evaluate_emergency_layer(nibe_state, weather_data, price_data) -> LayerDecision`
3. Add `should_block_dhw(dm, outdoor_temp) -> bool` for DHW
4. Update decision_engine to delegate
5. Update dhw_optimizer to use shared method
6. Run tests

**Effort:** 4h | **Risk:** High | **Lines moved:** 406

---

## Phase 7: Extract Proactive Debt Prevention Layer

### Current: `decision_engine.py` lines 1260-1458 (199 lines)

### Target: `thermal_layer.py` (same file as Phase 6)

### Steps
1. Add `evaluate_proactive_layer(nibe_state, weather_data) -> LayerDecision`
2. Move proactive debt prevention logic
3. Update decision_engine to delegate
4. Run tests

**Effort:** 2h | **Risk:** Medium | **Lines moved:** 199

---

## Phase 8: Extract Comfort Layer

### Current: `decision_engine.py` lines 2420-2681 (262 lines)

### Target: NEW `comfort_layer.py`

### Steps
1. Create new file `comfort_layer.py`
2. Add `evaluate_layer(nibe_state, weather_data, price_data) -> LayerDecision`
3. Move comfort logic
4. Update decision_engine to delegate
5. Run tests

**Effort:** 2h | **Risk:** Low | **Lines moved:** 262

---

## Final State

```
BEFORE                          AFTER
─────────────────────────────────────────────────────────
decision_engine.py    2860  →   ~1000 lines (orchestration)
effect_manager.py      345  →   effect_layer.py       ~460
price_analyzer.py      262  →   price_layer.py        ~740
weather_compensation   541  →   weather_layer.py      ~800
thermal_model.py        44  →   thermal_layer.py      ~650
thermal_predictor.py   492  →   prediction_layer.py   ~600
(new)                    0  →   comfort_layer.py      ~262
```

---

## Estimated Effort

| Phase | Description | Lines | Effort | Risk |
|-------|-------------|-------|--------|------|
| 0 | Rename files to `*_layer.py` | - | 2h | Medium |
| 1 | Extract Effect Layer | 114 | 2h | Low |
| 2 | Extract Price Layer | 480 | 4h | Medium |
| 3 | Extract Prediction Layer | 107 | 2h | Low |
| 4 | Extract Weather Layer | 104 | 2h | Low |
| 5 | Extract Weather Compensation Layer | 156 | 2h | Low |
| 6 | Extract Emergency Layer | 406 | 4h | High |
| 7 | Extract Proactive Debt Layer | 199 | 2h | Medium |
| 8 | Extract Comfort Layer | 262 | 2h | Low |
| **Total** | **~1,828 lines extracted** | | **~22h** | |

---

## Files Changed

### Renamed Files (Phase 0)
- `effect_manager.py` → `effect_layer.py`
- `thermal_model.py` → `thermal_layer.py`
- `thermal_predictor.py` → `prediction_layer.py`
- `weather_compensation.py` → `weather_layer.py`
- `price_analyzer.py` → `price_layer.py`

### New Files
- `comfort_layer.py`

### Modified Files
- `decision_engine.py` - Delegate to layer files
- `dhw_optimizer.py` - Import shared layer methods
- `coordinator.py` - Update imports
- `climate.py` - Update imports
- `__init__.py` - Update imports
- All test files - Update imports

---

## Benefits

1. **decision_engine.py manageable** - 1000 lines of pure orchestration
2. **Reusable layers** - DHW can import from same files
3. **Testable** - Each layer file can be unit tested independently
4. **Clearer ownership** - Price logic in price_layer, thermal in thermal_layer
5. **Consistent naming** - All layer files end with `_layer.py`

---

## Risk Mitigation

- **One phase at a time** - Never extract multiple layers in one PR
- **Keep old method as wrapper first** - `_effect_layer()` calls `effect_layer.evaluate_layer()`
- **Run full test suite after each phase**
- **No behavior changes** - Pure refactor, output must be identical
- **Git mv for renames** - Preserves file history

---

## Phase 9: Coordinator Refactor ✅ COMPLETED

### Results Achieved
- `coordinator.py`: **2551 → 2219 lines** (-332 lines)
- Methods moved to layer files for shared reuse
- All tests passing (1039 tests)

### What Was Moved

| Method | From | To | Lines |
|--------|------|-----|-------|
| `_calculate_dhw_recommendation` | coordinator.py | dhw_optimizer.py | ~180 |
| `_format_dhw_planning_summary` | coordinator.py | dhw_optimizer.py | ~60 |
| `_check_dhw_abort_conditions` | coordinator.py | dhw_optimizer.py | ~50 |
| `_estimate_power_consumption` | coordinator.py | effect_layer.py | ~40 |
| `_estimate_power_from_compressor` | coordinator.py | effect_layer.py | ~50 |
| `_get_thermal_debt_status` | coordinator.py | thermal_layer.py | ~20 |

### New Classes Added
- `DHWRecommendation` dataclass in dhw_optimizer.py - Complete recommendation result
- `IntelligentDHWScheduler.calculate_recommendation()` - Pure logic method

### New Constants Added to `const.py`
- `COMPRESSOR_HZ_MIN`, `COMPRESSOR_HZ_RANGE` - Compressor frequency bounds
- `COMPRESSOR_POWER_MIN_KW`, `COMPRESSOR_POWER_MAX_KW` - Power range
- `COMPRESSOR_TEMP_*_THRESHOLD/FACTOR` - Temperature-based power adjustments
- `POWER_STANDBY_KW` - Standby power for idle compressor

### Tests Created
- `tests/unit/test_shared_layer_methods.py` - 32 tests for new layer methods
- `tests/validation/test_dhw_abort_and_override.py` - DHW abort conditions
- `tests/validation/test_event_listener_integration.py` - HA event integration

### Refactor Pattern Used
The `_calculate_dhw_recommendation` was refactored using a thin wrapper pattern:
1. Coordinator gathers HA-specific data (entry.data, notifications, history)
2. Calls `dhw_optimizer.calculate_recommendation()` with all data as parameters
3. Pure logic in dhw_optimizer returns `DHWRecommendation` dataclass
4. Coordinator handles HA-specific side effects (notifications)

### Current State
- `coordinator.py`: **2551 lines** (largest file in codebase)
- Contains DHW logic, power estimation, peak tracking, persistence

### Analysis: What Can Move vs What Must Stay

#### MUST STAY in Coordinator (HA Integration)
These are core coordinator responsibilities that require Home Assistant context:

| Method | Lines | Reason |
|--------|-------|--------|
| `__init__` | ~150 | HA dependency injection, config entry |
| `_async_update_data` | ~500 | Main HA update loop, entity coordination |
| `async_initialize_learning` | ~90 | HA storage integration |
| `async_restore_peaks` | ~20 | HA storage |
| `setup_power_sensor_listener` | ~60 | HA event listeners |
| `async_shutdown` | ~35 | HA cleanup |
| `async_set_offset` | ~20 | HA service call to NIBE |
| `set_optimization_enabled` | ~20 | HA config updates |
| `async_update_config` | ~130 | HA options flow |
| `_save_learned_data` / `_load_learned_data` | ~100 | HA storage |
| `_schedule_aligned_refresh` | ~50 | HA timer scheduling |
| Properties / getters | ~50 | HA entity state |

**Total that MUST stay: ~1200-1300 lines**

#### CAN MOVE to Layer Files

| Method | Lines | Target | Status |
|--------|-------|--------|--------|
| `_calculate_dhw_recommendation` | ~200 | `dhw_optimizer.py` | ✅ MOVED |
| `_format_dhw_planning_summary` | ~60 | `dhw_optimizer.py` | ✅ MOVED |
| `_check_dhw_abort_conditions` | ~50 | `dhw_optimizer.py` | ✅ MOVED |
| `_get_thermal_debt_status` | ~20 | `thermal_layer.py` | ✅ MOVED |
| `_estimate_power_consumption` | ~40 | `effect_layer.py` | ✅ MOVED |
| `_estimate_power_from_compressor` | ~50 | `effect_layer.py` | ✅ MOVED |
| `_apply_airflow_decision` | ~80 | Keep (HA entity control) | - |
| `_apply_dhw_control` | ~130 | Keep (HA entity control) | - |
| `_update_peak_tracking` | ~180 | Already delegated to `EffectManager` | - |
| Climate detection | ~60 | Already in `climate_zones.py` | - |

**Total movable: ~400-500 lines**

#### CANNOT MOVE (Needs HA Context)

| Method | Reason |
|--------|--------|
| `_apply_airflow_decision` | Controls HA switch entity |
| `_apply_dhw_control` | Controls HA switch entity |
| `_record_learning_observations` | Uses HA time, multiple adapters |
| `_get_last_dhw_heating_time` | HA entity state history |

### Implementation Steps

#### Step 9.1: Move DHW Logic to `dhw_optimizer.py`

```python
# dhw_optimizer.py - ADD these methods to IntelligentDHWScheduler

def calculate_recommendation(
    self,
    nibe_data,
    price_data,
    weather_data,
    current_dhw_temp: float,
    now_time: datetime,
    climate_detector,
    thermal_predictor,
) -> DHWRecommendation:
    """Calculate DHW heating recommendation.
    
    Moved from coordinator._calculate_dhw_recommendation
    """
    pass

def format_planning_summary(self, decision: DHWScheduleDecision) -> str:
    """Format DHW planning summary for display.
    
    Moved from coordinator._format_dhw_planning_summary
    """
    pass

def check_abort_conditions(
    self,
    abort_conditions: list[str],
    thermal_debt: float,
    indoor_temp: float,
    target_indoor: float,
) -> tuple[bool, str | None]:
    """Check if DHW abort conditions are triggered.
    
    Moved from coordinator._check_dhw_abort_conditions
    """
    pass
```

**Coordinator becomes:**
```python
# coordinator.py - AFTER refactor
async def _calculate_dhw_recommendation(self, ...):
    return self.dhw_optimizer.calculate_recommendation(
        nibe_data, price_data, weather_data,
        current_dhw_temp, now_time,
        self.engine.climate_detector,
        self.thermal_predictor,
    )
```

#### Step 9.2: Move Power Estimation to `effect_layer.py`

```python
# effect_layer.py - ADD to EffectManager

def estimate_power_consumption(self, nibe_data) -> float:
    """Estimate heat pump power from NIBE state.
    
    Moved from coordinator._estimate_power_consumption
    """
    pass

def estimate_power_from_compressor(
    self,
    compressor_hz: int,
    outdoor_temp: float,
    heat_pump_model=None,
) -> float:
    """Estimate power from compressor frequency.
    
    Moved from coordinator._estimate_power_from_compressor
    """
    pass
```

#### Step 9.3: Move Thermal Status to `thermal_layer.py`

```python
# thermal_layer.py - ADD helper function

def get_thermal_debt_status(thermal_debt: float, dm_thresholds: dict) -> str:
    """Get human-readable thermal debt status.
    
    Moved from coordinator._get_thermal_debt_status
    """
    if thermal_debt < dm_thresholds.get("critical", -1500):
        return "CRITICAL"
    elif thermal_debt < dm_thresholds.get("warning", -700):
        return "WARNING"
    elif thermal_debt < dm_thresholds.get("normal_max", -450):
        return "ELEVATED"
    else:
        return "OK"
```

### Expected Results

| File | Before | After | Change |
|------|--------|-------|--------|
| `coordinator.py` | 2551 | ~2100 | -450 |
| `dhw_optimizer.py` | 1307 | ~1600 | +300 |
| `effect_layer.py` | 518 | ~610 | +90 |
| `thermal_layer.py` | 902 | ~930 | +30 |

### Not Worth Moving

| Method | Lines | Reason |
|--------|-------|--------|
| `_record_learning_observations` | ~60 | Tightly coupled to multiple adapters |
| `_save_thermal_predictor_immediate` | ~40 | HA storage specific |
| `_save_dhw_state_immediate` | ~30 | HA storage specific |

These are HA-specific glue code and don't benefit from extraction.

### Dependencies to Consider

The layers need access to shared data:
1. **ClimateZoneDetector** - Already passed via decision_engine
2. **ThermalPredictor** - Already passed to coordinator
3. **Heat pump model** - Already in coordinator and decision_engine

No new shared state needed - existing architecture supports this.

---

## Phase 10: DHW Layer Sharing Analysis

### Goal
Make DHW optimizer reuse the same layer logic as space heating to ensure consistent behavior.

### Current DHW Optimizer Logic vs Existing Layers

| DHW Logic | Lines | Existing Layer | Can Reuse? | Notes |
|-----------|-------|----------------|------------|-------|
| DM threshold check for blocking | ~30 | `thermal_layer.should_block_dhw()` | ✅ YES | Already exists! |
| Climate-aware DM thresholds | ~20 | `climate_zones.get_expected_dm_range()` | ✅ YES | Already injected |
| Price classification | ~10 | `price_layer.get_current_classification()` | ✅ YES | Already used |
| Find next cheap period | ~30 | `price_layer.get_next_cheap_period()` | ✅ YES | Already exists |
| Window-based scheduling | ~150 | NEW in `price_layer.py` | 🔄 EXTRACT | DHW has better impl |
| DM recovery estimation | ~50 | NEW in `thermal_layer.py` | 🔄 EXTRACT | Useful for space too |
| Demand period detection | ~60 | DHW-specific | ❌ NO | DHW-only concept |
| Legionella detection | ~100 | DHW-specific | ❌ NO | DHW-only concept |

### Layers DHW Should Import

```python
# dhw_optimizer.py - Current (duplicated logic)

# BEFORE: Inline DM check
if thermal_debt_dm <= dm_block_threshold:
    # Block DHW...

# AFTER: Use shared thermal layer
from .thermal_layer import EmergencyLayer

emergency_layer = EmergencyLayer(climate_detector)
if emergency_layer.should_block_dhw(thermal_debt_dm, outdoor_temp):
    # Block DHW...
```

### Shared Logic to Extract FROM dhw_optimizer TO layers

#### 1. Window-Based Scheduling → `price_layer.py`

DHW's `find_cheapest_dhw_window()` is more sophisticated than what space heating uses.
Extract to price_layer for both to use:

```python
# price_layer.py - ADD shared method

def find_cheapest_window(
    self,
    current_time: datetime,
    lookahead_hours: int,
    duration_minutes: int,
    price_periods: list,
) -> CheapestWindowResult | None:
    """Find cheapest continuous window for heating.
    
    Used by:
    - DHW optimizer: Find cheapest 45-min window for DHW heating
    - Space heating: Find cheapest 2-4h window for pre-heating
    
    Args:
        current_time: Current timestamp
        lookahead_hours: How far ahead to search
        duration_minutes: Required heating duration
        price_periods: List of QuarterPeriod objects
        
    Returns:
        CheapestWindowResult with start_time, end_time, avg_price, etc.
    """
    pass
```

**Benefits:**
- Space heating can use same logic for pre-heating before expensive periods
- Single algorithm, tested once, used by both

#### 2. DM Recovery Time Estimation → `thermal_layer.py`

DHW's `_estimate_dm_recovery_time()` is useful for space heating too:

```python
# thermal_layer.py - ADD shared method

def estimate_dm_recovery_time(
    current_dm: float,
    target_dm: float,
    outdoor_temp: float,
) -> float:
    """Estimate hours until DM recovers to target level.
    
    Used by:
    - DHW optimizer: Estimate when DM will be safe for DHW
    - Space heating: Estimate recovery time for user display
    
    Returns:
        Hours until recovery (0.5 - 12.0 range)
    """
    pass
```

### Implementation Steps

#### Step 10.1: Use Existing Layers in DHW

```python
# dhw_optimizer.py - REFACTOR

class IntelligentDHWScheduler:
    def __init__(
        self,
        demand_periods: list[DHWDemandPeriod] | None = None,
        climate_detector=None,
        user_target_temp: float | None = None,
        emergency_layer: EmergencyLayer | None = None,  # ADD: Shared thermal layer
        price_analyzer: PriceAnalyzer | None = None,    # ADD: Shared price layer
    ):
        self.emergency_layer = emergency_layer
        self.price = price_analyzer
        ...

    def should_start_dhw(self, ...):
        # BEFORE: Inline DM check
        if thermal_debt_dm <= dm_block_threshold:
            ...
        
        # AFTER: Use shared layer
        if self.emergency_layer and self.emergency_layer.should_block_dhw(
            thermal_debt_dm, outdoor_temp
        ):
            ...
```

#### Step 10.2: Extract Window Logic to price_layer

Move `find_cheapest_dhw_window()` → `price_layer.find_cheapest_window()`

#### Step 10.3: Extract DM Recovery to thermal_layer

Move `_estimate_dm_recovery_time()` → `thermal_layer.estimate_dm_recovery_time()`

### Expected Layer Reuse Matrix

| Layer | Space Heating | DHW | Notes |
|-------|---------------|-----|-------|
| `climate_zones.get_expected_dm_range()` | ✅ | ✅ | DM thresholds |
| `thermal_layer.EmergencyLayer` | ✅ | ✅ NEW | DM safety check |
| `thermal_layer.should_block_dhw()` | ❌ | ✅ | DHW-specific but uses shared DM logic |
| `thermal_layer.estimate_dm_recovery_time()` | ✅ NEW | ✅ | Recovery estimation |
| `price_layer.get_current_classification()` | ✅ | ✅ | Price classification |
| `price_layer.get_next_cheap_period()` | ✅ | ✅ | Next cheap period |
| `price_layer.find_cheapest_window()` | ✅ NEW | ✅ | Window-based scheduling |
| `ProactiveLayer` | ✅ | ❌ | Space heating only |
| `ComfortLayer` | ✅ | ❌ | Space heating only |

### What STAYS DHW-Specific

| Logic | Reason |
|-------|--------|
| `DHWDemandPeriod` | User-defined shower times |
| `_check_upcoming_demand_period()` | DHW-only concept |
| Legionella detection | DHW hygiene, no space heating equivalent |
| BT7 temperature tracking | DHW sensor, not space heating |
| Two-tier safety strategy | DHW-specific (heat to 30°C quickly, full later) |

### Benefits of Shared Layers

1. **Consistent DM safety logic** - Same thresholds, same behavior
2. **Single source of truth** - Price classification used identically
3. **Better testing** - Test layer once, works for both
4. **Easier maintenance** - Fix bug once, fixed everywhere
5. **Reduced code** - DHW optimizer shrinks by ~150 lines

---

## Phase 11: Price Forecast Sharing Analysis

### Current State: Duplicated Price Logic

Both space heating and DHW have sophisticated price forecast logic that should be unified:

| Logic | Space Heating (`price_layer.py`) | DHW (`dhw_optimizer.py`) | Can Share? |
|-------|----------------------------------|--------------------------|------------|
| Forward price analysis | `evaluate_layer()` ~200 lines | `find_cheapest_dhw_window()` ~150 lines | ✅ YES |
| Horizon calculation | `PRICE_FORECAST_BASE_HORIZON * thermal_mass` | `get_lookahead_hours()` | ✅ YES |
| Cheap period detection | `get_next_cheap_period()` | Inline in window search | ✅ YES |
| Expensive period detection | `get_next_expensive_period()` | Not used | ✅ ADD |
| Sliding window algorithm | Not implemented | `find_cheapest_dhw_window()` | 🔄 EXTRACT |
| Volatility detection | `is_volatile` logic | Not used | ✅ ADD |
| Peak cluster detection | `in_peak_cluster` logic | Not used | ✅ ADD |

### Price Layer Methods to Add/Extract

#### 1. Find Cheapest Continuous Window (FROM DHW)

DHW has a superior sliding window algorithm that finds the absolute cheapest continuous period:

```python
# price_layer.py - ADD shared method

@dataclass
class CheapestWindowResult:
    """Result of cheapest window search."""
    start_time: datetime
    end_time: datetime
    quarters: list[int]
    avg_price: float
    hours_until: float
    savings_vs_current: float  # % cheaper than current price

def find_cheapest_window(
    self,
    current_time: datetime,
    price_periods: list,  # QuarterPeriod list
    duration_minutes: int,
    lookahead_hours: int,
) -> CheapestWindowResult | None:
    """Find cheapest continuous window for heating.
    
    Sliding window algorithm from DHW optimizer, generalized for both:
    - DHW: 45-min window (3 quarters)
    - Space heating pre-heat: 2-4h window (8-16 quarters)
    
    Args:
        current_time: Search start
        price_periods: Combined today + tomorrow prices
        duration_minutes: Required heating duration
        lookahead_hours: How far ahead to search
        
    Returns:
        CheapestWindowResult or None if insufficient data
    """
    pass
```

**Used by:**
- DHW: Find 45-min window for DHW heating
- Space heating: Find optimal pre-heating window before expensive period

#### 2. Forecast Upcoming Periods (GENERALIZE)

Current `evaluate_layer()` has inline forecast logic. Extract to reusable method:

```python
# price_layer.py - ADD shared method

@dataclass
class PriceForecast:
    """Forward-looking price analysis result."""
    
    # Cheap period info
    next_cheap_quarters_away: int | None
    cheap_period_duration: int  # quarters
    cheap_price_ratio: float  # vs current
    
    # Expensive period info
    next_expensive_quarters_away: int | None
    expensive_period_duration: int  # quarters
    expensive_price_ratio: float  # vs current
    
    # Volatility
    is_volatile: bool
    current_run_length: int  # quarters
    volatile_reason: str
    
    # Cluster detection
    in_peak_cluster: bool  # EXPENSIVE sandwiched between PEAKs

def get_price_forecast(
    self,
    current_quarter: int,
    price_data: PriceData,
    lookahead_hours: float = 4.0,
) -> PriceForecast:
    """Analyze upcoming price periods.
    
    Consolidated forecast logic used by:
    - Space heating: Pre-heat before expensive, reduce before cheap
    - DHW: Wait for cheap if adequate, heat now if urgent
    
    Args:
        current_quarter: Current 15-min period (0-95)
        price_data: Today + tomorrow prices
        lookahead_hours: How far ahead to analyze
        
    Returns:
        PriceForecast with detailed upcoming period info
    """
    pass
```

**Used by:**
- `price_layer.evaluate_layer()` - Space heating decisions
- `dhw_optimizer.should_start_dhw()` - DHW decisions
- `coordinator._calculate_dhw_recommendation()` - User-facing summary

#### 3. Dynamic Lookahead Calculation (UNIFY)

Both have similar but different lookahead logic:

```python
# price_layer.py - ADD shared method

def calculate_lookahead_hours(
    self,
    heating_type: str,  # "space" or "dhw"
    thermal_mass: float = 1.0,  # For space heating
    next_demand_hours: float | None = None,  # For DHW
) -> float:
    """Calculate dynamic lookahead horizon.
    
    Space heating: Base hours × thermal_mass (2-8h range)
    DHW: Hours until next demand period (capped at 24h)
    
    Args:
        heating_type: "space" or "dhw"
        thermal_mass: Building thermal mass multiplier (0.5-2.0)
        next_demand_hours: Hours until next DHW demand period
        
    Returns:
        Lookahead hours (adaptive based on context)
    """
    if heating_type == "space":
        return PRICE_FORECAST_BASE_HORIZON * thermal_mass
    else:  # dhw
        if next_demand_hours is not None:
            return max(1.0, min(next_demand_hours, 24.0))
        return 24.0  # Default full day lookahead
```

### Implementation Steps

#### Step 11.1: Add `PriceForecast` Dataclass to `price_layer.py`

Extract forecast data structure for reuse.

#### Step 11.2: Add `get_price_forecast()` Method

Consolidate the forward-looking analysis from `evaluate_layer()`.

#### Step 11.3: Add `find_cheapest_window()` Method

Move from `dhw_optimizer.find_cheapest_dhw_window()` with generalized parameters.

#### Step 11.4: Add `calculate_lookahead_hours()` Method

Unify space heating and DHW lookahead calculation.

#### Step 11.5: Refactor DHW Optimizer to Use Shared Methods

```python
# dhw_optimizer.py - AFTER refactor

def should_start_dhw(self, ...):
    # BEFORE: Inline window search
    optimal_window = self.find_cheapest_dhw_window(...)
    
    # AFTER: Use shared price layer
    lookahead = self.price.calculate_lookahead_hours(
        "dhw", next_demand_hours=next_demand["hours_until"]
    )
    optimal_window = self.price.find_cheapest_window(
        current_time, price_periods, 
        duration_minutes=45, lookahead_hours=lookahead
    )
```

#### Step 11.6: Refactor `price_layer.evaluate_layer()` to Use Shared Methods

```python
# price_layer.py - AFTER refactor

def evaluate_layer(self, ...):
    # BEFORE: Inline forecast logic (~100 lines)
    
    # AFTER: Use shared method
    forecast = self.get_price_forecast(current_quarter, price_data, forecast_hours)
    
    if classification == QuarterClassification.CHEAP:
        if forecast.next_expensive_quarters_away and forecast.expensive_period_duration >= 3:
            # Pre-heat decision using shared forecast data
            ...
```

### Updated Layer Reuse Matrix

| Layer Method | Space Heating | DHW | Notes |
|--------------|---------------|-----|-------|
| `get_current_classification()` | ✅ | ✅ | Already shared |
| `get_next_cheap_period()` | ✅ | ✅ | Already shared |
| `get_next_expensive_period()` | ✅ | ✅ NEW | Add to DHW |
| `get_price_forecast()` | ✅ NEW | ✅ NEW | Unified forecast |
| `find_cheapest_window()` | ✅ NEW | ✅ | From DHW → shared |
| `calculate_lookahead_hours()` | ✅ NEW | ✅ NEW | Unified lookahead |
| Volatility detection | ✅ | ✅ NEW | Add to DHW |
| Peak cluster detection | ✅ | ✅ NEW | Add to DHW |

### Benefits

1. **Single forecast algorithm** - Same logic, same edge case handling
2. **DHW gets volatility awareness** - Won't heat during brief cheap spikes
3. **DHW gets peak cluster detection** - Respects EXPENSIVE sandwiched between PEAKs
4. **Space heating gets window search** - Can find optimal pre-heat windows
5. **Reduced code** - DHW loses ~150 lines, price_layer gains ~100 (net: -50)
6. **Better user display** - Coordinator can show consistent forecast info

### Risk Mitigation

- **Don't break existing space heating** - Keep `evaluate_layer()` working first
- **Add new methods, then migrate** - Don't remove old code until new code works
- **Test with real price data** - GE-Spot edge cases (missing data, extreme prices)

---

