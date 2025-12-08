# Layer Refactor Cleanup Plan

**Created:** December 8, 2025  
**Branch:** `layer-refactor`  
**PR:** #11  

## Overview

The layer refactor successfully moved ~1,400 lines from `decision_engine.py` to dedicated layer files. However, cleanup is needed to remove dead code, eliminate wrappers, and fix test issues.

---

## Phase 1: Remove Dead Code from decision_engine.py

### Problem
Three methods are duplicated between `decision_engine.py` and `thermal_layer.py`. The decision_engine versions are **dead code** - no longer called by production code, but tests still reference them.

### Methods to Delete

| Method | Lines | Duplicate In |
|--------|-------|--------------|
| `_apply_thermal_recovery_damping()` | 471-610 (~140 lines) | `thermal_layer.py:603` |
| `_calculate_expected_dm_for_temperature()` | 955-999 (~45 lines) | `thermal_layer.py:974` |
| `_get_thermal_mass_adjusted_thresholds()` | 1000-1066 (~67 lines) | `thermal_layer.py:564` |

**Total: ~252 lines to delete**

### Steps

1. **Update tests first** (before deleting methods):
   - `tests/unit/optimization/test_overshoot_aware_damping.py` - 13 calls to `engine._apply_thermal_recovery_damping()`
   - `tests/unit/optimization/test_decision_engine_climate_integration.py` - 20+ calls to `engine._calculate_expected_dm_for_temperature()`

2. **Migrate tests to use layer classes directly:**
   ```python
   # BEFORE: Testing dead code on engine
   damped_offset, reason = engine._apply_thermal_recovery_damping(...)
   
   # AFTER: Testing actual production code on EmergencyLayer
   from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer
   emergency_layer = EmergencyLayer(climate_detector, ...)
   damped_offset, reason = emergency_layer._apply_thermal_recovery_damping(...)
   ```

3. **Delete dead methods** from `decision_engine.py`

4. **Run full test suite** to verify

**Effort:** 2h | **Risk:** Medium (test updates required)

---

## Phase 2: Simplify Layer Wrappers

### Problem
`decision_engine.py` has 8 wrapper methods that just delegate to layer classes and convert dataclass types. This adds ~150 lines of boilerplate.

### Current Pattern (Wasteful)
```python
def _emergency_layer(self, nibe_state, weather_data=None, price_data=None) -> LayerDecision:
    """Emergency layer wrapper."""
    # Delegate to EmergencyLayer
    emergency_decision = self.emergency_layer.evaluate_layer(
        nibe_state=nibe_state,
        weather_data=weather_data,
        price_data=price_data,
        target_temp=self.target_temp,
        tolerance_range=self.tolerance_range,
    )
    # Convert EmergencyLayerDecision to LayerDecision
    return LayerDecision(
        name=emergency_decision.name,
        offset=emergency_decision.offset,
        weight=emergency_decision.weight,
        reason=emergency_decision.reason,
    )
```

### Target Pattern (Clean)

**Option A: Layers return `LayerDecision` directly (Recommended)**

Update all layer `evaluate_layer()` methods to return `LayerDecision` instead of `*LayerDecision`:

```python
# thermal_layer.py
from ..decision_engine import LayerDecision  # or move LayerDecision to shared module

def evaluate_layer(self, ...) -> LayerDecision:
    # ... logic ...
    return LayerDecision(
        name="Thermal Recovery T2",
        offset=damped_offset,
        weight=DM_CRITICAL_T2_WEIGHT,
        reason=reason,
    )
```

Then in decision_engine:
```python
layers = [
    self._safety_layer(nibe_state),
    self.emergency_layer.evaluate_layer(nibe_state, weather_data, price_data, ...),
    self.proactive_layer.evaluate_layer(nibe_state, weather_data, ...),
    # ... direct calls, no wrappers
]
```

**Option B: Keep wrappers but make them one-liners**

If we need to keep the wrapper methods for parameter injection:
```python
def _emergency_layer(self, nibe_state, weather_data=None, price_data=None) -> LayerDecision:
    return self.emergency_layer.evaluate_layer(
        nibe_state, weather_data, price_data, self.target_temp, self.tolerance_range
    ).to_layer_decision()  # Add conversion method
```

### Decision
**Choose Option A** - Move `LayerDecision` to a shared location and have all layers return it.

### Steps

1. **Create shared types module:**
   ```
   optimization/types.py  # Contains LayerDecision, shared dataclasses
   ```

2. **Move `LayerDecision` from decision_engine.py to types.py**

3. **Update layer files to import from types.py and return `LayerDecision`**

4. **Remove `EmergencyLayerDecision`, `ProactiveLayerDecision`, etc.** - keep only `LayerDecision`
   - BUT preserve diagnostic fields by adding them to `LayerDecision`

5. **Update `calculate_decision()` to call layer methods directly**

6. **Delete wrapper methods**

**Effort:** 3h | **Risk:** Medium

---

## Phase 3: Fix Hardcoded Test Values

### Problem
Tests use magic numbers instead of constants from `const.py`, violating project rules.

### Files to Fix

**`tests/unit/coordinator/test_power_measurement_fallback.py`:**
```python
# Line 41-42, 50-51, 59-60, 80-81, 90-91, etc.
# BEFORE:
assert power >= 1.5
assert power <= 2.5

# AFTER:
from custom_components.effektguard.const import (
    COMPRESSOR_POWER_MIN_KW,
    COMPRESSOR_POWER_MAX_KW,
)
# Use constants in assertions
assert power >= COMPRESSOR_POWER_MIN_KW  # 1.5
```

### Steps

1. Add missing constants to `const.py` if needed
2. Update test imports
3. Replace hardcoded values with constants
4. Run tests to verify

**Effort:** 1h | **Risk:** Low

---

## Phase 4: Document Logic Changes

### Problem
The refactor included actual logic changes that should be documented as features, not just "organization."

### Logic Changes to Document

1. **New `VERY_CHEAP` Price Classification**
   - Added `QuarterClassification.VERY_CHEAP` for bottom 10% prices
   - Offset: +4.0°C (aggressive pre-heating during exceptional prices)
   - Changed `PRICE_OFFSET_CHEAP` from 3.0 → 1.5

2. **Percentile-Based Price Classification**
   - Added explicit percentile thresholds instead of hardcoded values
   - More robust to price distribution changes

### Steps

1. Update PR description with "Features" section
2. Add to CHANGELOG.md (if exists) or release notes

**Effort:** 30min | **Risk:** None

---

## Execution Order

```
Phase 1: Remove Dead Code        [2h]  ← Must do first (unblocks Phase 2)
    ↓
Phase 2: Simplify Wrappers       [3h]  ← Largest cleanup
    ↓
Phase 3: Fix Test Values         [1h]  ← Can be parallel with Phase 2
    ↓
Phase 4: Document Changes        [30m] ← Final step
```

**Total Effort:** ~6.5 hours

---

## Verification Checklist

After each phase:
- [ ] `pytest tests/ -v` passes (1066+ tests)
- [ ] `black custom_components/effektguard/ --check --line-length 100` passes
- [ ] No grep matches for deleted method names in production code
- [ ] `python3 -c "from custom_components.effektguard.optimization.decision_engine import DecisionEngine"` succeeds

---

## Final State

| Metric | Before Cleanup | After Cleanup |
|--------|----------------|---------------|
| `decision_engine.py` lines | 1476 | ~1100 |
| Dead methods | 3 | 0 |
| Wrapper methods | 8 | 0 (or 1-liners) |
| Duplicated code | ~252 lines | 0 |
| Test hardcoded values | Many | 0 |

---

## Files Changed Summary

### Modified
- `custom_components/effektguard/optimization/decision_engine.py` - Remove dead code, simplify wrappers
- `custom_components/effektguard/optimization/thermal_layer.py` - Return `LayerDecision`
- `custom_components/effektguard/optimization/effect_layer.py` - Return `LayerDecision`
- `custom_components/effektguard/optimization/price_layer.py` - Return `LayerDecision`
- `custom_components/effektguard/optimization/weather_layer.py` - Return `LayerDecision`
- `custom_components/effektguard/optimization/prediction_layer.py` - Return `LayerDecision`
- `custom_components/effektguard/optimization/comfort_layer.py` - Return `LayerDecision`
- `tests/unit/optimization/test_overshoot_aware_damping.py` - Use EmergencyLayer directly
- `tests/unit/optimization/test_decision_engine_climate_integration.py` - Use layer classes
- `tests/unit/coordinator/test_power_measurement_fallback.py` - Use constants

### New
- `custom_components/effektguard/optimization/types.py` - Shared `LayerDecision` dataclass

### Deleted
- None (methods deleted from existing files)

---

## Risk Mitigation

1. **Phase 1 first** - Fix tests before deleting methods (avoids broken tests)
2. **One phase at a time** - Don't combine phases in single commit
3. **Run tests after each change** - Catch regressions immediately
4. **Keep layer-specific fields** - `LayerDecision` should have optional diagnostic fields
5. **Git stash before starting** - Easy rollback if needed
