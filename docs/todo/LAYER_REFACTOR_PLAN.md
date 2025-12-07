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

