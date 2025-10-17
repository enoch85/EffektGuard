# Climate-Aware DHW Threshold Fix

**Date:** October 17, 2025  
**Status:** ✅ Complete

## Issues Fixed

### 1. Translation Placeholder Mismatch

**Error:**
```
Logger: homeassistant.helpers.translation
Validation of translation placeholders for localized (sv) string 
component.effektguard.config.step.gespot.description failed: 
({'gespot_count'} != {'detection_info'})
```

**Root Cause:**
The Swedish translation (`sv.json`) used `{gespot_count}` placeholder while the English translation (`en.json`) used `{detection_info}`.

**Fix:**
Updated Swedish translation to match English translation:
- Changed `{gespot_count}` → `{detection_info}`
- Updated description text to match English format
- Added recommendation for `current_price` sensor

**File:** `custom_components/effektguard/translations/sv.json`

---

### 2. Incorrect "Critical Thermal Debt" DHW Blocking

**Issue:**
DHW recommendation showed:
```
Block DHW - Critical thermal debt (DM: -254)
```

**Problem:**
- DM -254 was marked as "critical" using fixed threshold of -240
- However, for **Moderate Cold zone (Malmö)** at **9°C outdoor temp**:
  - Normal range: -120 to -320
  - Warning threshold: -320
  - DM -254 is **WITHIN NORMAL RANGE**, not critical!

**Root Cause:**
DHW optimizer (`dhw_optimizer.py`) used hardcoded thresholds:
```python
DM_DHW_BLOCK = -240  # Fixed threshold, no climate awareness
DM_DHW_ABORT = -400  # Fixed threshold
```

This failed to account for:
1. **Climate zones** - Different regions have different normal DM ranges
2. **Current temperature** - Warmer weather = shallower DM expected

---

## Solution: Climate-Aware DHW Thresholds

### Architecture Changes

**1. DHW Optimizer Now Climate-Aware**

Updated `IntelligentDHWScheduler` class:
- Added `climate_detector` parameter (receives `ClimateZoneDetector` instance)
- Calculates dynamic thresholds based on:
  - Current outdoor temperature
  - Climate zone (from latitude)
  - Expected DM range for conditions

**2. Dynamic Threshold Calculation**

```python
if self.climate_detector:
    dm_thresholds = self.climate_detector.get_expected_dm_range(outdoor_temp)
    dm_block_threshold = dm_thresholds["warning"]  # Use warning threshold
    dm_abort_threshold = dm_thresholds["warning"] - 80  # 80 DM buffer
else:
    # Fallback to fixed thresholds if detector unavailable
    dm_block_threshold = self.DM_DHW_BLOCK_FALLBACK  # -240
    dm_abort_threshold = self.DM_DHW_ABORT_FALLBACK  # -400
```

**3. Context-Aware Logging**

```python
_LOGGER.debug(
    "DHW DM thresholds for %.1f°C (zone: %s): block=%.0f, abort=%.0f",
    outdoor_temp,
    self.climate_detector.zone_info.name,
    dm_block_threshold,
    dm_abort_threshold,
)
```

**4. Coordinator Integration**

```python
# Pass climate detector from decision engine to DHW optimizer
self.dhw_optimizer = IntelligentDHWScheduler(
    demand_periods=demand_periods,
    climate_detector=decision_engine.climate_detector,
)
```

**5. Improved User Message**

Old:
```
Block DHW - Critical thermal debt (DM: -254)
```

New:
```
Block DHW - Thermal debt warning (DM: -254, zone: Moderate Cold)
```

---

## Example Behavior

### Malmö (Moderate Cold, 55.6°N) at 9°C Outdoor

**Before Fix:**
- DM -254 → Blocked (fixed threshold -240)
- Message: "Critical thermal debt"
- **INCORRECT** - Too aggressive for warm conditions

**After Fix:**
- Normal range: -120 to -320
- Warning threshold: -320
- DM -254 → **ALLOWED** (within normal)
- Only blocks if DM < -320
- **CORRECT** - Climate-aware decision

### Stockholm (Cold, 59.3°N) at -10°C Outdoor

**Before Fix:**
- DM -450 → Blocked (fixed threshold -240)
- Message: "Critical thermal debt"

**After Fix:**
- Normal range: -450 to -700
- Warning threshold: -700
- DM -450 → **ALLOWED** (within normal for -10°C)
- Only blocks if DM < -700
- **CORRECT** - Adapted to colder climate + temperature

### Kiruna (Extreme Cold, 67.8°N) at -30°C Outdoor

**Before Fix:**
- DM -800 → Blocked (fixed threshold -240)
- Message: "Critical thermal debt"

**After Fix:**
- Normal range: -800 to -1200
- Warning threshold: -1200
- DM -800 → **ALLOWED** (normal for Arctic winter)
- Only blocks if DM < -1200
- **CORRECT** - Handles extreme cold properly

---

## Files Modified

1. **`custom_components/effektguard/translations/sv.json`**
   - Fixed placeholder from `{gespot_count}` → `{detection_info}`
   - Updated description text

2. **`custom_components/effektguard/optimization/dhw_optimizer.py`**
   - Added `climate_detector` parameter to `__init__`
   - Renamed fixed thresholds to `*_FALLBACK` versions
   - Updated `should_start_dhw()` to calculate dynamic thresholds
   - All abort conditions now use climate-aware thresholds
   - Added debug logging for threshold decisions

3. **`custom_components/effektguard/coordinator.py`**
   - Pass `decision_engine.climate_detector` to DHW optimizer
   - Updated DHW blocking message to show zone context

---

## Safety Validation

### Absolute Maximum Still Enforced

Climate-aware thresholds **NEVER exceed** absolute maximum:
```python
# From climate_zones.py
DM_ABSOLUTE_MAXIMUM: Final = -1500  # Swedish forum validation
```

All dynamic thresholds are capped:
```python
normal_min = max(normal_min, DM_ABSOLUTE_MAXIMUM + 100)  # -1400 max
normal_max = max(normal_max, DM_ABSOLUTE_MAXIMUM + 50)   # -1450 max
warning = max(warning, DM_ABSOLUTE_MAXIMUM + 50)         # -1450 max
```

### Fallback Protection

If `climate_detector` unavailable:
```python
dm_block_threshold = self.DM_DHW_BLOCK_FALLBACK  # -240 (safe default)
dm_abort_threshold = self.DM_DHW_ABORT_FALLBACK  # -400 (safe default)
```

---

## Testing Verification

### Expected Behavior After Fix

**Warm Weather (9°C outdoor, Moderate Cold zone):**
```
DM -254 → DHW allowed (within normal -120 to -320)
Message: "DHW OK - Temperature adequate (45.0°C)"
```

**Cold Weather (-10°C outdoor, Cold zone):**
```
DM -450 → DHW allowed (within normal -450 to -700)
Message: "DHW OK - Temperature adequate (45.0°C)"
```

**Extreme Cold (-30°C outdoor, Extreme Cold zone):**
```
DM -800 → DHW allowed (within normal -800 to -1200)
Message: "DHW OK - Temperature adequate (45.0°C)"
```

**True Critical (any zone):**
```
DM < warning_threshold → DHW blocked
Message: "Block DHW - Thermal debt warning (DM: -350, zone: Moderate Cold)"
```

---

## Documentation References

- **Architecture:** `architecture/02_emergency_thermal_debt.md`
- **Climate Zones:** `docs/CLIMATE_ZONES.md`
- **DHW Optimization:** `docs/DHW_OPTIMIZATION.md`
- **Forum Research:** `IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md`
- **Swedish NIBE:** `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md`

---

## Benefits

1. **Accurate for All Climates:**
   - Arctic to Mediterranean
   - Automatically adapts from latitude

2. **Temperature-Aware:**
   - Adjusts for current conditions
   - Warmer weather = shallower DM expected

3. **Less False Alarms:**
   - DM -254 in Malmö at 9°C = normal, not critical
   - Reduces unnecessary DHW blocking

4. **Maintains Safety:**
   - Absolute maximum -1500 always enforced
   - Fallback to conservative thresholds if detector unavailable

5. **Better User Experience:**
   - Messages show climate zone context
   - Clear explanation of why DHW blocked

---

## Commits

```bash
git add custom_components/effektguard/translations/sv.json
git add custom_components/effektguard/optimization/dhw_optimizer.py
git add custom_components/effektguard/coordinator.py
git add COMPLETED/CLIMATE_AWARE_DHW_FIX.md

git commit -m "Fix DHW optimizer with climate-aware thresholds

Issues fixed:
1. Translation placeholder mismatch (sv.json: gespot_count → detection_info)
2. Incorrect 'critical' DHW blocking at DM -254 (normal for Malmö at 9°C)

Changes:
- DHW optimizer now receives ClimateZoneDetector for dynamic thresholds
- Thresholds adapt to climate zone + current outdoor temperature
- Example: Malmö 9°C: normal -120 to -320, warning -320 (was fixed -240)
- DHW blocking message now shows zone context
- Maintains safety: absolute maximum -1500 always enforced

Tests: Syntax verified, Black formatted
References: Forum_Summary.md, Swedish_NIBE_Forum_Findings.md"
```

---

## Next Steps

1. **Test in Production:**
   - Monitor DHW recommendations
   - Verify climate-aware messages
   - Check for false alarms

2. **User Feedback:**
   - Collect reports from different climate zones
   - Validate threshold accuracy

3. **Documentation:**
   - Update user guide with climate zone explanation
   - Add troubleshooting for DHW blocking

---

**Status:** ✅ Complete - Ready for testing
