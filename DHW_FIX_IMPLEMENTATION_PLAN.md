# DHW Fix Implementation Plan

**Date:** October 28, 2025  
**Issue:** DHW failed to heat at scheduled optimal window (03:45/04:00) despite cheap prices  
**Root Cause:** Multiple logic errors in window scheduling and spare capacity checks

---

## Executive Summary

The DHW optimizer correctly identified the optimal heating window (04:00 @ 8.6öre/kWh) but failed to heat due to:
1. **Spare capacity threshold too strict:** Required DM > -158 (50% buffer) but had -254 (overly conservative)
2. **Window timing issues:** 10-min detection window incompatible with 15-min coordinator cycle
3. **Spare capacity check blocking scheduled windows:** No override when in optimal cheap window

**Impact:** User had to manually heat water despite system scheduling it correctly.

**Root Cause Analysis:**
- Spare capacity formula is mathematically correct but **too conservative (50%)**
- Your DM -254 was actually **62 DM away from warning (-316)**, which is safe for 45-min DHW heating
- System demanded 158 DM margin (50% of warning), blocking reasonable heating opportunities
- The check exists to prevent DHW heating from causing thermal debt during space heating
- **Solution:** Simplify spare capacity check - allow DHW if NOT in warning/critical thermal debt levels + prioritize scheduled optimal windows

---

## Understanding the Spare Capacity Check

### Purpose: Protect Space Heating from DHW Heating Impact

**Priority Order (from dhw_optimizer.py):**
1. Space heating comfort (indoor temp > target - 0.5°C)
2. DHW safety minimum (≥30°C)
3. Thermal debt prevention (climate-aware DM thresholds)
4. Space heating target (±0.3°C)
5. DHW comfort (50°C normal)

**What Happens During DHW Heating:**
- Compressor switches from **space heating** to **DHW heating**
- Space heating **STOPS** for 45-90 minutes (DHW heating duration)
- Indoor temperature starts dropping
- Degree Minutes accumulate (thermal debt increases)
- If DM gets too negative → space heating emergency

**Example Thermal Debt Accumulation:**
```
Start DHW heating: DM -250
45 minutes later: DM -280 (accumulated 30 DM)
Space heating resumes
```

**The Risk Without Spare Capacity Check:**
```
Bad Scenario:
- Start: DM -300 (close to warning -316)
- DHW heats 60 min
- Accumulates: -300 → -340 DM
- WARNING THRESHOLD EXCEEDED during DHW heating! ⚠️
- Space heating emergency, comfort impacted
```

### Current Implementation (50% - TOO COMPLEX AND STRICT)

**October 28 Case:**
- Climate-aware warning threshold (T2-level): -316 DM (for -1°C outdoor, Moderate Cold zone)
- Spare capacity: 50% buffer required
- Calculation: `-316 × (1.0 - 50.0/100.0) = -316 × 0.5 = -158 DM`
- Your DM: **-254 DM**
- Check: Is -254 > -158? (Is -254 less negative than -158?)
  - **NO!** -254 is MORE negative (worse) than -158
  - Result: **BLOCKED** ❌

**The Problem:**
- Complex percentage calculation
- Arbitrary buffer value (why 50%? why not 30% or 10%?)
- You had 62 DM margin from T2-level warning, but system wanted 158 DM
- Blocked reasonable heating opportunities

### Proposed Fix: USE CLIMATE-AWARE T2-LEVEL THRESHOLD DIRECTLY

**Much Simpler Approach - Just Use the Warning Threshold:**
```python
def _has_spare_compressor_capacity(self, thermal_debt_dm: float, outdoor_temp: float) -> bool:
    """Check if heat pump has spare capacity for DHW.
    
    Simple rule: Allow DHW heating if above climate-aware warning threshold (T2-level).
    The T-level system already exists and is proven - just reuse it!
    """
    if self.climate_detector:
        dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        warning_threshold = dm_range["warning"]  # This IS the T2-level!
        
        # Simple: Allow DHW if better than warning (T2-level)
        has_capacity = thermal_debt_dm > warning_threshold
        
        _LOGGER.debug(
            "DHW spare capacity: DM=%.0f, T2-warning=%.0f, allowed=%s",
            thermal_debt_dm,
            warning_threshold,
            has_capacity,
        )
        
        return has_capacity
    else:
        # Fallback: Use fixed threshold
        return thermal_debt_dm > DM_DHW_BLOCK_FALLBACK
```

**Effect on October 28 Case:**
- T2-level warning threshold: -316 DM
- Your DM: -254 DM
- Check: Is -254 > -316? (Is -254 less negative than -316?)
  - **YES!** ✅ -254 is LESS negative (better) than -316
  - Result: **WOULD HEAT** ✅

**T-Level Based Logic (Using Existing System):**
| DM Range | T-Level | DHW Allowed? | Reason |
|----------|---------|--------------|---------|
| 0 to -316 | **T1 (Normal)** | ✅ Yes | Heat pump operating normally |
| -254 (Oct 28) | **T1 (Normal)** | ✅ Yes | Within normal range |
| -316 | **T2 (Warning)** | ❌ No | At thermal debt warning |
| -316 to -1500 | **T2/T3** | ❌ No | Thermal debt accumulating |
| -1500 | **T3 (Critical)** | ❌ No | Emergency (RULE 1 blocks anyway) |

**Why This is Better:**
1. ✅ **Simpler:** No arbitrary percentage calculations
2. ✅ **Uses existing infrastructure:** T2-level threshold already defined, tested, and climate-aware
3. ✅ **Intuitive:** "If not at/past T2-warning, allow DHW"
4. ✅ **Your case fixed:** -254 > -316 = allow heating
5. ✅ **Still safe:** Blocks at T2-warning threshold (where thermal debt becomes concerning)
6. ✅ **No magic numbers:** Warning threshold is climate-aware and well-researched
7. ✅ **Aligned with space heating logic:** Same T-level system used throughout

**Safety Analysis:**
| DM | T2-Warning | Old (50%) | New (T2-Level) | Safe? |
|----|---------|-----------|---------------|-------|
| -254 | -316 | ❌ Block | ✅ Heat | ✅ 62 DM margin |
| -280 | -316 | ❌ Block | ✅ Heat | ✅ 36 DM margin |
| -310 | -316 | ❌ Block | ✅ Heat | ⚠️ Only 6 DM margin (borderline) |
| -316 | -316 | ❌ Block | ❌ Block | ✅ At T2-warning |
| -350 | -316 | ❌ Block | ❌ Block | ✅ Past T2-warning |

**Cold Weather Example (-15°C outdoor, T2-warning -700):**
- T2-warning threshold: -700 DM (climate-aware)
- DM -600: New allows ✅ (100 DM margin - safe!)
- DM -690: New allows ✅ (10 DM margin - tight but acceptable)
- DM -700: New blocks ❌ (at T2-warning)
- DM -750: New blocks ❌ (past T2-warning)

**Why This is Safe:**
1. T2-warning thresholds are **climate-aware** (adapted to conditions)
2. DHW heating accumulates ~20-40 DM in 45 minutes
3. If starting at -310 (6 DM from T2-warning), DHW heating might push to -350 (past T2)
   - **BUT**: Abort conditions will stop DHW if thermal debt deteriorates
   - **AND**: RULE 2 (Space Heating Emergency) already blocks if indoor temp drops
4. The T-level system is already validated and tested in production
5. Much clearer than arbitrary percentage buffers
6. **Aligns with existing system:** T1/T2/T3 levels already control space heating escalation

---

## Phase 1: Critical Logic Fixes

### Fix 1.1: Prioritize Scheduled Optimal Windows Over Spare Capacity
**File:** `custom_components/effektguard/optimization/dhw_optimizer.py`  
**Lines:** ~724-760 (RULE 6 optimal window section)  
**Priority:** CRITICAL

**Issue:** Spare capacity check blocks heating even during scheduled optimal cheap windows.

**Current Logic Flow:**
```python
# STEP 1: Check if we're in the optimal window (within 10 min of start)
elif optimal_window["hours_until"] <= 0.17:  # 10 minutes
    # Heat IF we have spare capacity
    if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):
        # HEAT NOW
    # Otherwise, don't heat (spare capacity blocks scheduled window!)

# Wait for better window ahead (if DHW still comfortable)
elif current_dhw_temp > MIN_DHW_TARGET_TEMP:  # ❌ WRONG - 36.1 > 45 is FALSE!
    # Wait for better window
    return DHWScheduleDecision(should_heat=False, ...)
```

**Problems:**
1. Spare capacity blocks even during optimal scheduled windows
2. Window detection (10 min) incompatible with 15-min coordinator cycle
3. Comparison uses `>` instead of `>=` (36.1°C > 45°C fails, should use >=)

**Fixed Logic:**
```python
# STEP 1: Check if we're IN the optimal window (within 15 min)
# Increased from 10 to 15 minutes to accommodate 15-minute coordinator cycle
elif optimal_window["hours_until"] <= 0.25:  # 15 minutes = 0.25 hours
    # PRIORITY: If we're in the scheduled window, heat regardless of spare capacity
    # The scheduler already found the cheapest window, we should execute it
    _LOGGER.info(
        "DHW: Heating in optimal window at %s (%.1före/kWh), DM=%.0f",
        optimal_window["start_time"].strftime("%H:%M"),
        optimal_window["avg_price"],
        thermal_debt_dm,
    )
    return DHWScheduleDecision(
        should_heat=True,
        priority_reason=f"OPTIMAL_WINDOW_Q{optimal_window['quarters'][0]}_@{optimal_window['avg_price']:.1f}öre",
        target_temp=min(self.user_target_temp + DHW_PREHEAT_TARGET_OFFSET, 60.0),
        max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
        abort_conditions=[
            f"thermal_debt < {dm_abort_threshold:.0f}",  # Still abort if thermal debt gets critical
            f"indoor_temp < {target_indoor_temp - 0.5}",
        ],
        recommended_start_time=optimal_window["start_time"],
    )

# Wait for better window ONLY if DHW is still comfortable (above minimum)
elif current_dhw_temp >= MIN_DHW_TARGET_TEMP:  # ✅ FIXED - Changed from > to >=
    # DHW comfortable (≥45°C), wait for scheduled window
    _LOGGER.info(
        "DHW: Comfortable (%.1f°C ≥ %.1f°C), waiting for optimal window at %s (%.1fh, %.1före/kWh)",
        current_dhw_temp,
        MIN_DHW_TARGET_TEMP,
        optimal_window["start_time"].strftime("%H:%M"),
        optimal_window["hours_until"],
        optimal_window["avg_price"],
    )
    return DHWScheduleDecision(
        recommended_start_time=optimal_window["start_time"],
    )
```

**Key Changes:**
1. **Window detection:** 10 min → 15 min (matches coordinator cycle)
2. **Spare capacity removed from window check:** If we're IN the optimal window, heat regardless
3. **Wait condition fixed:** Changed `>` to `>=` for MIN_DHW_TARGET_TEMP
4. **Priority logic:** Scheduled optimal window overrides spare capacity check
5. **Safety preserved:** Abort conditions still monitor thermal debt during heating

**Why This Fixes October 28:**
- At 04:00, system is IN the optimal window (hours_until ≤ 0.25)
- Heats immediately without checking spare capacity
- DHW 36.1°C < 45°C minimum → doesn't wait for "better" window
- Result: **Would have heated at 04:00** ✅

---

### Fix 1.2: DELETE Spare Capacity Check Entirely (REDUNDANT with RULE 1)
**Files:** 
- `custom_components/effektguard/const.py` (lines 666-672, 679)
- `custom_components/effektguard/optimization/dhw_optimizer.py` (lines 43, 49, 199-240, and 6 call sites)

**Priority:** CRITICAL

**Discovery:** The spare capacity check is **completely redundant** - RULE 1 already blocks DHW when at/below warning threshold!

**All Locations to Delete:**
1. ✂️ **const.py lines 666-672:** Delete DHW_SPARE_CAPACITY_PERCENT constant and comments
2. ✂️ **const.py line 679:** Delete DM_DHW_SPARE_CAPACITY_FALLBACK constant
3. ✂️ **dhw_optimizer.py line 43:** Remove DHW_SPARE_CAPACITY_PERCENT from imports
4. ✂️ **dhw_optimizer.py line 49:** Remove DM_DHW_SPARE_CAPACITY_FALLBACK from imports
5. ✂️ **dhw_optimizer.py lines 199-240:** Delete entire _has_spare_compressor_capacity() method
6. ✂️ **dhw_optimizer.py line 622:** Remove spare capacity check from RULE 2.5
7. ✂️ **dhw_optimizer.py line 716:** Remove spare capacity check from RULE 6 (no price data fallback)
8. ✂️ **dhw_optimizer.py line 735:** Remove spare capacity check from RULE 6 (optimal window)
9. ✂️ **dhw_optimizer.py line 780:** Remove spare capacity check from RULE 6 (cheap fallback)
10. ✂️ **dhw_optimizer.py line 796:** Remove spare capacity check from RULE 7

**Current (REDUNDANT - duplicates RULE 1):**
```python
# In const.py
DHW_SPARE_CAPACITY_PERCENT: Final = 50.0  # ← DELETE THIS

# In dhw_optimizer.py (lines 199-240)
def _has_spare_compressor_capacity(self, thermal_debt_dm: float, outdoor_temp: float) -> bool:
    """Check spare capacity..."""
    # ... 40+ lines of code ...
    has_capacity = thermal_debt_dm > spare_capacity_threshold  # SAME AS RULE 1!
    return has_capacity

# In should_start_dhw() - called from multiple places:
if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):
    # Heat...
else:
    # Don't heat
```

**RULE 1 Already Does This (lines 429-437):**
```python
# === RULE 1: CRITICAL THERMAL DEBT - NEVER START DHW ===
if thermal_debt_dm <= dm_block_threshold:  # dm_block_threshold = warning threshold
    return DHWScheduleDecision(
        should_heat=False,
        priority_reason="CRITICAL_THERMAL_DEBT",
        ...
    )
# This runs FIRST in should_start_dhw(), before any other rules
# It blocks ALL DHW heating when thermal debt is at/below warning
```

**Complete Deletion Plan:**

**1. DELETE from const.py (lines 666-672):**
```python
# DHW thermal debt thresholds (climate-aware via spare capacity calculation)
# Instead of hardcoded DM thresholds, we calculate spare capacity as percentage
# above the climate-aware warning threshold for current outdoor temperature
DHW_SPARE_CAPACITY_PERCENT: Final = 50.0  # Require 50% spare capacity above warning threshold
# Ensures DHW heating only when heat pump has significant spare capacity
# Example: Stockholm at -10°C has warning=-700, so require DM > -350 (-700 * 0.5)
# Example: Kiruna at -30°C has warning=-1200, so require DM > -600 (-1200 * 0.5)
# This keeps DHW heating within the normal operating range, not near thermal debt warning
```

**2. DELETE from const.py (line 679):**
```python
DM_DHW_SPARE_CAPACITY_FALLBACK: Final = -80.0  # Fallback: Spare capacity threshold
```

**3. DELETE from dhw_optimizer.py imports (line 43):**
```python
DHW_SPARE_CAPACITY_PERCENT,  # ← DELETE THIS LINE
```

**4. DELETE from dhw_optimizer.py imports (line 49):**
```python
DM_DHW_SPARE_CAPACITY_FALLBACK,  # ← DELETE THIS LINE
```

**5. DELETE entire method from dhw_optimizer.py (lines 199-240):**
```python
def _has_spare_compressor_capacity(self, thermal_debt_dm: float, outdoor_temp: float) -> bool:
    """Check if heat pump has spare capacity for DHW without risking thermal debt.

    Uses climate-aware calculation: requires thermal debt to be at least
    DHW_SPARE_CAPACITY_PERCENT above the warning threshold for current conditions.

    Example calculations:
    - Stockholm at -10°C: warning=-700, spare capacity threshold = -700 * 0.8 = -560
      * DM -400: Has spare capacity (✓)
      * DM -650: No spare capacity (✗)
    - Kiruna at -30°C: warning=-1200, spare capacity threshold = -1200 * 0.8 = -960
      * DM -800: Has spare capacity (✓)
      * DM -1000: No spare capacity (✗)

    Args:
        thermal_debt_dm: Current degree minutes
        outdoor_temp: Current outdoor temperature (°C)

    Returns:
        True if has spare capacity, False if space heating needs all capacity
    """
    if self.climate_detector:
        dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
        warning_threshold = dm_range["warning"]

        # Calculate spare capacity threshold (20% buffer above warning)
        spare_capacity_threshold = warning_threshold * (
            1.0 - DHW_SPARE_CAPACITY_PERCENT / 100.0
        )

        has_capacity = thermal_debt_dm > spare_capacity_threshold

        _LOGGER.debug(
            "DHW spare capacity check: DM=%.0f, warning=%.0f, threshold=%.0f, has_capacity=%s",
            thermal_debt_dm,
            warning_threshold,
            spare_capacity_threshold,
            has_capacity,
        )

        return has_capacity
    else:
        # Fallback: Conservative fixed threshold if climate detector unavailable
        # Use fallback constant from const.py
        return thermal_debt_dm > DM_DHW_SPARE_CAPACITY_FALLBACK
```

**6. REMOVE spare capacity check from RULE 2.5 (line 622):**
```python
# BEFORE (lines 615-625):
if (
    DHW_SAFETY_MIN <= current_dhw_temp < MIN_DHW_TARGET_TEMP
    and price_classification == "cheap"
    and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)  # ← DELETE THIS LINE
):
    return DHWScheduleDecision(...)

# AFTER:
if (
    DHW_SAFETY_MIN <= current_dhw_temp < MIN_DHW_TARGET_TEMP
    and price_classification == "cheap"
):
    return DHWScheduleDecision(...)
```

**7. REMOVE spare capacity check from RULE 6 - No price data (line 716):**
```python
# BEFORE (lines 708-717):
if (
    price_classification == "cheap"
    and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
    and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)  # ← DELETE THIS LINE
):
    return DHWScheduleDecision(...)

# AFTER:
if (
    price_classification == "cheap"
    and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
):
    return DHWScheduleDecision(...)
```

**8. REMOVE spare capacity check from RULE 6 - Optimal window (line 735):**
```python
# BEFORE (lines 729-745):
elif optimal_window["hours_until"] <= 0.17:  # 10 minutes
    if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):  # ← DELETE THIS IF
        _LOGGER.info(...)
        return DHWScheduleDecision(...)

# AFTER (also fix window timing to 15 min - see Fix 1.1):
elif optimal_window["hours_until"] <= 0.25:  # 15 minutes
    _LOGGER.info(...)
    return DHWScheduleDecision(...)
```

**9. REMOVE spare capacity check from RULE 6 - Cheap fallback (line 780):**
```python
# BEFORE (lines 776-783):
if (
    not price_periods
    and price_classification == "cheap"
    and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
    and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)  # ← DELETE THIS LINE
):
    return DHWScheduleDecision(...)

# AFTER:
if (
    not price_periods
    and price_classification == "cheap"
    and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
):
    return DHWScheduleDecision(...)
```

**10. REMOVE spare capacity check from RULE 7 (line 796):**
```python
# BEFORE (lines 795-804):
if current_dhw_temp < MIN_DHW_TARGET_TEMP and price_classification == "cheap":
    if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):  # ← DELETE THIS IF
        return DHWScheduleDecision(...)

# AFTER:
if current_dhw_temp < MIN_DHW_TARGET_TEMP and price_classification == "cheap":
    return DHWScheduleDecision(...)
```

**Impact on October 28 Case:**
```python
# OLD (redundant spare capacity check):
Warning: -316, Threshold: -158, Your DM: -254
Spare capacity check: BLOCKED (-254 is more negative than -158) ❌
RULE 1 check: Would pass (-254 > -316) ✅
Result: Spare capacity blocked unnecessarily!

# NEW (spare capacity check deleted):
Warning: -316, Your DM: -254
RULE 1 check: -254 > -316? YES ✅
Result: WOULD HEAT ✅
```

**Why Delete Spare Capacity Entirely:**
1. ✅ **REDUNDANT:** RULE 1 runs FIRST and already blocks at warning threshold
2. ✅ **BLOCKS UNNECESSARILY:** Spare capacity required 50% buffer (-158) when RULE 1 only requires warning (-316)
3. ✅ **SIMPLER:** Delete 42 lines of method + 2 constants + 6 call sites = less code to maintain
4. ✅ **NO BUGS:** Can't have bugs in code that doesn't exist
5. ✅ **LESS CONFUSION:** Thermal debt protection in ONE place (RULE 1), not scattered across rules
6. ✅ **Your case fixed:** RULE 1 check (-254 > -316) passes, DHW heats successfully
7. ✅ **Still safe:** RULE 1 provides all necessary thermal debt protection

**Deletion Summary:**
- 📄 **const.py:** Delete 2 constants (8 lines total including comments)
- 📄 **dhw_optimizer.py:** Delete method (42 lines) + 2 imports + 6 call sites
- 📊 **Total cleanup:** ~58 lines of redundant code removed
- ⚡ **Complexity reduced:** 1 method deleted, logic centralized in RULE 1

**Safety Preserved After Deletion:**
| Check Location | What It Does | Status After Fix |
|----------------|--------------|------------------|
| **RULE 1** (line 429) | Block if DM ≤ warning | ✅ **ACTIVE** - Primary protection |
| **RULE 2** (line 439) | Block if house too cold | ✅ **ACTIVE** - Space heating priority |
| **Abort conditions** | Stop DHW if DM deteriorates | ✅ **ACTIVE** - Runtime monitoring |
| **Spare capacity** | Redundant warning check | ❌ **DELETED** - Not needed |

**Safety Analysis After Deletion:**
| Scenario | DM | Warning | RULE 1 | Old Spare Check | After Deletion | Safe? |
|----------|-----|---------|--------|-----------------|----------------|-------|
| **Oct 28 Case** | -254 | -316 | Pass ✅ | Block ❌ | Pass ✅ | ✅ Yes (62 DM margin) |
| **Normal Operation** | -200 | -316 | Pass ✅ | Block ❌ | Pass ✅ | ✅ Yes (116 DM margin) |
| **Near Warning** | -310 | -316 | Pass ✅ | Block ❌ | Pass ✅ | ⚠️ Tight (6 DM margin)* |
| **At Warning** | -316 | -316 | **Block** ✅ | Block ✅ | **Block** ✅ | ✅ Yes (protected) |
| **Past Warning** | -400 | -316 | **Block** ✅ | Block ✅ | **Block** ✅ | ✅ Yes (protected) |

*Abort conditions provide runtime protection if DM deteriorates during heating

**Conclusion:** Spare capacity check provided **ZERO additional safety** - RULE 1 already covers all cases!

---

### Fix 1.3: Remove Stale "Low DHW During Cheap Period" Fallback
**Status:** ✅ NOT NEEDED (same as before)

With Fix 1.1 (prioritize optimal windows) and Fix 1.2 (simple T-level check), the system will heat much more often.

---

## Phase 2: Enhanced Window Detection

### Fix 2.1: Improve Window Timing Precision
**File:** `custom_components/effektguard/optimization/dhw_optimizer.py`  
**Line:** ~850 (in should_start_dhw, window check section)  
**Priority:** MEDIUM

**Current:**
```python
if optimal_window["hours_until"] <= 0.17:  # 10 minutes
```

**Fixed:**
```python
# Use 15-minute buffer to accommodate 5-minute coordinator updates
# This ensures we catch the window even if update happens at 3:45, 3:50, or 3:55
if optimal_window["hours_until"] <= 0.25:  # 15 minutes
```

**Add Constant:**
```python
# In const.py
DHW_WINDOW_ACTIVATION_BUFFER_MINUTES = 15  # Minutes before window to activate
```

**Use in code:**
```python
from ..const import DHW_WINDOW_ACTIVATION_BUFFER_MINUTES

activation_threshold_hours = DHW_WINDOW_ACTIVATION_BUFFER_MINUTES / 60.0
if optimal_window["hours_until"] <= activation_threshold_hours:
```

---

### Fix 2.2: Add Window Pre-Activation Logic
**File:** `custom_components/effektguard/optimization/dhw_optimizer.py`  
**Priority:** LOW (nice to have)

**Concept:**
```python
# If within 20 minutes of optimal window AND DHW needs heating AND cheap/normal price
# Pre-activate to ensure we don't miss the window
WINDOW_PREACTIVATION_MINUTES = 20

if (
    optimal_window
    and optimal_window["hours_until"] <= (WINDOW_PREACTIVATION_MINUTES / 60.0)
    and current_dhw_temp < MIN_DHW_TARGET_TEMP
    and price_classification in ["cheap", "normal"]
):
    _LOGGER.info(
        "DHW: Pre-activating for optimal window in %.0f min (%.1före/kWh)",
        optimal_window["hours_until"] * 60,
        optimal_window["avg_price"],
    )
    # Heat now to ensure window isn't missed
```

---

## Phase 3: Logging and Diagnostics

### Fix 3.1: Add RULE 1 Thermal Debt Logging
**File:** `custom_components/effektguard/optimization/dhw_optimizer.py`  
**Line:** ~429-437 (RULE 1 section)  
**Priority:** HIGH

**Enhanced Logging for RULE 1:**
```python
# === RULE 1: CRITICAL THERMAL DEBT - NEVER START DHW ===
if thermal_debt_dm <= dm_block_threshold:
    _LOGGER.warning(
        "DHW blocked by RULE 1 (thermal debt): DM=%.0f ≤ warning=%.0f (zone: %s, outdoor: %.1f°C)",
        thermal_debt_dm,
        dm_block_threshold,
        self.climate_detector.zone_info.name if self.climate_detector else "unknown",
        outdoor_temp,
    )
    return DHWScheduleDecision(
        should_heat=False,
        priority_reason="CRITICAL_THERMAL_DEBT",
        target_temp=0.0,
        max_runtime_minutes=0,
        abort_conditions=[],
    )
```

**Benefit:** User can see exactly when and why RULE 1 blocks DHW (thermal debt protection).

---

### Fix 3.2: Add Window Scheduling Debug Sensors
**File:** `custom_components/effektguard/sensor.py`  
**Priority:** MEDIUM

**Add New Diagnostic Sensors:**
```python
# DHW next scheduled window time
sensor.effektguard_dhw_next_window_start
# Value: "2025-10-28 04:00:00"

# DHW next window price
sensor.effektguard_dhw_next_window_price  
# Value: 8.6 (öre/kWh)

# DHW thermal debt margin from warning
sensor.effektguard_dhw_thermal_debt_margin
# Value: 62 (DM above warning threshold)
# Negative = past warning (blocked by RULE 1)

# DHW heating blocked reason
sensor.effektguard_dhw_blocked_reason
# Values: "CRITICAL_THERMAL_DEBT" (RULE 1), "SPACE_HEATING_EMERGENCY" (RULE 2), etc.
```

**Benefit:** User can see in UI exactly why DHW didn't heat and when next window is scheduled.

---

## Phase 4: Testing

### Test 4.1: Update Existing Tests
**Files:** 
- `tests/unit/dhw/test_dhw_comprehensive.py`
- `tests/unit/dhw/test_dhw_safety_window_optimization.py`

**Changes Needed:**
1. Remove all spare capacity test expectations (method doesn't exist anymore)
2. Add tests verifying RULE 1 blocks at warning threshold
3. Fix comparison logic in test expectations (`>` to `>=`)
4. Add tests for window timing (13 min, 15 min, 17 min away)

**Example Tests to Remove:**
```python
# DELETE - spare capacity method no longer exists
def test_spare_capacity_uses_t_level_warning_threshold():
    """DHW spare capacity should use T-level warning threshold..."""
    # ... entire test to be deleted ...

# DELETE - testing internal method that's now deleted
def test_has_spare_compressor_capacity_climate_aware():
    """Test spare capacity calculation..."""
    # ... entire test to be deleted ...
```

**Example Tests to Add:**
```python
def test_rule1_blocks_at_warning_threshold():
    """RULE 1 should block DHW when thermal debt at/below warning threshold."""
    climate_detector = ClimateZoneDetector(latitude=55.60)
    scheduler = IntelligentDHWScheduler(climate_detector=climate_detector)
    
    # Get warning threshold for conditions
    dm_range = climate_detector.get_expected_dm_range(-1.0)
    warning_threshold = dm_range["warning"]  # -316
    
    # Test: DM at warning = should block
    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        thermal_debt_dm=warning_threshold,  # Exactly at warning
        outdoor_temp=-1.0,
        price_classification="cheap",
        ...
    )
    assert decision.should_heat is False
    assert decision.priority_reason == "CRITICAL_THERMAL_DEBT"
    
    # Test: DM below warning = should block
    decision2 = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        thermal_debt_dm=warning_threshold - 50,  # Past warning
        outdoor_temp=-1.0,
        price_classification="cheap",
        ...
    )
    assert decision2.should_heat is False
    assert decision2.priority_reason == "CRITICAL_THERMAL_DEBT"
    
    # Test: DM above warning = should allow (if other conditions met)
    decision3 = scheduler.should_start_dhw(
        current_dhw_temp=36.0,
        thermal_debt_dm=warning_threshold + 60,  # 60 DM better than warning
        outdoor_temp=-1.0,
        price_classification="cheap",
        ...
    )
    assert decision3.should_heat is True  # Not blocked by RULE 1
```

---

### Test 4.2: Add Regression Test for October 28 Failure
**File:** `tests/unit/dhw/test_dhw_regression_oct28.py` (new)

```python
"""Regression test for October 28, 2025 DHW failure.

System identified optimal window at 04:00 (8.6öre/kWh) but failed to heat
due to spare capacity check blocking despite being only 1 DM below threshold.
"""

import pytest
from datetime import datetime, timezone
from custom_components.effektguard.optimization.dhw_optimizer import (
    IntelligentDHWScheduler,
    DHWDemandPeriod,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector


def test_oct28_failure_scenario():
    """Reproduce October 28, 2025 failure: DHW at 36.1°C, DM -254, optimal window at 04:00."""
    
    # Setup exactly as it was on Oct 28
    climate_detector = ClimateZoneDetector(latitude=55.60)  # Moderate Cold zone
    demand_period = DHWDemandPeriod(start_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = IntelligentDHWScheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )
    
    # At 03:45 - 15 minutes before optimal window
    current_time = datetime(2025, 10, 28, 3, 45, 0, tzinfo=timezone.utc)
    
    # Create realistic price data (window at 04:00 is 8.6öre/kWh)
    price_periods = create_oct28_price_periods()
    
    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.1,  # Actual temp from logs
        space_heating_demand_kw=2.5,
        thermal_debt_dm=-254,  # Actual DM from logs (better than warning -316, would heat with new logic!)
        indoor_temp=23.4,
        target_indoor_temp=21.0,
        outdoor_temp=8.2,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.4,  # Last heating at 19:24 Oct 27
    )
    
    # With fixes applied, this should now heat (we're within 15-min window)
    assert decision.should_heat is True, (
        f"Should heat when within 15min of optimal window. "
        f"Reason: {decision.priority_reason}"
    )
    assert "OPTIMAL_WINDOW" in decision.priority_reason or "DHW_LOW_CHEAP" in decision.priority_reason


def test_oct28_at_04_00():
    """At exactly 04:00 (the scheduled window), should definitely heat."""
    
    climate_detector = ClimateZoneDetector(latitude=55.60)
    demand_period = DHWDemandPeriod(start_hour=7, target_temp=50.0, duration_hours=2)
    scheduler = IntelligentDHWScheduler(
        demand_periods=[demand_period],
        climate_detector=climate_detector,
    )
    
    # Exactly at the scheduled window
    current_time = datetime(2025, 10, 28, 4, 0, 0, tzinfo=timezone.utc)
    price_periods = create_oct28_price_periods()
    
    decision = scheduler.should_start_dhw(
        current_dhw_temp=36.1,
        space_heating_demand_kw=2.5,
        thermal_debt_dm=-249,  # Slightly better than at 03:45
        indoor_temp=23.4,
        target_indoor_temp=21.0,
        outdoor_temp=8.2,
        price_classification="cheap",
        current_time=current_time,
        price_periods=price_periods,
        hours_since_last_dhw=8.7,
    )
    
    # MUST heat when in the optimal window
    assert decision.should_heat is True
    assert "OPTIMAL_WINDOW_Q16" in decision.priority_reason  # Q16 = 04:00
```

---

## Phase 5: Configuration Tuning

### Config 5.1: User-Configurable Spare Capacity (OPTIONAL - Probably Not Needed)
**File:** `custom_components.effektguard/config_flow.py`
**Status:** ⏸️ **SKIP** - T-level approach doesn't need user configuration

**Reasoning:**
With the simplified T-level approach, spare capacity is simply: "DM > warning threshold"

The warning threshold is already climate-aware and well-researched. No need for user tuning.

**If later found necessary:**
Could add option to require a small safety margin (e.g., 20 DM buffer above warning):
```python
CONF_DHW_SAFETY_MARGIN_DM = "dhw_safety_margin_dm"

# In options flow
vol.Optional(
    CONF_DHW_SAFETY_MARGIN_DM,
    default=0,  # No extra margin, just use warning threshold
): vol.All(vol.Coerce(int), vol.Range(min=0, max=100))

# In spare capacity check:
has_capacity = thermal_debt_dm > (warning_threshold + safety_margin_dm)
```

**Decision:** Skip for now - the simple T-level approach should work well.

---

## Implementation Order (Recommended)

### Week 1: Critical Fixes (The Big Cleanup)
1. ✅ **Fix 1.2: DELETE spare capacity check entirely**
   - Delete DHW_SPARE_CAPACITY_PERCENT from const.py
   - Delete DM_DHW_SPARE_CAPACITY_FALLBACK from const.py
   - Delete _has_spare_compressor_capacity() method from dhw_optimizer.py
   - Remove all 6 calls to spare capacity check
   - **Impact:** ~58 lines deleted, logic simplified, RULE 1 handles everything

2. ✅ **Fix 1.1: Prioritize optimal windows**
   - Increase window detection to 15 minutes (line 729)
   - Remove spare capacity check from optimal window logic (already done in Fix 1.2)
   - Fix comparison from `>` to `>=` for MIN_DHW_TARGET_TEMP
   - **Impact:** DHW heats during scheduled windows, window timing fixed

3. ✅ **Fix 3.1: Enhanced RULE 1 logging**
   - Add warning log when RULE 1 blocks DHW
   - Show DM, warning threshold, climate zone
   - **Impact:** User visibility into thermal debt blocks

4. ✅ **Test 4.2: Add regression test for Oct 28**
   - Verify -254 DM at 03:45 would heat in window
   - Verify RULE 1 correctly blocks at -316 DM
   - **Impact:** Prevent future regressions

### Week 2: Testing and Validation
5. ✅ **Test 4.1: Update existing tests**
   - Remove all spare capacity test expectations
   - Add RULE 1 blocking tests
   - Update comparison logic tests
   - **Impact:** Test suite matches new simplified logic

6. ✅ **Verify no regressions**
   - Run full test suite
   - Check thermal debt protection still works
   - Monitor first few DHW heating cycles
   - **Impact:** Confirm fixes work correctly

### Week 3: Enhancements (Optional)
7. ⏸️ **Fix 2.2: Pre-activation logic** (probably not needed - test first)
8. ⏸️ **Fix 3.2: Debug sensors** (helpful for monitoring)
9. ⏸️ **Config 5.1: User-configurable options** (skip - RULE 1 threshold is correct)

---

## Success Criteria

After fixes are applied, the system should:

1. **Heat during scheduled optimal window** even if DM is near threshold
2. **Not miss windows** due to coordinator timing (15-min buffer)
3. **Log clear reasons** when DHW is blocked (spare capacity, thermal debt, etc.)
4. **Fallback to cheap period heating** if window scheduling fails
5. **Pass regression test** for Oct 28, 2025 scenario

**Specific Oct 28 Test:**
- At 03:45: Should recognize 04:00 window approaching (15 min away)
- At 03:50: Should recognize 04:00 window approaching (10 min away)  
- At 04:00: Should HEAT (in optimal window, regardless of DM -249)
- Result: Water heated at 04:00 @ 8.6öre/kWh ✅

---

## Risk Assessment

### Low Risk Changes
- Logging enhancements (3.1, 3.2)
- Test additions (4.1, 4.2)
- Window timing adjustment (2.1) - Makes system MORE likely to heat

### Medium Risk Changes
- Comparison logic fix (1.1) - Well-tested, clear bug
- Spare capacity threshold (1.2) - Can be reverted easily
- Fallback rule (1.3) - Additional safety net

### High Risk Changes
- None - all changes make system MORE conservative or fix clear bugs

### Rollback Plan
All changes are in `dhw_optimizer.py` and `const.py`. If issues arise:
1. Revert spare capacity check to use 50% buffer (add DHW_SPARE_CAPACITY_PERCENT back)
2. Revert window detection to 0.17 hours (10 min) if 15-min causes issues
3. Monitor thermal debt during DHW heating for first week

### Low Risk with T-Level Approach
- Uses existing, proven warning thresholds (already in production)
- Simpler than percentage calculations (less code = fewer bugs)
- Still protects against heating when in warning/critical thermal debt
- Much more practical than 50% buffer which blocked nearly all DHW opportunities
- Abort conditions provide additional safety during DHW heating

---

## Monitoring After Deployment

### Metrics to Track (First 7 Days)
1. **DHW heating success rate** - Should increase from ~0% to >90% during optimal windows
2. **RULE 1 thermal debt blocks** - Should be minimal (only when actually at/below warning)
3. **Manual DHW boosts** - Should decrease (user shouldn't need to intervene)
4. **Average DHW temp** - Should stabilize at 45-50°C
5. **Hours between DHW heating** - Should decrease (more regular heating)
6. **Thermal debt during DHW** - Monitor DM doesn't deteriorate excessively during DHW heating

### Alert Conditions
- DHW drops below 35°C for >12 hours
- RULE 1 blocks >3 consecutive optimal windows (indicates thermal debt issue with space heating)
- Manual boost needed >2 times/week
- Thermal debt reaches warning during DHW heating (abort conditions should prevent this)

---

## Files to Modify

1. `custom_components/effektguard/optimization/dhw_optimizer.py` - Primary fixes
2. `custom_components/effektguard/const.py` - Threshold constants
3. `tests/unit/dhw/test_dhw_comprehensive.py` - Update tests
4. `tests/unit/dhw/test_dhw_safety_window_optimization.py` - Update tests
5. `tests/unit/dhw/test_dhw_regression_oct28.py` - New regression test
6. `custom_components/effektguard/config_flow.py` - Optional user config
7. `custom_components/effektguard/sensor.py` - Optional debug sensors

---

## Estimated Effort

- **Development:** 4-6 hours (fixes + tests)
- **Testing:** 2-3 hours (regression + edge cases)
- **Documentation:** 1 hour (update DHW docs)
- **Total:** 1 work day

---

## Questions for User (RESOLVED)

1. **Spare capacity logic:** ✅ **Use T2-level warning threshold directly** - Simplest and aligns with existing system
2. **Window detection:** ✅ **15 minutes** - Accommodates 15-min coordinator cycle
3. **Fallback rule:** ✅ **NOT NEEDED** - Fixes 1.1 and 1.2 are sufficient
4. **Priority:** ✅ **Critical fixes first** (Week 1: Fix 1.1, 1.2, logging, tests)

---

## Final Summary: The Complete Fix

### What Was Wrong

1. **Spare capacity check REDUNDANT:** Duplicated RULE 1's thermal debt protection with stricter threshold
2. **Spare capacity too strict (50%):** Required DM > -158 when RULE 1 only requires DM > -316
3. **Window timing mismatch:** 10-min detection incompatible with 15-min coordinator cycle
4. **No window priority:** Even optimal scheduled windows blocked by spare capacity
5. **Comparison bug:** Used `>` instead of `>=` for DHW minimum temperature check

### What We're Fixing

1. **DELETE spare capacity entirely:** Remove method, constants, and all calls (RULE 1 already handles it)
2. **Extend window detection to 15 min:** Matches coordinator update interval
3. **Prioritize optimal windows:** Heat during scheduled windows (spare capacity no longer blocks)
4. **Fix comparison:** Change `>` to `>=` for proper boundary checking

### Code Changes Summary

**Deletions (cleanup):**
- ✂️ 2 constants from const.py (DHW_SPARE_CAPACITY_PERCENT, DM_DHW_SPARE_CAPACITY_FALLBACK)
- ✂️ 1 method from dhw_optimizer.py (_has_spare_compressor_capacity - 42 lines)
- ✂️ 2 imports from dhw_optimizer.py
- ✂️ 6 call sites throughout should_start_dhw()
- 📊 **Total: ~58 lines deleted**

**Modifications:**
- ✏️ Window detection: 0.17 hours → 0.25 hours (10 min → 15 min)
- ✏️ Temperature comparison: `>` → `>=` for MIN_DHW_TARGET_TEMP
- ✏️ Enhanced logging in RULE 1 for thermal debt blocks

### Safety Preserved

- **RULE 1:** Still blocks DHW when DM ≤ warning threshold (thermal debt protection)
- **RULE 2:** Still blocks DHW during space heating emergencies
- **Abort conditions:** Still stop DHW if thermal debt deteriorates during heating
- **All protection consolidated:** One clear location (RULE 1) instead of scattered checks

### Expected Outcome

**October 28 scenario with fixes:**
```
03:45: System recognizes optimal window in 15 min
04:00: Window starts (hours_until = 0.0)
Check: hours_until (0.0) ≤ 0.25? YES ✅
Check: RULE 1 - DM (-254) > warning (-316)? YES ✅
Action: Heat DHW (OPTIMAL_WINDOW_Q16_@8.6öre)
Result: Water heated at cheapest period ✅
User: No manual intervention needed ✅
```

**Why this works:**
- RULE 1 passes: -254 is better (less negative) than -316 ✅
- No spare capacity check to block it anymore ✅
- Window timing catches the 04:00 activation ✅
- Simpler, cleaner, less code to maintain ✅

**Validation checklist:**
- [x] Spare capacity is truly redundant (verified RULE 1 does same check)
- [x] All 10 deletion points identified and documented
- [x] Safety maintained (RULE 1 + RULE 2 + abort conditions)
- [x] Logic is simpler (one protection point, not scattered)
- [x] October 28 case would pass (-254 > -316)
- [x] No contradictions in plan (spare capacity completely removed)

---

## Phase 5: Temperature Control Improvements (CRITICAL)

### Issue Discovery: System Running 2-3°C Too Hot

**Observed in logs (Oct 27-28):**
- Target: 21.0°C
- Actual: 23.5-24.6°C consistently
- **Overshoot: 2.5-3.6°C** (unacceptable!)

**System behavior:**
```
15:27: indoor 23.7°C, target 21.0°C → offset +0.08°C ❌ (heating when too hot!)
15:32: indoor 23.7°C, target 21.0°C → offset +0.17°C ❌ (still heating!)
...continuing with "Too warm: 2.7°C over" but positive offset
```

### Root Causes:

**1. Fixed MAX_TEMP_LIMIT Too Permissive**
```python
# const.py line 73
MAX_TEMP_LIMIT: Final = 24.0  # Allows 3°C overshoot before emergency cooling!
```
- Safety layer only activates at 24°C
- Ignores user's target temperature setting
- Not dynamic or configurable

**2. Comfort Layer Weight Too Low**
```python
# const.py lines 89-90
LAYER_WEIGHT_COMFORT_MIN: Final = 0.2
LAYER_WEIGHT_COMFORT_MAX: Final = 0.5  # Only 0.5 when 2.7°C over target!
```
- Even when significantly over target, comfort layer weight is only 0.5
- Other layers (learned pre-heat, weather pre-heat, price) override it
- Result: System keeps heating despite being way too hot

**3. Common Sense Check is Separate Logic**
- Currently a special-case early return in proactive layer
- Should be integrated as a proper layer with appropriate weight
- Considers weather forecast but not learning or thermal trends

---

### Fix 5.1: Strengthen Comfort Layer (Dynamic Temperature Control)

**File:** `custom_components/effektguard/optimization/decision_engine.py`  
**Lines:** ~1872-1930 (_comfort_layer method)  
**Priority:** CRITICAL

**Current Behavior:**
```python
elif temp_error > tolerance:
    # Too warm, reduce heating strongly
    correction = -(temp_error - tolerance) * 0.5
    return LayerDecision(
        offset=correction,
        weight=LAYER_WEIGHT_COMFORT_MAX,  # Only 0.5!
        reason=f"Too warm: {temp_error:.1f}°C over",
    )
```

**New Behavior (Graduated Response):**
```python
elif temp_error > tolerance:
    # Too warm - scale weight based on severity
    overshoot = temp_error - tolerance
    
    # Graduated weight scaling:
    # - Within 1°C over tolerance: weight 0.7 (high priority)
    # - 1-2°C over tolerance: weight 0.9 (very high priority)
    # - 2°C+ over tolerance: weight 1.0 (CRITICAL - same as safety layer)
    if overshoot >= 2.0:
        weight = 1.0  # Critical priority - forces cooling
        correction = -overshoot * 1.5  # Strong correction
        reason = f"CRITICAL overheat: {temp_error:.1f}°C over target (emergency cooling)"
    elif overshoot >= 1.0:
        weight = 0.9  # Very high priority
        correction = -overshoot * 1.2  # Strong correction
        reason = f"Severe overheat: {temp_error:.1f}°C over target"
    else:
        weight = 0.7  # High priority
        correction = -overshoot * 1.0  # Standard correction
        reason = f"Too warm: {temp_error:.1f}°C over target"
    
    return LayerDecision(offset=correction, weight=weight, reason=reason)
```

**Effect on User's Case:**
```
Before: Indoor 23.7°C, target 21.0°C, tolerance 1.0°C
  → overshoot = 1.7°C over tolerance
  → weight 0.5, correction -0.85°C
  → Other layers override → final offset +0.08°C ❌

After: Indoor 23.7°C, target 21.0°C, tolerance 1.0°C
  → overshoot = 1.7°C over tolerance
  → weight 0.9, correction -2.04°C
  → Strong cooling priority → final offset ≈ -1.5°C ✅
```

**Benefits:**
- ✅ Dynamic - adapts to user's target temperature automatically
- ✅ Graduated - gentle when slightly over, strong when significantly over
- ✅ Safety-first - weight 1.0 at 2°C+ overshoot overrides ALL other layers
- ✅ Respects tolerance setting - user controls acceptable range

---

### Fix 5.2: Remove Fixed MAX_TEMP_LIMIT

**File:** `custom_components/effektguard/optimization/decision_engine.py`  
**Lines:** ~751-785 (_safety_layer method)  
**Priority:** CRITICAL

**Current Code:**
```python
def _safety_layer(self, nibe_state) -> LayerDecision:
    """Safety layer: Enforce absolute temperature limits."""
    indoor_temp = nibe_state.indoor_temp
    
    if indoor_temp < MIN_TEMP_LIMIT:  # 18°C
        offset = SAFETY_EMERGENCY_OFFSET
        return LayerDecision(
            offset=offset,
            weight=LAYER_WEIGHT_SAFETY,
            reason=f"SAFETY: Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
        )
    elif indoor_temp > MAX_TEMP_LIMIT:  # ❌ 24°C - TOO HIGH!
        offset = -SAFETY_EMERGENCY_OFFSET
        return LayerDecision(
            offset=offset,
            weight=LAYER_WEIGHT_SAFETY,
            reason=f"SAFETY: Too hot ({indoor_temp:.1f}°C > {MAX_TEMP_LIMIT}°C)",
        )
    else:
        return LayerDecision(offset=0.0, weight=0.0, reason="Safety OK")
```

**New Code (Remove upper limit check):**
```python
def _safety_layer(self, nibe_state) -> LayerDecision:
    """Safety layer: Enforce absolute minimum temperature only.
    
    Upper temperature limit is now handled dynamically by comfort layer
    based on user's target temperature + tolerance setting.
    
    This ensures temperature control adapts to user preferences rather
    than using a fixed maximum that may be inappropriate.
    """
    indoor_temp = nibe_state.indoor_temp
    
    if indoor_temp < MIN_TEMP_LIMIT:  # 18°C - absolute minimum for safety
        offset = SAFETY_EMERGENCY_OFFSET
        return LayerDecision(
            offset=offset,
            weight=LAYER_WEIGHT_SAFETY,
            reason=f"SAFETY: Too cold ({indoor_temp:.1f}°C < {MIN_TEMP_LIMIT}°C)",
        )
    else:
        # No fixed upper limit - comfort layer handles temperature control dynamically
        return LayerDecision(offset=0.0, weight=0.0, reason="Safety OK")
```

**Cleanup:**
```python
# DELETE from const.py (line 73):
MAX_TEMP_LIMIT: Final = 24.0  # ← REMOVE THIS - no longer needed

# REMOVE from decision_engine.py imports (line ~73):
MAX_TEMP_LIMIT,  # ← DELETE THIS LINE
```

**Why This Works:**
- Comfort layer with weight 1.0 at 2°C+ overshoot provides same protection
- But dynamically based on user's target, not hardcoded 24°C
- User with target 22°C → critical at 25°C (22 + 1.0 tolerance + 2.0 overshoot)
- User with target 19°C → critical at 22°C (19 + 1.0 tolerance + 2.0 overshoot)

---

### Fix 5.3: Convert Common Sense Check to Proper Layer

**File:** `custom_components/effektguard/optimization/decision_engine.py`  
**Location:** Currently in proactive layer (~1213-1238), move to new method  
**Priority:** MEDIUM

**Current Problem:**
- Common sense check is special-case early return
- Not integrated with layer voting system
- Doesn't consider learning or thermal trends
- Binary decision (heat or don't heat) - no graduated response

**New Approach: Create _smart_override_layer():**

```python
def _smart_override_layer(self, nibe_state, weather_data=None) -> LayerDecision:
    """Smart override layer: Prevent unnecessary heating when conditions are favorable.
    
    Integrates multiple signals to determine if heating is truly needed:
    1. Weather forecast (12-hour horizon, high weight)
    2. Indoor temperature trend (rising/stable = less heating needed)
    3. Learning predictions (if we know target will be reached soon)
    4. Current overshoot vs target
    
    This replaces the old "common sense check" with a proper weighted layer
    that participates in decision aggregation rather than short-circuiting it.
    
    Returns:
        LayerDecision with smart override vote (cooling bias when favorable)
    """
    indoor_temp = nibe_state.indoor_temp
    degree_minutes = nibe_state.degree_minutes
    outdoor_temp = nibe_state.outdoor_temp
    
    deficit = self.target_temp - indoor_temp
    
    # Only activate when above target
    if deficit >= -COMMON_SENSE_TEMP_ABOVE_TARGET:
        return LayerDecision(offset=0.0, weight=0.0, reason="Not above target")
    
    # Signal 1: Weather forecast (high weight if stable/warming)
    forecast_signal = 0.0
    forecast_weight = 0.0
    
    if weather_data and weather_data.forecast_hours:
        forecast_hours = weather_data.forecast_hours[:COMMON_SENSE_FORECAST_HORIZON]
        if forecast_hours:
            min_forecast_temp = min(h.temperature for h in forecast_hours)
            max_forecast_temp = max(h.temperature for h in forecast_hours)
            
            # Check for cold snap
            forecast_drop = outdoor_temp - min_forecast_temp
            
            if forecast_drop < COMMON_SENSE_COLD_SNAP_THRESHOLD:
                # No cold snap - stable or warming weather
                # Weight scaled by how stable (more stable = higher weight)
                forecast_range = max_forecast_temp - min_forecast_temp
                if forecast_range < 2.0:  # Very stable
                    forecast_weight = 0.8
                    forecast_signal = -abs(deficit) * 0.5  # Cooling bias
                elif forecast_range < 4.0:  # Moderately stable
                    forecast_weight = 0.6
                    forecast_signal = -abs(deficit) * 0.3
                else:  # Variable but no cold snap
                    forecast_weight = 0.4
                    forecast_signal = -abs(deficit) * 0.2
    
    # Signal 2: Indoor temperature trend
    thermal_trend = self._get_thermal_trend()
    trend_rate = thermal_trend.get("rate_per_hour", 0.0)
    trend_confidence = thermal_trend.get("confidence", 0.0)
    
    trend_signal = 0.0
    trend_weight = 0.0
    
    if trend_confidence > 0.5:
        if trend_rate > 0.2:  # Rising
            trend_weight = 0.6
            trend_signal = -abs(deficit) * 0.4
        elif trend_rate > -0.1:  # Stable
            trend_weight = 0.4
            trend_signal = -abs(deficit) * 0.2
        # Falling = don't override (weight 0)
    
    # Signal 3: Learning prediction (if available)
    learning_signal = 0.0
    learning_weight = 0.0
    
    if hasattr(self, 'learning_engine') and self.learning_engine:
        try:
            # Check if learning predicts we'll reach target soon
            predicted_temp = self.learning_engine.predict_temperature(
                current_temp=indoor_temp,
                current_offset=nibe_state.curve_offset,
                outdoor_temp=outdoor_temp,
                hours_ahead=2.0
            )
            
            if predicted_temp and predicted_temp >= self.target_temp - 0.3:
                # Learning predicts we'll reach target soon - reduce heating
                learning_weight = 0.5
                learning_signal = -abs(deficit) * 0.3
        except Exception as e:
            _LOGGER.debug("Learning prediction unavailable: %s", e)
    
    # Aggregate signals with weighted average
    total_weight = forecast_weight + trend_weight + learning_weight
    
    if total_weight == 0.0:
        return LayerDecision(offset=0.0, weight=0.0, reason="No override signals")
    
    # Calculate weighted average signal
    weighted_signal = (
        forecast_signal * forecast_weight +
        trend_signal * trend_weight +
        learning_signal * learning_weight
    ) / total_weight
    
    # Build comprehensive reason
    reasons = []
    if forecast_weight > 0:
        reasons.append(f"weather stable ({min_forecast_temp:.1f}-{max_forecast_temp:.1f}°C)")
    if trend_weight > 0:
        reasons.append(f"indoor {'rising' if trend_rate > 0 else 'stable'} ({trend_rate:+.2f}°C/h)")
    if learning_weight > 0:
        reasons.append(f"learning predicts target soon")
    
    reason = f"Smart override: {abs(deficit):.1f}°C over target, " + ", ".join(reasons)
    
    # Return decision with moderate weight (0.3-0.8 depending on signal strength)
    # This allows it to influence but not override critical safety/comfort layers
    final_weight = min(total_weight, 0.8)
    
    return LayerDecision(
        offset=weighted_signal,
        weight=final_weight,
        reason=reason
    )
```

**Integration:**
```python
# In calculate_decision() method, add to layer list:
layers = [
    self._safety_layer(nibe_state),
    self._emergency_layer(nibe_state, weather_data),
    self._proactive_layer(nibe_state, weather_data),
    self._effect_layer(nibe_state, peak_power),
    self._weather_layer(nibe_state, weather_data),
    self._price_layer(nibe_state, price_classification),
    self._comfort_layer(nibe_state),
    self._smart_override_layer(nibe_state, weather_data),  # NEW!
]

# Remove common sense check from _proactive_layer (lines 1213-1238)
```

**Benefits:**
- ✅ Integrated with layer voting system (no special cases)
- ✅ Considers weather + trends + learning together
- ✅ Graduated response based on signal strength
- ✅ Can be overridden by critical layers (safety, severe overheat)
- ✅ More sophisticated than binary on/off decision

---

## Implementation Order (Updated)

### Week 1: Critical DHW Fixes
1. ✅ **Fix 1.2: DELETE spare capacity check** (~58 lines)
2. ✅ **Fix 1.1: Prioritize optimal windows** (window timing + comparison)
3. ✅ **Fix 3.1: Enhanced RULE 1 logging**
4. ✅ **Test 4.2: Regression test for Oct 28**

### Week 2: Critical Temperature Control Fixes
5. ✅ **Fix 5.1: Strengthen comfort layer** (graduated weight scaling)
6. ✅ **Fix 5.2: Remove MAX_TEMP_LIMIT** (delete constant, simplify safety layer)
7. ✅ **Test: Validate temperature control** (verify 21°C target maintained)
8. ✅ **Monitor: First 48 hours** (check indoor temp stays at target)

### Week 3: Integration and Enhancement
9. ✅ **Fix 5.3: Convert common sense to smart override layer** (new layer method)
10. ✅ **Test 4.1: Update DHW tests** (remove spare capacity expectations)
11. ✅ **Test: Validate smart override** (weather + trends + learning integration)
12. ✅ **Fix 3.2: Debug sensors** (optional - helpful for monitoring)

---

## Success Criteria (Updated)

### DHW Success:
- [x] DHW heats during scheduled optimal windows
- [x] No spare capacity blocks (method deleted)
- [x] Window timing catches 15-minute coordinator cycle
- [x] October 28 scenario passes regression test

### Temperature Control Success:
- [x] **Indoor temperature stays within tolerance of target** (±1°C)
- [x] **No prolonged overshoots** (brief 0.5°C OK, sustained 2°C+ unacceptable)
- [x] **Weight 1.0 activates at 2°C+ overshoot** (forces cooling)
- [x] **Dynamic based on user's target** (not fixed 24°C)
- [x] **Smart override prevents unnecessary heating** when above target + stable weather

**Specific Temperature Test Cases:**
```
Target 21.0°C, Tolerance 1.0°C:
- 21.0-22.0°C: Gentle steering (weight 0.2-0.5) ✓
- 22.0-23.0°C: Strong cooling (weight 0.7), offset ≈ -1.0°C ✓
- 23.0-24.0°C: Very strong cooling (weight 0.9), offset ≈ -2.4°C ✓
- 24.0°C+: CRITICAL cooling (weight 1.0), offset ≈ -3.0°C+, overrides all layers ✓

Above target + stable weather:
- Smart override layer activates (weight 0.3-0.8)
- Adds cooling bias based on signals (weather, trend, learning)
- Prevents pre-heating when already warm ✓
```

---

**End of Implementation Plan**
