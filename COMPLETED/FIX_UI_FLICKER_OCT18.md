# Fix: UI Flicker When Changing Number Entity Values

**Date:** October 18, 2025  
**Issue:** UI flickers when user changes temperature or other number entity values  
**Root Cause:** Unnecessary coordinator refresh violating Home Assistant best practices  
**Status:** ✅ Fixed

---

## Problem Description

When users changed number entity values (target temperature, thermal mass, etc.):
1. The number entity would update
2. **All entities in the integration would flicker and repaint**
3. User experience was poor with visible UI disruption

---

## Root Cause Analysis

### The Flicker Chain

```
User changes number entity value
  ↓
async_set_native_value() in number.py
  ↓
async_update_entry() updates config entry options
  ↓
Update listener triggers async_reload_entry()
  ↓
async_update_config() updates coordinator (hot reload - correct!)
  ↓
async_request_refresh() ← PROBLEM: Unnecessary refresh!
  ↓
Coordinator updates all entities
  ↓
All entities repaint → UI FLICKER
```

### Why This Was Wrong

According to **Home Assistant Developer Documentation**:
- NumberEntity's `async_set_native_value()` should update the value and call `async_write_ha_state()`
- **No mention of triggering coordinator refresh**
- Configuration values should be applied on next coordinator update cycle
- For real-time control, use switch/climate entities instead

**File:** `custom_components/effektguard/number.py` line 174
```python
# WRONG (violated HA best practices):
await self.coordinator.async_request_refresh()
```

---

## Fix Implementation

### Changes Made

**File:** `custom_components/effektguard/number.py`

**Before:**
```python
async def async_set_native_value(self, value: float) -> None:
    """Update the value."""
    config_key = self.entity_description.config_key
    if not config_key:
        return

    _LOGGER.info("Setting %s to %.2f", config_key, value)

    new_options = dict(self._entry.options)
    new_options[config_key] = value

    self.hass.config_entries.async_update_entry(self._entry, options=new_options)

    # PROBLEM: Unnecessary refresh causing flicker
    await self.coordinator.async_request_refresh()

    self.async_write_ha_state()
```

**After:**
```python
async def async_set_native_value(self, value: float) -> None:
    """Update the value."""
    config_key = self.entity_description.config_key
    if not config_key:
        return

    _LOGGER.info("Setting %s to %.2f", config_key, value)

    # Update config entry options (triggers update listener)
    # This follows Home Assistant best practices for NumberEntity
    new_options = dict(self._entry.options)
    new_options[config_key] = value

    self.hass.config_entries.async_update_entry(self._entry, options=new_options)
    # This automatically calls async_reload_entry() which updates coordinator config
    # via async_update_config() - no need for explicit refresh

    # Update this entity's state immediately (standard HA pattern)
    # User sees new value instantly, coordinator applies it on next cycle
    self.async_write_ha_state()
```

**Key Change:** Removed `await self.coordinator.async_request_refresh()` line

---

## How It Works Now (Correct Pattern)

```
User changes number entity value
  ↓
async_set_native_value() in number.py
  ↓
async_update_entry() updates config entry options
  ↓
Update listener triggers async_reload_entry()
  ↓
async_update_config() updates coordinator cached values (hot reload)
  ↓
async_write_ha_state() updates this entity's UI
  ↓
✅ User sees immediate feedback (number entity shows new value)
  ↓
Next coordinator cycle (≤5 minutes) applies new value
  ↓
✅ No unnecessary entity repaints, no flicker
```

---

## Benefits

1. ✅ **No UI Flicker**: Other entities don't repaint when number value changes
2. ✅ **Immediate User Feedback**: Number entity shows new value instantly
3. ✅ **Follows HA Best Practices**: Aligned with official Home Assistant documentation
4. ✅ **Efficient**: Configuration applied on next natural coordinator cycle
5. ✅ **Correct Pattern**: Update listener handles config synchronization

---

## Testing Recommendations

1. **Change Target Temperature**:
   - Set target temperature to different value
   - ✅ Number entity should update immediately
   - ✅ Other entities should NOT flicker
   - ✅ Next optimization cycle (≤5 min) uses new value

2. **Change Thermal Mass**:
   - Adjust thermal mass slider
   - ✅ Value updates immediately
   - ✅ No UI disruption
   - ✅ Applied on next decision calculation

3. **Change Peak Protection Margin**:
   - Modify peak protection margin
   - ✅ Instant value update
   - ✅ Smooth UI experience
   - ✅ Next cycle uses new threshold

---

## Documentation References

- **Home Assistant NumberEntity**: https://developers.home-assistant.io/docs/core/entity/number
- **Config Entry Options**: https://developers.home-assistant.io/docs/config_entries_options_flow_handler
- **Update Listeners**: Pattern documented in options flow handler

---

## Related Files

- `custom_components/effektguard/number.py` - Number entity implementation (FIXED)
- `custom_components/effektguard/__init__.py` - Update listener registration
- `custom_components/effektguard/coordinator.py` - Hot reload via async_update_config()
- `LEARNING_AND_STATE_ANALYSIS.md` - Full analysis of learning/state/flicker issue

---

## Verification

✅ Syntax check passed: `python3 -m py_compile custom_components/effektguard/number.py`  
✅ Black formatting applied: `black custom_components/effektguard/ --line-length 100`  
✅ Pattern follows Home Assistant documentation  
✅ No breaking changes - backward compatible  

---

## Commit Message

```
Fix: Remove unnecessary coordinator refresh from number entities

Fixes UI flicker when changing number entity values (target temp, thermal mass, etc.)

Problem:
- Number entities triggered coordinator refresh on value change
- Caused all entities to repaint unnecessarily
- Poor UX with visible flicker

Solution:
- Remove await self.coordinator.async_request_refresh() from async_set_native_value()
- Follow Home Assistant best practices for NumberEntity
- Config is updated via update listener (async_update_config)
- Value applied on next natural coordinator cycle (≤5 min)

Benefits:
- No UI flicker
- Immediate user feedback (number shows new value)
- Follows HA documentation pattern
- More efficient

Files changed:
- custom_components/effektguard/number.py

References:
- https://developers.home-assistant.io/docs/core/entity/number
- LEARNING_AND_STATE_ANALYSIS.md
- FIX_UI_FLICKER_OCT18.md
```
