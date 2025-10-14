# Service Rate Limiting Implementation

## Summary

Implemented rate limiting for EffektGuard service calls to prevent abuse and ensure safe operation.

## Constants Added to `const.py`

```python
# Service call rate limiting (boost, DHW, general)
BOOST_COOLDOWN_MINUTES: Final = 45  # Prevent boost spam
DHW_BOOST_COOLDOWN_MINUTES: Final = 60  # DHW boost cooldown (reserved for future)
SERVICE_RATE_LIMIT_MINUTES: Final = 5  # General service call cooldown
```

## Implementation Details

### Helper Functions (`__init__.py`)

Added two helper functions for clean, reusable rate limiting:

```python
def _check_service_cooldown(service_name: str, cooldown_minutes: int) -> tuple[bool, int]:
    """Check if service is in cooldown period.
    
    Returns:
        Tuple of (is_allowed, remaining_time_seconds)
    """

def _update_service_timestamp(service_name: str) -> None:
    """Update the last called timestamp for a service."""
```

### Service Cooldowns

| Service | Cooldown | Reason |
|---------|----------|--------|
| `boost_heating` | 45 minutes | Prevent comfort boost spam, allow thermal settling |
| `force_offset` | 5 minutes | General service protection, matches NIBE write limit |
| `dhw_boost` | 60 minutes | Reserved for future DHW boost implementation |

### Services with Rate Limiting

#### 1. **boost_heating**
- **Cooldown**: 45 minutes
- **Behavior**: Warns user with remaining time in minutes
- **Use case**: Emergency comfort boost during cold periods

#### 2. **force_offset**
- **Cooldown**: 5 minutes
- **Behavior**: Warns user with remaining time in seconds
- **Use case**: Manual heating curve adjustments, testing

#### 3. **reset_peak_tracking**
- **No rate limiting** (intentional)
- **Reason**: Monthly billing period reset, rarely called

#### 4. **calculate_optimal_schedule**
- **No rate limiting** (intentional)
- **Reason**: Read-only preview, no state changes

## Behavior

### When Cooldown Active

Services log a warning and return early:

```
WARNING: boost_heating called too soon. Cooldown: 23 minutes remaining
WARNING: force_offset called too soon. Cooldown: 180 seconds remaining
```

### User Experience

- Service calls during cooldown are silently rejected with log warning
- No errors thrown to Home Assistant
- Users can check logs to see cooldown status
- Prevents accidental rapid-fire automation calls

## Rationale

### Why These Cooldowns?

1. **45 minutes for boost_heating**:
   - Heat pump thermal inertia takes 10-30 minutes to show effect
   - Prevents oscillation from repeated boost calls
   - Allows time for system to stabilize before next boost
   - Comparable to PeaqHVAC's approach

2. **5 minutes for force_offset**:
   - Matches NIBE MyUplink API write rate limit
   - Prevents excessive manual interventions
   - Allows quick corrections if needed

3. **60 minutes for dhw_boost** (reserved):
   - DHW cycles typically 30-60 minutes
   - Prevents interference with natural DHW scheduling
   - Reserved for Phase 7+ implementation

## Comparison with PeaqHVAC

| Aspect | PeaqHVAC | EffektGuard |
|--------|----------|-------------|
| **Offset write interval** | 20 seconds | 5 minutes |
| **Service rate limiting** | Not implemented | Implemented |
| **Boost cooldown** | Not implemented | 45 minutes |
| **Approach** | Timer-based updates | Event-driven + cooldowns |

EffektGuard is **more conservative and safer** with additional user-facing protections.

## Future Enhancements

### Phase 7+: DHW Boost Service

When DHW boost is implemented:

```python
async def dhw_boost_handler(call) -> None:
    """Handle dhw_boost service call."""
    is_allowed, remaining = _check_service_cooldown("dhw_boost", DHW_BOOST_COOLDOWN_MINUTES)
    if not is_allowed:
        remaining_minutes = int(remaining / 60)
        _LOGGER.warning(
            "dhw_boost called too soon. Cooldown: %d minutes remaining",
            remaining_minutes
        )
        return
    
    # DHW boost implementation...
    _update_service_timestamp("dhw_boost")
```

### Potential Improvements

1. **Per-entry cooldowns**: Track cooldowns per config entry instead of globally
2. **User-configurable cooldowns**: Allow users to adjust in config
3. **Cooldown bypass**: Emergency override parameter for critical situations
4. **Telemetry**: Track cooldown violations for diagnostics

## Testing Recommendations

### Manual Testing

```python
# Call boost_heating twice rapidly
hass.services.async_call("effektguard", "boost_heating", {"duration": 120})
await asyncio.sleep(1)
hass.services.async_call("effektguard", "boost_heating", {"duration": 120})
# Second call should be rejected with warning

# Wait 45 minutes
await asyncio.sleep(45 * 60)
hass.services.async_call("effektguard", "boost_heating", {"duration": 120})
# Should succeed
```

### Unit Tests

Add to `tests/test_services.py`:

```python
async def test_boost_heating_cooldown(hass):
    """Test boost_heating respects cooldown."""
    # First call succeeds
    await hass.services.async_call(DOMAIN, "boost_heating", {})
    
    # Second immediate call fails
    await hass.services.async_call(DOMAIN, "boost_heating", {})
    # Assert warning logged
    
    # Advance time past cooldown
    with freeze_time(datetime.now() + timedelta(minutes=46)):
        await hass.services.async_call(DOMAIN, "boost_heating", {})
        # Should succeed
```

## Migration Notes

No migration needed - this is a new feature. Existing automations will work the same, but repeated rapid calls will now be rate-limited for safety.

## Documentation Updates Needed

Update user documentation to mention:
- Service cooldown periods
- What happens when cooldown is active
- How to check cooldown status in logs
- Why cooldowns exist (heat pump thermal inertia)
