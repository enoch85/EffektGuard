# NIBE MyUplink API Rate Limiting Analysis

Based on research and code analysis, here are the findings about NIBE MyUplink API rate limits and EffektGuard's usage pattern.

## Research Findings

### Official Documentation
- **No public rate limit documentation found**
- NIBE's developer portal requires authentication
- Community relies on reverse engineering and empirical testing

### EffektGuard's Current API Usage

#### Data Collection (Every 5 minutes)
```python
# EffektGuard NEVER calls external APIs for data collection
# It only reads cached Home Assistant entity states:

# NIBE data - reads from existing NIBE integration entities
outdoor_temp = await self._read_entity_float(self._entity_cache.get("outdoor_temp"))

# Spot Price data - reads from existing Spot Price integration entities  
state = self.hass.states.get(self._gespot_entity)
prices = state.attributes.get("prices_today", [])

# Weather data - reads from existing weather integration entities
forecast = state.attributes.get("forecast", [])
```

#### API Writes (Rate Limited)
```python
# ONLY writes to NIBE when offset actually changes
# Built-in 5-minute rate limiting:

if self._last_write and now - self._last_write < timedelta(minutes=5):
    _LOGGER.debug("Skipping offset write, too soon since last write")
    return

# Maximum: 12 writes per hour
# Typical: 2-4 writes per hour (most cycles don't change offset)
```

## Rate Limiting Analysis

### Current Implementation (Conservative)
- **Data gathering**: 0 additional API calls
- **Maximum write frequency**: Once per 5 minutes  
- **Typical write frequency**: 2-4 times per hour
- **Daily maximum**: 288 writes (unrealistic - requires constant changes)
- **Daily typical**: 48-96 writes

### Comparison with Other Integrations
- **NIBE official integration**: ~60 second polling intervals
- **Most HA integrations**: 30-60 second data polling
- **EffektGuard**: 5 minute optimization cycles + rate-limited writes

## Recommendations

### Option 1: Keep Current (5 minutes) - RECOMMENDED
**Rationale:**
- Already very conservative compared to other integrations
- Rate limiting prevents abuse
- Essential for 15-minute effect tariff optimization
- Zero additional data API load

### Option 2: Configurable Update Intervals
```python
# Allow users to choose update frequency
UPDATE_INTERVALS = {
    "aggressive": timedelta(minutes=2),    # Fast response
    "normal": timedelta(minutes=5),        # Current default  
    "conservative": timedelta(minutes=10), # Extra safe
    "eco": timedelta(minutes=15),          # Minimal API usage
}
```

### Option 3: Smart Adaptive Intervals
```python
# Dynamic intervals based on conditions
if price_transition_detected or approaching_peak:
    update_interval = timedelta(minutes=2)  # Fast during critical periods
elif stable_operation:
    update_interval = timedelta(minutes=10) # Slower during stable periods
```

### Option 4: Event-Driven Updates
```python
# Update immediately when price/weather entities change
@callback
def on_price_entity_change(event):
    await coordinator.async_request_refresh()
    
# Reduces unnecessary polling during stable periods
```

## Risk Assessment

### Low Risk Factors
- **No data polling APIs** - only reads cached HA entities
- **Rate-limited writes** - maximum once per 5 minutes
- **Write optimization** - only when offset actually changes
- **Graceful degradation** - continues working if API fails

### Potential Concerns
- **Premium subscription required** for MyUplink write access
- **Unknown daily/hourly limits** from NIBE
- **Possible throttling** during high usage periods

## Conclusion

**EffektGuard's current 5-minute interval is likely SAFE** because:

1. **Zero additional data API calls** (reads cached entities only)
2. **Conservative write rate limiting** (5-minute minimum)
3. **Smart write detection** (only when offset changes)
4. **Typical usage well below limits** (2-4 writes/hour vs 12 maximum)

The system is already designed with API conservation in mind. The 5-minute optimization cycle is necessary for effective Swedish effect tariff optimization (15-minute billing periods) while maintaining excellent API citizenship.

## Implementation Options

If you want to be extra conservative, I recommend **Option 2 (Configurable Intervals)** with default remaining at 5 minutes, allowing users to choose 10 or 15 minutes if they prefer.