# DHW (Domestic Hot Water) Optimization - Complete Documentation

**Last Updated:** 2025-01-28  
**Status:** ✅ Production Ready  
**Integration:** NIBE MyUplink via Home Assistant

---

## Table of Contents

1. [Overview](#overview)
2. [NIBE DHW Control](#nibe-dhw-control)
3. [Implementation Strategy](#implementation-strategy)
4. [Thermal Characteristics](#thermal-characteristics)
5. [Legionella Safety](#legionella-safety)
6. [Scheduling Logic](#scheduling-logic)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [References](#references)

---

## Overview

EffektGuard's DHW optimization schedules domestic hot water heating during cheap electricity periods while maintaining safety and comfort. Unlike space heating, DHW optimization is **immediate and responsive** due to fast thermal characteristics.

### Key Features

- ✅ **Smart Scheduling:** Heat DHW during cheapest hours before demand periods
- ✅ **MyUplink Integration:** Direct control via `switch.temporary_lux_50004`
- ✅ **Legionella Monitoring:** Detect NIBE's automatic boosts, don't trigger duplicates
- ✅ **Thermal Debt Protection:** Never start DHW when space heating in emergency
- ✅ **24h Forecast Window:** Maximize optimization with GE-Spot's 48h data
- ✅ **BT7 Monitoring:** Track hot water temperature for intelligent decisions

---

## NIBE DHW Control

### MyUplink API Entities

**Temperature Sensors:**
```yaml
sensor.bt7_hw_top_40013          # Hot water top temperature (°C)
sensor.bt6_hw_bottom_40014       # Hot water bottom temperature (°C) [optional]
```

**Control Switches:**
```yaml
switch.temporary_lux_50004       # Temporary lux boost (3-hour DHW priority)
```

### How Temporary Lux Works

**NIBE Temporary Lux Mode:**
- One-time DHW boost to comfort temperature (typically 50-55°C)
- Duration: Fixed 3 hours (180 minutes)
- Overrides normal DHW schedule
- Heat pump prioritizes DHW over space heating during lux period
- Automatically returns to normal mode after timeout

**EffektGuard Usage:**
```python
# Trigger DHW boost during cheap hours
await hass.services.async_call(
    "switch",
    "turn_on",
    {"entity_id": "switch.temporary_lux_50004"}
)

# Monitor via BT7 sensor
dhw_temp = float(hass.states.get("sensor.bt7_hw_top_40013").state)
```

### NIBE Internal DHW Settings

**From NIBE Menu 5.1.1 (Hot Water):**
```
Normal Mode:
├── Start temperature: 44°C (compressor starts)
├── Stop temperature: 50°C (6°C differential)
├── Periodic heating stop: 55°C (prevent condenser alarms)
└── Legionella boost: 65°C weekly (fixed schedule)
```

**EffektGuard does NOT change these settings** - we work with NIBE's built-in logic via temporary lux switch.

---

## Implementation Strategy

### Design Philosophy

**READ-ONLY Monitoring + Temporary Lux Control:**

1. **Monitor NIBE's DHW state** via BT7 temperature sensor
2. **Trigger boosts during cheap hours** via temporary lux switch
3. **Never interfere** with NIBE's internal safety logic
4. **Detect Legionella boosts** automatically (don't trigger duplicates)
5. **Respect thermal debt** (space heating always has priority)

### Decision Rules (Priority Order)

**Rule 1: CRITICAL THERMAL DEBT - NEVER START DHW**
- DM < -240: Space heating in crisis
- Block all DHW optimization
- Reasoning: Thermal debt > comfort > cost

**Rule 2: SPACE HEATING EMERGENCY - HOUSE TOO COLD**
- Indoor temp < target - 1.0°C AND outdoor < 0°C
- Block DHW to prioritize space heating
- Reasoning: Occupant comfort critical

**Rule 3: DHW SAFETY MINIMUM - MUST HEAT**
- DHW temp < 35°C (Legionella risk zone)
- Force heating regardless of price
- Limited runtime: 30 minutes (prevent thermal debt)
- Reasoning: Health safety > cost

**Rule 4: HIGH SPACE HEATING DEMAND - DELAY DHW**
- Indoor temp < target - 0.5°C AND space demand > 3.5 kW
- Delay DHW until space heating stabilizes
- Reasoning: Comfort before hot water luxury

**Rule 5: HIGH DEMAND PERIOD - TARGET TEMP BY START TIME**
- 1-24 hours before configured demand period (morning shower, etc.)
- Urgent (<2h): Heat immediately regardless of price
- Optimal (2-24h): Schedule during cheapest hours
- Reasoning: Ensure hot water ready when needed

**Rule 6: CHEAP ELECTRICITY - OPPORTUNISTIC HEATING**
- Price classification: "cheap"
- Indoor temp comfortable (within 0.3°C of target)
- Thermal debt healthy (DM > -150)
- Boost DHW to 55°C for extra buffer
- Reasoning: Use cheap electricity wisely

**Rule 7: NORMAL DHW HEATING - TEMPERATURE LOW**
- DHW temp < 45°C (normal start threshold)
- Normal conditions (no emergencies, no cheap periods)
- Heat to 50°C (standard comfort)
- Reasoning: Maintain baseline hot water availability

**Rule 8: ALL CONDITIONS FAIL - DON'T HEAT**
- DHW adequate, no demand periods, not cheap
- Let NIBE handle normal operation
- Reasoning: Don't optimize when unnecessary

---

## Thermal Characteristics

### CRITICAL: DHW vs Space Heating Thermal Lag

**⚠️ COMMON MISTAKE:** Confusing space heating thermal lag with DHW

**Space Heating (UFH):**
- Concrete slab UFH: **6+ hours** thermal lag
- Timber UFH: **2-3 hours** thermal lag
- Radiators: **<1 hour** thermal lag
- Changes take hours to manifest in indoor temperature

**DHW (Hot Water Tank):**
- Heating time: **1-2 hours** (typically 1.5 hours)
- Cooling rate: **Very slow** (well-insulated tanks)
- Response: **Immediate** (direct heating, no thermal mass delays)

**This is why DHW scheduling uses 24h window but acts quickly!**

### DHW Tank Behavior

**Heating Characteristics:**
```python
DHW_HEATING_TIME_HOURS = 1.5  # Time to heat from 40°C to 55°C
DHW_COOLING_RATE = 0.5        # °C per hour (well-insulated tank)
```

**Temperature Stratification:**
- BT7 (top): Hottest, used for delivery
- BT6 (bottom): Coolest, return from taps
- Stratification = efficiency (don't mix layers!)

**Scheduling Window:**
- Max preheat: 24 hours ahead
- Min preheat: 1 hour ahead
- Reasoning: Tank cools slowly (<12°C per day), but heating is fast (1.5h)

---

## Legionella Safety

### NIBE's Built-in Protection

**Automatic Legionella Boost:**
- Frequency: Weekly (user-configured day/time)
- Target: 65°C (kills Legionella bacteria)
- Duration: Until BT7 reaches 65°C
- Schedule: **FIXED** (cannot be changed via MyUplink API)

**Problem:** NIBE runs on fixed schedule regardless of electricity prices!

### EffektGuard Strategy: **MONITOR, DON'T TRIGGER**

**Why Monitor-Only:**
1. NIBE's schedule is **fixed** - we can't change it via API
2. Triggering our own boost would **waste energy** (double heating)
3. Better to **detect NIBE's boost** and skip our own scheduling

**Implementation:**

```python
class IntelligentDHWScheduler:
    """DHW scheduler with automatic Legionella detection."""
    
    def __init__(self):
        self.bt7_history: deque = deque(maxlen=48)  # 12 hours of 15-min readings
        self.last_legionella_boost: datetime | None = None
    
    def update_bt7_temperature(self, temp: float, timestamp: datetime):
        """Update BT7 history and detect NIBE's Legionella boost."""
        self.bt7_history.append((timestamp, temp))
        
        # Automatic detection when NIBE runs Legionella boost
        if self._detect_legionella_boost_completion():
            self.last_legionella_boost = timestamp
            _LOGGER.info("NIBE Legionella boost detected at %s", timestamp)
    
    def _detect_legionella_boost_completion(self) -> bool:
        """Detect when NIBE's Legionella boost completed.
        
        Detection logic:
        - BT7 peaked at ≥63°C recently (close to 65°C target)
        - Now cooling down (dropped 3°C from peak)
        - Confirms boost completed successfully
        """
        if len(self.bt7_history) < 10:
            return False
        
        temps = [temp for _, temp in self.bt7_history]
        max_temp = max(temps)
        current_temp = temps[-1]
        
        # Peak ≥63°C and now cooling
        if max_temp >= 63.0 and current_temp < (max_temp - 3.0):
            # Check not already recorded
            if self.last_legionella_boost:
                hours_since = (self.bt7_history[-1][0] - self.last_legionella_boost).total_seconds() / 3600
                if hours_since < 6.0:  # Same boost event
                    return False
            return True
        
        return False
```

**Benefits:**
- ✅ No duplicate heating (NIBE + EffektGuard)
- ✅ Automatic detection (no manual tracking needed)
- ✅ Works with NIBE's fixed schedule
- ✅ Future-proof for NIBE firmware changes

**Note:** We do NOT trigger Legionella ourselves. NIBE's fixed schedule handles it.

---

## Scheduling Logic

### 24-Hour Forecast Window

**Why 24h?**
- GE-Spot provides up to 48h price forecast
- DHW tank cools slowly (can preheat far ahead)
- Maximizes price optimization opportunities
- DHW heats fast (1.5h), so timing flexibility is high

**Smart Fallback:**
```python
def _check_upcoming_demand_period(self, current_time: datetime) -> dict | None:
    """Check for demand periods with 24h window and smart fallback.
    
    Window: 1-24 hours ahead
    Fallback: Use whatever forecast available (minimum 1h)
    """
    for period in self.demand_periods:
        hours_until = calculate_hours_until(period, current_time)
        
        # Extended window: 1-24h ahead
        # Minimum 1h: Need time for heating (1.5h typical)
        if 1 <= hours_until <= 24:
            return {
                "hours_until": hours_until,  # Keep as float for precision
                "target_temp": period.target_temp,
                "period_start": period.start_time,
            }
    
    return None
```

### Demand Periods

**User-Configured High Demand Times:**
```python
DHWDemandPeriod(
    start_hour=7,           # 7:00 AM - morning shower
    target_temp=55.0,       # Comfort temperature
    duration_hours=2,       # 7:00-9:00 AM window
)
```

**Scheduling Strategy:**
- **Urgent (<2h ahead):** Heat immediately, ignore prices
- **Optimal (2-24h ahead):** Find cheapest hours in window
- **No forecast:** Fall back to simple scheduling

### Price-Aware Scheduling

**Integration with GE-Spot:**
```python
def find_optimal_heating_time(
    demand_period: DHWDemandPeriod,
    price_forecast: list[QuarterPeriod],
) -> datetime:
    """Find cheapest 2-hour window before demand period.
    
    Uses GE-Spot's native 15-minute intervals.
    """
    # Filter quarters before demand period
    valid_quarters = [
        q for q in price_forecast
        if q.timestamp < demand_period.start_time
    ]
    
    # Find cheapest consecutive 8 quarters (2 hours)
    # DHW heating takes ~1.5h, 2h window provides margin
    cheapest_window = find_cheapest_consecutive_quarters(
        valid_quarters,
        num_quarters=8,
    )
    
    return cheapest_window.start_time
```

---

## Configuration

### Required Entities

**Must be configured in Home Assistant:**

1. **BT7 Temperature Sensor:**
   ```yaml
   sensor.bt7_hw_top_40013
   ```
   
2. **Temporary Lux Switch:**
   ```yaml
   switch.temporary_lux_50004
   ```

### EffektGuard Config Flow

**DHW Optimization Settings:**
```python
CONF_ENABLE_DHW_OPTIMIZATION = True        # Enable DHW feature
CONF_DHW_TEMP_ENTITY = "sensor.bt7_..."   # BT7 sensor entity
CONF_NIBE_TEMP_LUX_ENTITY = "switch...."  # Temporary lux switch
CONF_DHW_DEMAND_PERIODS = [                # High demand times (JSON)
    {
        "start_hour": 7,
        "target_temp": 55.0,
        "duration_hours": 2,
    },
    {
        "start_hour": 19,
        "target_temp": 50.0,
        "duration_hours": 1,
    }
]
```

### Constants

**DHW-Specific Constants (const.py):**
```python
# Temperature limits
DHW_MIN_TEMP = 40.0              # °C - Minimum safe DHW
DHW_MAX_TEMP = 55.0              # °C - Maximum normal comfort
DHW_COMFORT_TEMP = 50.0          # °C - Optimal comfort
DHW_ECO_TEMP = 45.0              # °C - Economy mode
DHW_LEGIONELLA_DETECT = 63.0     # °C - BT7 temp indicating boost

# Timing
DHW_HEATING_TIME_HOURS = 1.5     # Hours to heat tank
DHW_SCHEDULING_WINDOW_MAX = 24   # Max hours ahead for scheduling
DHW_SCHEDULING_WINDOW_MIN = 1    # Min hours ahead for scheduling

# MyUplink entity IDs (default patterns)
NIBE_TEMP_LUX_ENTITY_ID = "switch.temporary_lux_50004"
NIBE_BT7_SENSOR_ID = "sensor.bt7_hw_top_40013"
```

---

## Testing

### Unit Tests

**Test Coverage (tests/test_dhw_optimizer.py):**
- ✅ Smart scheduling during cheapest hours
- ✅ Thermal debt blocks DHW heating
- ✅ Space heating priority over DHW
- ✅ Safety minimum forces heating
- ✅ Legionella detection (monitor-only)
- ✅ BT7 history tracking
- ✅ Cheap electricity opportunistic heating

**Run Tests:**
```bash
pytest tests/test_dhw_optimizer.py -v
```

**Expected Result:** 7/7 tests pass

### Integration Testing

**Manual Testing Checklist:**
- [ ] Config flow accepts DHW entities
- [ ] BT7 sensor updates tracked in history
- [ ] Legionella detection logs "boost completed"
- [ ] `boost_dhw` service activates temp lux switch
- [ ] Service cooldown prevents spam (60 min)
- [ ] 24h scheduling finds cheap hours
- [ ] Smart fallback handles partial forecast
- [ ] Thermal debt blocks DHW correctly

### Production Validation

**Real-World Testing:**
1. Monitor BT7 sensor for 1 week
2. Verify Legionella detection during NIBE's scheduled boost
3. Test `boost_dhw` service during cheap hours
4. Confirm no interference with space heating
5. Check thermal debt never goes critical during DHW

---

## References

### NIBE Documentation

**MyUplink API:**
- `IMPLEMENTATION_PLAN/03_API/MyUplink_Complete_Guide.md`
- Entity IDs for F750/F2040/S-series
- Switch control methods

**NIBE Forum Research:**
- `IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md`
- Real-world DHW optimization cases
- Thermal debt impact studies

**Swedish NIBE Forums:**
- `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md`
- Menu 5.1.1 DHW settings
- Legionella protection schedules

### Implementation Files

**Core Logic:**
- `custom_components/effektguard/optimization/dhw_optimizer.py` - Scheduling engine
- `custom_components/effektguard/const.py` - DHW constants
- `custom_components/effektguard/__init__.py` - boost_dhw service

**Tests:**
- `tests/test_dhw_optimizer.py` - Unit tests
- `tests/conftest.py` - Test fixtures

**Documentation:**
- `docs/DHW_OPTIMIZATION.md` - This file
- `IMPLEMENTATION_PLAN/POST_PHASE_5_ROADMAP.md` - Future enhancements

### Research Decisions

**Key Architectural Decisions:**

1. **Monitor-Only Legionella:** NIBE's fixed schedule + MyUplink API limitations
2. **Temporary Lux Control:** Simplest reliable DHW boost method
3. **24h Window:** Balance tank cooling vs price optimization
4. **BT7 History Tracking:** Automatic detection without manual intervention
5. **Thermal Debt Priority:** Space heating safety > DHW comfort

---

## Changelog

### 2025-01-28 - Initial Implementation
- ✅ Temporary lux switch integration
- ✅ BT7 temperature monitoring
- ✅ Legionella detection (monitor-only)
- ✅ 24h scheduling window with smart fallback
- ✅ Thermal lag documentation fixes
- ✅ All unit tests passing

### Future Enhancements

**Planned Features:**
- Learning-based demand prediction (Phase 6+)
- Multi-zone DHW optimization (if multiple tanks)
- Integration with solar thermal (if available)
- Advanced Legionella cost optimization (if API improves)

---

**End of Document**
