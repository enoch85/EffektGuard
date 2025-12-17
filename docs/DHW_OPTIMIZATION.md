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
- ✅ **24h Forecast Window:** Maximize optimization with Spot Price's 48h data
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

All DHW decisions flow through `should_start_dhw()` in `dhw_optimizer.py`. The method has **18 decision return points** organized by priority:

| Rule | Priority Reason | Triggers When | Result |
|------|----------------|---------------|--------|
| **0** | `DHW_AMOUNT_TARGET_REACHED_*` | Within scheduled window + amount ≥ target | Stop heating |
| **1** | `CRITICAL_THERMAL_DEBT` | DM ≤ climate-aware warning threshold | Block - safety |
| **2** | `SPACE_HEATING_EMERGENCY` | Indoor >0.5°C below target + outdoor <0°C | Block - safety |
| **3a** | `DHW_SAFETY_WINDOW_Q*` | Temp <30°C + in optimal cheap window | Heat now |
| **3b** | `DHW_SAFETY_WAITING_WINDOW_*` | Temp <30°C + better window ahead | Wait |
| **3c** | `DHW_SAFETY_DEFERRED_PEAK_PRICE` | Temp 20-30°C + expensive + healthy DM | Wait |
| **3d** | `DHW_SAFETY_MINIMUM` | Temp <30°C (critical) | Heat immediately |
| **2.3a** | `DHW_HYGIENE_BOOST` | 14+ days since legionella + stable cheap | Heat to 56°C |
| **2.5** | `DHW_COMPLETE_EMERGENCY_HEATING` | Temp 30-45°C + stable cheap | Complete heating |
| **3.5** | `DHW_MAX_WAIT_EXCEEDED_*` | 36+ hours since last DHW + cheap/normal | Heat now |
| **4** | `HIGH_SPACE_HEATING_DEMAND` | Demand >6kW + DM < -60 (recovering) | Block DHW |
| **5a** | `DHW_ADEQUATE_WAITING_CHEAP_*` | Temp ≥45°C + not cheap price | Wait |
| **5b** | `OPTIMAL_WINDOW_Q*` | Within 15min of cheapest window | Heat now |
| **5c** | `WAITING_OPTIMAL_WINDOW_*` | Better window ahead + temp comfortable | Wait |
| **5d** | `CHEAP_NO_WINDOW_DATA` | No price data + stable cheap + low temp | Heat now |
| **7** | `DHW_COMFORT_LOW_CHEAP` | Temp <45°C + stable cheap | Heat now |
| **8** | `DHW_ADEQUATE` | All conditions fail | Don't heat |

### Why Recommendations Change Frequently

**1. Input Data Changes (most common)**
- `price_classification` changes every 15 min (cheap→normal→expensive)
- `thermal_debt_dm` fluctuates with compressor activity
- `current_dhw_temp` drifts as tank cools/heats
- `dhw_amount_minutes` changes with water usage
- `is_volatile` changes as price runs extend/shorten

**2. Time-Dependent Thresholds**
- `optimal_window.hours_until` crosses the 0.25h (15min) activation threshold
- `within_scheduled_window` changes when entering/leaving 6h window
- `hours_until_target` crosses demand period boundaries

**3. Rate Limiting vs Recommendation**
The coordinator has 60-minute rate limiting for actual control actions, but **recommendations are calculated every coordinator cycle** (typically 5 min). So:
- Recommendation can change 12x per hour
- But actual control action only changes 1x per hour

### Two-Lane Architecture

**Lane 1: Normal 24h Price Optimization**
- Always active, regardless of scheduling
- Finds cheapest windows over 24h lookahead
- Rules 1-8 (except 0) handle this lane

**Lane 2: Scheduled Window Check**
- Only active within `DHW_SCHEDULED_WINDOW_HOURS` (6h) of target time
- Rule 0 checks if target amount is reached
- Stops heating to avoid waste when target met
- Outside the 6h window, normal price optimization continues

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
        - BT7 peaked at ≥55°C recently (observed 55.3°C in production)
        - Now cooling down (dropped 3°C from peak)
        - Confirms boost completed successfully
        """
        if len(self.bt7_history) < 10:
            return False
        
        temps = [temp for _, temp in self.bt7_history]
        max_temp = max(temps)
        current_temp = temps[-1]
        
        # Peak ≥55°C and now cooling
        if max_temp >= 55.0 and current_temp < (max_temp - 3.0):
            # Check not already recorded
            if self.last_legionella_boost:
                hours_since = (self.bt7_history[-1][0] - self.last_legionella_boost).total_seconds() / 3600
                if hours_since < 6.0:  # Same boost event
                    return False
            return True
        
        return False
```

**Benefits:**
- ✅ Detects NIBE's automatic boosts to avoid conflicts
- ✅ Triggers our own hygiene boosts every 14 days during cheap periods (RULE 2.3)
- ✅ Automatic detection (no manual tracking needed)
- ✅ Optimizes cost by scheduling during cheapest electricity

**Note:** We BOTH monitor NIBE's automatic boosts AND trigger our own hygiene boosts every 14 days during cheap periods. This ensures bacteria prevention even with our lower safety thresholds (20°C critical/30°C minimum).

### DHW Immersion Heater Requirement for High-Temperature Cycles

**Heat Pump Compressor Limitation:**

Heat pumps (compressor-only operation) can typically only reach **50-55°C** maximum DHW temperature due to COP/efficiency constraints. To reach higher temperatures for Legionella prevention, **a DHW tank immersion heater is required**.

**Real-World Observation:**

User systems reach a maximum of **56°C** with compressor + immersion heater combined. While Boverket.se recommends 60°C ideally, the 56°C achieved in practice still provides significant bacterial reduction.

**How NIBE Handles This:**

1. **Compressor heats to ~50-55°C** (efficient operation)
2. **DHW tank immersion heater boosts to 56°C** (final boost, real-world max)
3. This is **automatic** in NIBE systems - no special configuration needed
4. Works even when auxiliary heating is **blocked for space heating** (Menu setting)

**EffektGuard's New Hygiene Boost (v0.1.2+):**

With lowered safety thresholds (20°C critical/30°C minimum), DHW can sit in the Legionella growth zone (20-45°C) for extended periods. To prevent bacterial growth:

```python
# RULE 2.3: HYGIENE BOOST
# If DHW hasn't been above 56°C in past 14 days, heat to 56°C during cheapest period
if days_since_legionella >= 14 and price_classification == "cheap":
    # Request 56°C target (real-world max with immersion heater)
    # NIBE will use compressor + DHW immersion heater automatically
    # Scheduled during cheap electricity to minimize cost
    return heat_to_56C()
```

**Official Guidelines (Boverket.se):**

- Legionella bacteria **grow at 20-45°C** (our optimization range!)
- Legionella **dormant below 20°C**
- Legionella **killed at ≥60°C** (ideal), **significantly reduced at 56°C** (real-world)
- Water heaters should **maintain ≥60°C** or periodic high-temp cycles

**Cost Optimization:**

By scheduling the 56°C hygiene boost during **cheapest electricity periods**, we minimize the cost of the immersion heater while ensuring bacterial safety.

**Terminology Clarification:**

- **DHW immersion heater** (Swedish: "elpatron") = In-tank electrical heating element for hot water
- **Auxiliary heating** (Swedish: "tilläggsvärme") = Separate space heating backup system
- These are **different systems** - DHW immersion heater is built into the hot water tank specifically for high-temperature cycles

**References:**
- [Boverket: Om legionella](https://www.boverket.se/sv/byggande/halsa-och-inomhusmiljo/om-legionella/)
- [Värmepumpsforum: F750 DHW temperature discussion](https://www.varmepumpsforum.com/vpforum/index.php?topic=53888.0)

---

## Scheduling Logic

### 24-Hour Forecast Window

**Why 24h?**
- Spot Price provides up to 48h price forecast
- DHW tank cools slowly (can preheat far ahead)
- Maximizes price optimization opportunities
- DHW heats fast (1.5h), so timing flexibility is high

**Two-Lane Architecture:**
```python
def _check_upcoming_demand_period(self, current_time: datetime) -> DemandPeriodInfoDict | None:
    """Check for demand periods with 24h monitoring and 6h active window.
    
    Lane 1: Normal 24h price optimization (always active)
    Lane 2: Scheduled window check (within DHW_SCHEDULED_WINDOW_HOURS of target)
    """
    for period in self.demand_periods:
        hours_until = calculate_hours_until(period, current_time)
        
        # Monitor within 24h for display, active scheduling within 6h
        if hours_until <= DHW_SCHEDULING_WINDOW_MAX:  # 24 hours
            return {
                "min_amount_minutes": period.min_amount_minutes,
                "target_temp": period.target_temp,
                "availability_time": availability_time,
                "hours_until": hours_until,
                "within_scheduled_window": hours_until <= DHW_SCHEDULED_WINDOW_HOURS,  # 6h
            }
    
    return None
```

### Demand Periods

**User-Configured High Demand Times:**
```python
DHWDemandPeriod(
    availability_hour=7,    # 7:00 AM - water should be READY by this time
    target_temp=55.0,       # Comfort temperature (fallback if no amount sensor)
    duration_hours=2,       # 7:00-9:00 AM window
    min_amount_minutes=5,   # Minimum 5 minutes of hot water required
)
```

**Note:** `availability_hour` is when hot water should be READY (available), not when heating should start. The optimizer calculates backwards to determine when to start heating based on current DHW amount.

**Scheduling Strategy (Two Lanes):**
- **Lane 1 (Always Active):** Normal 24h price optimization finds cheapest windows
- **Lane 2 (Within 6h of target):** Amount-based scheduling checks if target met
  - If DHW amount ≥ target → Stop heating (Rule 0)
  - If DHW amount < target → Continue to price optimization rules

### Price-Aware Scheduling

**Integration with Spot Price:**
```python
def find_optimal_heating_time(
    demand_period: DHWDemandPeriod,
    price_forecast: list[QuarterPeriod],
) -> datetime:
    """Find cheapest 2-hour window before demand period.
    
    Uses Spot Price's native 15-minute intervals.
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
CONF_NIBE_TEMP_LUX_ENTITY = "switch....." # Temporary lux switch
CONF_DHW_DEMAND_PERIODS = [                # High demand times (JSON)
    {
        "availability_hour": 7,
        "target_temp": 55.0,
        "duration_hours": 2,
        "min_amount_minutes": 5,
    },
    {
        "availability_hour": 19,
        "target_temp": 50.0,
        "duration_hours": 1,
        "min_amount_minutes": 5,
    }
]
```
```

### Constants

**DHW-Specific Constants (const.py):**
```python
# Temperature hierarchy
DHW_SAFETY_CRITICAL = 20.0       # °C - Hard floor, always heat (emergency)
DHW_SAFETY_MIN = 30.0            # °C - Price optimization minimum
DHW_MIN_TEMP = 40.0              # °C - User-configurable minimum
MIN_DHW_TARGET_TEMP = 45.0       # °C - Minimum user target / NIBE start
DEFAULT_DHW_TARGET_TEMP = 50.0   # °C - Default comfort target
DHW_MAX_TEMP = 60.0              # °C - Maximum normal DHW
DHW_LEGIONELLA_DETECT = 55.0     # °C - BT7 temp indicating boost complete
DHW_LEGIONELLA_PREVENT_TEMP = 56.0  # °C - Hygiene boost target

# Timing
DHW_HEATING_TIME_HOURS = 1.5     # Hours to heat tank
DHW_COOLING_RATE = 0.5           # °C per hour cooling
DHW_SCHEDULING_WINDOW_MAX = 24   # Max hours ahead for scheduling
DHW_SCHEDULED_WINDOW_HOURS = 6   # Hours before target when scheduling active
DHW_MAX_WAIT_HOURS = 36.0        # Max hours between DHW heating
DHW_LEGIONELLA_MAX_DAYS = 14.0   # Days without high-temp = trigger hygiene boost

# Space heating thresholds for DHW blocking
DHW_SPACE_HEATING_DEFICIT_THRESHOLD = 0.5  # °C indoor deficit
DHW_SPACE_HEATING_OUTDOOR_THRESHOLD = 0.0  # °C outdoor temp
SPACE_HEATING_DEMAND_HIGH_THRESHOLD = 6.0  # kW - Rule 4 blocking

# Runtime limits (monitoring only - NIBE controls actual completion)
DHW_SAFETY_RUNTIME_MINUTES = 30  # Safety minimum heating
DHW_NORMAL_RUNTIME_MINUTES = 45  # Normal heating window
DHW_EXTENDED_RUNTIME_MINUTES = 60  # High demand period
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
