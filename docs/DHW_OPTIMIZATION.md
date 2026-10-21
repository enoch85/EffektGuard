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

All DHW decisions flow through `should_start_dhw()` in `dhw_optimizer.py`: rule-ordered early
returns, the first rule that applies decides. The scheduled window (Lane 1) is checked first; outside
it the rules run in this order:

| Rule | Priority reason | Triggers when | Result |
|------|-----------------|---------------|--------|
| **0 - Lane 1** | `DHW_SCHEDULED_BLOCKED_BY_SAFETY` | Inside the scheduled window, the window's safety gate refuses | Wait; owed to Rule 0.5 |
| | `DHW_SCHEDULED_TARGET_REACHED_TEMP_*C` / `_*MIN` | Inside the window, target temperature or amount reached | Stop heating |
| | `DHW_SCHEDULED_OPTIMAL_Q*` | Inside the window, in its cheapest window now | Heat now |
| | `DHW_SCHEDULED_WAITING_OPTIMAL_*` | A cheaper window ahead and slack to wait for it | Wait |
| | `DHW_PEAK_AVOIDED_SCHEDULED` | No window found and the quarter is PEAK | Wait |
| | `DHW_SCHEDULED_PRIORITY_*H` | Otherwise inside the window | Heat now |
| **0.5** | `DHW_SCHEDULED_RETRY_AFTER_SAFETY` | A window safety refused, now safe | Heat now |
| **1** | `CRITICAL_THERMAL_DEBT` | DM at or below the climate-aware block threshold | Block - safety |
| **2** | `SPACE_HEATING_EMERGENCY` | Indoor deficit above `DHW_SPACE_HEATING_DEFICIT_THRESHOLD` and outdoor below `DHW_SPACE_HEATING_OUTDOOR_THRESHOLD` | Block - safety |
| **3** | `DHW_SAFETY_WINDOW_Q*` / `DHW_SAFETY_WAITING_WINDOW_*` / `DHW_SAFETY_DEFERRED_PEAK_PRICE` / `DHW_SAFETY_MINIMUM` | Tank below `DHW_SAFETY_MIN` (30 °C) | Heat in a cheap window, wait for one, defer at peak, or heat at once when critical |
| **2.6** | `DHW_PEAK_AVOIDED` | PEAK price | Wait |
| **2.3** | `DHW_HYGIENE_BOOST` | No 55 °C cycle for `DHW_LEGIONELLA_MAX_DAYS` (14) and a stable cheap period | Heat to 56 °C |
| **2.5** | `DHW_COMPLETE_EMERGENCY_HEATING` | Tank 30-45 °C and a stable cheap period | Complete to the target |
| **3.5** | `DHW_MAX_WAIT_EXCEEDED_*H` | More than `DHW_MAX_WAIT_HOURS` (36) since the last heating, cheap or normal price | Heat now |
| **4** | `HIGH_SPACE_HEATING_DEMAND` | Demand above 6 kW and DM below `DM_THRESHOLD_START` | Block DHW |
| **4.7** | `PEAK_AVOIDANCE_PREHEAT_*H` | Cheap or normal now, and a PEAK ahead would otherwise force heating then | Heat now |
| **5 - Lane 2** | `DHW_ADEQUATE_WAITING_CHEAP_*` / `CHEAP_NO_WINDOW_DATA` / `OPTIMAL_WINDOW_Q*` / `WAITING_OPTIMAL_WINDOW_IN_*` / `CHEAP_ELECTRICITY_OPPORTUNITY` | Normal price optimisation | Heat in the cheapest window, or wait for it |
| **7** | `DHW_COMFORT_OPTIMAL_*` / `DHW_COMFORT_WAITING_OPTIMAL_*` / `DHW_COMFORT_LOW_*` | Tank getting low | Heat now or in a cheaper window ahead |
| **8** | `DHW_ADEQUATE` | Nothing above applies | Don't heat |

### Why Recommendations Change Frequently

**1. Input Data Changes (most common)**
- `price_classification` changes every 15 min (cheap→normal→expensive)
- `thermal_debt_dm` fluctuates with compressor activity
- `current_dhw_temp` drifts as tank cools/heats
- `dhw_amount_minutes` changes with water usage
- `is_volatile` changes as price runs extend/shorten

**2. Time-Dependent Thresholds**
- `optimal_window.hours_until` crosses the 0.25h (15min) activation threshold
- `within_scheduled_window` changes when entering or leaving the scheduled window
- `hours_until_target` crosses demand period boundaries

**3. Rate Limiting vs Recommendation**
The coordinator has 60-minute rate limiting for actual control actions, but **recommendations are calculated every coordinator cycle** (typically 5 min). So:
- Recommendation can change 12x per hour
- But actual control action only changes 1x per hour

### Two-Lane Architecture

**Lane 1: the scheduled window** (Rule 0)
- Active from `DHW_SCHEDULED_WINDOW_HOURS` (6 h) before a demand period's target time - or earlier
  for a tank too slow to heat in 6 h: the window opens at the larger of 6 h and the tank's heating
  time plus `DHW_AMOUNT_HEATING_BUFFER` (`_check_upcoming_demand_period`)
- Stops heating once the target temperature or amount is reached
- Waits for a cheaper window only when there is slack for it; otherwise heats (never at PEAK unless
  the tank is critically low)

**Lane 2: normal price optimisation** (Rules 0.5-8)
- Everything outside the scheduled window: safety first, then the cheapest windows over the price
  horizon

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

**Implementation** (`IntelligentDHWScheduler.update_bt7_temperature`, fed every cycle):

```python
reached = temp >= DHW_LEGIONELLA_DETECT and (
    not self.bt7_history or self.bt7_history[-1][1] < DHW_LEGIONELLA_DETECT
)
self.bt7_history.append((timestamp, temp))  # deque(maxlen=48): 4 hours at the 5-minute cycle
if reached:
    self.last_legionella_boost = timestamp
    _LOGGER.info("Hot water reached %.1f°C at %s: high-temperature cycle seen", temp, timestamp)
```

A cycle is seen the moment BT7 reaches `DHW_LEGIONELLA_DETECT` (55 °C) - the same rule the start-up
restore applies to the recorder's history. (It used to need the tank to cool 3 °C below its peak
inside the 4-hour history; the owner's F750 takes 5.5-17.7 h to do that, so NIBE's own cycles went
unseen. Fixed in #55.)

**What the hygiene rule does and does not do:**
- RULE 2.3 (`DHW_HYGIENE_BOOST`) is an opportunistic top-up into a stable cheap period when no 55 °C
  cycle was seen for `DHW_LEGIONELLA_MAX_DAYS` (14). It **cannot perform a hygiene cycle**: the
  temporary-lux boost stops at the pump's luxury stop temperature (53-54 °C on factory settings),
  below the 55 °C threshold.
- Hygiene is NIBE's own periodic increase (menu 2.9.1 F-series / 2.4 S-series, factory every 14 days
  to 55 °C). If no cycle is seen for `DHW_LEGIONELLA_OVERDUE_DAYS` (21), EffektGuard warns that it may
  have been switched off on the pump; it does not substitute for it.

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

**The scheduled window** (`_check_upcoming_demand_period`):
```python
def _check_upcoming_demand_period(self, current_time, current_dhw_temp=None):
    # The window opens DHW_SCHEDULED_WINDOW_HOURS before the target, or as early as a tank
    # too slow to heat in that time needs.
    window = DHW_SCHEDULED_WINDOW_HOURS
    if current_dhw_temp is not None and current_dhw_temp < self.user_target_temp:
        window = max(
            window,
            self.estimate_heating_time(current_dhw_temp, self.user_target_temp)
            + DHW_AMOUNT_HEATING_BUFFER,
        )
    # ... for the closest demand period within DHW_SCHEDULING_WINDOW_MAX (24 h), in real hours:
    #     "within_scheduled_window": hours_until <= window
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
- **Lane 1 (inside the scheduled window):** heat the tank to its target before the demand period;
  stop once the target temperature or amount is reached (Rule 0)
- **Lane 2 (outside it):** normal price optimisation finds the cheapest windows (Rules 0.5-8)

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

The hot-water tests live in `tests/unit/dhw/` (one behaviour per file, e.g.
`test_a_slow_tank_is_heated_in_time.py`, `test_nibe_s_own_hot_water_cycle_is_seen.py`,
`test_every_dhw_abort_condition_is_honoured.py`):

```bash
pytest tests/unit/dhw/ -q
```

### Integration Testing

**Manual Testing Checklist:**
- [ ] Config flow accepts DHW entities
- [ ] BT7 sensor updates tracked in history
- [ ] A 55 °C reading logs "high-temperature cycle seen"
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

The NIBE facts this optimiser relies on (the periodic increase, temporary lux and its luxury stop
temperature) are sourced in the comments at RULE 2.3 in `dhw_optimizer.py`: the NIBE F750/F730/F1155
installer manuals, menus 2.9.1 and 5.1.1, and the register map (47046/47050/47051).

### Implementation Files

**Core Logic:**
- `custom_components/effektguard/optimization/dhw_optimizer.py` - Scheduling engine
- `custom_components/effektguard/const.py` - DHW constants
- `custom_components/effektguard/__init__.py` - boost_dhw service

**Tests:**
- `tests/unit/dhw/` - Unit tests
- `tests/conftest.py` - Test fixtures

**Documentation:**
- `docs/DHW_OPTIMIZATION.md` - This file

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
