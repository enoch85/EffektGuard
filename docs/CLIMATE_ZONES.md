# Climate Zone System

## Overview

EffektGuard automatically adapts to your climate using latitude-based zone detection. This ensures that thermal debt (degree minutes) thresholds are appropriate for your location - what's normal in Kiruna is an emergency in Paris!

## Automatic Detection

The system uses your Home Assistant's configured latitude to detect your climate zone. **No configuration needed.**

```yaml
# In your Home Assistant configuration.yaml
homeassistant:
  latitude: 59.3293  # Stockholm
  longitude: 18.0686
```

EffektGuard automatically detects: **Cold Zone** (56-60.5°N)

## Climate Zones

### 🥶 Extreme Cold Zone (66.5°N - 90°N)

**Examples:** Kiruna (SWE), Tromsø (NOR), Fairbanks (USA)

**Winter characteristics:**
- Average winter low: -20°C
- Severe heating demands
- Heat pump works continuously in winter

**Expected degree minutes (DM):**

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -30°C        | -1000 to -1400 | -1400             |
| -20°C        | -800 to -1200  | -1200             |
| -10°C        | -600 to -1000  | -1000             |

**Safety margin:** +2.5°C additional flow temperature headroom

**Why it matters:** DM -1000 at -30°C is **perfectly normal** - your heat pump is working hard, but not in distress!

---

### ❄️ Very Cold Zone (60.5°N - 66.5°N)

**Examples:** Luleå (SWE), Umeå (SWE), Oulu (FIN), Trondheim (NOR)

**Winter characteristics:**
- Average winter low: -12°C
- Heavy heating demands
- Extended compressor run times

**Expected degree minutes (DM):**

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -20°C        | -760 to -1160  | -1160             |
| -15°C        | -660 to -1060  | -1060             |
| -10°C        | -560 to -960   | -960              |
| -5°C         | -460 to -860   | -860              |

**Safety margin:** +1.5°C additional flow temperature headroom

---

### 🌨️ Cold Zone (56.0°N - 60.5°N)

**Examples:** Stockholm (SWE), Oslo (NOR), Göteborg (SWE), Helsinki (FIN)

**Winter characteristics:**
- Average winter low: -8°C
- Substantial heating demands
- Standard Nordic winter operation

**Expected degree minutes (DM):**

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -15°C        | -590 to -840   | -840              |
| -10°C        | -490 to -740   | -740              |
| -5°C         | -390 to -640   | -640              |
| 0°C          | -290 to -540   | -540              |

**Safety margin:** +1.0°C additional flow temperature headroom

**Most common zone for Swedish users!**

---

### 🌡️ Moderate Cold Zone (54.5°N - 56.0°N)

**Examples:** Copenhagen (DEN), Malmö (SWE), Aarhus (DEN), Helsingborg (SWE)

**Winter characteristics:**
- Average winter low: -1°C
- Moderate heating demands
- Milder Nordic climate (Øresund region)

**Expected degree minutes (DM):**

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -5°C         | -380 to -580   | -580              |
| 0°C          | -280 to -480   | -480              |
| 5°C          | -180 to -380   | -380              |

**Safety margin:** +0.5°C additional flow temperature headroom

---

### 🏠 Standard Zone (<54.5°N)

**Examples:** Paris (FRA), London (UK), Berlin (GER), and all other locations

**Winter characteristics:**
- Average winter low: 0°C
- Minimal heating demands
- Mild climate, less optimization benefit

**Expected degree minutes (DM):**

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| 0°C          | -200 to -350   | -350              |
| 5°C          | -100 to -250   | -250              |
| 10°C         | 0 to -150      | -150              |

**Safety margin:** No additional headroom needed

---

## How It Works

### Temperature Adjustment

The base DM thresholds adjust dynamically based on current outdoor temperature:

**Formula:** `adjustment = (outdoor_temp - zone_avg_winter_low) × 20 DM/°C`

Colder than the zone average ⇒ the difference is negative ⇒ the threshold moves **deeper**.
Warmer ⇒ positive ⇒ **shallower**. (`climate_zones.py:231` — `temp_delta = outdoor_temp -
self.zone_info.winter_avg_low`. This document previously stated the subtraction the other way
round, which yields the opposite sign, and then wrote the correct number anyway — so the formula
and its own worked example disagreed. If you are deriving a threshold by hand, use the code.)

**Example (Stockholm — Cold Zone, winter avg −8°C):**
- Zone winter average: **−8°C**
- Current outdoor: −20°C
- `temp_delta` = −20 − (−8) = **−12°C** (colder than average)
- Adjustment: −12 × 20 = **−240 DM** (deeper threshold)
- Base warning: −700 DM
- **Adjusted: −940 DM** (allows deeper DM in colder weather)

**Conversely at 0°C:**
- `temp_delta` = 0 − (−8) = **+8°C** (warmer than average)
- Adjustment: +8 × 20 = **+160 DM** (shallower threshold)
- **Adjusted: −540 DM** (tighter tolerance in mild weather)

**⚠️ The WARNING band has zero width.** In every zone `normal_max == warning`
(`DM_CRITICAL_T1_MARGIN` is 0), so a DM that is "past normal" is already "past warning". The
intermediate WARNING/CAUTION states are unreachable. This is a known open finding — do not
"correct" the tables to conceal it.

**⚠️ On a NIBE F750 the pump acts first.** Its own "start addition" (menu 4.9.3) defaults to
**−700**, so the immersion heater engages *before* Stockholm's −740 warning threshold is ever
reached, and works DM back up. Do not reason about −1500 as though the elpatron were not there.

### Absolute Safety Limit

**All zones respect:** DM -1500 absolute maximum (Swedish research validated)

Even in extreme Arctic conditions, the system will trigger emergency recovery before exceeding -1500 DM.

---

## Southern Hemisphere

The system uses **absolute latitude** values, so it works equally well in the southern hemisphere:

- Melbourne (37.8°S) → Same as 37.8°N → Standard Zone
- Punta Arenas (53.2°S) → Same as 53.2°N → Standard Zone
- Antarctica research stations (70°S+) → Same as 70°N+ → Extreme Cold Zone

Climate zones are symmetric around the equator for heating purposes.

---

## Code Reference

### Implementation

```python
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

# Automatic detection from Home Assistant latitude
detector = ClimateZoneDetector(latitude=hass.config.latitude)

# Access zone information
zone_info = detector.zone_info
print(f"Zone: {zone_info.zone_key}")
print(f"Winter avg low: {zone_info.winter_avg_low}°C")

# Get expected DM range for current conditions
dm_range = detector.get_expected_dm_range(outdoor_temp=-10.0)
print(f"Normal DM range: {dm_range['normal_min']} to {dm_range['normal_max']}")
print(f"Warning threshold: {dm_range['warning']}")
```

### Module Location

`custom_components/effektguard/optimization/climate_zones.py`

### Integration Points

1. **Decision Engine** (`decision_engine.py`): Uses climate-aware DM thresholds for emergency layer
2. **Weather Compensation** (`weather_compensation.py`): Adds climate-based safety margins to flow temperature
3. **Configuration** (`const.py`): Imports zone definitions from dedicated module

---

## Benefits

### Global Applicability

- Works from Arctic Circle to Mediterranean
- No configuration needed beyond Home Assistant latitude
- Automatically adapts to unusual weather

### Safety-First

- Context-aware thresholds prevent false alarms
- Arctic users aren't constantly in "warning" state
- Mild climate users get tighter monitoring

### Real-World Validated

- Based on Swedish NIBE forum research
- DM -1500 absolute limit validated across Nordic conditions
- Zone thresholds derived from actual F2040/F750 case studies

---

## FAQ

### Why heating-focused zone names?

The original implementation used geographic terms (Arctic, Oceanic) that didn't clearly indicate heating needs. The new names (Extreme Cold → Standard) immediately convey how much heating optimization matters for your location.

### What if I'm between zones?

Zone boundaries are designed to align with real climate transitions:
- 66.5°N = Arctic Circle (actual climate boundary)
- 60.5°N = Northern vs Southern Nordics
- 54.5°N = Nordic vs Central Europe

If you're exactly on a boundary, you'll get the colder zone (more conservative approach).

### Can I override the detection?

Not currently needed - the automatic detection is very accurate. If you have a microclimate situation, the system's temperature-based adjustments will compensate automatically.

### Does this replace weather compensation?

No! Climate zones work **with** weather compensation:
- **Weather compensation** (André Kühne formula): Calculates optimal flow temperature
- **Climate zones**: Add safety margins and set DM expectations
- **Together**: Maximum efficiency with appropriate safety buffers

---

## Testing

Comprehensive test coverage in `tests/test_climate_zones.py` and `tests/test_decision_engine_climate_integration.py`:

- ✅ Zone detection for all 5 zones
- ✅ DM calculations across temperature ranges
- ✅ Arctic scenario (DM -1200 normal at -30°C)
- ✅ Standard scenario (DM -400 warning at 0°C)
- ✅ Boundary conditions
- ✅ Temperature adjustments
- ✅ Absolute maximum safety limits
- ✅ Global applicability (7+ cities tested)

**Total: 61 dedicated climate zone tests, all passing**

---

## References

- Implementation plan: `IMPLEMENTATION_PLAN/FUTURE/CLIMATE_ZONE_DM_INTEGRATION.md`
- Architecture: `architecture/10_adaptive_climate_zones.md`
- Test suite: `tests/test_climate_zones.py`
- Integration tests: `tests/test_decision_engine_climate_integration.py`
- Swedish research: `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md`
