# Climate zones

EffektGuard reads Home Assistant's latitude, picks one of five climate zones, and from the zone
and the current outdoor temperature derives what degree-minute (DM) reading is normal. Nothing is
configured. The tables below are what `ClimateZoneDetector.get_expected_dm_range()` returns, and a
test keeps them equal to the code. How the numbers are made:
[architecture/10_adaptive_climate_zones.md](architecture/10_adaptive_climate_zones.md).

In every zone the warning threshold equals the deep end of the normal range: a DM reading past
"normal" is already past "warning", and the T1 recovery tier starts there. There is no WARNING or
CAUTION tier.

## Zones

### 🥶 Extreme Cold Zone (66.5°N - 90°N)

Kiruna (SWE), Tromsø (NOR), Fairbanks (USA). Average winter low: -20°C. Flow safety margin
+2.5°C.

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -30°C        | -1000 to -1400 | -1400             |
| -20°C        | -800 to -1200  | -1200             |
| -10°C        | -600 to -1000  | -1000             |

### ❄️ Very Cold Zone (60.5°N - 66.5°N)

Luleå (SWE), Umeå (SWE), Oulu (FIN), Trondheim (NOR). Average winter low: -12°C. Flow safety
margin +1.5°C.

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -20°C        | -760 to -1160  | -1160             |
| -15°C        | -660 to -1060  | -1060             |
| -10°C        | -560 to -960   | -960              |
| -5°C         | -460 to -860   | -860              |

### 🌨️ Cold Zone (56.0°N - 60.5°N)

Stockholm (SWE), Oslo (NOR), Göteborg (SWE), Helsinki (FIN). Average winter low: -8°C. Flow
safety margin +1.0°C.

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -15°C        | -590 to -840   | -840              |
| -10°C        | -490 to -740   | -740              |
| -5°C         | -390 to -640   | -640              |
| 0°C          | -290 to -540   | -540              |

### 🌡️ Moderate Cold Zone (54.5°N - 56.0°N)

Copenhagen (DEN), Malmö (SWE), Aarhus (DEN), Helsingborg (SWE). Average winter low: -1°C. Flow
safety margin +0.5°C.

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| -5°C         | -380 to -580   | -580              |
| 0°C          | -280 to -480   | -480              |
| 5°C          | -180 to -380   | -380              |

### 🏠 Standard Zone (<54.5°N)

Paris (FRA), London (UK), Berlin (GER) and everywhere else. Average winter low: 0°C. No extra
flow safety margin.

| Outdoor Temp | Normal DM Range | Warning Threshold |
|--------------|----------------|-------------------|
| 0°C          | -200 to -350   | -350              |
| 5°C          | -100 to -250   | -250              |
| 10°C         | 0 to -150      | -150              |

## How the thresholds move

`adjustment = (outdoor_temp - zone_avg_winter_low) × 20 DM/°C`, added to the zone's base range.
Colder than the zone's winter average gives a negative adjustment and a deeper threshold; warmer
gives a shallower one.

Stockholm, Cold zone, winter average -8°C, base warning -700:

- at -20°C: (-20 - (-8)) × 20 = -240, warning -940
- at 0°C: (0 - (-8)) × 20 = +160, warning -540

The warning threshold never goes shallower than -110 DM, where NIBE's compressor cycles on every
normal run, and never deeper than -1450. Degree minutes at -1500 (`DM_THRESHOLD_AUX_LIMIT`) are the
EMERGENCY tier in every zone. A concrete slab or timber floor then has its thresholds divided by
1.3 or 1.15 so the slow house starts recovering sooner.

## Other facts

- Latitude is used as an absolute value: Melbourne at 37.8°S is treated like 37.8°N (Standard),
  an Antarctic station at 75°S like 75°N (Extreme Cold).
- A latitude exactly on a boundary gets the colder zone.
- The zone cannot be overridden.
- Weather compensation uses the zone too, for the flow safety margin and the layer weight
  ([architecture/10](architecture/10_adaptive_climate_zones.md)).

## Code

```python
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector

detector = ClimateZoneDetector(latitude=59.33)
detector.zone_info.zone_key          # "cold"
detector.get_expected_dm_range(-10.0)
# {"normal_min": -490.0, "normal_max": -740.0, "warning": -740.0, "critical": -1500}
```

Tests: `tests/unit/climate/test_climate_zones.py`,
`tests/validation/test_climate_zones_doc_matches_the_code.py`. Sources for the degree-minute
limits: [research/01_degree_minutes.md](research/01_degree_minutes.md).
