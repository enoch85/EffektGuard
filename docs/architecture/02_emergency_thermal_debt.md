# Scenario 2: Thermal Debt - the Degree-Minute Ladder

**Description**: how two layers answer deepening degree minutes (DM). `ProactiveLayer` (Z1-Z5) acts
before the climate-aware warning threshold, `EmergencyLayer` (T1-T3, EMERGENCY) from it. Both live
in `optimization/thermal_layer.py`; their thresholds come from `ClimateZoneDetector`.

The ladder is `Z1 → Z2 → Z3 → Z4 → Z5 → T1 → T2 → T3 → EMERGENCY`. There is no WARNING or CAUTION
tier: `normal_max == warning` in every climate zone, so they could never fire, and they were removed
(see the open-findings register).

```mermaid
flowchart TD
    A[DM, outdoor, indoor] --> B[ClimateZoneDetector.get_expected_dm_range<br/>latitude + outdoor temperature]
    B --> C[Thermal-mass buffer<br/>concrete ×1.3, timber ×1.15, radiator ×1.0<br/>thresholds divided: a slab acts earlier]

    C --> E
    C --> P

    subgraph "EmergencyLayer"
    E{DM ≤ DM_THRESHOLD_AUX_LIMIT<br/>and room not above its band?}
    E -->|yes| EM[EMERGENCY<br/>no cuts: the others' verdict floored at 0<br/>no raise of its own]
    E -->|no| AW{Anti-windup cooldown,<br/>or DM falling ≥ 50/h after a raise?}
    AW -->|yes| AWD[Hold the offset, or cut by |rate| / 100<br/>weight 0.7]
    AW -->|no| W{Room above its band, or at target<br/>while the price is not cheap?}
    W -->|yes| OK1[No vote]
    W -->|no| T3{DM ≤ T3?}
    T3 -->|yes| T3D[T3]
    T3 -->|no| T2{DM ≤ T2?}
    T2 -->|yes| T2D[T2]
    T2 -->|no| T1{DM ≤ T1 = warning?}
    T1 -->|yes| T1D[T1]
    T1 -->|no| OK2[No vote]
    end

    subgraph "ProactiveLayer"
    P{Indoor cooling rapidly<br/>in the cold?}
    P -->|yes| RC[Rapid-cooling boost]
    P -->|no| PZ{DM between normal_max × 2 %<br/>and the warning?}
    PZ -->|yes| Z[Z1 ... Z5<br/>weight trimmed for a warm house]
    PZ -->|no| OK3[No vote]
    end
```

Both layers vote, and `_aggregate_layers` combines them with the rest
([08_layer_priority_system.md](08_layer_priority_system.md)).

## The rungs

Values as in `const.py` on 2026-10-19; the names are what to read.

| Rung | Fires from | Offset | Weight |
|---|---|---|---|
| Z1 | `normal_max × PROACTIVE_ZONE1_THRESHOLD_PERCENT` (2 %) | `PROACTIVE_ZONE1_OFFSET` (+1.0) | `LAYER_WEIGHT_PROACTIVE_MIN` (0.3) |
| Z2 | × `PROACTIVE_ZONE2_THRESHOLD_PERCENT` (30 %) | `PROACTIVE_ZONE2_OFFSET` (+1.5) | `PROACTIVE_ZONE2_WEIGHT` (0.4) |
| Z3 | × `PROACTIVE_ZONE3_THRESHOLD_PERCENT` (50 %) | `PROACTIVE_ZONE3_OFFSET_MIN` + severity × `_RANGE` (+2.0..+2.5) | `PROACTIVE_ZONE3_WEIGHT` (0.5) |
| Z4 | × `PROACTIVE_ZONE4_THRESHOLD_PERCENT` (75 %) | `PROACTIVE_ZONE4_OFFSET` (+2.5) | `PROACTIVE_ZONE4_WEIGHT` (0.55) |
| Z5 | × `PROACTIVE_ZONE5_THRESHOLD_PERCENT` (87.5 %), down to the warning | `PROACTIVE_ZONE5_OFFSET` (+3.0) | `PROACTIVE_ZONE5_WEIGHT` (0.6) |
| T1 | warning − `DM_CRITICAL_T1_MARGIN` (0) | `DM_CRITICAL_T1_OFFSET` (+4.0) | `DM_CRITICAL_T1_WEIGHT` (0.65) |
| T2 | warning − `DM_CRITICAL_T2_MARGIN` (200) | `DM_CRITICAL_T2_OFFSET` (+7.0) | `DM_CRITICAL_T2_WEIGHT` (0.81) |
| T3 | max(warning − `DM_CRITICAL_T3_MARGIN` (400), `DM_CRITICAL_T3_MAX` (−1450)) | `DM_CRITICAL_T3_OFFSET` (+8.5) | `DM_CRITICAL_T3_WEIGHT` (0.91) |
| EMERGENCY | `DM_THRESHOLD_AUX_LIMIT` (−1500) | floor at 0 | 1.0 (applied by `_aggregate_layers`) |

While a T-tier is recovering, a critical cost layer may only moderate it down to
`DM_CRITICAL_PEAK_AWARE_OFFSETS` - see [08_layer_priority_system.md](08_layer_priority_system.md).

## Worked example (computed by the code, 2026-10-19)

Stockholm (latitude 59.33, the Cold zone) at −10 °C outdoors:

| | warning = normal_max | Z1 / Z2 / Z3 / Z4 / Z5 | T1 / T2 / T3 |
|---|---|---|---|
| Radiators | −740 | −15 / −222 / −370 / −555 / −648 | −740 / −940 / −1140 |
| Concrete slab | −569 | −11 / −171 / −285 / −427 / −498 | −569 / −769 / −969 |

The same pump at the same DM is "normal" in Kiruna and "in trouble" in Malmö. The zones, and how
the outdoor temperature moves them (more in [10_adaptive_climate_zones.md](10_adaptive_climate_zones.md);
`tests/validation/test_no_document_misquotes_the_safety_thresholds.py` holds these to the code):

| Zone | Latitude | Winter Avg | Base Normal Range | Base Warning |
|------|----------|------------|-------------------|--------------|
| Extreme Cold | 66.5°+ | -20°C | -800 to -1200 | -1200 |
| Very Cold | 60.5°-66.5° | -12°C | -600 to -1000 | -1000 |
| Cold | 56°-60.5° | -8°C | -450 to -700 | -700 |
| Moderate Cold | 54.5°-56° | -1°C | -300 to -500 | -500 |
| Standard | <54.5° | 0°C | -200 to -350 | -350 |

```
adjusted_warning = zone_warning + (outdoor_temp - zone_winter_avg) × 20
```

**Example: Stockholm (Cold zone, winter avg -8°C)**
- At -8°C: warning = -700 (no adjustment - this is the zone average)
- At -10°C: warning = -700 + (-10 - (-8)) × 20 = -700 - 40 = **-740** (deeper)
- At 0°C: warning = -700 + (0 - (-8)) × 20 = -700 + 160 = **-540** (shallower)
- At -20°C: warning = -700 + (-20 - (-8)) × 20 = -700 - 240 = **-940** (deeper)

Expected ranges at a cold hour:
- Stockholm (-10°C, Cold zone): expects DM -490 to -740, warning at -740
- Kiruna (-25°C, Extreme Cold zone): expects DM -900 to -1300, warning at -1300

⚠️ The Cold zone's winter average is **-8.0 °C**, not -10. This worked example used to say -10,
and every threshold derived from it was 40 degree-minutes shallower than the code's. -450 to -700
IS a real Stockholm range - at -8 °C. It is not the range at -10 °C.

## The aux limit

Past **DM −1500** the offset may not go below 0 - no price, peak or comfort cut - unless the room is
above its band, where a deep DM means our own target is too high. The tier raises nothing itself:
the compressor is flat out there and the immersion heater already running, so a raise only drives DM
down faster and steps the heater up (F-124 - on the owner's F750 the old +10 latched for hours).

⚠️ It is **not** the number that governs a real F750: the pump's own "start addition" (menu 4.9.3)
is set per model and per installation (factory −700; the owner's F750 starts it at about −1450).
The −1500 figure is attributed to Swedish forums and is **not sourced in this repository** - see
`docs/research/01_degree_minutes.md`.

## Recovery damping (T1, T2 and T3)

`_apply_thermal_recovery_damping` lowers a T-tier's offset when the house is warming on its own, so
a slab does not bank heat it will release hours later as overshoot:

- **Warming**: indoor rising at `THERMAL_RECOVERY_WARMING_THRESHOLD` (0.3 °C/h) or more, outdoor not
  falling faster than `THERMAL_RECOVERY_OUTDOOR_DROPPING_THRESHOLD`: the offset is multiplied by
  `THERMAL_RECOVERY_DAMPING_FACTOR`, or `THERMAL_RECOVERY_RAPID_FACTOR` from
  `THERMAL_RECOVERY_RAPID_THRESHOLD` (0.5 °C/h).
- **Overshoot**: a room 0.5 / 1.0 / 1.5 °C over target multiplies it by the mild / moderate / severe
  `THERMAL_RECOVERY_OVERSHOOT_*_DAMPING` factor.
- **Never below** `THERMAL_RECOVERY_T1/T2/T3_MIN_OFFSET` (+1.0 / +1.5 / +2.0), and **no damping**
  when the forecast shows a drop past `THERMAL_RECOVERY_FORECAST_DROP_THRESHOLD`, or when the
  trend's confidence is under `THERMAL_RECOVERY_MIN_CONFIDENCE` (overshoot damping alone then
  applies).

## Also on this path

- **Anti-windup** cuts or holds when DM falls at `ANTI_WINDUP_DM_DROPPING_RATE` or faster within
  `ANTI_WINDUP_CAUSATION_WINDOW_MINUTES` of our own raise, then stays for
  `ANTI_WINDUP_COOLDOWN_MINUTES`; it also caps the T-tiers' offsets while active.
- **Volatile prices**: during a cheap run shorter than `VOLATILE_MIN_DURATION_QUARTERS` the T- and
  Z-boosts are skipped and their weight cut by `VOLATILE_WEIGHT_REDUCTION`. The EMERGENCY floor is
  checked first, so it is never suppressed.
