# Thermal debt: the degree-minute ladder

Two layers in `optimization/thermal_layer.py` answer deepening degree minutes (DM).
`ProactiveLayer` (Z1-Z5) acts before the warning threshold, `EmergencyLayer` (T1-T3, EMERGENCY)
from it. Both read the same thresholds: `ClimateZoneDetector.get_expected_dm_range` from latitude
and outdoor temperature ([docs/CLIMATE_ZONES.md](../CLIMATE_ZONES.md)), divided by the thermal-mass
buffer (`apply_thermal_mass_buffer`: concrete 1.3, timber 1.15, radiators 1.0), so a slab acts
earlier. The ladder is Z1, Z2, Z3, Z4, Z5, T1, T2, T3, EMERGENCY. There is no WARNING or CAUTION
tier: `normal_max` equals `warning` in every zone, so they could never fire.

## The rungs

| Rung | Fires at DM | Offset | Weight |
|---|---|---|---|
| Z1 | `normal_max` x 2 % (`PROACTIVE_ZONE1_THRESHOLD_PERCENT`) | +1.0, x1.3 when cooling 0.2 °C/h or faster | 0.3 |
| Z2 | x 30 % | +1.5 | 0.4 |
| Z3 | x 50 % | +2.0 to +2.5 by depth | 0.5 |
| Z4 | x 75 % | +2.5 | 0.55 |
| Z5 | x 87.5 %, down to the warning | +3.0 | 0.6 |
| T1 | the warning (`DM_CRITICAL_T1_MARGIN` 0) | +4.0 | 0.65 |
| T2 | warning - 200 | +7.0 | 0.81 |
| T3 | warning - 400, never below -1450 (`DM_CRITICAL_T3_MAX`) | +8.5 | 0.91 |
| EMERGENCY | `DM_THRESHOLD_AUX_LIMIT` -1500 | floor at 0 | 1.0 |

Constants: `PROACTIVE_ZONE*_OFFSET`, `PROACTIVE_ZONE*_WEIGHT`, `DM_CRITICAL_T*_OFFSET`,
`DM_CRITICAL_T*_WEIGHT`, `DM_CRITICAL_T*_MARGIN` in `const.py`.

The normal range `get_expected_dm_range` returns, computed by the code; its deep end is the
warning:

| Place | Outdoor | Normal DM range |
|---|---|---|
| Stockholm (Cold zone) | -10 °C | -490 to -740 |
| Stockholm (Cold zone) | -8 °C | -450 to -700 |
| Stockholm (Cold zone) | 0 °C | -290 to -540 |
| Kiruna (Extreme Cold zone) | -25 °C | -900 to -1300 |
| Kiruna (Extreme Cold zone) | -30 °C | -1000 to -1400 |
| Malmö (Moderate Cold zone) | 0 °C | -280 to -480 |

The whole ladder for Stockholm at -10 °C (warning -740 before the buffer):

| Emitter | Warning | Z1 / Z2 / Z3 / Z4 / Z5 | T1 / T2 / T3 |
|---|---|---|---|
| Radiators | -740 | -15 / -222 / -370 / -555 / -648 | -740 / -940 / -1140 |
| Concrete slab | -569 | -11 / -171 / -285 / -427 / -498 | -569 / -769 / -969 |

## EmergencyLayer

Checked in this order, every cycle:

1. **EMERGENCY**: DM at or past -1500 and the room not above its band. The layer votes 0 at weight
   1.0 and `_aggregate_layers` lets the other layers decide, floored at 0: no price, peak or comfort
   cut, and no raise of its own. The compressor is flat out there, so a raise only buys immersion
   heat. A room above its band skips this: a deep DM there means the target is too high.
2. **Anti-windup**: DM falling 50 DM/h or faster (`ANTI_WINDUP_DM_DROPPING_RATE`) while the offset
   is at least +0.5 and EffektGuard raised it within the last 90 minutes
   (`ANTI_WINDUP_CAUSATION_WINDOW_MINUTES`). The offset is held, or, once DM falls faster than
   100 DM/h, cut by 1 °C per 100 DM/h of fall, at weight 0.7, and no raise is allowed for 30
   minutes (`ANTI_WINDUP_COOLDOWN_MINUTES`). A fall with no recent raise is treated as weather and
   falls through to the tiers.
3. **No vote** when the room is above its band, or at or above target while the price is not
   cheap. Both need a real room sensor; without one the tiers run on DM alone.
4. **T3, T2, T1** by the thresholds above. Each tier's offset is damped by
   `_apply_thermal_recovery_damping` when the house is warming on its own (0.3 °C/h or more: x0.6;
   0.5 °C/h or more: x0.4) or already over target (+0.5 / +1.0 / +1.5 °C: x0.95 / x0.9 / x0.8),
   never below +1.0 / +1.5 / +2.0 (`THERMAL_RECOVERY_T*_MIN_OFFSET`), and not at all when the
   6-hour forecast drops more than 2 °C. Anti-windup caps a tier at the current offset.

While a tier is recovering, a critical cost layer (a PEAK price quarter, or the effect layer at the
monthly peak) may only moderate it down to `DM_CRITICAL_PEAK_AWARE_OFFSETS` (+0.5 / +0.75 / +1.0),
never reverse it ([08](08_layer_priority_system.md)).

## ProactiveLayer

- **Rapid cooling** first: indoor falling 0.3 °C/h or faster (`RAPID_COOLING_THRESHOLD`) below
  0 °C outdoors with a deficit over 0.3 °C votes 2 x the rate, at most +3.0, weight 0.8; x1.3 when
  the next 3 forecast hours drop more than 1 °C, x0.8 otherwise.
- Otherwise the Z-zone the DM is in, from the table.
- A room more than 0.5 °C above target (`PROACTIVE_WARM_HOUSE_THRESHOLD`) cuts the weight to 30 %;
  Z5 is exempt.

## Volatile prices

During a price run shorter than 3 quarters (`VOLATILE_MIN_DURATION_QUARTERS`) the Z and T boosts
are skipped and their weight multiplied by 0.3 (`VOLATILE_WEIGHT_REDUCTION`). The EMERGENCY check
comes first, so it is never suppressed.

## The aux limit is not sourced

The -1500 figure (`DM_THRESHOLD_AUX_LIMIT`) is not sourced in this repository. A real pump starts
its immersion heater at its own "start addition" setting (menu 4.9.3), which differs per model and
installation; see [docs/research/01_degree_minutes.md](../research/01_degree_minutes.md).
