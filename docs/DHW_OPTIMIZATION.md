# Hot water (DHW) optimisation

EffektGuard moves NIBE's hot-water heating into cheap quarter-hours and keeps it out of the way of
space heating. It has one hot-water actuator, the pump's **temporary lux** switch, which it turns on
and off. It never changes the pump's own hot-water settings. The code is `should_start_dhw()` in
`custom_components/effektguard/optimization/dhw_optimizer.py` and `_apply_dhw_control()` in
`coordinator.py`. Switches, sensors, logs and troubleshooting are in
[DHW_SETUP.md](DHW_SETUP.md).

## What it needs

| Input | Where it comes from |
|---|---|
| Hot-water top temperature (BT7) | Auto-discovered from the NIBE integration (an entity id containing `_bt7`, `bt7_`, `hw_top`, `hot_water_top` or `40013`), or "Override: hot water top temperature (BT7)" in the config flow. Without it the hot-water sensors read `not_configured` and nothing runs. |
| Temporary-lux entity | "Temporary lux switch (DHW)" in the config flow, auto-detected when a `switch.` id contains `temporary` or `temp` plus `lux`, or `50004` or `48132`, and comes from myuplink or nibe_heatpump or has a NIBE model name in its id. myuplink creates `switch.<device>_temporary_lux` on F-series pumps (parameter 50004). nibe_heatpump exposes register 48132 as a select (F-series, SMO, VVM), and nothing for the S-series; bridge it with an `input_boolean` helper. Without an entity EffektGuard raises the repair issue "No hot-water control entity" and touches nothing. |
| Hot-water amount (optional) | A sensor whose id contains `hot_water_amount` or `hw_amount` (minutes of hot water). With it, the scheduled window can stop on amount instead of temperature. |

## What you configure

Options (Configure, section "Domestic Hot Water (DHW)"):

| Option | Range | Default |
|---|---|---|
| DHW target temperature | 45-60 °C | 50 °C |
| Minimum hot water at scheduled time | 1-30 min | 5 min |
| Morning heating, evening heating | each on or off | both on |
| Morning availability hour | 00:00-23:00 | 07:00 |
| Evening availability hour | 00:00-23:00 | 18:00 |

The availability hour is when the water must be **ready**, not when heating starts. Automatic
control runs only while `switch.effektguard_hot_water_optimization` is on (off by default).

## How it decides

Every 5-minute cycle the rules below run in this order; the first one that applies decides. Price
classes are the price layer's `very_cheap`, `cheap`, `normal`, `expensive`, `peak`; "stable cheap"
means cheap or very cheap and not volatile. Before any rule: if the house is more than 0.3 °C below
target and cooling faster than 0.3 °C/h, no decision is made this cycle (`INDOOR_COOLING_RAPIDLY`)
and the switch is left as it is.

**The scheduled window** opens 6 h (`DHW_SCHEDULED_WINDOW_HOURS`) before the next enabled
availability hour, or earlier if the tank needs longer: the heating-time estimate (learned rate,
default 14 °C/h) plus a 0.5 h buffer. Only the closest availability hour within 24 h counts.

| Rule | `priority_reason` | When | Result |
|---|---|---|---|
| 0 | `DHW_SCHEDULED_BLOCKED_BY_SAFETY` | In the window; indoor below 18 °C or DM at or below −1500 | Wait; the window is owed |
| 0 | `DHW_SCHEDULED_TARGET_REACHED_TEMP_*C` / `_*MIN` | In the window; tank at target, or amount at the minimum | Stop |
| 0 | - | In the window; tank at or above 45 °C | Fall through to rules 1-8 |
| 0 | `DHW_SCHEDULED_OPTIMAL_Q*` / `DHW_SCHEDULED_WAITING_OPTIMAL_*` | In the window; tank below 45 °C; a window at least 15 % cheaper that still finishes in time | Heat when it starts; wait until then |
| 0 | `DHW_PEAK_AVOIDED_SCHEDULED` | In the window; tank below 45 °C but at least 20 °C; peak price | Wait for the next non-peak quarter |
| 0 | `DHW_SCHEDULED_PRIORITY_*H` | Otherwise in the window with the tank below 45 °C | Heat now |
| 0.5 | `DHW_SCHEDULED_RETRY_AFTER_SAFETY` | A window was owed and the house is safe again | Heat now |
| 1 | `CRITICAL_THERMAL_DEBT` | DM at or below the block threshold (below) | Block |
| 2 | `SPACE_HEATING_EMERGENCY` | Indoor more than 0.5 °C below target and outdoor below 0 °C | Block |
| 3 | `DHW_SAFETY_MINIMUM` | Tank below 30 °C | Heat to 30 °C now |
| 3 | `DHW_SAFETY_WINDOW_Q*` / `DHW_SAFETY_WAITING_WINDOW_IN_*` / `DHW_SAFETY_DEFERRED_PEAK_PRICE` | Tank 20-30 °C at expensive or peak price with DM more than 20 above the block | Heat in the cheapest window instead; wait for it; or just defer |
| 2.6 | `DHW_PEAK_AVOIDED` | Peak price, tank at least 20 °C | Wait for the next non-peak quarter |
| 2.3 | `DHW_HYGIENE_BOOST` | No 55 °C reading for 14 days (or never), stable cheap | Heat (see Hygiene) |
| 2.5 | `DHW_COMPLETE_EMERGENCY_HEATING` | Tank 30-45 °C, stable cheap | Heat to target |
| 3.5 | `DHW_MAX_WAIT_EXCEEDED_*H` | More than 36 h since the last heating, price `cheap` or `normal` | Heat |
| 4 | `HIGH_SPACE_HEATING_DEMAND` | Heating demand above 6 kW and DM below −60 | Block |
| 4.7 | `PEAK_AVOIDANCE_PREHEAT_*H` | Price very cheap, cheap or normal; a peak within 24 h with DM predicted below the block 2 h before it and the tank predicted below 30 °C | Heat now |
| 5 | `DHW_ADEQUATE_WAITING_CHEAP_*` | Tank at least 45 °C and the price class is not `cheap` | Wait for the next cheap window |
| 5 | `OPTIMAL_WINDOW_Q*` / `WAITING_OPTIMAL_WINDOW_IN_*` | Tank below target + 5 °C; the cheapest window in the look-ahead starts within 15 min, or later while the tank is at least 45 °C | Heat to target + 5 °C (at most 60 °C); or wait |
| 5 | `CHEAP_NO_WINDOW_DATA` / `CHEAP_ELECTRICITY_OPPORTUNITY` | No price window data; stable cheap; tank below 50 °C | Heat |
| 7 | `DHW_COMFORT_OPTIMAL_*` / `DHW_COMFORT_WAITING_OPTIMAL_*` | Tank 30-45 °C at normal price; a window at least 15 % cheaper within 6 h | Heat when it starts; wait until then |
| 7 | `DHW_COMFORT_LOW_*` | Tank below 45 °C at cheap or normal price | Heat now |
| 8 | `DHW_ADEQUATE` | Nothing above applied | Do not heat |

Every "wait" or "block" carries a `recommended_start_time`, shown by
`sensor.effektguard_dhw_scheduled_start`.

### Thermal-debt thresholds

- **Block** (rule 1): DM at or below −1500 (`DM_THRESHOLD_AUX_LIMIT`), or at or below the climate
  zone's warning threshold, adjusted for thermal mass, minus 200 (`DM_CRITICAL_T2_MARGIN`) -
  `EmergencyLayer.should_block_dhw()`. The zone thresholds come from
  [CLIMATE_ZONES.md](CLIMATE_ZONES.md).
- **Abort** (a running cycle): 160 below the block (`DM_DHW_ABORT_BUFFER`), never below −1500.
- Without a climate detector the fallbacks are −340 and −500 (`DM_DHW_BLOCK_FALLBACK`,
  `DM_DHW_ABORT_FALLBACK`).
- The scheduled window (rules 0 and 0.5) outranks the block but not the safety gate: indoor below
  18 °C (`MIN_TEMP_LIMIT`) or DM at or below −1500.

## When a running boost stops

`_apply_dhw_control()` runs after the decision, only when optimisation and the hot-water switch are
on, a lux entity is configured, and not during the start-up grace period.

- **Abort**: while the switch is on, the decision's abort conditions are checked every cycle:
  `thermal_debt <` the abort threshold and `indoor_temp <` target − 0.5 °C; rule 4.7 also aborts
  at `dhw_temp >=` target; a scheduled cycle aborts only on indoor below 18 °C or DM below −1500.
- **Decision flips to "do not heat"**: the switch is turned off, unless a `boost_dhw` service window
  is open.
- **Starts are rate-limited**: at most one start per 60 min (`DHW_CONTROL_MIN_INTERVAL_MINUTES`)
  after the last start or stop. Stops are never delayed.
- On unload EffektGuard turns off a boost it started; one the household started is left alone.
- NIBE's own lux timeout still ends the boost underneath; the decision's `max_runtime_minutes` is
  for display only.

## Hygiene (Legionella)

EffektGuard does not provide Legionella protection. NIBE's **periodic increase** does (menu 2.9.1 on
F-series, 2.4 on S-series; factory-enabled, every 14 days, 7 on S-series, to 55 °C with compressor
and immersion heater), and EffektGuard cannot block it. What EffektGuard does:

- Records a high-temperature cycle the moment BT7 reaches 55 °C (`DHW_LEGIONELLA_DETECT`), live
  and from 14 days of recorder history at start-up; shown as `last_legionella_boost` on
  `sensor.effektguard_dhw_status`.
- Warns once in the log if no such cycle is seen for 21 days (`DHW_LEGIONELLA_OVERDUE_DAYS`): check
  that the periodic increase is still enabled on the pump.
- Rule 2.3 is a cheap-period top-up, not a hygiene cycle: temporary lux stops at the pump's luxury
  stop temperature (53-54 °C on factory settings), below 55 °C. The 56 °C
  `DHW_LEGIONELLA_PREVENT_TEMP` is never written to the pump.

## Where things live

- Constants, with their sources: `DHW_*`, `DM_DHW_*`, `SPACE_HEATING_DEMAND_*` in `const.py`
- Manual boost: `effektguard.boost_dhw`, see
  [DHW_SETUP.md](DHW_SETUP.md)
- Tests: `tests/unit/dhw/`
