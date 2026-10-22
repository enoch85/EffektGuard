# Hot-water control in practice

EffektGuard heats hot water by switching NIBE's **temporary lux** on and off from the decision in
[DHW_OPTIMIZATION.md](DHW_OPTIMIZATION.md). This page is how to turn it on, read it and debug it.

## Turn it on

1. Configure the temporary-lux entity: Settings, Devices & services, EffektGuard, Reconfigure,
   "Temporary lux switch (DHW)". Without it EffektGuard raises the repair issue "No hot-water
   control entity" and the hot-water sensors show plans that never run.
2. Turn on `switch.effektguard_hot_water_optimization` (off by default).
3. Set the target temperature, the morning and evening hours and the minimum amount under
   Configure, "Domestic Hot Water (DHW)".

| NIBE integration | Temporary-lux entity |
|---|---|
| myuplink, F-series | `switch.<device>_temporary_lux` (parameter 50004), for example `switch.f750_cu_3x400v_temporary_lux`. Auto-detected. |
| nibe_heatpump (Modbus), F-series, SMO, VVM | Register 48132 is a **select** (off, 3 h, 6 h, 12 h, one-time increase), which EffektGuard cannot drive. Create an `input_boolean` helper, and an automation that sets the select to its one-time increase when the helper turns on and to off when it turns off; choose the helper as the lux entity. EffektGuard switches it with `homeassistant.turn_on` and `turn_off`. |
| nibe_heatpump, S-series | No temporary-lux register. Hot-water control is not available. |

## Read it

| Entity | State | Attributes |
|---|---|---|
| `sensor.effektguard_dhw_status` | `heating` (the pump reports hot-water priority, or the lux switch is on), `low` below 30 °C, `pending` below 45 °C, `ready` below 52 °C, `hot` from 52 °C, `not_configured` | `dhw_current_temp`, `dhw_target_temp`, `last_legionella_boost`, `heating_start`, `heating_end`, `last_cycle_duration`, `within_scheduled_window`, `water_amount_current`, `water_amount_target` |
| `sensor.effektguard_dhw_recommendation` | One line, e.g. "Block DHW - Thermal debt warning (DM: -620, zone: Standard)", "Heat now - Safety minimum (25.0°C < 30.0°C)", "DHW OK - Temperature adequate (48.0°C)" | `should_heat`, `priority_reason` (the code in the rule table), `thermal_debt_threshold_block`, `thermal_debt_threshold_abort`, `current_price_classification`, `planning_summary` |
| `sensor.effektguard_dhw_scheduled_start` | Next planned start (timestamp) | `dhw_status` `heating` or `blocked`, `heating_reason` or `blocking_reason`, `summary` |

Dashboard card:

```yaml
type: entities
title: Hot water
entities:
  - entity: switch.effektguard_hot_water_optimization
  - entity: sensor.effektguard_dhw_status
  - entity: sensor.effektguard_dhw_recommendation
  - entity: sensor.effektguard_dhw_scheduled_start
  - entity: sensor.effektguard_degree_minutes
  - entity: switch.f750_cu_3x400v_temporary_lux
    name: Temporary lux
  - entity: sensor.f750_cu_3x400v_hot_water_top_bt7
    name: Hot water top (BT7)
```

The last two are myuplink entity ids for an F750; use your pump's.

## Manual boost

`effektguard.boost_dhw` turns temporary lux on at once, with `duration` 30-180 min (default 90).
The pump heats to its own luxury stop temperature; EffektGuard keeps the switch on for the duration
and then turns it off, with or without hot-water optimisation. Price does not cancel it; the safety
gate does (DM at −1500 or the house below 18 °C), and refuses to start it.
Calls are 60 min apart at least, and refused while optimisation is off.

```yaml
automation:
  - alias: Boost hot water for guests
    triggers:
      - trigger: state
        entity_id: input_boolean.guests_staying
        to: "on"
    actions:
      - action: effektguard.boost_dhw
        data:
          duration: 180
```

## Logs

```yaml
logger:
  logs:
    custom_components.effektguard.coordinator: debug
    custom_components.effektguard.optimization.dhw_optimizer: debug
```

At `info` you see every switch action; at `debug` also the rate limiter and "No change needed":

```
DHW control: Activating temporary lux - OPTIMAL_WINDOW_Q12_@31.2öre (DHW: 42.0°C, DM: -120)
DHW control: Deactivating temporary lux - CRITICAL_THERMAL_DEBT (DHW: 38.9°C, DM: -370)
DHW heating aborted early: Thermal debt -820 < -800. Stopping DHW to prioritize space heating.
DHW start rate limited: 23.4 min since last change (min 60 min)
```

## Troubleshooting

- **Nothing is ever switched**: the hot-water switch is off, the lux entity is missing (repair issue
  "No hot-water control entity"), or the entity no longer exists ("Temporary lux entity ... not
  found" in the log).
- **Status is `not_configured`**: no BT7 sensor was found ("DHW sensor (BT7) not found" in the
  log). Set the BT7 override in the config flow. While BT7 is unreadable EffektGuard starts nothing
  and stops a boost it started.
- **A boost started on the pump or in MyUplink is not stopped**: by design - EffektGuard only stops
  the boosts it started.
- **A start does not happen when expected**: one start per 60 min after the last start or stop;
  nothing is controlled during the start-up grace period after a restart.
- **The pump heats hot water while EffektGuard says "block"**: temporary lux is a boost, not an
  enable. The pump still heats on its own comfort settings and its periodic increase; EffektGuard
  only withholds the boost.
