# DHW Automatic Control

## How It Works

EffektGuard **automatically controls DHW heating** using NIBE's temporary lux switch (50004) based on the
hot-water optimiser's decisions (`docs/DHW_OPTIMIZATION.md`).

### Automatic Control Logic

**When enabled**, EffektGuard:
1. ✅ **Monitors** DHW temperature and thermal debt every cycle (5 minutes)
2. ✅ **Calculates** the hot-water decision from:
   - Thermal debt (degree minutes)
   - Electricity prices (spot price)
   - Space heating demand
   - Indoor temperature vs target
   - The configured demand periods (morning/evening)
3. ✅ **Controls** NIBE's temporary lux switch:
   - **Turns ON** when `should_heat = True` (safe to heat DHW)
   - **Turns OFF** when `should_heat = False`, or when a running boost hits a safety abort

### Safety Features

- **Thermal-debt blocking**: no DHW start at or below the climate-aware block threshold
  (`EmergencyLayer.should_block_dhw`; fallback `DM_DHW_BLOCK_FALLBACK` = −340), and a running boost is
  aborted past the abort threshold (fallback `DM_DHW_ABORT_FALLBACK` = −500)
- **Space heating priority**: blocks DHW when the house is more than 0.5 °C below target and it is
  below 0 °C outside (`DHW_SPACE_HEATING_DEFICIT_THRESHOLD`, `DHW_SPACE_HEATING_OUTDOOR_THRESHOLD`)
- **Rate limiting**: at most one automatic START per `DHW_CONTROL_MIN_INTERVAL_MINUTES` (60); a safety
  stop is never delayed
- **Graceful degradation**: if the temporary lux entity is unavailable, recommendations only

## Setup Requirements

### 1. Enable Hot Water Optimization

**Via Home Assistant UI:**
1. Go to Settings → Devices & Services → EffektGuard
2. Find the "Hot Water Optimization" switch (`switch.effektguard_hot_water_optimization`)
3. Turn it ON

EffektGuard is configured through the UI (config and options flows); there is no YAML configuration.

### 2. Configure Temporary Lux Entity

EffektGuard needs to know which switch controls NIBE's temporary lux:

**During setup:**
- Entity ID: `switch.temporary_lux_50004` (or your MyUplink device name)
- This is the NIBE parameter 50004 exposed via the MyUplink integration (or the equivalent nibe_heatpump
  or Modbus switch)

Writes through MyUplink may require a valid myUplink subscription.

## Understanding DHW Status

The `sensor.effektguard_dhw_status` shows the DHW state:

| Status | Meaning |
|--------|---------|
| `heating` | Compressor actively heating DHW |
| `low` | Below `DHW_SAFETY_MIN` (30 °C), waiting to heat |
| `pending` | Below the minimum target (45 °C), will heat soon |
| `ready` | At normal target (45-52 °C) |
| `hot` | Above 52 °C (high demand met, or NIBE's periodic increase) |
| `not_configured` | No DHW temperature sensor bound |

The `sensor.effektguard_dhw_recommendation` explains WHY, from the decision's priority reason (the
full list is the rule table in `docs/DHW_OPTIMIZATION.md`). For example:

| Priority reason | Meaning | Temporary Lux |
|----------------|---------|---------------|
| `CRITICAL_THERMAL_DEBT` | Thermal debt too deep | **OFF** (blocking) |
| `SPACE_HEATING_EMERGENCY` | House too cold | **OFF** (space heating first) |
| `HIGH_SPACE_HEATING_DEMAND` | High heating demand | **OFF** (temporary delay) |
| `CHEAP_ELECTRICITY_OPPORTUNITY` | Cheap electricity | **ON** |
| `DHW_SAFETY_MINIMUM` | Tank critically low | **ON** |
| `DHW_ADEQUATE` | No action needed | **OFF** |

## How Temporary Lux Works

### NIBE Temporary Lux (Parameter 50004)

When **ON**:
- NIBE heats the tank to its luxury stop temperature (53-54 °C on factory settings) with priority
- Space heating may be delayed during this period
- EffektGuard turns it OFF again when its decision or a safety abort says so

When **OFF**:
- NIBE follows its normal schedule (space heating + its own DHW comfort settings)

### Automatic vs Manual Control

**Automatic control** (when Hot Water Optimization is enabled):
- EffektGuard evaluates every 5 minutes
- Turns temporary lux ON/OFF based on the optimiser; at most one start per 60 minutes

**Manual control** (always available):
- Service: `effektguard.boost_dhw` with an optional `duration` (minutes, 30-180, default 90)
- The pump heats to its own lux temperature; EffektGuard keeps the boost running for the duration
  (price optimisation will not cancel it - only the thermal-debt safety stop will) and then turns it off
- `DHW_BOOST_COOLDOWN_MINUTES` (60) between manual boosts

## Monitoring Automatic Control

### Logs

Enable debug logging to see DHW control decisions:

```yaml
logger:
  default: info
  logs:
    custom_components.effektguard.coordinator: debug
    custom_components.effektguard.optimization.dhw_optimizer: debug
```

**Example log output:**
```
INFO DHW control: Deactivating temporary lux - CRITICAL_THERMAL_DEBT (DHW: 38.9°C, DM: -370)
INFO DHW control: Activating temporary lux - CHEAP_ELECTRICITY_OPPORTUNITY (DHW: 42.0°C, DM: -120)
```

### Dashboard Card

```yaml
type: entities
title: DHW Automatic Control
entities:
  - entity: switch.effektguard_hot_water_optimization
    name: Auto Control Enabled
  - entity: sensor.effektguard_dhw_status
    name: Current Status
  - entity: sensor.effektguard_dhw_recommendation
    name: Recommendation
  - entity: switch.temporary_lux_50004
    name: Temporary Lux
  - entity: sensor.f750_cu_3x400v_hot_water_top_bt7
    name: DHW Temperature
  - entity: sensor.effektguard_degree_minutes
    name: Thermal Debt
```

## Troubleshooting

### DHW Still Heating During "Block" Period

**Possible causes:**

1. **Hot Water Optimization switch is OFF**
   - Check: `switch.effektguard_hot_water_optimization`
   - Fix: Turn it ON to enable automatic control

2. **Temporary lux entity not configured**
   - Debug log: "No temporary lux entity configured"
   - Fix: Reconfigure EffektGuard and set the temporary lux entity

3. **Temporary lux entity not found**
   - Check: `switch.temporary_lux_50004` exists in the MyUplink integration

4. **A start is rate limited**
   - Debug log: "DHW start rate limited: ... min since last change (min 60 min)"

5. **NIBE's own schedule**
   - Temporary lux controls DHW priority, not a full enable/disable
   - NIBE may still heat DHW on its own comfort settings and its periodic increase

### Logs Show "No change needed" But Should Be Blocking

This happens when temporary lux is already OFF and NIBE is heating on its own schedule.
**This is expected behavior** - EffektGuard blocks lux-triggered heating; NIBE may still heat DHW
periodically on its own.

## Advanced Configuration

### DHW Demand Periods

Configured in EffektGuard's options (Configure button): a morning and an evening hour at which hot water
should be ready, each switchable, and the minimum hot-water amount (minutes). EffektGuard opens a
scheduled window before each (`docs/DHW_OPTIMIZATION.md`, Two-Lane Architecture) and:
- Heats in the cheapest window that still finishes in time
- Ensures the target is reached by the demand time
- Avoids PEAK prices unless the tank is critically low

### Integration with Automations

```yaml
automation:
  - alias: "Boost hot water for guests"
    trigger:
      - platform: state
        entity_id: input_boolean.guests_staying
        to: "on"
    action:
      - service: effektguard.boost_dhw
        data:
          duration: 180
```

## Technical Details

### Decision Priority

The rules and their order are in `docs/DHW_OPTIMIZATION.md` (Decision Rules), taken from
`should_start_dhw()` in `dhw_optimizer.py`.

### Rate Limiting

- **Coordinator update**: every 5 minutes, clock-aligned
- **Automatic DHW starts**: at most one per 60 minutes (`DHW_CONTROL_MIN_INTERVAL_MINUTES`); stops are not delayed
- **Manual boost service**: 60-minute cooldown (`DHW_BOOST_COOLDOWN_MINUTES`)

## Best Practices

1. **Enable gradually**: Start with monitoring only, enable auto control after understanding patterns
2. **Monitor logs**: Watch for 1-2 days to verify control decisions are correct
3. **Set demand periods**: Configure your real high-demand times for good pre-heating
4. **Manual override available**: Use the `boost_dhw` service for special situations

## Safety Guarantees

- ✅ Below `DHW_SAFETY_CRITICAL` (20 °C) price never defers heating; below `DHW_SAFETY_MIN` (30 °C) the
  tank is heated in the next good window or at once
- ✅ Never starts DHW during a space-heating emergency or past the thermal-debt block threshold
- ✅ Aborts a running boost past the thermal-debt abort threshold
- ✅ Hygiene is NIBE's own periodic increase (factory every 14 days to 55 °C); EffektGuard warns if it
  sees none for 21 days and cannot perform it itself (temporary lux stops below 55 °C)

## Related Documentation

- `docs/DHW_OPTIMIZATION.md` - Full DHW optimization algorithm
- `architecture/07_manual_override_services.md` - Manual control services

## Version History

- **v0.0.1-alpha19**: Automatic DHW control via temporary lux implemented
- **v0.0.1-alpha18**: DHW monitoring and recommendations only (previous version)
