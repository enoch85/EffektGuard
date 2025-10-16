# DHW Automatic Control

## How It Works

EffektGuard now **automatically controls DHW heating** using NIBE's temporary lux switch (50004) based on intelligent optimization decisions.

### Automatic Control Logic

**When enabled**, EffektGuard:
1. ✅ **Monitors** DHW temperature and thermal debt continuously
2. ✅ **Calculates** optimal DHW heating decisions based on:
   - Thermal debt (degree minutes)
   - Electricity prices (GE-Spot)
   - Space heating demand
   - Indoor temperature vs target
   - High demand periods (morning/evening)
3. ✅ **Controls** NIBE temporary lux switch automatically:
   - **Turns ON** when `should_heat = True` (safe to heat DHW)
   - **Turns OFF** when `should_heat = False` (block DHW to prevent thermal debt)

### Safety Features

- **Critical thermal debt blocking** (DM < -240): Automatically turns OFF temporary lux
- **Space heating priority**: Blocks DHW if house is cold (indoor < target - 0.5°C)
- **Rate limiting**: Minimum 10 minutes between control changes
- **Graceful degradation**: If temporary lux entity unavailable, reverts to recommendations only

## Setup Requirements

### 1. Enable Hot Water Optimization

The automatic DHW control is **OFF by default** (experimental feature). To enable:

**Via Home Assistant UI:**
1. Go to Settings → Devices & Services → EffektGuard
2. Find the "Hot Water Optimization" switch
3. Turn it ON

**Or via configuration:**
```yaml
# configuration.yaml (if using YAML config)
effektguard:
  enable_hot_water_optimization: true
```

### 2. Configure Temporary Lux Entity

EffektGuard needs to know which switch controls NIBE's temporary lux:

**During setup:**
- Entity ID: `switch.temporary_lux_50004` (or your MyUplink device name)
- This is the NIBE parameter 50004 exposed via MyUplink integration

**MyUplink Premium required** for write access to control the switch.

## Understanding DHW Status

## Understanding DHW Status

The `sensor.effektguard_dhw_status` shows real-time DHW state:

| Status | Meaning | Automatic Action |
|--------|---------|------------------|
| `heating` | Compressor actively heating DHW | Monitor, may turn off if thermal debt critical |
| `pending` | Below target, will heat soon | Turn ON temporary lux if conditions safe |
| `low` | Below 35°C safety minimum | Turn ON temporary lux immediately (safety) |
| `ready` | At normal target (45-52°C) | Keep temporary lux OFF |
| `hot` | Above 52°C (high demand met) | Keep temporary lux OFF |

The `sensor.effektguard_dhw_recommendation` explains WHY:

| Recommendation | Meaning | Temporary Lux State |
|----------------|---------|---------------------|
| `Block DHW - Critical thermal debt (DM: -XXX)` | **CRITICAL** - Thermal debt too high | **OFF** (force blocking) |
| `Block DHW - House too cold (XX°C)` | Space heating emergency | **OFF** (prioritize heating) |
| `Delay DHW - High heating demand (X kW)` | Wait for space heating | **OFF** (temporary delay) |
| `Heat now - Cheap electricity (cheap)` | Optimal time to heat | **ON** (boost DHW) |
| `Heat now - Safety minimum (XX°C < 35°C)` | Health/safety critical | **ON** (must heat) |
| `DHW OK - Temperature adequate (XX°C)` | No action needed | **OFF** (maintain) |

## How Temporary Lux Works

### NIBE Temporary Lux (Parameter 50004)

When **ON**:
- NIBE prioritizes DHW heating for **3 hours**
- Compressor focuses on reaching DHW target temperature
- Space heating may be delayed during this period

When **OFF**:
- NIBE follows its normal schedule (space heating + periodic DHW)
- DHW heated based on NIBE's own comfort mode settings
- EffektGuard can block unwanted DHW cycles during high thermal debt

### Automatic vs Manual Control

**Automatic control** (when Hot Water Optimization enabled):
- EffektGuard monitors conditions every 5 minutes
- Turns temporary lux ON/OFF based on optimization algorithm
- Respects 10-minute rate limiting between changes

**Manual control** (always available):
- Service: `effektguard.boost_dhw`
- Turns ON temporary lux for 3 hours (NIBE default)
- Useful for manual DHW boost before high demand periods

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
2025-10-16 03:42:16 INFO DHW control: Deactivating temporary lux - CRITICAL_THERMAL_DEBT (DHW: 38.9°C, DM: -370)
2025-10-16 04:15:00 INFO DHW control: Activating temporary lux - CHEAP_ELECTRICITY_OPPORTUNITY (DHW: 42.0°C, DM: -120)
```

### Dashboard Card

Monitor DHW control status:

### Dashboard Card

Monitor DHW control status:

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
  - entity: sensor.effektguard_current_degree_minutes
    name: Thermal Debt
```

## Troubleshooting

### DHW Still Heating During "Block" Period

**Possible causes:**

1. **Hot Water Optimization switch is OFF**
   - Check: `switch.effektguard_hot_water_optimization`
   - Fix: Turn it ON to enable automatic control

2. **Temporary lux entity not configured**
   - Check logs for: "DHW control disabled: No temporary lux entity configured"
   - Fix: Reconfigure EffektGuard integration, set temporary lux entity

3. **Temporary lux entity not found**
   - Check: `switch.temporary_lux_50004` exists in MyUplink integration
   - Fix: Ensure MyUplink Premium is active, entity is exposed

4. **Rate limiting active**
   - Check logs for: "DHW control rate limited"
   - Info: Changes limited to every 10 minutes for stability

5. **NIBE's own schedule overriding**
   - Temporary lux controls DHW priority, not full enable/disable
   - NIBE may still heat DHW based on comfort mode settings
   - Consider lowering NIBE's DHW comfort temp if needed

### Logs Show "No change needed" But Should Be Blocking

This happens when:
- Temporary lux is already OFF (correct state)
- NIBE is heating from its own schedule (not lux-triggered)
- EffektGuard has correctly disabled lux boost

**This is expected behavior** - EffektGuard blocks lux-triggered heating, but NIBE may still heat DHW periodically based on its internal schedule.

## Advanced Configuration

### Custom DHW Demand Periods

Configure high-demand periods (e.g., morning showers, evening baths):

```yaml
# Via EffektGuard config flow
dhw_demand_periods:
  - start_hour: 7
    target_temp: 55
    duration_hours: 2
  - start_hour: 18
    target_temp: 55
    duration_hours: 3
```

EffektGuard will:
- Pre-heat DHW during cheap electricity before demand periods
- Ensure target temperature is reached by start time
- Optimize timing to avoid expensive peak hours

### Integration with Automations

Create custom rules that work with EffektGuard's DHW control:

```yaml
automation:
  - alias: "Override DHW blocking for guests"
    trigger:
      - platform: state
        entity_id: input_boolean.guests_staying
        to: "on"
    action:
      # Temporarily disable auto control
      - service: switch.turn_off
        target:
          entity_id: switch.effektguard_hot_water_optimization
      # Manually boost DHW
      - service: effektguard.boost_dhw
        data:
          target_temp: 55
          duration: 180
```

## Technical Details

### Decision Priority (from DHW Optimizer)

1. **CRITICAL_THERMAL_DEBT** (DM ≤ -240): Block DHW immediately
2. **SPACE_HEATING_EMERGENCY** (indoor < target - 0.5°C, outdoor < 0°C): Block DHW
3. **DHW_SAFETY_MINIMUM** (DHW < 35°C): Heat DHW (limited runtime)
4. **HIGH_SPACE_HEATING_DEMAND** (>6kW, DM < -60): Delay DHW
5. **URGENT_DEMAND** (<2h until high-demand period): Heat DHW now
6. **OPTIMAL_PREHEAT** (2-24h until demand, cheap prices): Heat DHW
7. **CHEAP_ELECTRICITY** (cheap period, indoor OK): Opportunistic heating
8. **NORMAL_DHW_HEATING** (DHW < 45°C, indoor OK): Standard heating

### Rate Limiting

- **Coordinator update**: Every 5 minutes
- **DHW control changes**: Minimum 10 minutes between ON/OFF
- **Manual boost service**: 60-minute cooldown
- **NIBE offset changes**: 5-minute rate limit

This prevents excessive API calls and allows NIBE to stabilize between changes.

## Best Practices

1. **Enable gradually**: Start with monitoring only, enable auto control after understanding patterns
2. **Monitor logs**: Watch for 1-2 days to verify control decisions are correct
3. **Adjust demand periods**: Configure your actual high-demand times for optimal pre-heating
4. **Trust the algorithm**: DHW optimizer is conservative - prioritizes comfort and safety over savings
5. **Manual override available**: Use `boost_dhw` service for special situations

## Safety Guarantees

EffektGuard's DHW control includes multiple safety layers:

- ✅ Never blocks DHW below 35°C (Legionella risk)
- ✅ Never blocks during space heating emergency
- ✅ Limits DHW runtime during high thermal debt
- ✅ Aborts DHW if thermal debt reaches -400 during cycle
- ✅ Respects NIBE's automatic Legionella protection (weekly 65°C boost)

## Related Documentation

- `docs/DHW_OPTIMIZATION.md` - Full DHW optimization algorithm
- `architecture/07_manual_override_services.md` - Manual control services  
- `.github/copilot-instructions.md` - DHW control implementation guidelines

## Version History

- **v0.0.1-alpha19**: Automatic DHW control via temporary lux implemented
- **v0.0.1-alpha18**: DHW monitoring and recommendations only (previous version)
