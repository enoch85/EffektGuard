# DHW Planning Display

## Overview

The DHW (Domestic Hot Water) Recommendation sensor now provides detailed planning information in **human-readable format**.

## Sensor: `sensor.effektguard_dhw_recommendation`

### State
Shows the current recommendation (e.g., "Heat now - Safety minimum (33.4°C < 35°C)")

### Attributes

#### `planning_summary` (Human-Readable)

Multi-line formatted summary showing all planning details:

```
📋 DHW Planning Summary
━━━━━━━━━━━━━━━━━━━━
🌡️  Current: 33.4°C → Target: 55°C
💰 Price: CHEAP
🟢 Thermal Debt: OK (DM -207)
✅ Low heating demand: 0.0 kW
🌤️  Unusually warm (+7.8°C), good for DHW heating

⏰ Optimal Heating Windows:
   1. 02:00-05:45 (3.8h, CHEAP)
   2. 12:30-14:00 (1.5h, NORMAL)
   3. 20:00-23:00 (3.0h, CHEAP)

💡 Recommendation: Heat now - Safety minimum (33.4°C < 35°C)
```

#### Machine-Readable Attributes

Detailed technical data for automations:

| Attribute | Example | Description |
|-----------|---------|-------------|
| `should_heat` | `true` | Should DHW heat now? |
| `priority_reason` | `DHW_SAFETY_MINIMUM` | Why this decision? |
| `current_temperature` | `33.4` | Current DHW temp (°C) |
| `target_temperature` | `55.0` | Target DHW temp (°C) |
| `thermal_debt` | `-207` | Current degree minutes |
| `thermal_debt_threshold_block` | `-320` | Block threshold for zone |
| `thermal_debt_threshold_abort` | `-400` | Abort threshold for zone |
| `thermal_debt_status` | `OK - 113 DM safety margin` | Human-readable status |
| `space_heating_demand_kw` | `0.0` | Current heating demand |
| `current_price_classification` | `CHEAP` | Current price level |
| `outdoor_temperature` | `9.0` | Outdoor temp (°C) |
| `indoor_temperature` | `21.0` | Indoor temp (°C) |
| `climate_zone` | `Moderate Cold` | Your climate zone |
| `weather_opportunity` | `Unusually warm...` | Weather note (if applicable) |
| `optimal_windows_count` | `3` | Number of windows found |
| `window_1_time` | `02:00-05:45` | First optimal window |
| `window_1_price` | `CHEAP` | Price classification |
| `window_1_duration_hours` | `3.75` | Duration in hours |
| `window_1_thermal_debt_ok` | `true` | Safe to heat? |
| `window_2_time` | ... | Second window |
| `window_3_time` | ... | Third window |
| `next_window_time` | `02:00-05:45` | Next optimal start |
| `next_window_price` | `CHEAP` | Next window price |
| `next_window_duration` | `3.8h` | Next window duration |

## Lovelace Card Example

```yaml
type: markdown
content: >
  {{ state_attr('sensor.effektguard_dhw_recommendation', 'planning_summary') }}
title: DHW Planning
```

## Automation Example

```yaml
automation:
  - alias: "Notify when DHW should heat"
    trigger:
      - platform: state
        entity_id: sensor.effektguard_dhw_recommendation
        attribute: should_heat
        to: true
    condition:
      - condition: template
        value_template: "{{ state_attr('sensor.effektguard_dhw_recommendation', 'current_price_classification') == 'CHEAP' }}"
    action:
      - service: notify.mobile_app
        data:
          message: >
            DHW heating recommended!
            Next window: {{ state_attr('sensor.effektguard_dhw_recommendation', 'next_window_time') }}
            Price: {{ state_attr('sensor.effektguard_dhw_recommendation', 'next_window_price') }}
```

## Thermal Debt Status Icons

- 🟢 **OK** - DM above block threshold, safe to heat
- 🟡 **WARNING** - DM between block and abort threshold
- 🔴 **CRITICAL** - DM below abort threshold, emergency

## Price Classifications

- **CHEAP** - Below P25, best time to heat
- **NORMAL** - P25-P75, acceptable
- **EXPENSIVE** - P75-P90, avoid if possible
- **PEAK** - Above P90, avoid heating

## Notes

- Planning updates every 5 minutes
- Windows are calculated for the next 24 hours
- Thermal debt thresholds adapt to your climate zone (Malmö: -320 DM block, -400 DM abort)
- Weather opportunities are flagged when outdoor temp is >5°C above zone average
