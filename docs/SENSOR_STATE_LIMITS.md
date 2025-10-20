# Sensor State Character Limits

## Home Assistant 255-Character Limit

Home Assistant has a hard limit of **255 characters** for sensor states. When a sensor state exceeds this limit, it falls back to "unknown" and logs a warning.

## EffektGuard Solution

### Optimization Reasoning Sensor

The `sensor.effektguard_optimization_reasoning` can generate very detailed reasoning strings, especially during critical situations (thermal debt, unusual weather, etc.).

**Implementation:**
- **State**: Truncated to 252 characters + "..." if needed
- **Attributes**: Full reasoning available in `full_reasoning` attribute
- **Layer Breakdown**: Individual layer decisions in `layers` attribute

### Example

**Truncated State (255 chars max):**
```
State WARNING: DM -522 beyond expected for 7.0°C (expected: -360) | Math WC: kuehne; Zone: Moderate Cold; Optimal: 26.2°C; Safety: +1.5°C; Adjusted: 27.7°C; Unusual weather (severity=0.5); Current: 35.0°C → offset: -4.8°C; Weight: 0.49...
```

**Full Reasoning in Attributes:**
```yaml
attributes:
  full_reasoning: "State WARNING: DM -522 beyond expected for 7.0°C (expected: -360) | Math WC: kuehne; Zone: Moderate Cold; Optimal: 26.2°C; Safety: +1.5°C; Adjusted: 27.7°C; Unusual weather (severity=0.5); Current: 35.0°C → offset: -4.8°C; Weight: 0.49 | GE-Spot Q0: CHEAP (night)"
  applied_offset: -4.8
  decision_timestamp: "2025-10-16T00:04:31"
  layers:
    - reason: "WARNING: DM -522 beyond expected for 7.0°C (expected: -360)"
      offset: 1.5
      weight: 0.8
    - reason: "Math WC: kuehne; Zone: Moderate Cold; Optimal: 26.2°C"
      offset: -4.8
      weight: 0.49
    - reason: "GE-Spot Q0: CHEAP (night)"
      offset: 0.0
      weight: 0.3
```

### Accessing Full Reasoning

**Developer Tools → States:**
```
sensor.effektguard_optimization_reasoning
```

**Automations/Templates:**
```yaml
{{ state_attr('sensor.effektguard_optimization_reasoning', 'full_reasoning') }}
```

**Lovelace Card:**
```yaml
type: entities
entities:
  - entity: sensor.effektguard_optimization_reasoning
    secondary_info: last-changed
    attribute: full_reasoning
```

### Layer-by-Layer Analysis

```yaml
{% for layer in state_attr('sensor.effektguard_optimization_reasoning', 'layers') %}
{{ layer.reason }} (offset: {{ layer.offset }}°C, weight: {{ layer.weight }})
{% endfor %}
```

## Other Sensors

All other EffektGuard sensors stay well below the 255-character limit. Only `optimization_reasoning` required special handling due to its detailed nature.

## Debug Logging

When reasoning is truncated, a debug log is generated:
```
DEBUG: Reasoning truncated to 255 chars, full reasoning in attributes
```

Enable debug logging to see truncation events:
```yaml
logger:
  default: info
  logs:
    custom_components.effektguard.sensor: debug
```
