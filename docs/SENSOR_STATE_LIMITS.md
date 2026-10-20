# Sensor State Character Limits

## Home Assistant 255-Character Limit

Home Assistant has a hard limit of **255 characters** for sensor states. When a sensor state exceeds this limit, it falls back to "unknown" and logs a warning.

## EffektGuard Solution

### Optimization Reasoning Sensor

The `sensor.effektguard_optimization_reasoning` can generate very detailed reasoning strings, especially during critical situations (thermal debt, unusual weather, etc.).

**Implementation** (`sensor.py`, key `optimization_reasoning`):
- **State**: the reasoning, cut to 252 characters + "..." when it is longer than 255 (silently - nothing is logged)
- **Attributes**: the full reasoning in `full_reasoning`, the decided offset in `applied_offset`, and
  one attribute per layer, named after the layer (`Safety`, `Thermal Debt`, `Proactive`, `Peak`,
  `Learned Pre-heat`, `Math WC`, `Weather Pre-heat`, `Spot Price`, `Comfort`), each with its
  `reason`, `offset` and `weight`

### Example

A production record (2026-10-18):

**State (cut to 255 characters):**
```
[Math WC] Math WC: en442_design_point; Zone: Moderate Cold; Optimal: 27.7°C; Safety: +0.5°C; Adjusted: 28.2°C; Current: 29.5°C → offset: -0.8°C; Weight: 0.20; Deferred: Moderate debt (DM -280) | [Spot Price] Q42: NORMAL (day) | Adapter: gespot_current_...
```

**Attributes:**
```yaml
attributes:
  full_reasoning: "[Math WC] Math WC: en442_design_point; Zone: Moderate Cold; ... | [Spot Price] Q42: NORMAL (day) | Adapter: gespot_current_price_se4 | Horizon: 10.0h | ..."
  applied_offset: -3.17
  Thermal Debt:
    reason: "DM -280 but 1.2°C over target - let cool naturally"
    offset: 0
    weight: 0
  Spot Price:
    reason: "Q42: NORMAL (day) | Adapter: gespot_current_price_se4 | Horizon: 10.0h"
    offset: 0
    weight: 0.8
  # ... one entry per layer
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
{% set s = 'sensor.effektguard_optimization_reasoning' %}
{% for name in ['Safety', 'Thermal Debt', 'Proactive', 'Peak', 'Learned Pre-heat', 'Math WC', 'Weather Pre-heat', 'Spot Price', 'Comfort'] %}
{% set layer = state_attr(s, name) %}
{% if layer %}{{ name }}: {{ layer.reason }} (offset: {{ layer.offset }}°C, weight: {{ layer.weight }}){% endif %}
{% endfor %}
```

## Other Sensors

All other EffektGuard sensors stay well below the 255-character limit. Only `optimization_reasoning` required special handling due to its detailed nature.
