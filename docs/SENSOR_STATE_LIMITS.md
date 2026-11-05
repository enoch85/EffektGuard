# Sensor state limits

Home Assistant caps a state at 255 characters. A longer state is logged as an error and shown
as `unknown`.

`sensor.effektguard_optimization_reasoning` carries the decision engine's reasoning, which is
often longer. `sensor.py` cuts the state to 252 characters plus `...` when the reasoning is longer
than 255, and puts the full text in attributes:

| Attribute | Content |
|---|---|
| `full_reasoning` | the whole reasoning string |
| `applied_offset` | the decided offset, °C |
| one per layer, named after the layer | `{reason, offset, weight}` |

The layer names are `Safety`, `Thermal Debt` (or `T1`, `T2`, `T3`, `Anti-windup`,
`Anti-windup Cooldown`, whichever tier is active), `Proactive`, `Peak`, `Learned Pre-heat`,
`Math WC`, `Weather Pre-heat`, `Spot Price`, `Comfort`. During a service override the only layer
is `Manual Override`; when the engine fails it is `Safe Mode`.

## Reading it

Developer Tools → States → `sensor.effektguard_optimization_reasoning`, or a template:

```jinja
{{ state_attr('sensor.effektguard_optimization_reasoning', 'full_reasoning') }}
```

One line per layer, whatever the active tier is called:

```jinja
{% for name, layer in states.sensor.effektguard_optimization_reasoning.attributes.items()
   if layer is mapping and 'weight' in layer %}
{{ name }}: {{ layer.reason }} (offset {{ layer.offset }}, weight {{ layer.weight }})
{% endfor %}
```
