# Sensor state limits

Home Assistant caps a state at 255 characters. A longer state is logged as an error and shown
as `unknown`. It also shows a state on one line: a line break in it renders as a space.

`sensor.effektguard_optimization_reasoning` carries the decision engine's reasoning: entries
separated by ` · `. First what changed the offset after the layers voted, the last one first, as
before → after:

```
Heater guard +0.0 → -1.0 (DM -582, heater starts -700) · Compressor at max +1.4 → +0.0
```

Then each layer that voted, heaviest first, as offset@weight (why). Where more than one thing made
the vote, each part is cause → what it adds, so the parts add up to the vote; a part that changed
nothing is left out:

```
Spot Price +3.5@0.80 (very cheap night → +1.7, price +71% in 2h → pre-heat +2.0, room 0.7°C over → -0.2)
T2 +4.2@0.81 (DM -800 past -646 → +7.0, room warming 0.30°C/h → -2.8)
```

When the reasoning is longer than 255 characters, `sensor.py` keeps whole entries from the front
and ends the state with ` · …`.

| Attribute | Content |
|---|---|
| `full_reasoning` | the whole reasoning |
| `applied_offset` | the decided offset, °C |
| `layers` | one line per layer that voted, heaviest first: `{name: "offset@weight (why)"}` |

The layer names are `Safety`, `Thermal Debt` (or `T1`, `T2`, `T3`, `Anti-windup`,
`Anti-windup Cooldown`, whichever tier is active), `Proactive`, `Peak`, `Learned Pre-heat`,
`Math WC`, `Weather Pre-heat`, `Spot Price`, `Comfort`. During a service override the only layer
is `Manual Override`; when the engine fails it is `Safe Mode`. Every layer, voting or not, is an
attribute of `sensor.effektguard_current_offset` as `{reason, offset, weight}`.

## Reading it

Open the sensor and choose Details: `layers` shows one line per vote. Or a template:

```jinja
{{ state_attr('sensor.effektguard_optimization_reasoning', 'full_reasoning') }}
```
