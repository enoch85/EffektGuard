# EffektGuard

<img src="icons/logo.png" alt="EffektGuard logo" width="160"/>

![Version](https://img.shields.io/badge/version-0.7.1-blue)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.11%2B-blue)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github)](https://github.com/sponsors/enoch85)

NIBE heat pump control for Home Assistant. It moves the heating-curve offset against spot prices,
the Swedish effect tariff, the weather forecast and the pump's own degree minutes, with comfort and
pump health ahead of savings.

**Use at your own risk.** This software writes to a real heat pump. You are responsible for your
heating, your bill and your pump. Watch it closely the first weeks and turn it off if anything
looks wrong.

## Requirements

- Home Assistant 2025.11 or newer.
- A NIBE heat pump exposed as Home Assistant entities by one of:
  - [MyUplink](https://www.home-assistant.io/integrations/myuplink/) (cloud)
  - [nibe_heatpump](https://www.home-assistant.io/integrations/nibe_heatpump/) (local, NibeGW or
    Modbus TCP; most of its entities are disabled by default, enable the ones you need)
  - [Modbus](https://www.home-assistant.io/integrations/modbus/) sensors plus a template number
    for the offset register, see [Local Modbus](#local-modbus)
- A model profile: F730, F750, F1155, F2040 or S1155. Pick the nearest for another NIBE model.
- [GE-Spot](https://github.com/enoch85/ge-spot) for 15-minute spot prices. Without it there is no
  price optimization.
- Optional: a weather entity with an hourly forecast (Met.no works) for pre-heating before cold
  days, and a whole-house power meter for peak tracking (without one the pump's own phase currents
  are used when it exposes them).

## Install

HACS: *Integrations, three-dot menu, Custom repositories*, add
`https://github.com/enoch85/EffektGuard` as an integration, install, restart Home Assistant.
Manual: copy `custom_components/effektguard/` into your config directory and restart. Then
*Settings, Devices & services, Add integration, EffektGuard*.

## Setup

The config flow asks, in this order:

1. **Offset entity**: the number entity for the heating-curve offset (register 47011 on the
   F-series, for example `number.heat_offset_s1_47011`).
2. **Price sensor**: a GE-Spot `current_price` sensor, and whether price optimization is on.
3. **Heat pump model**: default F750.
4. **Weather entity**, and whether peak protection is on.
5. **Sensors**: degree minutes, power meter, the temporary-lux switch (needed for hot-water
   control), extra room sensors, and override fields for setups discovery cannot name (generic
   Modbus, renamed entities).

Everything else is found by entity-name pattern. *Reconfigure* changes the entities later.
*Configure* holds the options: mode (comfort, balanced, savings), tolerance (0.5-3.0 °C), thermal
mass and insulation (0.5-2.0), hot-water target, amount and schedule, and airflow rates on the
F750/F730. The target temperature is set on the climate entity.

## What you get

- `climate.effektguard`: target temperature (18.5-25 °C), HEAT or OFF, presets none / eco / away
  / comfort (balanced / savings / savings and 3 °C lower / comfort). OFF holds the offset at 0 and
  runs no layer.
- Sensors (`sensor.effektguard_current_offset`, `sensor.effektguard_degree_minutes`, ...): current
  offset, degree minutes, supply, outdoor and indoor temperature, current electricity price, peak
  today, peak this month, NIBE power, compressor frequency and health, optimization reasoning,
  quarter of day, price period classification, indoor and outdoor temperature trend, estimated
  monthly savings, heat pump model, DHW status, recommendation and scheduled start. The F750/F730
  also get airflow enhancement and thermal gain.
- Switches (`switch.effektguard_enable_optimization`, ...): enable optimization, price
  optimization, peak protection and weather prediction (on by default); hot water optimization and
  airflow optimization (off by default, the latter F750/F730 only).
- Services: `effektguard.force_offset` (offset -10 to +10 for 0-480 min, 0 = one cycle),
  `effektguard.boost_heating` (+10 for 30-360 min), `effektguard.boost_dhw` (temporary lux for
  30-180 min), `effektguard.reset_peak_tracking`, `effektguard.calculate_optimal_schedule`
  (returns a 24-hour price preview).

## How it decides

Every 5 minutes, 10 seconds past the mark, the coordinator reads the pump, the prices and the
forecast, runs nine layers (safety, thermal debt, proactive debt prevention, effect tariff, learned
pre-heat, weather compensation, weather pre-heat, spot price, comfort), each voting an offset with a
weight, and writes one integer offset. A layer at weight 1.0 wins outright; otherwise the weighted
average. The degree-minute thresholds come from Home Assistant's latitude and the outdoor
temperature, not from settings. Details in [docs/architecture/](docs/architecture/00_overview.md).

## Safety

- Indoor below 18 °C: the safety layer asks +10 at full weight (+7.5 while the room is already
  warming fast).
- Degree minutes past -1500: no layer may cut below 0 unless the room is above its band, and
  nothing raises into the flat-out compressor.
- At most one write every 5 minutes, whole degrees only, clamped to -10..+10 and to the number
  entity's own range.
- A cycle that cannot read the pump writes nothing and the entities go unavailable.
- `force_offset` cannot hold the pump below these floors.

## Local Modbus

With the generic `modbus` integration, define sensors for the registers you need (temperatures and
degree minutes are raw x10, use `scale: 0.1`; the offset register is not scaled) and wrap the
offset register in a template number so EffektGuard can write it:

```yaml
template:
  - number:
      - name: "Heat Offset S1 47011"
        unique_id: nibe_heat_offset_s1_47011
        state: "{{ states('sensor.heat_offset_s1_47011') | float(0) }}"
        availability: "{{ has_value('sensor.heat_offset_s1_47011') }}"
        min: -10
        max: 10
        step: 1
        set_value:
          - action: modbus.write_register
            data:
              hub: nibe
              slave: 1
              address: 47011
              value: "{{ (value | int) % 65536 }}"
```

`min`, `max` and `step` are required: a template number defaults to 0-100, and EffektGuard clamps
to the entity's range, so a negative offset would never reach the pump. The displayed value catches
up on the sensor's next poll (`scan_interval`, 15 s default). Select this number in the first setup
step.

## Documentation

- [Architecture](docs/architecture/00_overview.md): the cycle, the layers, the aggregation
- [Climate zones](docs/CLIMATE_ZONES.md): the degree-minute tables per zone
- [Hot water](docs/DHW_OPTIMIZATION.md)
- [Development](docs/dev/README.md): setup, tests, code standards
- [Contributor rulebook](CLAUDE.md)

## License

MIT, see [LICENSE](LICENSE). Author: [@enoch85](https://github.com/enoch85).
