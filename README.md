# EffektGuard

**Intelligent heat pump optimizer for dynamic electricity markets**

<img src="icons/logo.png" alt="EffektGuard Logo" width="200"/>

[![hacs_badge](https://img.shields.io/badge/HACS-Default-41BDF5.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.5.0-beta.1-blue)
![HA](https://img.shields.io/badge/Home%20Assistant-2025.10%2B-blue)
[![Sponsor on GitHub](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github&style=for-the-badge)](https://github.com/sponsors/enoch85)

---

## ⚠️ Important Disclaimer

**USE AT YOUR OWN RISK.** This integration controls your heat pump's heating curve offset. While extensively tested and designed with safety-first principles, we are **not responsible** for:
- Heat pump damage or malfunction
- Uncomfortable indoor temperatures
- Increased energy costs
- Any other issues arising from use of this integration

This is experimental software controlling real heating systems. Monitor your system closely, especially during the first weeks. If anything seems wrong, disable the integration immediately.

---

## What It Does

Automatically optimizes your heat pump to minimize electricity costs (spot prices + effect tariff) while maintaining comfort and heat pump health. Continuously adjusts heating curve offset based on prices, peak risk, weather forecasts, and learned building characteristics.

**Lower bills without sacrificing comfort or longevity.**

Currently supports NIBE heat pumps via MyUplink integration, with plans to add support for additional brands in the future.

## Key Features

### 🎯 Multi-Layer Optimization Engine
9-layer decision system that balances competing priorities:
- **Safety** (temperature limits) - always enforced
- **Emergency** (thermal debt prevention) - climate-aware DM thresholds
- **Proactive debt prevention** - trend-based future DM prediction
- **Effect tariff** (peak avoidance) - predictive 15-min protection
- **Prediction/Learning** (self-tuning) - learned thermal model for pre-heating
- **Weather compensation** (mathematical flow temp) - André Kühne + Timbones formulas
- **Weather prediction** (pre-heating) - time-aware cold snap protection
- **Spot price** (cost reduction) - forward-looking optimization with adaptive horizon
- **Comfort** (tolerance) - reactive temperature correction

### 🌍 Global Climate Adaptation
Automatic latitude-based zone detection (Arctic to Mediterranean):
- **Extreme Cold** (66.5°N+): Kiruna, Tromsø - DM -800 to -1200 normal
- **Very Cold** (60.5-66.5°N): Luleå, Umeå - DM -600 to -1000 normal  
- **Cold** (56-60.5°N): Stockholm, Oslo, Helsinki - DM -450 to -700 normal
- **Moderate Cold** (54.5-56°N): Copenhagen, Malmö - DM -300 to -500 normal
- **Standard** (<54.5°N): Paris, London - DM -200 to -350 normal

**No configuration needed** - uses Home Assistant latitude. DM -1500 absolute maximum enforced globally.

### 🧠 Self-Learning Capability (observing only — not yet driving your pump)
EffektGuard continuously observes your building and estimates:
- **UFH type detection** - concrete slab (6h lag) vs timber (2-3h lag) vs radiators (<1h lag)
- **Thermal mass** - building heat storage capacity (kWh/°C)
- **Heat loss coefficient** - envelope performance (W/°C)
- **Heating efficiency** - system response to offset changes (°C/°C)
- **Weather patterns** - seasonal adaptation with unusual weather detection

**These estimates do not currently influence control.** Learned parameters are used for
pre-heating only once the model's confidence exceeds 70%, and confidence cannot presently
reach that: the indoor temperature is read to 0.1 °C every 5 minutes, and a building's
response over 5 minutes is smaller than the sensor's own resolution. Nothing reliable can be
learned from that, so nothing is claimed from it.

Control today is physics-based and deterministic — the heating curve, the EN 442 emitter law,
weather compensation and the degree-minute safety net — none of which depends on learning.

### ⚡ Effect Tariff Optimization
Native 15-minute (quarterly) integration:
- **Top-3 monthly peak tracking** - prevents creating new peaks
- **Predictive peak avoidance** - acts before spikes using temp trends
- **Day/night weighting** - full effect daytime, reduced nighttime
- **Savings calculation** - estimates monthly savings (effect + spot)

Works with any 15-min price source (GE-Spot, Nordpool, Tibber, etc.).

### 💰 Smart Price Forecasting
Multi-factor forward-looking optimization combining:
- **Price + thermal state** - considers current indoor temp overshoot for strategic thermal storage
- **Price + building characteristics** - adaptive horizon scales with thermal mass (2.0-8.0 hours)  
- **Price + compressor dynamics** - filters brief spikes < 45 min using ramp-up/cool-down constraints

### 🌡️ Weather Compensation (Mathematical)
Physics-based flow temperature optimization:
- **André Kühne formula** - validated across manufacturers (Vaillant, Daikin, NIBE, etc.)
- **Timbones method** - radiator-specific calculations (BS EN442)
- **UFH adjustments** - concrete slab (-8°C), timber (-5°C)
- **Climate-aware margins** - automatic safety headroom by zone

Uses proper heat transfer mathematics to optimize flow temperature beyond standard outdoor temp curves.

### 🔒 Safety-First Design
Production-ready safety mechanisms:
- **Climate-aware thermal debt** - DM thresholds adapt to outdoor temp + zone
- **Trend-aware damping** - prevents overshoot/undershoot (±0.3°C/h detection)
- **Configuration validation** - warns about potentially problematic setups
- **DHW coordination** - prevents thermal debt from hot water cycles
- **Manual override** - services for diagnostic control

## Requirements

- **Home Assistant** 2025.10+
- **Compatible NIBE heat pump** connected through ONE of these data sources:
  - [MyUplink integration](https://www.home-assistant.io/integrations/myuplink/) (cloud;
    writes may require a valid myUplink subscription)
  - [nibe_heatpump integration](https://www.home-assistant.io/integrations/nibe_heatpump/)
    (local: NibeGW/MODBUS40 for F-series, built-in Modbus TCP for S-series; no
    subscription needed — recommended for local Modbus users)
  - Generic [Modbus](https://www.home-assistant.io/integrations/modbus/) YAML sensors plus
    [template numbers](https://www.home-assistant.io/integrations/template/) for writable
    registers (see [Local Modbus setups](#local-modbus-setups))
  - Models: F750, F730, F1155, F2040, S1155 (profiles); similar NIBE models work with the
    nearest profile. Additional brands: planned for future releases
- **Price integration** with 15-min data (GE-Spot, Nordpool, Tibber, etc.)
- **Weather integration** (Met.no or equivalent)

## Installation

### HACS (Recommended)

1. Open HACS → Integrations
2. Click ⋮ → Custom repositories
3. Add `https://github.com/enoch85/EffektGuard` as Integration
4. Search for "EffektGuard" and install
5. Restart Home Assistant
6. Add integration via Settings → Devices & Services → Add Integration

### Manual

1. Download latest release
2. Extract to `custom_components/effektguard/`
3. Restart Home Assistant
4. Add integration via Settings → Devices & Services

## Configuration

Guided setup flow with validation:
1. **Select heat pump entity** - heating curve offset (e.g., number.xxx_offset_s1_47011 for NIBE)
2. **Select price entity** - quarterly price sensor (any spot price integration)
3. **Select weather entity** - forecast integration
4. **Configure targets** - indoor temperature, tolerance, optimization mode
5. **Optional features** - DHW optimization, power meter, extra sensors

System auto-detects:
- Climate zone (from latitude)
- UFH type (from thermal lag)
- Heat pump model (from entity patterns)
- Pump configuration (validates against system type)

Sensors are found automatically by entity-name patterns (MyUplink, nibe_heatpump,
and common Modbus namings). If discovery misses a sensor — typical for generic
Modbus YAML setups — set it manually via the **override fields** in the Sensors
step, or later via **Reconfigure** on the integration.

### Local Modbus setups

The recommended local path is the official
[nibe_heatpump](https://www.home-assistant.io/integrations/nibe_heatpump/) integration:
it exposes read-only registers as sensors (e.g. `sensor.bt1_outdoor_temperature_40004`)
and writable registers as **numbers** (e.g. `number.heat_offset_s1_47011`,
`number.degree_minutes_16_bit_43005`). All its entities are **disabled by default** —
enable at least BT1, BT50, BT2/BT25, BT3, BT7, BT6, degree minutes, compressor state,
and Heat Offset S1 in Settings → Devices & Services → NIBE → entities.

With the generic `modbus` integration, define sensors for the registers above
(F-series values are raw ×10 — use `scale: 0.1`) and wrap the offset register in a
template number so EffektGuard can write it:

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

`min`/`max`/`step` are required as shown — a template number defaults to
0–100 and would reject negative offsets. `unique_id` registers the entity so
it is UI-editable and preferred by discovery; the `availability` guard stops
the number from erroring while the Modbus hub is down. Writes take effect
immediately; the displayed value catches up on the sensor's next poll
(`scan_interval`, 15 s default).

Select that number as the offset entity in the first setup step.

## Architecture

### Clean Separation of Concerns
```
Integration Layer (HA-specific)
└── coordinator.py (DataUpdateCoordinator pattern)
    ├── climate.py (main UI entity)
    ├── sensor.py (monitoring entities)
    ├── switch.py (DHW control switch)
    ├── config_flow.py (setup wizard)
    ├── options.py (runtime settings)
    └── services.yaml (manual control)

Optimization Engine (pure Python)
└── decision_engine.py (9-layer aggregation)
    ├── thermal_layer.py (thermal debt + emergency)
    ├── effect_layer.py (peak protection)
    ├── prediction_layer.py (learned pre-heating)
    ├── price_layer.py (spot price optimization)
    ├── weather_layer.py (mathematical WC + cold snap)
    ├── comfort_layer.py (temperature correction)
    ├── dhw_optimizer.py (18 decision rules)
    ├── adaptive_learning.py (self-tuning)
    ├── savings_calculator.py (cost estimation)
    ├── airflow_optimizer.py (S-series supply air)
    └── climate_zones.py (latitude detection)

Data Adapters (external interfaces)
├── nibe_adapter.py (MyUplink read/write)
├── gespot_adapter.py (price data)
└── weather_adapter.py (forecast)

Models (heat pump abstractions)
└── models/
    ├── base.py (abstract interface)
    ├── registry.py (model discovery)
    └── nibe/ (NIBE-specific implementations)

Utilities
└── utils/
    └── compressor_monitor.py (runtime tracking)
```

### Data Flow
```
Heat Pump/Price/Weather Entities → Adapters → Coordinator →
Decision Engine → Optimization → Climate Entity → Curve Offset
```

5-minute update cycle with instant responses to entity state changes (power sensor availability listener).

## Technical Details

### Decision Aggregation
Weighted average of active layers with critical layer override:
- **Critical layers** (weight ≥ 1.0): Safety, Emergency, Effect @ peak
- **Advisory layers** (weight < 1.0): Weighted aggregation
- **Emergency always wins** - thermal safety > peak cost protection

### Thermal Debt (Degree Minutes)
Heat deficit tracking used by many heat pumps (NIBE Menu 4.9.3, others may vary):
```
DM = ∫(BT25 - S1) dt
```
- **BT25**: Actual flow temperature
- **S1**: Target flow temperature  
- **Negative DM**: Heat deficit (compressor catching up)

Climate-aware thresholds prevent heat pump damage. DM -1500 absolute maximum enforced.

### Price Integration
Native quarterly (15-min) price periods:
- **96 periods/day** - matches effect tariff measurement
- **4-tier classification** - cheap/normal/expensive/peak (percentile-based)
- **Day/night weighting** - full optimization daytime, reduced nighttime
- **Auto-discovery** - finds price entity automatically

### Weather Compensation Math
```python
# André Kühne formula (universal)
TFlow = 2.55 × (HC × (Tset - Tout))^0.78 + Tset

# Timbones method (radiator-specific)  
TFlow = ((Pin / Pout)^(1/1.3) × (DTout / DTin)) × (Tset - Tout) + Tset
```
Combined with climate-aware safety margins (0.0-2.5°C by zone).

### Self-Learning Status

Observations are recorded every 5 minutes into a rolling 672-entry window — **56 hours**, not
the week the window was sized for. Confidence is scored from observation count, data
consistency and time span, and learned parameters are used for pre-heating only above 70%.

At the present observation cadence that threshold is not reachable, and the integration says so
rather than pretending otherwise: a 0.1 °C indoor sensor sampled every 5 minutes quantises the
building's response into steps of 1.2 °C/h, which is larger than any real heating rate. The
consistency term is therefore honestly zero and confidence settles around 0.47.

**Learning does not drive your heat pump today.** Making it do so means sampling slowly enough
for the signal to exceed the sensor's resolution — a deliberate change, not a tuning tweak,
because it would put a learned model in the control path of real heating equipment.

## Documentation

- **[Climate Zones](docs/CLIMATE_ZONES.md)** - Global climate adaptation system
- **[Architecture Overview](docs/architecture/00_overview.md)** - System design with flow diagrams
- **[DHW Optimization](docs/DHW_OPTIMIZATION.md)** - Hot water scheduling
- **[Release Process](docs/RELEASE_PROCESS.md)** - Version management
- **[Development Guide](docs/dev/README.md)** - Contributing and testing guidelines

## Development Status

**Status:** Production-ready but still not perfect
**Active users:** Running in Swedish homes with real NIBE systems

## Contributing

Production code affecting real homes. Contributions welcome, quality standards apply:
- Read entire files before editing
- Use const.py for all thresholds
- Safety-first approach
- Black formatting (line length 100)
- Test safety-critical code

See `.github/copilot-instructions.md` for guidelines.

## License

MIT License - See LICENSE file

## Credits

**Author:** [@enoch85](https://github.com/enoch85)

**Built for the Swedish community, works globally.**
