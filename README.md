# EffektGuard

**Intelligent NIBE heat pump optimizer**

<img src="EffektGuard-logo.png" alt="EffektGuard Logo" width="200"/>

[![hacs_badge](https://img.shields.io/badge/HACS-Default-41BDF5.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.0.23-blue)
![HA](https://img.shields.io/badge/Home%20Assistant-2025.10%2B-blue)

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

Automatically optimizes your NIBE heat pump to minimize electricity costs (spot prices + effect tariff) while maintaining comfort and heat pump health. Continuously adjusts heating curve offset based on prices, peak risk, weather forecasts, and learned building characteristics.

**Lower bills without sacrificing comfort or longevity.**

## Key Features

### 🎯 Multi-Layer Optimization Engine
8-layer decision system that balances competing priorities:
- **Safety** (temperature limits) - always enforced
- **Emergency** (thermal debt prevention) - climate-aware DM thresholds
- **Effect tariff** (peak avoidance) - predictive 15-min protection
- **Weather compensation** (mathematical flow temp) - André Kühne + Timbones formulas
- **Weather prediction** (pre-heating) - time-aware cold snap protection
- **Spot price** (cost reduction) - native GE-Spot 15-min integration
- **Learning** (predictive control) - self-tuning thermal model
- **Comfort** (tolerance) - reactive temperature correction

### 🌍 Global Climate Adaptation
Automatic latitude-based zone detection (Arctic to Mediterranean):
- **Extreme Cold** (66.5°N+): Kiruna, Tromsø - DM -800 to -1200 normal
- **Very Cold** (60.5-66.5°N): Luleå, Umeå - DM -600 to -1000 normal  
- **Cold** (56-60.5°N): Stockholm, Oslo, Helsinki - DM -450 to -700 normal
- **Moderate Cold** (54.5-56°N): Copenhagen, Malmö - DM -300 to -500 normal
- **Standard** (<54.5°N): Paris, London - DM -200 to -350 normal

**No configuration needed** - uses Home Assistant latitude. DM -1500 absolute maximum enforced globally.

### 🧠 Self-Learning Capability
Learns your building over 7-14 days:
- **UFH type detection** - concrete slab (6h lag) vs timber (2-3h lag) vs radiators (<1h lag)
- **Thermal mass** - building heat storage capacity (kWh/°C)
- **Heat loss coefficient** - envelope performance (W/°C)
- **Heating efficiency** - system response to offset changes (°C/°C)
- **Weather patterns** - seasonal adaptation with unusual weather detection

Predictive pre-heating uses learned parameters for intelligent load shifting.

### ⚡ Effect Tariff Optimization
Native 15-minute (quarterly) integration:
- **Top-3 monthly peak tracking** - prevents creating new peaks
- **Predictive peak avoidance** - acts before spikes using temp trends
- **Day/night weighting** - full effect daytime, reduced nighttime
- **Savings calculation** - estimates monthly savings (effect + spot)

Works with any 15-min price source ([GE-Spot](https://github.com/Tropos-AS/ge-spot), Nordpool, Tibber, etc.).

### 🌡️ Weather Compensation (Mathematical)
Physics-based flow temperature optimization:
- **André Kühne formula** - validated across manufacturers (Vaillant, Daikin, NIBE)
- **Timbones method** - radiator-specific calculations (BS EN442)
- **UFH adjustments** - concrete slab (-8°C), timber (-5°C)
- **Climate-aware margins** - automatic safety headroom by zone

Replaces NIBE's simple outdoor temp curves with proper heat transfer mathematics.

### 🔒 Safety-First Design
Production-ready safety mechanisms:
- **Climate-aware thermal debt** - DM thresholds adapt to outdoor temp + zone
- **Trend-aware damping** - prevents overshoot/undershoot (±0.3°C/h detection)
- **Configuration validation** - warns about potentially problematic setups
- **DHW coordination** - prevents thermal debt from hot water cycles
- **Manual override** - services for diagnostic control

## Requirements

- **Home Assistant** 2025.10+
- **NIBE heat pump** with MyUplink (F2040, F750, F730, S1155, S-series)
- **MyUplink integration** configured
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
1. **Select NIBE entity** - heating curve offset (number.xxx_offset_s1_47011)
2. **Select GE-Spot entity** - quarterly price sensor
3. **Select weather entity** - forecast integration
4. **Configure targets** - indoor temperature, tolerance, optimization mode
5. **Optional features** - DHW optimization, power meter, extra sensors

System auto-detects:
- Climate zone (from latitude)
- UFH type (from thermal lag)
- Heat pump model (from entity patterns)
- Pump configuration (validates against system type)

## Architecture

### Clean Separation of Concerns
```
Integration Layer (HA-specific)
└── coordinator.py (DataUpdateCoordinator pattern)
    ├── climate.py (main UI entity)
    ├── sensor.py (monitoring entities)
    └── services.yaml (manual control)

Optimization Engine (pure Python)
└── decision_engine.py (8-layer aggregation)
    ├── thermal_model.py (physics)
    ├── price_analyzer.py (GE-Spot classification)
    ├── effect_manager.py (peak tracking)
    ├── weather_compensation.py (mathematical WC)
    ├── adaptive_learning.py (self-tuning)
    ├── thermal_predictor.py (pre-heating)
    └── climate_zones.py (latitude detection)

Data Adapters (external interfaces)
├── nibe_adapter.py (MyUplink read/write)
├── gespot_adapter.py (price data)
└── weather_adapter.py (forecast)
```

### Data Flow
```
NIBE/GE-Spot/Weather Entities → Adapters → Coordinator →
Decision Engine → Optimization → Climate Entity → NIBE Offset
```

5-minute update cycle with instant responses to entity state changes (power sensor availability listener).

## Technical Details

### Decision Aggregation
Weighted average of active layers with critical layer override:
- **Critical layers** (weight ≥ 1.0): Safety, Emergency, Effect @ peak
- **Advisory layers** (weight < 1.0): Weighted aggregation
- **Emergency always wins** - thermal safety > peak cost protection

### Thermal Debt (Degree Minutes)
NIBE's heat deficit tracking (Menu 4.9.3):
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

### Self-Learning Timeline
- **Day 1-3**: Low confidence (0.0-0.3), conservative defaults
- **Day 4-7**: Medium confidence (0.3-0.7), starts using learned params
- **Day 8-14**: High confidence (0.7-1.0), fully optimized
- **Ongoing**: Continuous refinement, seasonal adaptation

672 observations (1 week @ 15-min) minimum for reliable learning.

## Documentation

- **[Climate Zones](docs/CLIMATE_ZONES.md)** - Global climate adaptation system
- **[Architecture](architecture/00_overview.md)** - System design with flow diagrams
- **[DHW Optimization](docs/DHW_OPTIMIZATION.md)** - Hot water scheduling
- **[Release Process](docs/RELEASE_PROCESS.md)** - Version management

## Development Status

**Current version:** v0.0.23 (pre-release)  
**Phase 6 complete:** Self-learning capability integrated  
**Production testing:** Active in Swedish homes

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
