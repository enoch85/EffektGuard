# EffektGuard Architecture Overview

This document provides a high-level overview of the EffektGuard system architecture and the detailed flow analysis documents in this folder.

## System Overview

EffektGuard is a sophisticated Swedish heat pump optimization system with these key components:
- **Multi-layer decision engine** with 9 priority layers
- **Native 15-minute spot price integration** for Swedish Effektavgift
- **Context-aware thermal debt prevention** (adapts to outdoor temperature)
- **Self-learning capabilities** with thermal prediction
- **Effect tariff peak protection** (monthly top-3 tracking)

## Architecture Documents

This folder contains detailed Mermaid diagrams showing how EffektGuard works in different scenarios:

1. **[01_normal_optimization_cycle.md](01_normal_optimization_cycle.md)** - Standard 5-minute optimization cycle
2. **[02_emergency_thermal_debt.md](02_emergency_thermal_debt.md)** - Context-aware emergency response
3. **[03_effect_tariff_protection.md](03_effect_tariff_protection.md)** - 15-minute peak avoidance
4. **[04_weather_preheating.md](04_weather_preheating.md)** - Predictive pre-heating
5. **[05_spot_price_optimization.md](05_spot_price_optimization.md)** - Spot price classification
6. **[06_learning_integration.md](06_learning_integration.md)** - Phase 6 self-learning
7. **[07_manual_override_services.md](07_manual_override_services.md)** - Service-based control
8. **[08_layer_priority_system.md](08_layer_priority_system.md)** - Decision aggregation

## Key Architectural Insights

### 1. Context-Aware Safety System
- **Not fixed thresholds** - adapts degree minutes limits based on outdoor temperature
- Malmö (0°C): Expects DM -600, warns at -800
- Kiruna (-25°C): Expects DM -1150, warns at -1350
- **Absolute maximum -1500 DM** never exceeded regardless of conditions

### 2. Native Swedish Integration
- **Spot price integration provides exactly 96 quarterly periods** (15-minute intervals)
- Perfect match for Swedish Effektavgift requirements
- Day/night weighting: Full effect 06:00-22:00, 50% weight 22:00-06:00

### 3. Multi-Layer Decision Engine
- **9 prioritized decision layers** with weighted aggregation
- Critical layers can override others (Safety 1.0, Effect up to 1.0)
- Advisory layers use dynamic weighted averaging (Emergency 0.8, Price 0.8, Weather 0.85, etc.)

### 4. Self-Learning Capabilities (Phase 6)
- **Thermal state predictor** learns building characteristics
- **Weather pattern learning** adapts to seasonal changes
- **Adaptive thermal model** adjusts thermal mass estimates
- Uses 672 observations (1 week) minimum for learning

## Layer Architecture

The optimization system uses a **shared layer architecture** where reusable layer components are consumed by multiple systems:

```mermaid
flowchart TB
    subgraph Layers["Shared Layers (Reusable)"]
        direction LR
        TL[ThermalLayer<br/>DM recovery, thermal debt]
        EL[EmergencyLayer<br/>Safety gating, DHW blocking]
        PL[ProactiveLayer<br/>Preemptive protection]
        CL[ComfortLayer<br/>Indoor temp targeting]
        PrL[PriceLayer<br/>Spot price analysis]
        EfL[EffectLayer<br/>Peak power tracking]
        WL[WeatherLayer<br/>Forecast integration]
        PredL[PredictionLayer<br/>Trend analysis]
    end

    subgraph Consumers["Layer Consumers"]
        DE[Decision Engine<br/>Space heating optimization]
        DHW[DHW Optimizer<br/>Hot water scheduling]
        COORD[Coordinator<br/>Status & display]
    end

    Layers --> DE
    Layers --> DHW
    Layers --> COORD

    subgraph Shared["Shared Functions"]
        F1[estimate_dm_recovery_time]
        F2[get_thermal_debt_status]
        F3[should_block_dhw]
        F4[find_cheapest_window]
    end

    EL -.-> F3
    TL -.-> F1
    TL -.-> F2
    PrL -.-> F4
```

### Key Integration Points

| Layer | Shared By | Purpose |
|-------|-----------|---------|
| `EmergencyLayer` | Decision Engine, DHW Optimizer | `should_block_dhw()` gates DHW during thermal debt |
| `ThermalLayer` | All consumers | `estimate_dm_recovery_time()`, `get_thermal_debt_status()` |
| `PriceLayer` | Decision Engine, DHW Optimizer | `find_cheapest_window()` for cost optimization |
| `ComfortLayer` | Decision Engine | Indoor temperature targeting and heat loss estimation |

## Production Quality Features

The analysis reveals this is a **production-quality system** designed for real Swedish homes:

- **Multiple safety layers** with graceful degradation
- **Context-aware algorithms** that adapt from Malmö to Kiruna automatically
- **Research-based thresholds** from Swedish NIBE forums and real-world validation
- **Effect tariff optimization** with native quarterly precision
- **Manual override capabilities** for diagnostic and emergency use

The system balances cost optimization with heat pump health and comfort, using proven Swedish research and real-world validation.