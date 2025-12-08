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

The optimization system uses a **shared layer architecture** where reusable layer components are created once and shared across consumers:

```mermaid
flowchart TB
    subgraph Layers["LAYERS (shared, reusable)"]
        direction TB
        TL["<b>thermal_layer.py</b><br/>→ EmergencyLayer, ProactiveLayer<br/>+ estimate_dm_recovery_time()<br/>+ get_thermal_debt_status()"]
        CL["<b>comfort_layer.py</b><br/>→ ComfortLayer"]
        PrL["<b>price_layer.py</b><br/>→ PriceAnalyzer<br/>+ find_cheapest_window()"]
        EfL["<b>effect_layer.py</b><br/>→ EffectManager"]
        WL["<b>weather_layer.py</b><br/>→ WeatherLayer"]
        PredL["<b>prediction_layer.py</b><br/>→ ThermalStatePredictor"]
    end

    Layers --> DE

    subgraph DE["DECISION ENGINE (space heating)"]
        DE1["Creates: EmergencyLayer, ProactiveLayer, ComfortLayer"]
        DE2["Uses: PriceAnalyzer, EffectManager"]
        DE3["Exposes: emergency_layer, price for sharing"]
    end

    DE --> DHW

    subgraph DHW["DHW OPTIMIZER (DHW scheduling)"]
        DHW1["Receives: emergency_layer (shared from DecisionEngine)"]
        DHW2["Receives: price_analyzer (shared from DecisionEngine)"]
        DHW3["Uses: estimate_dm_recovery_time(), find_cheapest_window()"]
    end

    DHW --> COORD

    subgraph COORD["COORDINATOR"]
        COORD1["Creates DecisionEngine (with all layers)"]
        COORD2["Creates DHWOptimizer (with shared layers)"]
        COORD3["Uses get_thermal_debt_status() for display"]
        COORD4["Uses ThermalStatePredictor for predictions"]
    end
```

### Layer Sharing Flow

1. **Coordinator** creates `DecisionEngine` which instantiates all layers
2. **DecisionEngine** exposes `emergency_layer` and `price` (PriceAnalyzer) as properties
3. **Coordinator** passes these shared instances to `DHWOptimizer`
4. **DHWOptimizer** uses shared layers to gate DHW during thermal debt and find cheap windows

## Production Quality Features

The analysis reveals this is a **production-quality system** designed for real Swedish homes:

- **Multiple safety layers** with graceful degradation
- **Context-aware algorithms** that adapt from Malmö to Kiruna automatically
- **Research-based thresholds** from Swedish NIBE forums and real-world validation
- **Effect tariff optimization** with native quarterly precision
- **Manual override capabilities** for diagnostic and emergency use

The system balances cost optimization with heat pump health and comfort, using proven Swedish research and real-world validation.