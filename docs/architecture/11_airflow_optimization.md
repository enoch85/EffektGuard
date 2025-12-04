# Airflow Optimization for Exhaust Air Heat Pumps

Thermodynamic optimization for exhaust air heat pump ventilation (NIBE F750, F730).

## Overview

Exhaust air heat pumps extract heat from indoor exhaust air. When the compressor is working hard but struggling to meet demand, increasing ventilation can help by:

1. **More heat available** — Higher airflow brings more warm exhaust air to the evaporator
2. **Better COP** — Evaporator runs warmer, improving coefficient of performance by ~20%
3. **Faster recovery** — Combined effect accelerates heating during deficits

The trade-off is **ventilation penalty** — more cold outdoor air enters and must be heated.

## The Physics

```
Net Benefit = (Extra heat extracted) + (COP improvement) - (Ventilation penalty)
```

| Component | Formula | Typical Value |
|-----------|---------|---------------|
| Heat extraction | Q = ṁ × cp × ΔT | +0.41 kW |
| COP improvement | 20% × baseline output | +1.32 kW |
| Ventilation penalty | ṁ × cp × (T_in - T_out) | -0.70 kW (at 0°C) |
| **Net gain** | | **+1.03 kW** |

### Physical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Air density | 1.2 kg/m³ | At ~20°C |
| Specific heat | 1.005 kJ/kg·K | Air at constant pressure |
| Evaporator ΔT | 12°C | Typical temp drop through evaporator |
| COP improvement | 20% | Empirical gain from warmer evaporator |
| Standard flow | 150 m³/h | NIBE F750 normal ventilation |
| Enhanced flow | 252 m³/h | NIBE F750 maximum ventilation |

## When Enhanced Airflow Helps

### ✅ Beneficial (Green Zone)

| Outdoor Temp | Min Compressor | Expected Gain | Max Duration |
|--------------|----------------|---------------|--------------|
| +5°C to +10°C | ≥50% | +1.0 to +1.4 kW | Until recovered |
| 0°C to +5°C | ≥50% | +0.8 to +1.1 kW | 45-60 min |
| -5°C to 0°C | ≥62% | +0.5 to +0.9 kW | 30-45 min |

### ⚠️ Marginal (Yellow Zone)

| Outdoor Temp | Min Compressor | Expected Gain | Max Duration |
|--------------|----------------|---------------|--------------|
| -10°C to -5°C | ≥75% | +0.3 to +0.6 kW | 20-30 min |
| -15°C to -10°C | ≥87% | +0.1 to +0.3 kW | 15-20 min |

### ❌ Don't Use (Red Zone)

| Condition | Reason |
|-----------|--------|
| Outdoor < -15°C | Ventilation penalty exceeds all gains |
| Compressor < threshold | Not limited by heat source |
| Indoor ≥ target - 0.2°C | No deficit to recover |
| Indoor trend > +0.1°C/h | Already warming, let stabilize |

## Compressor Threshold Formula

The minimum compressor % increases as outdoor temperature drops:

```
min_compressor_% = max(50, 50 + (-2.5 × outdoor_temp))
```

| Outdoor °C | Minimum Compressor % |
|------------|---------------------|
| ≥0 | 50% |
| -5 | 62% |
| -10 | 75% |
| -15 | 87% |

## Duration Guidelines

| Indoor Deficit | Base Duration | Cold Weather Adjustment |
|----------------|---------------|------------------------|
| 0.2 - 0.3°C | 15 min | — |
| 0.3 - 0.5°C | 20 min | — |
| 0.5 - 1.0°C | 45 min | Max 30 min if < -5°C |
| > 1.0°C | 60 min | Max 20 min if < -10°C |

### Stop Conditions

End enhanced ventilation when **any** of:
- Indoor temp reaches target
- Indoor trend turns positive (> +0.1°C/h)
- Compressor drops below threshold
- Maximum duration reached (anti-cycling protection)

## EffektGuard Implementation

### Entities Created

| Entity | Type | Description |
|--------|------|-------------|
| `switch.effektguard_airflow_optimization` | Switch | Enable/disable automatic airflow optimization |
| `sensor.effektguard_airflow_enhancement` | Sensor | Current decision status and reason |
| `sensor.effektguard_airflow_thermal_gain` | Sensor | Expected thermal gain (kW) |

### Model Support

Airflow optimization is only available for exhaust air heat pumps:

| Model | `supports_exhaust_airflow` | Notes |
|-------|---------------------------|-------|
| NIBE F750 | ✅ True | Exhaust air ASHP |
| NIBE F730 | ✅ True | Exhaust air ASHP |
| NIBE F2040 | ❌ False | Outdoor unit (no exhaust) |
| NIBE S1155 | ❌ False | Ground source (no exhaust) |

### NIBE Control Entity

EffektGuard controls the NIBE "Increased Ventilation" switch:
- **Entity pattern**: `switch.{device}_increased_ventilation`
- **Example**: `switch.f750_cu_3x400v_increased_ventilation`

### Automatic Control Flow

```
Coordinator Update Cycle
        │
        ▼
┌───────────────────────┐
│ Is airflow_optimization│
│ switch enabled?       │
└───────────┬───────────┘
            │ Yes
            ▼
┌───────────────────────┐
│ Evaluate thermal state │
│ (outdoor, indoor,      │
│  target, compressor,   │
│  trend)                │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Calculate net benefit  │
│ from enhanced airflow  │
└───────────┬───────────┘
            │
      ┌─────┴─────┐
      │           │
   benefit > 0  benefit ≤ 0
      │           │
      ▼           ▼
┌─────────────┐ ┌─────────────┐
│ Turn ON     │ │ Turn OFF    │
│ increased   │ │ increased   │
│ ventilation │ │ ventilation │
└─────────────┘ └─────────────┘
```

### Rate Limiting & Safety

| Protection | Value | Purpose |
|------------|-------|---------|
| Minimum enhanced duration | 5 min | Prevent rapid cycling |
| API rate limit | 5 min | Reduce MyUplink API calls |
| Redundant state check | — | Skip call if already in desired state |

## Configuration

### Options Panel

When enabled for a supported model, the Options panel shows:

| Option | Default | Description |
|--------|---------|-------------|
| Enable Airflow Optimization | Off | Master toggle |

### Advanced Tuning (const.py)

```python
# Flow rates (m³/h) - adjust for your heat pump
AIRFLOW_DEFAULT_STANDARD = 150.0  # Normal ventilation
AIRFLOW_DEFAULT_ENHANCED = 252.0  # Maximum ventilation

# Decision thresholds
AIRFLOW_OUTDOOR_TEMP_MIN = -15.0  # Don't enhance below this
AIRFLOW_INDOOR_DEFICIT_MIN = 0.2  # Minimum deficit to trigger
AIRFLOW_TREND_WARMING_THRESHOLD = 0.1  # °C/h - already warming

# Compressor threshold formula
AIRFLOW_COMPRESSOR_BASE_THRESHOLD = 50.0  # % at 0°C outdoor
AIRFLOW_COMPRESSOR_SLOPE = -2.5  # % per °C below 0

# Duration limits (minutes)
AIRFLOW_DURATION_SMALL_DEFICIT = 15
AIRFLOW_DURATION_MODERATE_DEFICIT = 20
AIRFLOW_DURATION_LARGE_DEFICIT = 45
AIRFLOW_DURATION_EXTREME_DEFICIT = 60
AIRFLOW_DURATION_COOL_CAP = 30  # Max if outdoor < -5°C
AIRFLOW_DURATION_COLD_CAP = 20  # Max if outdoor < -10°C
```

## Code Usage

### AirflowOptimizer Class

```python
from custom_components.effektguard.optimization.airflow_optimizer import (
    AirflowOptimizer,
    FlowDecision,
)

optimizer = AirflowOptimizer(
    flow_standard=150.0,  # m³/h
    flow_enhanced=252.0,  # m³/h
)

decision = optimizer.evaluate(
    temp_outdoor=0.0,       # °C
    temp_indoor=20.5,       # °C
    temp_target=21.0,       # °C
    compressor_pct=80.0,    # % of max Hz
    trend_indoor=-0.2,      # °C/hour
)

if decision.should_enhance:
    print(f"Enhance for {decision.duration_minutes} min")
    print(f"Expected gain: {decision.expected_gain_kw:.2f} kW")
    print(f"Reason: {decision.reason}")
```

### Simple Function Interface

```python
from custom_components.effektguard.optimization.airflow_optimizer import (
    should_enhance_airflow,
)

should_enhance, duration = should_enhance_airflow(
    temp_outdoor=0.0,
    temp_indoor=20.5,
    temp_target=21.0,
    compressor_pct=80.0,
    trend_indoor=-0.2,
)
# Returns: (True, 45)
```

### Integration with NIBE Adapter

```python
# Coordinator automatically calls this
decision = self.airflow_optimizer.evaluate(...)

if decision.should_enhance:
    await self.nibe.set_enhanced_ventilation(True)
else:
    await self.nibe.set_enhanced_ventilation(False)
```

## References

- Heat transfer: `Q = ṁ × cp × ΔT`
- Carnot COP: `COP = T_hot / (T_hot - T_cold)`
- NIBE F750 specifications: 90-252 m³/h airflow range
- Implementation: `custom_components/effektguard/optimization/airflow_optimizer.py`
- Tests: `tests/unit/optimization/test_airflow_optimizer.py` (33 tests)
