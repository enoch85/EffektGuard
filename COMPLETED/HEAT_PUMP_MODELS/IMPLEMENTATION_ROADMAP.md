```markdown
# Heat Pump Model-Specific Optimization

## Overview

EffektGuard now supports model-specific optimization for NIBE heat pumps. Each heat pump model has vastly different power consumption characteristics, efficiency curves, and optimal operating parameters.

## Why Model-Specific Profiles Matter

### ❌ **OLD ASSUMPTION (INCORRECT)**:
"3 kW is normal for heat pumps"

### ✅ **REALITY**:
Power consumption varies **massively** based on:

1. **Heat pump model** (6kW to 16kW capacity)
2. **House size** (100m² to 250m²)  
3. **Insulation quality** (pre-1990 vs modern)
4. **Heating medium** (floor heating vs radiators)
5. **Outdoor temperature** (-30°C to +7°C range in Sweden)

## Supported Models

### NIBE F-Series (Air Source Heat Pumps)

#### **F730** - 6kW ASHP
- **Target**: 80-120m² or well-insulated houses
- **Electrical**: 1.0-2.0kW typical, max 4.5kW
- **COP Range**: 2.0-5.0
- **Best For**: Smaller homes, excellent insulation
- **Typical Power**:
  - At 0°C: ~1.2kW (0°C × 4.0 COP = 4.8kW heat with 1.2kW electrical)
  - At -15°C: ~1.8kW (COP 2.7)

#### **F750** - 8kW ASHP ⭐ **Most Common**
- **Target**: 100-150m² standard insulation houses
- **Electrical**: 1.2-2.8kW typical, max 6.5kW (3-phase)
- **COP Range**: 2.0-5.0
- **Best For**: Typical Swedish single-family homes
- **Typical Power**:
  - At 0°C: ~1.5kW (Malmö/Gothenburg average winter)
  - At -15°C: ~2.3kW (Northern Sweden design temp)
  - At -25°C: ~3.5kW (Kiruna extreme cold)

#### **F2040** - 12-16kW ASHP
- **Target**: 180-250m² or poorly insulated houses
- **Electrical**: 2.5-6.5kW typical, max 10kW
- **COP Range**: 1.8-4.8
- **Best For**: Large or older homes
- **Typical Power**:
  - At 0°C: ~2.5kW
  - At -15°C: ~4.0kW
  - At -25°C: ~6.5kW+ (may activate auxiliary)

### NIBE S-Series (Ground Source Heat Pumps)

#### **S1155** - 12kW GSHP
- **Target**: 120-180m² with ground source
- **Electrical**: **1.5-3.5kW typical** (much lower than F2040!)
- **COP Range**: 3.5-5.5 (consistently high)
- **Best For**: Best efficiency year-round
- **Typical Power**:
  - At 0°C: ~1.8kW (COP 5.0!)
  - At -15°C: ~2.2kW (COP 4.3)
  - At -25°C: ~2.6kW (COP 3.8) - still excellent

#### **S1255** - 15kW GSHP
- **Target**: 180-250m² with ground source
- **Electrical**: **2.0-4.5kW typical** (vs 3-8kW for F2040)
- **COP Range**: 3.5-5.5
- **Best For**: Large homes with best efficiency

### Key Differences

| Model | Heat Output | Electrical (typical) | COP at -15°C | Use Case |
|-------|-------------|---------------------|--------------|----------|
| **F730** | 6kW | 1.0-2.0kW | 2.7 | Small/efficient homes |
| **F750** | 8kW | 1.2-2.8kW | 2.7 | Typical Swedish homes |
| **F2040** | 12-16kW | 2.5-6.5kW | 2.5 | Large/poor insulation |
| **S1155** | 12kW | 1.5-3.5kW | 4.3 | Efficient GSHP |
| **S1255** | 15kW | 2.0-4.5kW | 4.3 | Large efficient GSHP |

## House Size Impact

Heat demand scales with house size and insulation:

### **100m² Well-Insulated** (modern building codes)
- Heat loss: ~50 W/m² at ΔT=30°C
- At -15°C (ΔT=36°C): **6kW heat demand**
- F730 perfect match: 6kW ÷ 2.7 COP = **2.2kW electrical**

### **150m² Standard** (1990-2010 construction)
- Heat loss: ~70 W/m² at ΔT=30°C
- At -15°C: **12.6kW heat demand**
- F750 slightly undersized but OK: 8kW ÷ 2.7 COP = **3.0kW electrical** (auxiliary may activate)

### **200m² Poor Insulation** (pre-1990)
- Heat loss: ~100 W/m² at ΔT=30°C
- At -15°C: **24kW heat demand** ⚠️
- F2040 undersized: 16kW ÷ 2.5 COP = **6.4kW electrical** (auxiliary definitely needed)
- **Recommendation**: Insulation upgrade + larger heat pump or dual F2040 units

## Heating Medium Impact

### Floor Heating (UFH)
- **Flow temp**: 25-35°C
- **COP boost**: +10-15% (lower flow temp = better efficiency)
- **F750 example**: COP 3.0 → 3.3-3.4 with UFH
- **Power savings**: ~10% lower electrical consumption

### Radiators
- **Flow temp**: 45-55°C
- **COP penalty**: Standard performance
- **F750 example**: COP 3.0 at -10°C

### Air Heating
- **Flow temp**: 55-60°C (highest)
- **COP penalty**: -10-15%
- **F750 example**: COP 3.0 → 2.6-2.7 with air heating
- **Power increase**: ~15% higher electrical consumption

## How Model Profiles Are Used

### 1. **Power Validation**
```python
# Example: F750 at -10°C outdoor, 2.5kW electrical consumption
profile = HeatPumpModelRegistry.get_model("nibe_f750")
validation = profile.validate_power_consumption(
    current_power_kw=2.5,
    outdoor_temp=-10.0,
    flow_temp=35.0,
)

# Result: ✓ Normal (expected ~2.3kW for 150m² house at -10°C)
```

### 2. **Optimal Flow Temperature**
```python
# Calculate best flow temp for efficiency
optimal_flow = profile.calculate_optimal_flow_temp(
    outdoor_temp=-10.0,
    indoor_target=21.0,
    heat_demand_kw=7.0,
)

# Result: ~32°C (outdoor -10°C + 27°C efficiency target + UFH adjustment)
```

### 3. **COP Estimation**
```python
# Get expected COP for conditions
cop = profile.get_cop_at_temperature(-10.0)

# F750: ~3.0 COP at -10°C
# S1155: ~4.5 COP at -10°C (GSHP much better!)
```

### 4. **Sizing Validation**
```python
# Check if F750 is correctly sized for 150m² house
validator = HeatPumpSizingValidator()
sizing = validator.validate_sizing(
    heat_pump_profile=profile,
    house_area_m2=150,
    insulation_period="standard",
)

# Result: "optimal" - F750 good match for 150m² standard insulation
```

## Configuration

### In config_flow.py

Users select their model during setup:

```python
CONF_HEAT_PUMP_MODEL = "heat_pump_model"

# In configuration step
vol.Required(CONF_HEAT_PUMP_MODEL): vol.In({
    "nibe_f730": "NIBE F730 (6kW ASHP)",
    "nibe_f750": "NIBE F750 (8kW ASHP)",
    "nibe_f2040": "NIBE F2040 (12-16kW ASHP)",
    "nibe_s1155": "NIBE S1155 (12kW GSHP)",
    "nibe_s1255": "NIBE S1255 (15kW GSHP)",
})
```

### Optional: House Size + Insulation

For even better optimization:

```python
vol.Optional(CONF_HOUSE_AREA_M2): vol.All(
    vol.Coerce(int),
    vol.Range(min=50, max=500)
),
vol.Optional(CONF_INSULATION_PERIOD): vol.In([
    "pre_1990",   # 80-120 W/m²
    "standard",   # 60-80 W/m² (1990-2010)
    "modern",     # 40-60 W/m² (post-2010)
    "passive",    # <30 W/m² (passive house)
]),
```

## Benefits

### 1. **Accurate Power Monitoring**
- Know if 2.5kW is normal or concerning
- Detect auxiliary heating activation
- Identify sizing problems

### 2. **Optimal Efficiency**
- Model-specific flow temperature targets
- COP-aware optimization
- Minimize electrical consumption

### 3. **Better Diagnostics**
- "F750 consuming 4kW at -10°C" → Warning: Check auxiliary
- "S1155 consuming 2kW at -10°C" → Normal for GSHP
- "F730 consuming 3kW at 0°C" → Critical: System problem

### 4. **Smarter Optimization**
- F750 can pre-heat more aggressively (8kW capacity)
- F730 must be more conservative (6kW capacity)
- S-series can run lower flow temps (better COP)

## Future Expansion

### More NIBE Models
- F1145, F1155, F1245, F1255 (F-series)
- S735, S2125 (S-series)
- VVM225, VVM310, VVM320, VVM500 (VVM series)

### Other Brands
With the registry system, easy to add:
- Vaillant (aroTHERM, flexoTHERM)
- Daikin (Altherma series)
- Mitsubishi (Ecodan series)

## Implementation Status

✅ **Model profile architecture** - Complete
✅ **Registry system** - Complete
✅ **NIBE F730/F750/F2040** - Complete
✅ **NIBE S1155/S1255** - Complete
⏳ **Integration into coordinator** - TODO
⏳ **Configuration flow update** - TODO
⏳ **Power monitoring sensor** - TODO

## Next Steps

1. Update `coordinator.py` to load and use model profile
2. Update `config_flow.py` to include model selection
3. Create power validation sensor
4. Add sizing validation warnings
5. Test with real heat pumps

---

**Bottom Line**: The old "3 kW is normal" assumption is gone. Now we have **model-specific, house-size-aware, insulation-aware** power validation and optimization. This will prevent false alarms and enable much better efficiency optimization.
```
