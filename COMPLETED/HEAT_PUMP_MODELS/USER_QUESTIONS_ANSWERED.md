```markdown
# EffektGuard User Questions - Comprehensive Answers

**Date**: October 15, 2025  
**Status**: Complete Analysis

---

## Question 1: End User Temperature Control - How It Works

### **Short Answer**

✅ **The user sets a target temperature (e.g., 21°C)**, and EffektGuard automatically optimizes the NIBE heating curve offset to maintain that temperature while minimizing cost and protecting heat pump health.

❌ **No "baseline" configuration needed** - EffektGuard will automatically find and apply optimal offsets regardless of current NIBE settings.

### **Detailed Explanation**

#### **User Configuration Flow**:

1. **User Sets Target**: Via climate entity or configuration
   ```yaml
   # User simply sets desired indoor temperature
   Target Temperature: 21.0°C
   Tolerance: ±0.5°C (comfort range: 20.5-21.5°C)
   ```

2. **EffektGuard Reads Current State**:
   - Indoor temperature (from NIBE BT50 sensor): e.g., 20.8°C
   - Outdoor temperature (from NIBE BT1 sensor): e.g., -5°C
   - Current curve offset (from NIBE): e.g., -2.0°C (whatever user had set)
   - Degree Minutes (thermal debt): e.g., -120

3. **EffektGuard Calculates Optimal Offset**:
   - Multi-layer decision engine votes on best offset
   - Considers: current temp error, outdoor temp, prices, weather forecast, thermal debt
   - **Calculates absolute optimal offset** (not relative to current)
   - Example output: "Optimal offset = +0.5°C"

4. **EffektGuard Applies Offset**:
   - Sets NIBE heating curve offset to calculated value
   - Completely **replaces** whatever offset was there before
   - Updates every 5 minutes based on conditions

#### **Example Scenario**:

**Initial Situation** (user has wrong NIBE settings):
- User's manual NIBE offset: -2.0°C (too low, house cold)
- Indoor temp: 19.5°C (target is 21.0°C, deficit of 1.5°C!)
- EffektGuard first measurement

**What EffektGuard Does**:

**Minute 0** (first run):
```python
# Decision engine calculates
indoor_deficit = 21.0 - 19.5 = 1.5°C  # Too cold!

# Comfort layer votes: "Need +3.0°C offset to warm up"
# Safety layer votes: "Allow it, not dangerous"
# Price layer votes: "+1.0°C (cheap period, OK to heat)"

# Aggregated decision: +2.5°C offset

# EffektGuard sets NIBE offset to +2.5°C
# (Completely ignores the old -2.0°C, replaces it)
```

**Minute 15** (house warming up):
```python
# New measurement
indoor_temp = 20.2°C  # Getting closer
indoor_deficit = 21.0 - 20.2 = 0.8°C  # Still cold but improving

# Comfort layer votes: "+1.5°C offset to continue warming"
# Aggregated decision: +1.8°C

# EffektGuard sets NIBE offset to +1.8°C
```

**Minute 30** (target reached):
```python
# New measurement
indoor_temp = 21.0°C  # Perfect!
indoor_deficit = 0.0°C

# Comfort layer votes: "0.0°C, maintain current"
# Price layer votes: "-0.5°C (expensive period, reduce slightly)"
# Aggregated decision: -0.3°C

# EffektGuard sets NIBE offset to -0.3°C
# House will maintain 21°C with minimal heating during expensive period
```

### **Key Points**:

✅ **Automatic Correction**: If user has completely wrong NIBE offsets, EffektGuard will automatically correct them within ~30-60 minutes

✅ **No Manual Intervention**: User doesn't need to touch NIBE settings. Just set target temp in EffektGuard.

✅ **Continuous Optimization**: Every 5 minutes, recalculates optimal offset based on:
- Current indoor temperature vs target
- Outdoor temperature
- Electricity prices (15-minute periods)
- Weather forecast
- Thermal debt (Degree Minutes)
- Monthly peak tracking (Effektavgift)

✅ **Tolerance Handling**: 
- **Comfort preset**: Keeps indoor very close to target (±0.3°C), higher cost
- **Balanced preset**: Allows ±0.5°C deviation, moderate cost
- **Eco preset**: Allows ±1.0°C deviation, maximum savings

### **Configuration Example**:

```yaml
# climate.yaml
climate:
  - platform: effektguard
    target_temperature: 21.0  # User's desired temp
    tolerance: 5  # Scale 1-10 (5 = balanced)
    optimization_mode: balanced
```

**User sets once, EffektGuard handles everything else.**

---

## Question 2: DHW (Domestic Hot Water) vs Space Heating Priority

### **Short Answer**

🏠 **Space heating (comfort heat) has HIGHER priority than water heating.**

The integration will **block or abort DHW cycles** if they threaten indoor comfort or cause thermal debt.

### **Priority Order** (Highest to Lowest)

Based on `POST_PHASE_5_ROADMAP.md` Phase 8 specification:

1. **🚨 CRITICAL: Space Heating Comfort**
   - Indoor temperature > target - 0.5°C
   - **DHW blocked** if indoor temp drops below 20.5°C (when target is 21°C)

2. **🛡️ SAFETY: DHW Minimum Temperature**
   - DHW ≥ 35°C for health/safety (Legionella risk)
   - **Only heats DHW to safety minimum** if space heating needs are high

3. **⚠️ WARNING: Thermal Debt Prevention**
   - Degree Minutes (DM) > -400
   - **DHW completely blocked** if DM < -240
   - **DHW aborted mid-cycle** if DM drops below -400

4. **🏠 NORMAL: Space Heating Target**
   - Maintain ±0.3°C comfort
   - **DHW delayed** if space heating demand > 6kW

5. **💧 COMFORT: DHW Target Temperature**
   - Normal DHW comfort (50°C)
   - **Only heats** when space heating is satisfied

6. **🦠 HYGIENE: Legionella Prevention**
   - Weekly 65°C boost
   - **Only runs** when indoor temp is comfortable

### **Detailed DHW Blocking Rules**

#### **Rule 1: Critical Thermal Debt** ⛔
```python
if degree_minutes <= -240:
    # NEVER start DHW - thermal debt too high
    return DHWScheduleDecision(should_heat=False, reason="CRITICAL_THERMAL_DEBT")
```

**Effect**: DHW completely disabled until space heating recovers

#### **Rule 2: Space Heating Emergency** ⛔
```python
indoor_deficit = target_temp - indoor_temp
if indoor_deficit > 0.5 and outdoor_temp < 0:
    # Indoor temp 0.5°C below target + it's cold outside
    return DHWScheduleDecision(should_heat=False, reason="SPACE_HEATING_EMERGENCY")
```

**Effect**: DHW blocked when house is too cold in winter

#### **Rule 3: DHW Safety Minimum** ✅ (Limited)
```python
if dhw_temp < 35.0:  # Health/safety critical
    return DHWScheduleDecision(
        should_heat=True,
        target_temp=50.0,  # Normal target, not just minimum
        max_runtime=30,  # LIMITED to 30 minutes
        abort_conditions=[
            "thermal_debt < -400",  # Abort if DM gets critical
            "indoor_temp < target - 0.5",  # Abort if house gets cold
        ]
    )
```

**Effect**: DHW can heat to safety minimum, but with strict limits and abort conditions

#### **Rule 4: High Space Heating Demand** ⛔
```python
if space_heating_demand > 6.0 and thermal_debt < -60:
    # High heat demand (>6kW) + already some thermal debt
    return DHWScheduleDecision(should_heat=False, reason="HIGH_SPACE_HEATING_DEMAND")
```

**Effect**: DHW delayed during high space heating load

#### **Rule 5: Normal DHW Heating** ✅ (Conditional)
```python
if dhw_temp < 45.0:  # 5°C below target
    if indoor_deficit < 0.3 and thermal_debt > -100:
        # Only if:
        # - Indoor temp comfortable (within 0.3°C)
        # - Thermal debt OK (> -100)
        return DHWScheduleDecision(
            should_heat=True,
            target_temp=50.0,
            max_runtime=30,
            abort_conditions=[
                "thermal_debt < -400",
                "indoor_temp < target - 0.5",
            ]
        )
```

**Effect**: DHW heats normally only when space heating is satisfied

### **Real-World Example**

**Scenario**: Cold winter evening, -10°C outside

**17:00** - Family gets home, indoor temp 20.2°C (target 21.0°C), DHW temp 48°C

```python
# Decision: Block DHW, prioritize space heating
indoor_deficit = 21.0 - 20.2 = 0.8°C  # Too cold!
outdoor_temp = -10.0  # Cold outside

# Rule 2 triggers: Space heating emergency
# DHW heating: BLOCKED
# Space heating offset: +2.0°C to warm house

_LOGGER.info("DHW blocked: Space heating priority (indoor 0.8°C below target)")
```

**18:00** - House warmed up, indoor 21.0°C, DHW temp 45°C, DM -80

```python
# Decision: Allow DHW heating
indoor_deficit = 0.0°C  # Perfect!
thermal_debt = -80  # Acceptable

# Rule 5 triggers: Normal DHW heating
# DHW heating: ALLOWED (target 50°C, max 30 min)
# Space heating offset: Maintain 0.0°C

_LOGGER.info("DHW heating allowed: Space heating satisfied, DM OK")
```

**18:20** - DHW heating active, DM drops to -380 (approaching danger)

```python
# Decision: Continue DHW but watch carefully
thermal_debt = -380  # Approaching -400 abort threshold

# DHW continues but will abort at -400
_LOGGER.warning("DHW continuing, thermal debt -380, will abort at -400")
```

**18:25** - DM hits -400

```python
# Decision: ABORT DHW immediately
thermal_debt = -400  # Abort threshold!

# Abort condition triggered!
# DHW heating: ABORTED
# Space heating offset: +3.0°C (emergency recovery)

_LOGGER.error("DHW ABORTED: Thermal debt critical, emergency space heating boost")
```

### **Summary Table**

| Condition | Space Heating | DHW Heating | Winner |
|-----------|---------------|-------------|---------|
| Indoor 0.8°C below target, -10°C outside | Priority | Blocked | 🏠 **Space** |
| Indoor OK, DHW 48°C, DM -80 | Maintain | Allowed | 💧 DHW allowed |
| Indoor OK, DHW active, DM -400 | Emergency boost | **ABORTED** | 🏠 **Space** |
| Indoor OK, DHW 32°C (safety!) | Maintain | Limited (30 min) | 💧 DHW (safety) |
| Indoor OK, DHW 48°C, cheap electricity | Maintain | Opportunistic | 💧 DHW (cheap) |

### **Key Insight**

**Space heating ALWAYS wins** when there's a conflict. DHW is nice-to-have, but **indoor comfort and heat pump health come first**.

This prevents the classic NIBE problem: "DHW cycle caused thermal debt → house got cold → massive power spike → £500 electricity bill".

---

## Question 3: Model-Specific Configuration

### **Status**

✅ **Model profiles created** (F730, F750, F2040, S1155, S1255)  
⏳ **Integration into config flow** - TODO (next step)  
⏳ **Power validation** - TODO

### **What's Ready**

5 NIBE models fully profiled in `/custom_components/effektguard/models/nibe/`:
- **F730**: 6kW ASHP, 1.0-2.0kW electrical, 80-120m² houses
- **F750**: 8kW ASHP, 1.2-2.8kW electrical, 100-150m² houses ⭐ Most common
- **F2040**: 12-16kW ASHP, 2.5-6.5kW electrical, 180-250m² houses
- **S1155**: 12kW GSHP, 1.5-3.5kW electrical (excellent efficiency!)
- **S1255**: 15kW GSHP, 2.0-4.5kW electrical

### **What Each Profile Contains**

- **Power characteristics**: Rated power, electrical consumption ranges
- **COP curves**: Temperature-dependent efficiency (2.0-5.5 COP)
- **Optimal flow temps**: Model-specific efficiency targets
- **Power validation**: "Is 2.5kW normal for F750 at -10°C?" → Yes/No + suggestions
- **Sizing validation**: "Is F750 right for 150m² house?" → Optimal/Oversized/Undersized

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for complete details.

### **Next Steps**

1. Update `config_flow.py` to add model selection dropdown
2. Update `coordinator.py` to load and use model profile
3. Create power validation sensor entity
4. Add sizing warnings during setup
5. Test with real heat pumps

---

## Bottom Line

1. **✅ Self-learning**: Implemented (Phase 6)
2. **✅ Model-specific profiles**: Created for 5 NIBE models, registry system ready
3. **✅ DHW vs space heating priority**: Space heating wins, DHW blocked if threatens comfort/thermal debt
4. **✅ User temperature control**: Set target once, EffektGuard handles everything, automatic correction of wrong NIBE offsets

**No "baseline" needed - EffektGuard calculates absolute optimal offsets regardless of current NIBE settings.**
```
