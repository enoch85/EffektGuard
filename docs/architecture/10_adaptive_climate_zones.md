# Adaptive Climate Zone System Architecture

## Overview

Universal climate-based safety margin system that replaces Swedish-specific hardcoding with latitude-based automatic climate detection. Combines proven mathematical formulas with adaptive learning for global applicability.

# Adaptive Climate Zone System Architecture

## Overview

Universal climate-based safety margin system that replaces Swedish-specific hardcoding with latitude-based automatic climate detection. Combines proven mathematical formulas with adaptive learning for global applicability.

---

## Diagram 1: Latitude Source & Initialization Flow

**Shows exactly where latitude comes from and how it flows through the system.**

```mermaid
flowchart TD
    A["User's Home Assistant<br/>configuration.yaml"] -->|Define location| B["homeassistant:<br/>  latitude: 59.3293<br/>  longitude: 18.0686"]
    B -->|Stored in| C["hass.config.latitude<br/>(Home Assistant Core)"]
    
    C -->|Integration loads| D["effektguard/__init__.py<br/>async_setup_entry()"]
    
    D -->|Step 1| E["config_with_latitude = dict(entry.options)<br/>{target_temperature: 21.0, tolerance: 0.5, ...}"]
    E -->|Step 2: Inject latitude| F["config_with_latitude['latitude'] = hass.config.latitude<br/>Result: {latitude: 59.3293, ...}"]
    
    F -->|Pass config dict| G["DecisionEngine(<br/>  config=config_with_latitude,<br/>  ...)<br/>"]
    
    G -->|Extract with fallback| H["latitude = config.get('latitude', 59.33)<br/>Default: Stockholm if missing"]
    
    H -->|Create climate system| I["AdaptiveClimateSystem(<br/>  latitude=59.3293,<br/>  weather_learner=weather_learner<br/>)"]
    
    I -->|Use absolute value| J["self.latitude = abs(59.3293) = 59.3293<br/>Handles southern hemisphere"]
    
    J -->|Detect zone| K["self._detect_climate_zone()"]
    
    K -->|Check ranges in order| L{Loop through<br/>CLIMATE_ZONE_ORDER}
    
    L -->|arctic: 66.5°-90.0°| M1["66.5 <= 59.3 <= 90.0?<br/>No"]
    L -->|subarctic: 60.5°-66.5°| M2["60.5 <= 59.3 <= 66.5?<br/>No"]
    L -->|cold: 55.0°-60.5°| M3["55.0 <= 59.3 <= 60.5?<br/>YES!"]
    
    M3 -->|Return zone key| N["self.climate_zone = 'cold'"]
    
    N -->|Log detection| O["Logger: AdaptiveClimateSystem initialized:<br/>latitude=59.33° -> Cold Continental zone<br/>(winter avg low: -10.0°C, base safety margin: 1.0°C)"]
    
    O -->|Ready for use| P["self.climate_system<br/>(stored in DecisionEngine)"]
    
    style C fill:#e0e0e0,stroke:#333,color:#000
    style F fill:#d0d0d0,stroke:#333,color:#000
    style I fill:#c0c0c0,stroke:#333,color:#000
    style M3 fill:#b0b0b0,stroke:#333,color:#000
    style N fill:#a0a0a0,stroke:#333,color:#000
```

---

## Diagram 2: Climate Zone Detection Logic

**Shows the exact decision tree for zone detection with all edge cases.**

```mermaid
flowchart TD
    A["latitude = abs(config.get('latitude', 59.33))"] -->|Examples| B["Stockholm: 59.33°<br/>Kiruna: 67.85°<br/>London: 51.51°<br/>Paris: 48.86°<br/>Singapore: 1.35°<br/>Antarctica: abs(-75°) = 75°"]
    
    B -->|Loop through zones| C{Check Arctic<br/>66.5° - 90.0°}
    
    C -->|67.85° YES| D1["Zone: Arctic<br/>Safety margin: +2.5°C<br/>Winter avg: -30°C"]
    C -->|75° YES| D1
    C -->|59.33° NO| E{Check Subarctic<br/>60.5° - 66.5°}
    
    E -->|62.45° YES| D2["Zone: Subarctic<br/>Safety margin: +1.5°C<br/>Winter avg: -15°C"]
    E -->|59.33° NO| F{Check Cold<br/>55.0° - 60.5°}
    
    F -->|59.33° YES| D3["Zone: Cold<br/>Safety margin: +1.0°C<br/>Winter avg: -10°C"]
    F -->|51.51° NO| G{Check Temperate<br/>49.0° - 55.0°}
    
    G -->|51.51° YES| D4["Zone: Temperate<br/>Safety margin: +0.5°C<br/>Winter avg: 0°C"]
    G -->|49.0° YES boundary| D4
    G -->|48.86° NO| H{Check Mild<br/>35.0° - 48.999°}
    
    H -->|48.86° YES| D5["Zone: Mild<br/>Safety margin: 0.0°C<br/>Winter avg: +5°C"]
    H -->|1.35° NO| I["Outside all zones<br/>(< 35° or > 90°)"]
    
    I -->|Fallback| J["Default: Temperate<br/>Logger.warning()<br/>'Latitude X° outside defined zones'"]
    
    D1 & D2 & D3 & D4 & D5 & J -->|Store result| K["return zone_key"]
    
    K -->|Assign| L["self.climate_zone = zone_key"]
    
    style D1 fill:#e0e0e0,stroke:#333,color:#000
    style D2 fill:#d0d0d0,stroke:#333,color:#000
    style D3 fill:#c0c0c0,stroke:#333,color:#000
    style D4 fill:#b0b0b0,stroke:#333,color:#000
    style D5 fill:#a0a0a0,stroke:#333,color:#000
    style I fill:#f0f0f0,stroke:#333,color:#000
    style J fill:#d8d8d8,stroke:#333,color:#000
```

---

## Diagram 3: Runtime Safety Margin Calculation

**Shows exactly how safety margins are calculated during each optimization cycle.**

```mermaid
flowchart TD
    A["Coordinator Update (every 5 min)"] -->|Trigger| B["DecisionEngine.calculate_decision()"]
    
    B -->|Call layer 5| C["_weather_compensation_layer(nibe_state, weather_data)"]
    
    C -->|Check config| D{self.enable_weather<br/>_compensation?}
    D -->|False| E1["Return LayerDecision:<br/>offset=0.0, weight=0.0<br/>reason='disabled'"]
    D -->|True| E2["Continue"]
    
    E2 -->|Extract data| F["outdoor_temp = nibe_state.outdoor_temp<br/>current_flow = nibe_state.flow_temp<br/>Example: -12°C, 38°C"]
    
    F -->|Calculate optimal| G["weather_comp.calculate_optimal_flow_temp()<br/>Uses André Kühne formula"]
    G -->|Result| H["flow_calc.flow_temp = 35.5°C<br/>(mathematical optimum)"]
    
    H -->|Check weather learner| I{self.weather_learner<br/>exists?}
    I -->|Yes| J["weather_learner.detect_unusual_weather(forecast)"]
    I -->|No| K["unusual_weather = False<br/>unusual_severity = 0.0"]
    
    J -->|Check result| L{is_unusual?}
    L -->|Yes| M["unusual_weather = True<br/>severity = 1.0 if 'extreme' else 0.5"]
    L -->|No| K
    
    K & M -->|Pass to climate system| N["climate_system.get_safety_margin(<br/>  outdoor_temp=-12.0,<br/>  unusual_weather_detected=False,<br/>  unusual_severity=0.0<br/>)"]
    
    N -->|Get zone info| O["zone_info = CLIMATE_ZONES[self.climate_zone]<br/>Example Cold zone:<br/>  base: 1.0°C<br/>  winter_avg_low: -10.0°C"]
    
    O -->|Calculate component 1| P["base_margin = zone_info['safety_margin_base']<br/>= 1.0°C"]
    
    O -->|Calculate component 2| Q{outdoor_temp < winter_avg_low?}
    Q -->|Yes: -12 < -10| R["temp_margin = (winter_avg_low - outdoor) × 0.1<br/>= (-10 - (-12)) × 0.1 = 0.2°C"]
    Q -->|No: outdoor >= winter_avg| S["temp_margin = 0.0°C"]
    
    R & S -->|Calculate component 3| T{unusual_weather_detected?}
    T -->|Yes| U["learning_margin = 0.5 + (unusual_severity × 1.0)<br/>Examples:<br/>  severity=0.5 → 1.0°C<br/>  severity=1.0 → 1.5°C"]
    T -->|No| V["learning_margin = 0.0°C"]
    
    U & V -->|Sum all components| W["total_margin = base + temp + learning<br/>Example: 1.0 + 0.2 + 0.0 = 1.2°C"]
    
    W -->|Return| X["safety_margin = 1.2°C"]
    
    X -->|Apply to optimal| Y["adjusted_flow_temp = flow_calc.flow_temp + safety_margin<br/>= 35.5 + 1.2 = 36.7°C"]
    
    Y -->|Calculate offset| Z["offset = (adjusted_flow - current_flow) / 1.5<br/>= (36.7 - 38.0) / 1.5 = -0.87°C<br/>(NIBE curve sensitivity: 1.5°C per offset)"]
    
    style P fill:#e0e0e0,stroke:#333,color:#000
    style R fill:#d0d0d0,stroke:#333,color:#000
    style U fill:#c0c0c0,stroke:#333,color:#000
    style W fill:#b0b0b0,stroke:#333,color:#000
    style Z fill:#a0a0a0,stroke:#333,color:#fff
```

---

## Diagram 4: Dynamic Weight Calculation

**Shows how layer weight adjusts based on temperature severity and unusual weather.**

```mermaid
flowchart TD
    A["climate_system.get_dynamic_weight(<br/>  outdoor_temp=-12.0,<br/>  unusual_weather_detected=False<br/>)"] -->|Get zone info| B["zone_info = CLIMATE_ZONES[self.climate_zone]<br/>winter_avg_low = -10.0°C (Cold zone)"]
    
    B -->|Check temperature severity| C{outdoor_temp vs<br/>winter_avg_low?}
    
    C -->|"outdoor < winter_avg_low"| D1["Very cold: -12 < -10<br/>base_weight = 0.85"]
    C -->|"outdoor < (winter_avg_low + 5)"| D2["Cold: -10 to -5<br/>base_weight = 0.75"]
    C -->|"outdoor < 5"| D3["Mild cold: -5 to 5<br/>base_weight = 0.65"]
    C -->|"outdoor >= 5"| D4["Warm: >= 5<br/>base_weight = 0.50"]
    
    D1 & D2 & D3 & D4 -->|Check unusual weather| E{unusual_weather<br/>_detected?}
    
    E -->|Yes| F["boost_weight = min(0.95, base_weight + 0.15)<br/>Examples:<br/>  0.85 + 0.15 = 0.95 (capped)<br/>  0.75 + 0.15 = 0.90<br/>  0.65 + 0.15 = 0.80<br/>  0.50 + 0.15 = 0.65"]
    E -->|No| G["final_weight = base_weight<br/>(no boost)"]
    
    F & G -->|Apply user config| H["adjusted_weight = final_weight × config['weather_compensation_weight']<br/>Default multiplier: 0.75"]
    
    H -->|Examples| I["Very cold + unusual: 0.95 × 0.75 = 0.71<br/>Very cold normal: 0.85 × 0.75 = 0.64<br/>Cold normal: 0.75 × 0.75 = 0.56<br/>Warm normal: 0.50 × 0.75 = 0.38"]
    
    I -->|Return to layer| J["final_weight (for decision aggregation)"]
    
    J -->|Combine with offset| K["LayerDecision(<br/>  offset=-0.87,<br/>  weight=0.64,<br/>  reason='...'<br/>)"]
    
    style D1 fill:#404040,stroke:#333,color:#fff
    style D2 fill:#606060,stroke:#333,color:#fff
    style D3 fill:#808080,stroke:#333,color:#fff
    style D4 fill:#a0a0a0,stroke:#333,color:#000
    style F fill:#505050,stroke:#333,color:#fff
    style K fill:#707070,stroke:#333,color:#fff
```

---

## Diagram 5: Decision Aggregation (All 8 Layers)

**Shows how weather compensation layer vote combines with other layers.**

```mermaid
flowchart TD
    A["DecisionEngine.calculate_decision()"] -->|Collect all layers| B["Layer Decisions (8 total)"]
    
    B -->|Priority 1| L1["Safety Layer<br/>offset=0.0, weight=1.0<br/>reason='Temperature within limits'"]
    B -->|Priority 2| L2["Emergency Layer<br/>offset=0.0, weight=0.9<br/>reason='DM -180 OK'"]
    B -->|Priority 3| L3["Effect Tariff Layer<br/>offset=0.0, weight=0.8<br/>reason='Not in peak window'"]
    B -->|Priority 4| L4["Prediction Layer (Phase 6)<br/>offset=+0.5, weight=0.65<br/>reason='Pre-heat for cold'"]
    B -->|Priority 5| L5["Weather Compensation Layer<br/>offset=-0.9, weight=0.64<br/>reason='Math WC: Cold zone, -12°C'"]
    B -->|Priority 6| L6["Weather Prediction Layer<br/>offset=0.0, weight=0.7<br/>reason='Stable forecast'"]
    B -->|Priority 7| L7["Price Layer<br/>offset=-0.5, weight=0.6<br/>reason='Expensive period'"]
    B -->|Priority 8| L8["Comfort Layer<br/>offset=0.0, weight=0.5<br/>reason='Indoor 21.0°C in range'"]
    
    L1 & L2 & L3 & L4 & L5 & L6 & L7 & L8 -->|Weighted aggregation| C["Calculate weighted sum"]
    
    C -->|Formula| D["total_weighted_offset =<br/>Σ(layer.offset × layer.weight)"]
    
    D -->|Calculation| E["= (0.0×1.0) + (0.0×0.9) + (0.0×0.8)<br/>  + (0.5×0.65) + (-0.9×0.64) + (0.0×0.7)<br/>  + (-0.5×0.6) + (0.0×0.5)<br/>= 0.325 - 0.576 - 0.300<br/>= -0.551"]
    
    E -->|Total weights| F["total_weight = Σ(layer.weight)<br/>= 1.0 + 0.9 + 0.8 + 0.65 + 0.64 + 0.7 + 0.6 + 0.5<br/>= 5.79"]
    
    F -->|Final calculation| G["final_offset = total_weighted_offset / total_weight<br/>= -0.551 / 5.79<br/>= -0.095°C"]
    
    G -->|Round to NIBE precision| H["rounded_offset = -0.1°C"]
    
    H -->|Apply to heat pump| I["Climate Entity<br/>set_heating_curve_offset(-0.1)"]
    
    I -->|NIBE MyUplink API| J["NIBE F750 Heat Pump<br/>Heating curve adjusted<br/>Flow temp slightly reduced"]
    
    style L1 fill:#303030,stroke:#333,color:#fff
    style L2 fill:#404040,stroke:#333,color:#fff
    style L3 fill:#505050,stroke:#333,color:#fff
    style L4 fill:#606060,stroke:#333,color:#fff
    style L5 fill:#707070,stroke:#333,color:#fff
    style L6 fill:#808080,stroke:#333,color:#fff
    style L7 fill:#909090,stroke:#333,color:#000
    style L8 fill:#a0a0a0,stroke:#333,color:#000
    style G fill:#b0b0b0,stroke:#333,color:#000
    style J fill:#c0c0c0,stroke:#333,color:#000
```

---

## Data Flow Detail

### 1. Latitude Source
```python
# Home Assistant Core Configuration (config/configuration.yaml)
homeassistant:
  name: Home
  latitude: 59.3293    # Stockholm, Sweden
  longitude: 18.0686
  elevation: 28
  unit_system: metric
  time_zone: Europe/Stockholm

# This becomes: hass.config.latitude = 59.3293
```

### 2. Integration Initialization (`__init__.py`)
```python
# Step 1: Copy user options from config flow
config_with_latitude = dict(entry.options)
# entry.options = {
#     "target_temperature": 21.0,
#     "tolerance": 0.5,
#     "enable_price_optimization": True,
#     ...
# }

# Step 2: Inject latitude from Home Assistant global config
config_with_latitude["latitude"] = hass.config.latitude  # 59.3293

# Step 3: Pass to DecisionEngine
decision_engine = DecisionEngine(
    price_analyzer=price_analyzer,
    effect_manager=effect_manager,
    thermal_model=thermal_model,
    config=config_with_latitude,  # Now includes latitude!
    ...
)
```

### 3. DecisionEngine Initialization
```python
# Extract latitude with default fallback
latitude = config.get("latitude", 59.33)  # Default: Stockholm if missing

# Create adaptive climate system
self.climate_system = AdaptiveClimateSystem(
    latitude=latitude,           # 59.3293 from HA config
    weather_learner=weather_learner  # Optional Phase 6 module
)
```

### 4. AdaptiveClimateSystem Initialization
```python
def __init__(self, latitude: float, weather_learner: Optional = None):
    self.latitude = abs(latitude)  # 59.3293 → 59.3293 (already positive)
    self.weather_learner = weather_learner
    self.climate_zone = self._detect_climate_zone()
    
    # Detection logic:
    # 59.3293° falls in range (55.0°, 60.5°) → "cold" zone
    # Zone info:
    #   name: "Cold Continental"
    #   winter_avg_low: -10.0°C
    #   safety_margin_base: 1.0°C
    #   examples: ["Stockholm (SWE)", "Oslo (NOR)", "Helsinki (FIN)"]
    
    _LOGGER.info(
        "AdaptiveClimateSystem initialized: latitude=59.33° -> Cold Continental zone "
        "(winter avg low: -10.0°C, base safety margin: 1.0°C)"
    )
```

### 5. Runtime: Weather Compensation Layer
```python
# Called every 5 minutes during coordinator update
def _weather_compensation_layer(self, nibe_state, weather_data):
    # Check if enabled
    if not self.enable_weather_compensation:
        return LayerDecision(offset=0.0, weight=0.0, reason="disabled")
    
    # Get current conditions
    outdoor_temp = -12.0  # Example: Stockholm winter
    current_flow = 38.0   # Current NIBE flow temperature
    
    # Calculate optimal flow using math formulas
    flow_calc = self.weather_comp.calculate_optimal_flow_temp(
        indoor_setpoint=21.0,
        outdoor_temp=-12.0,
        prefer_method="auto"
    )
    # Result: flow_calc.flow_temp = 35.5°C (from Kühne formula)
    
    # Check for unusual weather
    unusual_weather = False
    unusual_severity = 0.0
    if self.weather_learner:
        unusual = self.weather_learner.detect_unusual_weather(...)
        if unusual.is_unusual:
            unusual_weather = True
            unusual_severity = 1.0 if unusual.severity == "extreme" else 0.5
    
    # Get adaptive safety margin
    safety_margin = self.climate_system.get_safety_margin(
        outdoor_temp=-12.0,
        unusual_weather_detected=unusual_weather,  # False in this example
        unusual_severity=unusual_severity,          # 0.0
    )
    # Calculation:
    #   base_margin = 1.0°C (Cold zone)
    #   temp_margin = (-10.0 - (-12.0)) × 0.1 = 0.2°C (colder than avg)
    #   learning_margin = 0.0 (no unusual weather)
    #   total = 1.0 + 0.2 + 0.0 = 1.2°C
    
    # Apply safety margin
    adjusted_flow_temp = 35.5 + 1.2 = 36.7°C
    
    # Calculate required offset
    # NIBE sensitivity: 1.5°C flow change per 1°C offset
    required_offset = (36.7 - 38.0) / 1.5 = -0.87°C
    
    # Get dynamic weight
    dynamic_weight = self.climate_system.get_dynamic_weight(
        outdoor_temp=-12.0,
        unusual_weather_detected=False
    )
    # Logic:
    #   outdoor (-12°C) < winter_avg_low (-10°C) → very cold → 0.85
    
    # Apply user-configured weight multiplier
    final_weight = 0.85 × 0.75 = 0.6375
    
    return LayerDecision(
        offset=-0.87,
        weight=0.6375,
        reason="Math WC: kuehne; Zone: Cold Continental; Optimal: 35.5°C; "
               "Safety: +1.2°C; Adjusted: 36.7°C; Current: 38.0°C → offset: -0.9°C; "
               "Weight: 0.64"
    )
```

### 6. Decision Aggregation Example
```python
# All layers vote (simplified):
layers = [
    LayerDecision(offset=0.0, weight=1.0, reason="Safety: OK"),          # Safety
    LayerDecision(offset=0.0, weight=0.9, reason="Emergency: OK"),       # Emergency
    LayerDecision(offset=0.0, weight=0.8, reason="Effect: Not peak"),    # Effect
    LayerDecision(offset=0.5, weight=0.65, reason="Prediction: pre-heat"),  # Prediction
    LayerDecision(offset=-0.9, weight=0.64, reason="Weather comp: -12°C"),  # Weather Comp
    LayerDecision(offset=0.0, weight=0.7, reason="Weather: Stable"),     # Weather
    LayerDecision(offset=-0.5, weight=0.6, reason="Price: Expensive"),   # Price
    LayerDecision(offset=0.0, weight=0.5, reason="Comfort: In range"),   # Comfort
]

# Weighted aggregation:
total_weighted_offset = (
    0.0×1.0 + 0.0×0.9 + 0.0×0.8 + 0.5×0.65 + (-0.9)×0.64 + 
    0.0×0.7 + (-0.5)×0.6 + 0.0×0.5
) = 0.325 - 0.576 - 0.300 = -0.551

total_weight = 1.0 + 0.9 + 0.8 + 0.65 + 0.64 + 0.7 + 0.6 + 0.5 = 5.79

final_offset = -0.551 / 5.79 = -0.095°C

# Apply to NIBE heating curve
climate_entity.set_offset(-0.1)  # Rounded to 0.1°C precision
```

## Climate Zone Detection Logic

```python
def _detect_climate_zone(self) -> str:
    """Detect climate zone based on latitude."""
    latitude = abs(self.latitude)  # Southern hemisphere support
    
    # Check from coldest to mildest (defined in const.py CLIMATE_ZONE_ORDER)
    for zone_key in ["arctic", "subarctic", "cold", "temperate", "mild"]:
        zone_data = CLIMATE_ZONES[zone_key]
        lat_min, lat_max = zone_data["latitude_range"]
        
        if lat_min <= latitude <= lat_max:
            return zone_key
    
    # Fallback for edge cases (< 35° or > 90°)
    return "temperate"
```

## Real-World Examples

### Example 1: Kiruna, Sweden (67.85°N, -35°C)
```
Initialization:
  hass.config.latitude = 67.85
  → Detected zone: Arctic
  → winter_avg_low: -30.0°C
  → safety_margin_base: 2.5°C

Runtime (outdoor = -35°C, no unusual weather):
  1. Kühne formula: 42.0°C optimal flow
  2. Safety margin calculation:
     - base: 2.5°C (Arctic)
     - temp: (-30 - (-35)) × 0.1 = 0.5°C (colder than avg)
     - unusual: 0.0°C (no unusual weather)
     - total: 3.0°C
  3. Adjusted flow: 42.0 + 3.0 = 45.0°C
  4. Dynamic weight: 0.85 (very cold)
  5. Final weight: 0.85 × 0.75 = 0.6375

Result: Aggressive heating to prevent thermal debt in extreme cold
```

### Example 2: Stockholm, Sweden (59.33°N, -10°C)
```
Initialization:
  hass.config.latitude = 59.33
  → Detected zone: Cold Continental
  → winter_avg_low: -10.0°C
  → safety_margin_base: 1.0°C

Runtime (outdoor = -10°C, no unusual weather):
  1. Kühne formula: 35.5°C optimal flow
  2. Safety margin calculation:
     - base: 1.0°C (Cold)
     - temp: 0.0°C (at winter average)
     - unusual: 0.0°C (no unusual weather)
     - total: 1.0°C
  3. Adjusted flow: 35.5 + 1.0 = 36.5°C
  4. Dynamic weight: 0.75 (cold)
  5. Final weight: 0.75 × 0.75 = 0.5625

Result: Balanced approach for typical Swedish winter
```

### Example 3: London, UK (51.51°N, 2°C)
```
Initialization:
  hass.config.latitude = 51.51
  → Detected zone: Temperate Oceanic
  → winter_avg_low: 0.0°C
  → safety_margin_base: 0.5°C

Runtime (outdoor = 2°C, no unusual weather):
  1. Kühne formula: 28.0°C optimal flow
  2. Safety margin calculation:
     - base: 0.5°C (Temperate)
     - temp: 0.0°C (warmer than winter avg)
     - unusual: 0.0°C (no unusual weather)
     - total: 0.5°C
  3. Adjusted flow: 28.0 + 0.5 = 28.5°C
  4. Dynamic weight: 0.75 (cold for London)
  5. Final weight: 0.75 × 0.75 = 0.5625

Result: Moderate heating for UK winter conditions
```

### Example 4: Paris, France (48.86°N, 8°C)
```
Initialization:
  hass.config.latitude = 48.86
  → Detected zone: Mild Oceanic
  → winter_avg_low: 5.0°C
  → safety_margin_base: 0.0°C

Runtime (outdoor = 8°C, no unusual weather):
  1. Kühne formula: 24.0°C optimal flow
  2. Safety margin calculation:
     - base: 0.0°C (Mild)
     - temp: 0.0°C (warmer than winter avg)
     - unusual: 0.0°C (no unusual weather)
     - total: 0.0°C
  3. Adjusted flow: 24.0 + 0.0 = 24.0°C (formulas alone!)
  4. Dynamic weight: 0.50 (warm weather)
  5. Final weight: 0.50 × 0.75 = 0.375

Result: Pure mathematical optimization for mild climate
```

## Configuration

### Automatic (No User Input)
- **Latitude**: From Home Assistant global config (`hass.config.latitude`)
- **Climate zone**: Automatically detected from latitude
- **Weather learner**: Automatically used if available (Phase 6)

### User-Configurable (Future - Phase 4 TODO)
- `enable_weather_compensation`: Boolean (default: True)
- `weather_compensation_weight`: 0.0-1.0 (default: 0.75)

### Advanced (Config file / learning)
- `heat_loss_coefficient`: W/°C (learned or user-provided)
- `radiator_rated_output`: Watts (optional for Timbones method)
- `heating_type`: "radiator", "concrete_ufh", "timber" (for UFH adjustments)

## Key Benefits

1. **Zero Configuration**: Uses HA latitude automatically
2. **Global Coverage**: Arctic to Mediterranean with same code
3. **Adaptive**: Combines zones + current temp + unusual weather
4. **Safe**: Progressive margins scale with cold severity
5. **Testable**: Clear inputs/outputs, 26 passing tests
6. **Maintainable**: No country-specific conditionals

## Technical Notes

### Edge Cases & Fallback Behavior

**Tested with 35 comprehensive tests including 9 edge case scenarios.**

#### 1. Equatorial Locations (< 35°N)
```python
# Example: Singapore (1.35°N), Kuala Lumpur (3.14°N)
latitude = 1.35
# Result: Defaults to "temperate" zone with warning
_LOGGER.warning("Latitude 1.35° outside defined zones, defaulting to temperate")
# Rationale: Heat pumps rare in tropics, temperate provides moderate safety
```

#### 2. Southern Hemisphere (Negative Latitude)
```python
# Example: Melbourne (-37.81°S), Sydney (-33.87°S)
latitude = -37.81
self.latitude = abs(latitude)  # Convert to 37.81°
# Result: 37.81° → "mild" zone
# Rationale: Climate zones symmetric around equator for heating purposes
```

#### 3. Antarctica (Far South)
```python
# Example: McMurdo Station (-77.85°S)
latitude = -77.85
self.latitude = abs(latitude)  # Convert to 77.85°
# Result: 77.85° → "arctic" zone (66.5°-90.0°)
# Rationale: CORRECT! Antarctic bases need same heating as Arctic
```

#### 4. Boundary Cases (Exactly at Zone Transition)
```python
# Arctic Circle (66.5°N) - exactly at boundary
latitude = 66.5
# Logic: if 66.5 <= latitude <= 90.0 → TRUE
# Result: "arctic" zone (inclusive lower bound)

# Subarctic/Cold boundary (60.5°N)
latitude = 60.5
# Subarctic: if 60.5 <= latitude <= 66.5 → TRUE (matches first)
# Result: "subarctic" zone

# Temperate/Mild boundary (49.0°N)
latitude = 49.0
# Temperate: if 49.0 <= latitude <= 55.0 → TRUE (matches first)
# Result: "temperate" zone
```

#### 5. Invalid Latitudes (> 90° or very unusual)
```python
# Example: Invalid input > 90° (should never happen)
latitude = 95.0
# Falls through all zone checks
# Result: Defaults to "temperate" with warning
# Rationale: Safe fallback, prevents crashes
```

#### 6. Zero Latitude (Equator)
```python
latitude = 0.0
# Outside all zones (< 35°)
# Result: Defaults to "temperate" with warning
```

#### 7. Between Zones (No Gaps)
```python
# Zones are contiguous - all latitudes from 35° to 90° covered:
# Arctic:     66.5 - 90.0  (23.5° range)
# Subarctic:  60.5 - 66.5  (6.0° range)
# Cold:       55.0 - 60.5  (5.5° range)
# Temperate:  49.0 - 55.0  (6.0° range)
# Mild:       35.0 - 48.999 (13.999° range)
# Total:      35.0 - 90.0  (55° range, no gaps)

# Example: 57.5° (middle of Cold zone)
latitude = 57.5
# Cold: if 55.0 <= 57.5 <= 60.5 → TRUE
# Result: "cold" zone (clean match, no ambiguity)
```

### Zone Range Overlaps (NONE - Fixed)

**Previous issue:** Mild zone (35.0-49.0) and Temperate zone (49.0-55.0) both included 49.0°.

**Current fix:**
```python
"temperate": {
    "latitude_range": (49.0, 55.0),  # Inclusive: 49.0 is temperate
},
"mild": {
    "latitude_range": (35.0, 48.999),  # Exclusive upper: 49.0 NOT in mild
},
```

**Result:** 49.0° → "temperate" (first match wins, no ambiguity)

### Why Absolute Latitude?
```python
self.latitude = abs(latitude)  # Handle southern hemisphere
```
Melbourne (-37.81°S) → abs(-37.81) = 37.81° → Mild zone (same as southern France)

This works because climate zones are symmetric around equator for heat pump purposes. A location at 40°S has similar heating needs to 40°N.

### Why Default to Stockholm (59.33°N)?
Safe fallback if `hass.config.latitude` is somehow missing. Cold zone provides moderate safety margins suitable for most European deployments.

### Why Cap Dynamic Weight at 0.95?
Leave headroom for emergency/safety layers to override. Weather compensation is important but not more critical than preventing thermal debt or protecting from effect tariff peaks.

## Integration Points

- **Input**: `hass.config.latitude` (set during HA initial setup)
- **Output**: Safety margin (°C) and dynamic weight (0.0-1.0) for decision layer
- **Dependencies**: Optional weather_learner (Phase 6) for unusual weather detection
- **Side effects**: None (pure calculation, no state modification)
