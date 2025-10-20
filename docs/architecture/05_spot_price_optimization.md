# Scenario 5: Spot Price Optimization with GE-Spot

**Description**: Base optimization using native 15-minute GE-Spot price classification.

```mermaid
flowchart TD
    subgraph "GE-Spot Data Processing"
        A[96 Native 15-min Periods<br/>Perfect Effektavgift Match]
        B[Today: Q0-Q95<br/>Tomorrow: Q0-Q95 if available]
        C[Parse Quarter Periods<br/>quarter, hour, minute, price, is_daytime]
    end

    subgraph "Price Classification"
        D[Calculate Percentiles<br/>P25, P50, P75, P90]
        E[Price ≤ P25:<br/>CHEAP]
        F[P25 < Price ≤ P75:<br/>NORMAL]
        G[P75 < Price ≤ P90:<br/>EXPENSIVE]
        H[Price > P90:<br/>PEAK]
    end

    subgraph "Current Quarter Analysis"
        I[Current Quarter: Q32<br/>08:00-08:15]
        J[Classification: EXPENSIVE]
        K[Is Daytime: Yes<br/>Q24-Q87 = full weight]
    end

    subgraph "Price Layer Decision"
        L[Base Offsets:<br/>CHEAP: +2.0°C<br/>NORMAL: 0.0°C<br/>EXPENSIVE: -1.0°C<br/>PEAK: -2.0°C]
        M[Daytime Multiplier<br/>Expensive/Peak × 1.5]
        N[User Tolerance Factor<br/>Scale 1-10 → 0.2-2.0]
        O[Final: -1.0 × 1.5 × tolerance<br/>Result: -1.5°C tolerance=1.0]
        P[Price Layer<br/>Offset: -1.5°C<br/>Weight: 0.6]
    end

    subgraph "Extended Optimization"
        Q[Find Next Cheap Period<br/>Search today + tomorrow]
        R[Pre-heat Opportunity<br/>Q45 11:15 = CHEAP]
        S[Plan Pre-heat Strategy<br/>3 hours ahead]
    end

    %% Flow
    A --> B
    B --> C
    C --> D
    D --> E
    D --> F  
    D --> G
    D --> H
    
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    
    H --> Q
    Q --> R
    R --> S

    %% Styling
    style A fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    style D fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    style I fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    
    style E fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style F fill:#999,stroke:#fff,stroke-width:1px,color:#fff
    style G fill:#bbb,stroke:#000,stroke-width:1px,color:#000
    style H fill:#ddd,stroke:#000,stroke-width:1px,color:#000
    
    style P fill:#666,stroke:#fff,stroke-width:2px,color:#fff
```

## GE-Spot Native Integration

### Perfect Quarter Alignment

GE-Spot provides **exactly 96 quarterly periods per day**, eliminating the need for:
- Hourly price interpolation
- Complex time zone calculations  
- Approximation errors

Each quarter maps directly to Swedish Effektavgift measurement windows:
- **Q0-Q23**: 00:00-06:00 (Night, 50% effect weight)
- **Q24-Q87**: 06:00-22:00 (Day, full effect weight)
- **Q88-Q95**: 22:00-00:00 (Night, 50% effect weight)

### Dynamic Price Classification

Rather than fixed price thresholds, the system uses **percentile-based classification**:

#### Daily Price Distribution Analysis
1. **Collect all 96 prices** for the day
2. **Calculate percentiles**: P25, P50, P75, P90
3. **Classify each quarter** relative to the day's distribution

#### Classification Bands
- **CHEAP (≤P25)**: Bottom 25% of prices - pre-heating opportunity
- **NORMAL (P25-P75)**: Middle 50% of prices - maintain status quo
- **EXPENSIVE (P75-P90)**: Top 25-10% of prices - reduce consumption
- **PEAK (>P90)**: Top 10% of prices - minimize consumption

This approach automatically adapts to:
- Seasonal price variations
- Market volatility
- Regional price differences
- Supply/demand fluctuations

### Base Offset Strategy

Each classification has a **base offset value**:

```
CHEAP: +2.0°C    # Pre-heat opportunity
NORMAL: 0.0°C    # Neutral operation
EXPENSIVE: -1.0°C # Reduce consumption  
PEAK: -2.0°C     # Minimize consumption
```

### Day/Night Multiplier

During **daytime hours (Q24-Q87)**, expensive and peak periods get amplified:
- **Expensive**: -1.0°C × 1.5 = -1.5°C
- **Peak**: -2.0°C × 1.5 = -3.0°C

**Rationale**: Daytime consumption has full effect tariff weight, making reductions more valuable.

### User Tolerance Scaling

The **tolerance setting (1-10 scale)** allows users to control optimization aggressiveness:

```
tolerance_factor = user_tolerance / 5.0  # Range: 0.2-2.0

final_offset = base_offset × daytime_multiplier × tolerance_factor
```

#### Tolerance Examples
- **Tolerance 1**: Very conservative (0.2× scaling)
- **Tolerance 5**: Balanced (1.0× scaling)  
- **Tolerance 10**: Aggressive (2.0× scaling)

### Extended Optimization Horizon

When **tomorrow's prices are available**, the system can:

1. **Find next cheap period** across today + tomorrow
2. **Plan extended pre-heating** before expensive periods
3. **Optimize 48-hour strategy** rather than just current quarter

### Example Price Day Analysis

**Sample Day Prices** (SEK/kWh):
- **Minimum**: 0.85 (night)
- **P25**: 1.20 (cheap threshold)
- **P50**: 1.45 (median)
- **P75**: 1.75 (expensive threshold)  
- **P90**: 2.10 (peak threshold)
- **Maximum**: 2.45 (evening peak)

**Q32 (08:00-08:15)**: Price 1.80 SEK/kWh
- **Classification**: EXPENSIVE (1.80 > 1.75)
- **Is Daytime**: Yes (Q32 in Q24-Q87 range)
- **Base Offset**: -1.0°C
- **Daytime Multiplier**: 1.5×
- **Tolerance Factor**: 1.0 (user setting 5)
- **Final Offset**: -1.0 × 1.5 × 1.0 = **-1.5°C**

### Integration with Other Layers

The price layer (weight 0.6) coordinates with:

- **Effect Layer (1.0)**: Don't pre-heat during cheap periods if it creates peaks
- **Weather Layer (0.7)**: Prioritize cheap periods for weather-based pre-heating  
- **Comfort Layer (0.2-0.5)**: Balance cost savings with temperature maintenance

This ensures cost optimization never compromises safety or creates unacceptable comfort deviations.