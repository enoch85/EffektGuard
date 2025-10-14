# Scenario 2: Emergency Thermal Debt Response

**Description**: Context-aware emergency response when degree minutes approach critical thresholds.

```mermaid
flowchart TD
    subgraph "NIBE State Reading"
        A[Read Degree Minutes]
        B[Read Outdoor Temperature]
        C[Read Indoor Temperature]
    end

    subgraph "Context-Aware Analysis"
        D[Calculate Expected DM Range]
        E[Outdoor > 0°C:<br/>Expected: -600 DM<br/>Warning: -800 DM]
        F[Outdoor -10 to 0°C:<br/>Expected: -800 DM<br/>Warning: -1000 DM]
        G[Outdoor < -20°C:<br/>Expected: -1150 DM<br/>Warning: -1350 DM]
        H[Distance from Absolute Max<br/>-1500 DM NEVER EXCEED]
    end

    subgraph "Emergency Layer Decision"
        I{DM ≤ -1500?}
        J[ABSOLUTE MAX<br/>Offset: +5.0°C<br/>Weight: 1.0]
        
        K{Margin < 300 DM?}
        L[CRITICAL<br/>Offset: +3.0°C<br/>Weight: 1.0]
        
        M{DM < Expected Warning?}
        N[WARNING<br/>Offset: +1.0 to +2.0°C<br/>Weight: 0.8]
        
        O{DM < Expected Normal?}
        P[CAUTION<br/>Offset: +0.5°C<br/>Weight: 0.5]
        
        Q[THERMAL DEBT OK<br/>Offset: 0.0°C<br/>Weight: 0.0]
    end

    subgraph "Decision Examples"
        R[Malmö 0°C: DM -900<br/>→ WARNING: Beyond expected -800]
        S[Stockholm -5°C: DM -900<br/>→ CAUTION: Near expected -800]  
        T[Kiruna -25°C: DM -900<br/>→ OK: Well within expected -1150]
    end

    %% Flow
    A --> D
    B --> D
    C --> D
    D --> E
    D --> F
    D --> G
    D --> H
    
    E --> I
    F --> I
    G --> I
    H --> I
    
    I -->|Yes| J
    I -->|No| K
    K -->|Yes| L
    K -->|No| M
    M -->|Yes| N
    M -->|No| O
    O -->|Yes| P
    O -->|No| Q

    %% Examples
    E -.-> R
    F -.-> S
    G -.-> T

    %% Styling
    style I fill:#000,stroke:#fff,stroke-width:2px,color:#fff
    style K fill:#000,stroke:#fff,stroke-width:2px,color:#fff
    style M fill:#000,stroke:#fff,stroke-width:2px,color:#fff
    style O fill:#000,stroke:#fff,stroke-width:2px,color:#fff
    
    style J fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    style L fill:#666,stroke:#fff,stroke-width:2px,color:#fff
    style N fill:#999,stroke:#fff,stroke-width:2px,color:#fff
    style P fill:#bbb,stroke:#000,stroke-width:2px,color:#000
    style Q fill:#ddd,stroke:#000,stroke-width:2px,color:#000
```

## Context-Aware Safety Philosophy

### Smart Adaptation vs Fixed Thresholds

Instead of using fixed degree minutes thresholds, EffektGuard employs **context-aware analysis** that understands what's "normal" for the current outdoor temperature:

- **At -30°C in Kiruna**: DM -1000 might be completely normal
- **At 0°C in Malmö**: DM -1000 indicates a serious problem

This approach automatically adapts to ANY Swedish climate without complex configuration.

### Temperature-Based Expected Ranges

The system calculates expected DM ranges based on outdoor temperature:

#### Mild Weather (> 0°C)
- **Expected normal**: -600 DM
- **Warning threshold**: -800 DM
- **Rationale**: Light heat demand, DM should stay shallow

#### Moderate Cold (0°C to -10°C)
- **Expected normal**: -800 DM
- **Warning threshold**: -1000 DM
- **Rationale**: Standard Swedish winter conditions

#### Cold Weather (-10°C to -20°C)
- **Expected normal**: -800 to -1000 DM (scales with temperature)
- **Warning threshold**: -1200 DM
- **Rationale**: Heavy heat demand, deeper DM expected

#### Extreme Cold (< -20°C)
- **Expected normal**: -1000 to -1150 DM
- **Warning threshold**: -1350 DM
- **Rationale**: Very heavy demand, very deep DM is normal

### Absolute Safety Limit

**DM -1500 is NEVER exceeded** regardless of outdoor temperature. This is the hard safety limit validated by Swedish NIBE forums and represents the point where heat pump damage becomes likely.

### Graduated Response System

The emergency layer provides graduated responses:

1. **ABSOLUTE MAX** (-1500 DM): Maximum emergency recovery (+5.0°C)
2. **CRITICAL** (within 300 DM of limit): Strong recovery (+3.0°C)
3. **WARNING** (beyond expected range): Moderate recovery (+1.0 to +2.0°C)
4. **CAUTION** (approaching expected limit): Gentle correction (+0.5°C)
5. **OK** (within normal range): No intervention (0.0°C)

This ensures the system responds appropriately to the severity of the thermal debt situation while accounting for normal operational variations in different climates.