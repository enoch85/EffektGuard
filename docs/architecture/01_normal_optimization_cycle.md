# Scenario 1: Normal Optimization Cycle

**Description**: Standard 5-minute optimization cycle with all systems operating normally.

```mermaid
flowchart TD
    subgraph "HA Core"
        A[NIBE MyUplink Entities]
        B[Spot Price Entities] 
        C[Weather Entities]
        D[HA Services]
    end

    subgraph "Coordinator Update Cycle (5min)"
        E[EffektGuardCoordinator._async_update_data]
        F[Gather NIBE Data]
        G[Gather Spot Price Prices]
        H[Gather Weather Forecast]
        I[Run Decision Engine]
        J[Update Peak Tracking]
        K[Record Learning Data]
        L[Update All Entities]
    end

    subgraph "Decision Engine Layers"
        M[1. Safety Layer]
        N[2. Emergency Layer]
        O[3. Effect Layer]
        P[4. Prediction Layer]
        Q[5. Weather Layer]
        R[6. Price Layer]
        S[7. Comfort Layer]
        T[Aggregate Weighted Votes]
    end

    subgraph "Data Adapters"
        U[NibeAdapter]
        V[GESpotAdapter]
        W[WeatherAdapter]
    end

    subgraph "Entities Update"
        X[Climate Entity]
        Y[Sensor Entities]
        Z[Number Entities]
    end

    %% Data Flow
    A --> U
    B --> V
    C --> W
    
    E --> F
    F --> U
    U --> F
    F --> G
    G --> V
    V --> G
    G --> H
    H --> W
    W --> H
    
    H --> I
    I --> M
    M --> N
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> I
    
    I --> J
    J --> K
    K --> L
    L --> X
    L --> Y
    L --> Z
    
    %% Apply Decision
    I -.->|Apply Offset| D
    D -.->|number.set_value| A

    %% Styling
    style E fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    style I fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    style T fill:#333,stroke:#fff,stroke-width:2px,color:#fff
    
    style M fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style N fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style O fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style P fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style Q fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style R fill:#666,stroke:#fff,stroke-width:1px,color:#fff
    style S fill:#666,stroke:#fff,stroke-width:1px,color:#fff
```

## Flow Description

### 1. Data Collection (Every 5 minutes)
The coordinator orchestrates data gathering from three main sources:
- **NIBE MyUplink**: Heat pump state, temperatures, degree minutes
- **Spot Price**: Native 15-minute electricity prices (96 quarters/day)
- **Weather**: Forecast data for predictive optimization

### 2. Decision Engine Execution
The decision engine processes data through 7 prioritized layers:
1. **Safety Layer**: Hard temperature limits (18-24°C)
2. **Emergency Layer**: Context-aware thermal debt prevention
3. **Effect Layer**: 15-minute peak protection
4. **Prediction Layer**: Learned pre-heating (Phase 6)
5. **Weather Layer**: Weather-based pre-heating
6. **Price Layer**: Spot price optimization
7. **Comfort Layer**: Temperature error correction

### 3. Decision Aggregation
- Critical layers (weight 1.0) override all others
- Non-critical layers use weighted averaging
- Final offset applied via NIBE MyUplink entity

### 4. State Updates
All entities are updated with:
- Current optimization decision and reasoning
- Peak tracking information
- Learning data for Phase 6 capabilities
- Diagnostic information for monitoring

### 5. Learning Integration (Phase 6)
- Records observations for adaptive learning
- Updates thermal model parameters
- Saves learned data periodically
- Enhances future predictions

This cycle ensures continuous optimization while maintaining safety and providing full visibility into the decision-making process.