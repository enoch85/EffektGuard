# Developer Documentation

Complete development documentation for EffektGuard contributors.

## Quick Links

- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Set up your development environment
- **[TESTING.md](TESTING.md)** - Running and writing tests
- **[CODE_STANDARDS.md](CODE_STANDARDS.md)** - Code style and best practices
- **[CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md)** - How to contribute

## Getting Started

New to EffektGuard development? Follow these steps:

1. **Environment Setup** → [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
   - Clone repository
   - Activate Python 3.13 virtual environment
   - Verify installation with tests

2. **Understand Architecture** → `../../architecture/`
   - Read `00_overview.md` for system overview
   - Review layer priority system
   - Understand climate-aware safety

3. **Learn Code Standards** → [CODE_STANDARDS.md](CODE_STANDARDS.md)
   - Black formatting (line length 100)
   - Type hints and docstrings
   - NIBE-specific knowledge
   - Configuration constants

4. **Write Tests** → [TESTING.md](TESTING.md)
   - Unit tests for pure logic
   - Integration tests for HA components
   - Safety-critical code requires 100% coverage

5. **Contribute** → [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md)
   - Create feature branch
   - Follow commit guidelines
   - Submit pull request

## Core Principles

This is **production code for real homes**. We prioritize:

1. **Safety First** - Heat pump health over cost savings
2. **Configuration-Driven** - No hardcoded values
3. **Read Before Editing** - Understand full context
4. **Test Everything** - Especially safety-critical code
5. **Document Decisions** - Reference research for NIBE behavior

## Project Structure

```
custom_components/effektguard/
├── __init__.py              # Integration setup
├── manifest.json            # HA integration metadata
├── const.py                 # All constants and thresholds
├── coordinator.py           # Data update coordination
├── climate.py               # Main climate entity
├── config_flow.py           # User configuration flow
│
├── optimization/            # Pure Python optimization logic
│   ├── decision_engine.py   # Multi-layer decision making
│   ├── thermal_model.py     # Thermal calculations
│   ├── effect_manager.py    # Peak power tracking
│   ├── price_analyzer.py    # Price classification
│   └── climate_zones.py     # Climate-aware thresholds
│
├── adapters/                # External data interfaces
│   ├── nibe_adapter.py      # NIBE MyUplink reader
│   ├── gespot_adapter.py    # Spot Price price reader
│   └── weather_adapter.py   # Weather forecast reader
│
└── utils/                   # Cross-cutting utilities
    ├── validators.py        # Configuration validation
    └── safety.py            # Thermal debt tracking
```

## Key Concepts

### Climate-Aware Safety

EffektGuard adapts safety thresholds based on climate zone and outdoor temperature:

```python
from .optimization.climate_zones import ClimateZoneDetector

climate_detector = ClimateZoneDetector(latitude=59.33)
dm_range = climate_detector.get_expected_dm_range(outdoor_temp=-10.0)

# Stockholm at -10°C: warning=-700, critical=-1500
# Kiruna at -30°C: warning=-1200, critical=-1500
```

### NIBE Degree Minutes (DM)

Critical safety metric tracking thermal balance:
- **Standard start**: DM -60
- **Extended runs**: DM -240 (acceptable)
- **Climate-aware warning**: Varies by zone (e.g., -700 for Stockholm at -10°C)
- **Absolute critical**: DM -1500 (never exceed)

### Multi-Layer Decision Engine

Optimization uses weighted voting across layers:
1. Emergency Recovery (blocks all else)
2. Effect Tariff Protection
3. Weather Pre-heating
4. Spot Price Optimization
5. Normal Operation

## Common Tasks

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=custom_components/effektguard --cov-report=html

# Specific test
pytest tests/unit/test_thermal_debt.py::test_dm_500_emergency -v
```

### Formatting Code

```bash
# Format all code
black custom_components/effektguard/ --line-length 100

# Check formatting
black custom_components/effektguard/ --check --line-length 100
```

### Debugging

```bash
# Check logs for decision reasoning
grep "Decision:" logs/effektguard.log

# Verify entity readings
grep "NIBE state:" logs/effektguard.log

# Enable debug logging
# In configuration.yaml:
# logger:
#   logs:
#     custom_components.effektguard: debug
```

### Adding Constants

1. Add to `custom_components/effektguard/const.py` with research reference
2. Update relevant logic to use constant
3. Add tests for new threshold/value
4. Document in docstrings

## Research Documentation

NIBE-specific behavior is based on research:
- `IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md` - F2040 real-world cases
- `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md` - F750 optimization
- `docs/CLIMATE_ZONES.md` - Climate-aware safety system

**Never guess NIBE behavior** - always reference research!

## Testing Requirements

- **Unit tests**: Pure logic, fast execution
- **Integration tests**: HA components, mock entities
- **Safety tests**: 100% coverage for safety-critical code
- **All tests must pass** before merge

## Code Review Checklist

Before submitting PR:
- [ ] Black formatted (`black . --check --line-length 100`)
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No hardcoded values (use `const.py`)
- [ ] Type hints on all functions
- [ ] Docstrings with research references
- [ ] Safety implications documented
- [ ] Test coverage for new code

## Getting Help

- Read relevant architecture documents
- Check existing research documentation
- Open GitHub issue with specific questions
- Reference NIBE manuals/forums when applicable

## Additional Resources

### User Documentation
- `../../README.md` - User installation and setup
- `../../docs/CLIMATE_ZONES.md` - Climate zone system
- `../../docs/DHW_OPTIMIZATION.md` - DHW scheduling

### Architecture Documentation
- `../../architecture/00_overview.md` - System overview
- `../../architecture/08_layer_priority_system.md` - Decision engine
- `../../architecture/10_adaptive_climate_zones.md` - Climate adaptation

### Research Documentation
- `../../IMPLEMENTATION_PLAN/02_Research/` - NIBE research findings
- `../../IMPLEMENTATION_PLAN/01_Algorithm/` - Algorithm specifications
- `../../IMPLEMENTATION_PLAN/03_API/` - MyUplink API guide

---

**Remember**: This code controls heating in real homes. Quality, safety, and correctness over speed!
