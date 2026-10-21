# EffektGuard Test Suite

Organized test structure for comprehensive coverage of the EffektGuard integration.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── requirements.txt         # Test dependencies
│
├── unit/                    # One behaviour per file, named for the behaviour
│   ├── adapters/           # NIBE, GE-Spot and weather adapters
│   ├── climate/            # Climate zones and weather compensation
│   ├── coordinator/        # The control loop
│   ├── dhw/                # Hot-water scheduling
│   ├── effect/             # Effect tariff and peaks
│   ├── learning/           # Self-learning and prediction
│   ├── models/             # Heat pump model profiles
│   ├── optimization/       # Decision engine and layers
│   ├── utils/              # Offset, volatility gate, helpers
│   └── test_*.py           # Behaviours that span areas
│
├── validation/             # The code against its documents, physics and ratchets
│   ├── test_no_hardcoded_values.py   # with hardcoded_values_baseline.json
│   ├── test_climate_zones_doc_matches_the_code.py
│   ├── test_research_docs_still_hold.py
│   └── ...                 # 22 files
│
├── integration/            # Reserved for future integration tests
│
└── [Integration tests at root]     # High-level system tests
    ├── test_config_flow_model_selection.py
    ├── test_config_reload.py
    ├── test_coordinator_learning_initialization.py
    ├── test_entity_comprehensive.py
    ├── test_optional_features.py
    ├── test_regression_imports.py
    ├── test_service_rate_limiting.py
    └── test_services.py

Note: These 8 integration tests remain at /tests root because they test
      multiple components together (config flow, entities, services, etc.).
      They're intentionally separate from unit tests.
```

## Running Tests

### Using the Test Runner Script (Recommended)

```bash
# Run all tests
bash scripts/run_all_tests.sh

# Run specific category
bash scripts/run_all_tests.sh unit
bash scripts/run_all_tests.sh validation
bash scripts/run_all_tests.sh integration

# Run specific component
bash scripts/run_all_tests.sh optimization
bash scripts/run_all_tests.sh climate
bash scripts/run_all_tests.sh learning

# With options
bash scripts/run_all_tests.sh -v unit              # Verbose
bash scripts/run_all_tests.sh -c all               # With coverage
bash scripts/run_all_tests.sh -p all               # Parallel
bash scripts/run_all_tests.sh -x -v optimization   # Fail fast + verbose

# Get help
bash scripts/run_all_tests.sh --help
```

### Using pytest Directly

```bash
# All Tests
```bash
python -m pytest tests/ -v
```

### Specific Category
```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Specific component
python -m pytest tests/unit/optimization/ -v
python -m pytest tests/unit/climate/ -v
python -m pytest tests/unit/dhw/ -v

# Validation tests
python -m pytest tests/validation/ -v
```

### Specific File
```bash
python -m pytest tests/unit/optimization/test_decision_engine_climate_integration.py -v
```

### With Coverage
```bash
python -m pytest tests/ --cov=custom_components/effektguard --cov-report=html
```

### Quick Run (Parallel)
```bash
python -m pytest tests/ -n auto
```

## Test Categories

### Unit Tests (`tests/unit/`, files as of 2026-10-20)

| Folder | Files | Covers |
|---|---|---|
| `optimization/` | 54 | Decision engine, layers, aggregation, damping, savings |
| `coordinator/` | 24 | The control loop: cycle order, failures, peaks, gates |
| `dhw/` | 14 | Hot-water scheduling, safety temperatures, the scheduled window |
| `adapters/` | 13 | NIBE, GE-Spot and weather adapters |
| `learning/` | 7 | Adaptive learning, observations, prediction |
| `effect/` | 6 | Effect tariff, peaks (at most one per day) |
| `utils/` | 5 | The offset integer, the volatility gate, helpers |
| `climate/` | 4 | Climate zones, weather compensation |
| `models/` | 3 | Heat pump profiles (F750, F730, F1155, F2040, S1155) |
| (root) | 20 | Behaviours that span areas |

### Integration Tests (8 files at root)
- Config flow with model selection
- Configuration reload functionality
- Coordinator learning initialization
- Entity comprehensive testing
- Optional features
- Regression import tests
- Service rate limiting
- Service handlers

### Validation Tests (`tests/validation/`, 22 files)
- The hardcoded-value ratchet (`test_no_hardcoded_values.py` + its baseline)
- Documents against the code: climate-zone tables, research docs, the rulebook, quoted safety thresholds
- Physics and datasheets: the emitter law against OpenEnergyMonitor, pump models, the simulated plant
- Hygiene: naive datetimes, translations, the clock at import time

## Test Statistics

The live count: `pytest tests/ --collect-only -q | tail -1` (2833 collected on 2026-10-20: 2832 pass,
and F-132b is a strict xfail).

## Writing New Tests

### Unit Test Template
```python
"""Tests for [component] in EffektGuard."""
import pytest
from custom_components.effektguard.optimization.[module] import [Class]

class Test[Feature]:
    """Test [feature] functionality."""
    
    def test_[scenario](self):
        """Test that [expected behavior]."""
        # Arrange
        instance = [Class](...)
        
        # Act
        result = instance.method(...)
        
        # Assert
        assert result == expected
```

### Integration Test Template
```python
"""Integration tests for [system]."""
import pytest
from homeassistant.core import HomeAssistant

async def test_[integration_scenario](hass: HomeAssistant):
    """Test [system behavior]."""
    # Setup
    await async_setup_component(hass, ...)
    
    # Execute
    await hass.async_block_till_done()
    
    # Verify
    assert hass.states.get(...) is not None
```

## CI/CD Integration

`.github/workflows/validate.yml` runs the suite on pull requests, on pushes to `main` and in the merge
queue, with the other gates (see `docs/dev/TESTING.md`).

## Related Documentation
- `/docs/dev/` - Developer documentation

