# EffektGuard Test Suite

Organized test structure for comprehensive coverage of the EffektGuard integration.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── requirements.txt         # Test dependencies
│
├── unit/                    # Unit tests for individual components
│   ├── adapters/           # NIBE adapter tests
│   ├── climate/            # Climate zone and weather tests
│   ├── coordinator/        # Coordinator logic tests
│   ├── dhw/                # DHW optimization tests
│   ├── effect/             # Effect tariff and peak tests
│   ├── learning/           # Self-learning and prediction tests
│   ├── models/             # Heat pump model tests
│   └── optimization/       # Decision engine and optimization tests
│
├── validation/             # Validation and regression tests
│   ├── test_no_hardcoded_values.py
│   ├── test_phase1_fixes.py
│   └── test_phase2_fixes.py
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

### Unit Tests (37 tests organized)

#### Optimization (9 files)
- Decision engine integration with climate zones
- Peak protection and effect tariff
- Trend integration and forecasting
- Critical scenarios and edge cases
- Thermal recovery damping
- Savings calculations

#### Climate (6 files)
- Adaptive climate zones (Stockholm to Kiruna)
- Swedish climate region detection
- Weather compensation
- Weather-based pre-heating timing
- Trend and forecast integration

#### Learning (5 files)
- Adaptive thermal model API
- Learning data persistence
- Observation recording
- Prediction layer integration
- Thermal predictor persistence

#### Models (3 files)
- Heat pump model definitions (F750, F2040, S735, etc.)
- Model integration with codebase
- F750 realistic scenarios

#### DHW (2 files)
- Comprehensive DHW optimization
- Safety temperature thresholds

#### Effect (2 files)
- Effect manager and peak tracking
- Predictive peak avoidance

#### Coordinator (1 file)
- Power measurement fallback logic

#### Adapters (1 file)
- NIBE power calculation from current sensors

### Integration Tests (8 files at root)
- Config flow with model selection
- Configuration reload functionality
- Coordinator learning initialization
- Entity comprehensive testing
- Optional features
- Regression import tests
- Service rate limiting
- Service handlers

### Validation Tests (3 files)
- No hardcoded values verification
- Phase 1 fixes validation
- Phase 2 fixes validation

## Test Statistics

- **Total Tests:** 743
- **Pass Rate:** 100%
- **Coverage:** Comprehensive
- **Execution Time:** ~6 seconds

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

Tests are automatically run on:
- Every commit (via GitHub Actions)
- Pull requests
- Release builds

## Related Documentation
- `/docs/dev/` - Developer documentation

---

**Last Updated:** October 20, 2025
