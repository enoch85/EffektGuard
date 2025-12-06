# Testing Guide

Complete guide to running and writing tests for EffektGuard.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_entities.py -v

# Run specific test
pytest tests/unit/test_thermal_debt.py::test_dm_500_emergency -v

# With coverage report
pytest tests/ --cov=custom_components/effektguard --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests (pure logic, no HA dependencies)
│   ├── test_thermal_debt.py
│   ├── test_decision_engine.py
│   └── ...
├── integration/             # Integration tests (requires HA entities)
│   ├── test_coordinator.py
│   └── ...
└── test_*.py               # Root level tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test pure Python logic
- No Home Assistant dependencies
- Fast execution
- Mock external dependencies

**Example:**
```python
def test_thermal_debt_prevention():
    """Test that DM -500 triggers emergency recovery."""
    tracker = ThermalDebtTracker()
    tracker.degree_minutes = -500
    
    assert tracker.get_severity() == "CRITICAL"
    assert tracker.get_recovery_offset() == 3.0
```

### Integration Tests (`tests/integration/`)
- Test Home Assistant integration
- Require mock HA entities
- Test coordinator patterns
- Verify entity behavior

**Example:**
```python
async def test_coordinator_update(hass):
    """Test coordinator data update cycle."""
    coordinator = EffektGuardCoordinator(hass, ...)
    await coordinator.async_refresh()
    
    assert coordinator.last_update_success
```

## Running Tests

### Basic Commands

```bash
# All tests with verbose output
pytest tests/ -v

# Quiet mode (only show failures)
pytest tests/ -q

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Run only tests that match pattern
pytest tests/ -k "thermal_debt"
```

### Coverage Reports

```bash
# Terminal coverage report
pytest tests/ --cov=custom_components/effektguard

# HTML coverage report (opens in browser)
pytest tests/ --cov=custom_components/effektguard --cov-report=html
# Then open htmlcov/index.html

# Coverage with missing lines
pytest tests/ --cov=custom_components/effektguard --cov-report=term-missing
```

### Debug Mode

```bash
# Show print statements
pytest tests/ -s

# Drop into debugger on failure
pytest tests/ --pdb

# Full traceback
pytest tests/ --tb=long
```

## Writing Tests

### Test Safety-Critical Code

All safety thresholds must have tests:

```python
def test_prevents_catastrophic_thermal_debt():
    """Verify DM -500 threshold prevents heat pump damage."""
    # Based on stevedvo's real failure: DM -500 = 15kW spikes
    engine = DecisionEngine()
    state = NibeState(degree_minutes=-500)
    
    decision = engine.calculate_decision(state, prices, weather)
    
    # Must trigger emergency recovery, ignore cost optimization
    assert decision.offset >= 2.0
    assert "EMERGENCY" in decision.reasoning
```

### Test NIBE-Specific Behavior

```python
def test_blocks_activation_wrong_pump_config():
    """Verify open-loop with Intermittent mode blocks activation."""
    # Based on glyn.hudson's 8-hour off period failure
    validator = SystemConfigValidator()
    
    warnings = validator.validate_pump_configuration(
        system_type="open_loop",
        pump_mode="INTERMITTENT"
    )
    
    assert len(warnings) > 0
    assert warnings[0].severity == "CRITICAL"
    assert warnings[0].blocks_activation is True
```

### Use Realistic NIBE Values

```python
# ✅ Good - realistic NIBE behavior
def test_thermal_model_concrete_ufh():
    """Test thermal lag for concrete slab UFH."""
    model = ThermalModel(ufh_type="concrete_slab")
    
    # Concrete slab: 6+ hours thermal lag (glyn.hudson case study)
    lag = model.calculate_thermal_lag()
    assert 6.0 <= lag <= 12.0

# ❌ Bad - arbitrary values
def test_thermal_model():
    """Test thermal model."""
    model = ThermalModel(ufh_type="concrete_slab")
    lag = model.calculate_thermal_lag()
    assert lag > 0  # Too vague!
```

### Mock External Dependencies

```python
from unittest.mock import Mock, AsyncMock

async def test_nibe_adapter(hass):
    """Test NIBE adapter reads correct entities."""
    # Mock Home Assistant state
    hass.states.async_set("sensor.nibe_degree_minutes", "-120")
    
    adapter = NibeAdapter(hass, "nibe_device")
    state = await adapter.get_current_state()
    
    assert state.degree_minutes == -120
```

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def hass():
    """Home Assistant instance for testing."""
    # Returns configured HA instance
    
@pytest.fixture
def nibe_state():
    """Sample NIBE state data."""
    # Returns realistic NIBE state
    
@pytest.fixture
def price_data():
    """Sample Spot Price price data."""
    # Returns 15-minute price data
```

## Best Practices

### ✅ Do

- Test real NIBE behavior patterns
- Use realistic values from research documents
- Test safety-critical thresholds thoroughly
- Mock external APIs and network calls
- Use descriptive test names
- Add docstrings explaining what's tested
- Reference research documents in test docstrings

### ❌ Don't

- Test implementation details
- Use hardcoded magic numbers without context
- Skip testing safety-critical code
- Make network calls in tests
- Test private methods directly
- Use arbitrary test values

## Continuous Integration

Tests run automatically on:
- Pull requests
- Push to main branch
- Manual workflow dispatch

All tests must pass before merging.

## Troubleshooting

### Import Errors

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +

# Reinstall in development mode
pip install -e .
```

### Async Test Issues

Ensure async tests use `pytest-asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Fixture Not Found

Check that fixture is defined in `conftest.py` or imported correctly.

### Test Hangs

Check for:
- Missing `await` in async code
- Infinite loops
- Network calls (should be mocked)

## Test Coverage Goals

- **Core logic**: 90%+ coverage
- **Safety systems**: 100% coverage
- **Decision engine**: 95%+ coverage
- **Adapters**: 80%+ coverage (mock external APIs)

Run coverage check:
```bash
pytest tests/ --cov=custom_components/effektguard --cov-report=term-missing
```

## Related Documentation

- [CODE_STANDARDS.md](CODE_STANDARDS.md) - Code style conventions
- [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) - How to contribute
