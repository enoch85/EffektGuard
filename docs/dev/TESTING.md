# Testing Guide

How the EffektGuard tests are laid out and run. **What deserves a test** - and what does not - is in
[`docs/agent/IMPLEMENTATION.md`](../agent/IMPLEMENTATION.md) (Tests); this page does not repeat it.

## Quick Start

```bash
# Everything, organized output
bash scripts/run_all_tests.sh

# One category (see below)
bash scripts/run_all_tests.sh dhw

# Direct
pytest tests/ -q
pytest tests/unit/dhw/test_a_slow_tank_is_heated_in_time.py -q
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and helpers (mock hass, entry, price analyzer)
├── test_*.py                # Integration-level tests: config flow, services, entities, reload
├── unit/                    # One behaviour per file, named for the behaviour
│   ├── adapters/  climate/  coordinator/  dhw/  effect/
│   ├── learning/  models/   optimization/ utils/
│   └── test_*.py            # Behaviours that span areas
├── validation/              # The code against its documents, physics and ratchets
│   ├── test_climate_zones_doc_matches_the_code.py
│   ├── test_no_hardcoded_values.py         # with hardcoded_values_baseline.json
│   ├── test_research_docs_still_hold.py
│   └── ...
└── integration/             # (empty package)
```

Test files are named for the behaviour they hold, e.g.
`tests/unit/dhw/test_nibe_s_own_hot_water_cycle_is_seen.py`.

## Categories (`scripts/run_all_tests.sh <category>`)

| Category | Runs |
|---|---|
| `all` | `tests/` |
| `unit` | `tests/unit/` |
| `validation` | `tests/validation/` |
| `integration` | `tests/test_*.py` |
| `optimization`, `climate`, `learning`, `models`, `dhw`, `effect` | `tests/unit/<category>/` |
| `simulation` | The month-long simulation (real DecisionEngine + plant model) |

## Running Tests

```bash
pytest tests/ -q             # Quiet
pytest tests/ -x             # Stop on first failure
pytest tests/ --lf           # Last failed
pytest tests/ -k "peak"      # Name pattern
pytest tests/ -n 4           # In parallel (pytest-xdist)
```

pytest's options live in `pyproject.toml` (`asyncio_mode = "auto"`: async tests need no marker).
`pytest-homeassistant-custom-component` provides the real `hass` fixture for tests that need Home
Assistant itself.

## Fixtures and helpers

`tests/conftest.py`:
- `create_mock_hass(latitude=59.3)`, `create_mock_entry()`, `make_mock_async_all(...)` - helpers
- `mock_price_analyzer` - a fixture wrapping `create_mock_price_analyzer()`
- an autouse fixture that sets up Home Assistant's frame helper

## The other gates

Tests are one of several gates; the full list (Black, the hardcoded-value ratchet, the simulator's
`--selftest`, `--dst` and `--all-scenarios`, the live bench, the closed loop for control changes) is in
`CLAUDE.md` ("Before claiming it works") and `docs/agent/IMPLEMENTATION.md`.

## Continuous Integration

`.github/workflows/validate.yml` runs on pull requests, on pushes to `main` and in the merge queue:
Black and the ratchet, `pip check`, `compileall`, the full suite, the simulator's three runs, the HACS
build, HACS and Hassfest validation, workflow lint and a development-container build.

## Troubleshooting

- **Import errors**: clear `__pycache__` (`find . -type d -name __pycache__ -exec rm -r {} +`) and check
  that the venv is the one `scripts/setup_dev.sh` created (`.venv`, Python 3.15)
- **A test hangs**: a missing `await`, or something real (network, clock) that should be mocked - tests
  use no real network and no real clock

## Related Documentation

- [CODE_STANDARDS.md](CODE_STANDARDS.md) - Code style conventions
- [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) - How to contribute
- [`tests/README.md`](../../tests/README.md) - The test tree in detail
