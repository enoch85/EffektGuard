# Tests

How to run them: [docs/dev/TESTING.md](../docs/dev/TESTING.md). What earns a test:
[docs/agent/IMPLEMENTATION.md](../docs/agent/IMPLEMENTATION.md) (Tests).

## Layout

```
tests/
├── conftest.py          shared helpers and fixtures
├── requirements.txt     the dev dependencies
├── test_*.py            integration-level: config flow, reload, entities, services, imports
├── unit/                one behaviour per file, named for the behaviour
│   ├── adapters/        NIBE, GE-Spot and weather adapters
│   ├── climate/         climate zones, weather compensation
│   ├── coordinator/     the control loop: cycle order, failures, peaks, gates
│   ├── dhw/             hot-water scheduling
│   ├── effect/          effect tariff and peaks
│   ├── learning/        adaptive learning, observations, prediction
│   ├── models/          heat pump profiles
│   ├── optimization/    decision engine and layers
│   ├── utils/           the offset integer, the volatility gate, helpers
│   └── test_*.py        behaviours that span areas
├── validation/          the code against its documents, physics and ratchets
│   ├── hardcoded_values_baseline.json   the hardcoded-value ratchet's baseline
│   └── test_*.py        climate-zone tables, research docs, the rulebook, the emitter law,
│                        pump datasheets, the simulated plant, naive datetimes, translations
└── integration/         empty package
```

Name a file for the behaviour it pins: `test_nibe_s_own_hot_water_cycle_is_seen.py`, not
`test_dhw_optimizer.py`.

## Writing one

- Async tests need no marker (`asyncio_mode = "auto"` in `pyproject.toml`).
- Pure-logic tests build what they need with `create_mock_hass()`, `create_mock_entry()` and
  `make_mock_async_all()` from `conftest.py`; a test that needs Home Assistant itself takes the
  `hass` fixture from `pytest-homeassistant-custom-component`.
- No real network, no real clock, nothing written outside a temp folder.
- A known defect stays visible: `xfail(strict=True)` here, or a known red in the simulator, each
  listed in the open-findings register.

Live count: `pytest tests/ --collect-only -q | tail -1`.
