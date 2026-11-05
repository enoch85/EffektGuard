# Developer documentation

The rules live in [CLAUDE.md](../../CLAUDE.md) and `docs/agent/`:
[IMPLEMENTATION.md](../agent/IMPLEMENTATION.md) (workflow, verification, what earns a test, the
safety rules) and [PROJECT_NOTES.md](../agent/PROJECT_NOTES.md) (conventions, the open-findings
register, the traps). Read those first. This folder holds the mechanics.

| Doc | What |
|---|---|
| [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) | Python, the venv, the dev container |
| [TESTING.md](TESTING.md) | Running the tests and the other gates |
| [CODE_STANDARDS.md](CODE_STANDARDS.md) | Formatting, imports, type hints, docstrings, async |
| [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) | Branch, change, check, commit, pull request |

## Code layout

```
custom_components/effektguard/
├── __init__.py          entry setup and the services
├── config_flow.py       the config flow (options.py: the options flow)
├── coordinator.py       the control loop; the only place that writes to the pump
├── climate.py, sensor.py, switch.py, diagnostics.py
├── const.py             every numeric value, with its source
├── adapters/            nibe_adapter (MyUplink, nibe_heatpump or Modbus entities),
│                        gespot_adapter (prices), weather_adapter
├── models/              heat pump profiles (nibe/f730, f750, f1155, f2040, s1155), registry
├── optimization/        decision_engine, the *_layer.py modules, climate_zones,
│                        dhw_optimizer, airflow_optimizer, adaptive_learning, savings_calculator
└── utils/               offset (the register integer), volatile_helpers (the volatility
                         gate), emitter (EN 442), degree_minutes, compressor_monitor, power,
                         price_math
```

How the layers combine: [docs/architecture/](../architecture/), starting with `00_overview.md`.
The evidence behind the numbers in `const.py`: [docs/research/](../research/).

## Debug logging

In Home Assistant's `configuration.yaml`:

```yaml
logger:
  logs:
    custom_components.effektguard: debug
```

The engine logs its verdict at `info` as `Decision: offset ...`. Home Assistant writes to
`home-assistant.log` in its config directory, so from there: `grep "Decision:" home-assistant.log`.
