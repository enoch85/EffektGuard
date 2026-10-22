# Testing

How to run the tests. What earns a test is in
[docs/agent/IMPLEMENTATION.md](../agent/IMPLEMENTATION.md) (Tests); the tree and the conventions
for writing one are in [tests/README.md](../../tests/README.md).

## Run

```bash
bash scripts/run_all_tests.sh                 # Black check, then the whole suite
bash scripts/run_all_tests.sh -x dhw          # one category, stop at the first failure
pytest tests/ -q -n 4                         # direct, in parallel (pytest-xdist)
pytest tests/unit/dhw/test_a_slow_tank_is_heated_in_time.py -q
pytest tests/ -q --cov=custom_components/effektguard --cov-report=term-missing
```

The whole suite takes several minutes; `-n 4` helps.

`run_all_tests.sh` takes `-v` (verbose), `-x` (stop at the first failure) and `--tail N` (lines of
pytest output to show; default 20). Its `-p` and `-c` switches do nothing: the script looks for
executables named `pytest-xdist` and `pytest-cov`, which do not exist, and falls back to a
sequential run without coverage. Use the `pytest` commands above for both.

Categories: `all` (default), `unit`, `validation`, `integration` (`tests/test_*.py`),
`optimization`, `climate`, `learning`, `models`, `dhw`, `effect` (each `tests/unit/<name>/`), and
`simulation` (the simulator's `--selftest`; not listed by `--help`).

pytest's own options are in `pyproject.toml`: `asyncio_mode = "auto"`, `--strict-markers`,
`--tb=short`. `pytest-homeassistant-custom-component` supplies the `hass` fixture.

## The other gates

Tests are one gate of several. The list - Black, the hardcoded-value ratchet, the simulator's
`--selftest`, `--dst` and `--all-scenarios`, the live bench, the owner's closed loop for a change to
what the register receives - is in [CLAUDE.md](../../CLAUDE.md) under "Before claiming it works".

## CI

`.github/workflows/validate.yml` runs on pull requests, pushes to `main` and the merge queue: Black
and the ratchet; `pip check`, `compileall`, the full suite, the simulator's three runs, a HACS ZIP
build, HACS and Hassfest validation; workflow lint and a dev-container build.

## If a test hangs

A missing `await`, or something real (network, clock) that should be mocked: tests use no real
network and no real clock.
