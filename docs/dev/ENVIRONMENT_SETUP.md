# Development environment

## Requirements

- Python 3.15: what `.python-version`, CI and `scripts/setup_dev.sh` use (`pyproject.toml` allows
  3.13 and up)
- Git

## Set up

```bash
git clone https://github.com/enoch85/EffektGuard.git
cd EffektGuard
bash scripts/setup_dev.sh        # creates .venv with python3.15, installs tests/requirements.txt
source .venv/bin/activate        # or: source docs/dev/activate.sh
bash scripts/run_all_tests.sh    # verify
```

`setup_dev.sh` installs `python3.15` from the deadsnakes PPA if it is missing (Ubuntu/Debian only;
elsewhere install it first), creates `.venv`, checks that it is 3.15, and installs
`tests/requirements.txt`. It is safe to re-run.

## Dependencies

`tests/requirements.txt` pins three things: `pytest-homeassistant-custom-component`, which in turn
pins Home Assistant, NumPy, pytest, pytest-asyncio, pytest-cov and pytest-xdist; `voluptuous`; and
the Black version CI uses. Reinstall with `pip install -r tests/requirements.txt`.

## Dev container and Codespaces

`.devcontainer/` builds Ubuntu 24.04 with Python 3.15 and runs `scripts/setup_dev.sh` when the
container is created. In VS Code: "Dev Containers: Reopen in Container". A GitHub Codespace built
from the repository uses the same files.

## Troubleshooting

- `which python` must point into `.venv`; if not, `source .venv/bin/activate`.
- Stale imports after a rename: `find . -type d -name __pycache__ -exec rm -r {} +`.
