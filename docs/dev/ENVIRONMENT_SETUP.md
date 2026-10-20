# Development Environment Setup

This guide covers setting up the EffektGuard development environment.

## Requirements

- **Python 3.15** - what CI (`.python-version`) and `scripts/setup_dev.sh` use; `pyproject.toml` sets
  the floor at 3.13
- Git
- Dev container support (optional but recommended)

## Quick Start

```bash
git clone https://github.com/enoch85/EffektGuard.git
cd EffektGuard
bash scripts/setup_dev.sh        # creates .venv with python3.15 and installs tests/requirements.txt
source .venv/bin/activate        # or: source docs/dev/activate.sh
```

### Verify

```bash
python --version                 # Python 3.15.x
which python                     # <project>/.venv/bin/python
bash scripts/run_all_tests.sh    # the full suite
```

## Dependencies

`tests/requirements.txt` pins `pytest-homeassistant-custom-component`, which in turn pins Home
Assistant, NumPy and the pytest toolchain (including pytest-xdist), plus the Black version CI uses.
To reinstall:

```bash
pip install -r tests/requirements.txt
```

## GitHub Codespaces (Option 1: Cloud)

Use this option if you want a cloud-hosted development environment.
Codespaces already runs inside a container, so you do not use the Dev Containers menu.

1. Create a Codespace from the repository.
2. Open the workspace and wait for it to finish starting.
3. Run `bash scripts/setup_dev.sh`, then `source .venv/bin/activate` when needed.

## Dev Container (Option 2: Local Docker)

Use this option if you want a local containerized environment in VS Code.

1. Install the "Dev Containers" extension.
2. Open the project folder in VS Code.
3. Run "Dev Containers: Reopen in Container".

The container (`.devcontainer/Dockerfile`: Ubuntu 24.04 with Python 3.15) runs the setup script to
create `.venv` and install the test dependencies.

## Next Steps

- [TESTING.md](TESTING.md) - Running and writing tests
- [CODE_STANDARDS.md](CODE_STANDARDS.md) - Code style and conventions
- [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) - How to contribute

## Troubleshooting

- **Wrong Python**: `which python` must point into `.venv`; if not, `source .venv/bin/activate`
- **Missing dependencies**: `pip install -r tests/requirements.txt`
- **Import errors**: clear `__pycache__` (`find . -type d -name __pycache__ -exec rm -r {} +`)
