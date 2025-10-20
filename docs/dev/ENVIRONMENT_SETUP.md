# Development Environment Setup

This guide covers setting up the EffektGuard development environment.

## Requirements

- **Python 3.13.8** or later
- Git
- Dev container support (optional but recommended)

## Quick Start

### 1. Clone and Enter Project

```bash
git clone https://github.com/enoch85/EffektGuard.git
cd EffektGuard
```

### 2. Activate Virtual Environment

The Python 3.13 virtual environment should auto-activate when you're in the project directory.

If it doesn't, manually activate it:
```bash
source docs/dev/activate.sh
# or
source venv313/bin/activate
```

### 3. Verify Environment

```bash
python --version
# Should output: Python 3.13.8

which python
# Should output: /workspaces/EffektGuard/venv313/bin/python
```

## Virtual Environment Details

### Location
- `venv313/` - Python 3.13 virtual environment with all dependencies

### Auto-Activation
The environment auto-activates via:
- `.envrc` - direnv configuration (if direnv is installed)
- `~/.bashrc` - Shell integration for automatic activation

### Manual Activation
```bash
source docs/dev/activate.sh
```

### Dependencies
All dependencies are managed in `venv313`. To reinstall:

```bash
pip install -r tests/requirements.txt
```

Key dependencies:
- Home Assistant 2025.10.2
- pytest and testing tools
- Black formatter (line length 100)
- Type checking tools

## Dev Container

This project includes a dev container configuration for consistent development environments.

The container runs:
- Ubuntu 24.04.2 LTS
- Python 3.13.8
- All required development tools

## Verify Installation

Run the test suite to verify everything is working:

```bash
pytest tests/ -v
```

Expected output:
```
======================== 715 passed, 75 warnings in ~5s =========================
```

## Next Steps

See the following guides:
- [TESTING.md](TESTING.md) - Running and writing tests
- [CODE_STANDARDS.md](CODE_STANDARDS.md) - Code style and conventions
- [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) - How to contribute

## Troubleshooting

### Virtual Environment Not Activating

Manually activate:
```bash
source venv313/bin/activate
```

### Python Version Issues

Verify Python version:
```bash
python --version
```

Should be 3.13.8 or later. If not, check that you're using the venv Python:
```bash
which python
```

### Missing Dependencies

Reinstall dependencies:
```bash
pip install -r tests/requirements.txt
```

### Import Errors

Clear Python cache and reinstall:
```bash
find . -type d -name __pycache__ -exec rm -r {} +
pip install -e .
```
