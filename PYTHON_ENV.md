# Python 3.13 Environment Setup

This project uses **Python 3.13.8** for development and testing.

## Quick Start

The Python 3.13 virtual environment should auto-activate when you're in the project directory.

If it doesn't, manually activate it:
```bash
source activate.sh
# or
source venv313/bin/activate
```

## Verify Environment

```bash
python --version
# Should output: Python 3.13.8
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_entities.py -v

# With coverage
pytest tests/ --cov=custom_components/effektguard --cov-report=html
```

## Dependencies

All dependencies are installed in the `venv313` virtual environment. To reinstall:

```bash
pip install -r tests/requirements.txt
```

## Why Python 3.13?

- Latest Python features and performance improvements
- Better async/await handling
- Improved type checking support
- All 715 tests passing successfully
