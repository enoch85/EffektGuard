#!/bin/bash
# Quick activation of the development virtual environment that scripts/setup_dev.sh creates (.venv)
# Get the project root (two directories up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/.venv/bin/activate"
echo "✓ Development virtual environment activated"
python --version
