#!/bin/bash
# Quick activation script for Python 3.13 virtual environment
# Get the project root (two directories up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/venv313/bin/activate"
echo "✓ Python 3.13 virtual environment activated"
python --version
