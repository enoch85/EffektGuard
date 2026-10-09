#!/bin/bash
# Setup development environment for EffektGuard
# Runtime is shared with CI through .python-version.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== EffektGuard Development Environment Setup ==="
echo ""

PYTHON_VERSION="$(cat "$PROJECT_DIR/.python-version")"
PYTHON_MINOR="${PYTHON_VERSION%.*}"
PYTHON_BIN="python${PYTHON_MINOR}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Install Python $PYTHON_VERSION or newer in the $PYTHON_MINOR series, then re-run."
    exit 1
fi
"$PYTHON_BIN" -c "import sys; assert sys.version_info >= tuple(map(int, '$PYTHON_VERSION'.split('.')))"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$PYTHON_MINOR'.split('.'))), 'Recreate .venv for the configured runtime'"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo ""
echo "Installing test requirements..."
pip install -q -r "$PROJECT_DIR/tests/requirements.txt"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  bash scripts/run_all_tests.sh"
echo ""
echo "To deactivate:"
echo "  deactivate"
