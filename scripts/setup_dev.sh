#!/bin/bash
# Setup development environment for EffektGuard
# Requires Python 3.14 for Home Assistant compatibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== EffektGuard Development Environment Setup ==="
echo ""

# Check if Python 3.14 is available
if ! command -v python3.14 &> /dev/null; then
    echo "Python 3.14 not found. Installing..."
    
    # Check if we're on Ubuntu/Debian
    if command -v apt-get &> /dev/null; then
        echo "Adding deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install -y python3.14 python3.14-venv python3.14-dev
    else
        echo "ERROR: Automatic installation only supported on Ubuntu/Debian."
        echo "Please install Python 3.14 manually and re-run this script."
        exit 1
    fi
fi

echo "✓ Python 3.14 found: $(python3.14 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3.14 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

python -c 'import sys; assert sys.version_info[:2] == (3, 14), "Recreate .venv with Python 3.14"'

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
