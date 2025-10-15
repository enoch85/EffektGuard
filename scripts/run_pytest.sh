#!/bin/bash
# Run pytest for EffektGuard integration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Running EffektGuard test suite...${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}ERROR: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-homeassistant-custom-component"
    exit 1
fi

# Run pytest with verbose output
if pytest tests/ -v --tb=short; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
