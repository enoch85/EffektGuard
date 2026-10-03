#!/bin/bash
# EffektGuard Comprehensive Test Runner
# Runs all tests with organized output and reporting

# Don't use set -e because we need to capture test failures
set -o pipefail

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default options
VERBOSE=false
COVERAGE=false
PARALLEL=false
CATEGORY=""
FAIL_FAST=false
MARKERS=""
TAIL_LINES=20  # Default tail output for summary

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -x|--fail-fast)
            FAIL_FAST=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        --tail)
            TAIL_LINES="$2"
            shift 2
            ;;
        unit|validation|integration|all|optimization|climate|learning|models|dhw|effect|simulation)
            CATEGORY="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [CATEGORY]"
            echo ""
            echo "Categories:"
            echo "  all            Run all tests (default)"
            echo "  unit           Run only unit tests"
            echo "  validation     Run only validation tests"
            echo "  integration    Run only integration tests"
            echo "  optimization   Run optimization unit tests"
            echo "  climate        Run climate unit tests"
            echo "  learning       Run learning unit tests"
            echo "  models         Run model unit tests"
            echo "  dhw            Run DHW unit tests"
            echo "  effect         Run effect unit tests"
            echo ""
            echo "Options:"
            echo "  -v, --verbose      Verbose output"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -p, --parallel     Run tests in parallel"
            echo "  -x, --fail-fast    Stop on first failure"
            echo "  -m, --markers      Run tests matching markers"
            echo "  --tail N           Show last N lines of output (default: 20)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Run all tests"
            echo "  $0 unit                 # Run unit tests only"
            echo "  $0 optimization         # Run optimization tests"
            echo "  $0 -c -p all            # Run all tests with coverage in parallel"
            echo "  $0 -x unit              # Run unit tests, stop on first fail"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Set default category if not specified
if [ -z "$CATEGORY" ]; then
    CATEGORY="all"
fi

# Header
echo ""
echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║${NC}                ${BOLD}EffektGuard Test Suite${NC}                              ${BOLD}${CYAN}║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run Black formatting check first
echo -e "${BLUE}🎨 Checking Black formatting...${NC}"
if command -v black &> /dev/null; then
    if black custom_components/effektguard/ --check --line-length 100 &> /dev/null; then
        echo -e "${GREEN}✓ Black formatting: PASS${NC}"
    else
        # A gate that silently rewrites the tree and reports success is not a gate.
        echo -e "${RED}✗ Black formatting: FAIL - run: black custom_components/effektguard/ --line-length 100${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Black not installed, skipping formatting check${NC}"
fi
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ ERROR: pytest is not installed${NC}"
    echo "Install with: pip install -r tests/requirements.txt"
    exit 1
fi

# Build pytest command
PYTEST_CMD="python -m pytest"
PYTEST_ARGS=""

# Add path based on category
case $CATEGORY in
    all)
        echo -e "${BLUE}📊 Running: ${BOLD}All Tests${NC}"
        PYTEST_ARGS="tests/"
        ;;
    unit)
        echo -e "${BLUE}📊 Running: ${BOLD}Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/"
        ;;
    validation)
        echo -e "${BLUE}📊 Running: ${BOLD}Validation Tests${NC}"
        PYTEST_ARGS="tests/validation/"
        ;;
    integration)
        echo -e "${BLUE}📊 Running: ${BOLD}Integration Tests${NC}"
        PYTEST_ARGS="tests/test_*.py"
        ;;
    optimization)
        echo -e "${BLUE}📊 Running: ${BOLD}Optimization Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/optimization/"
        ;;
    climate)
        echo -e "${BLUE}📊 Running: ${BOLD}Climate Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/climate/"
        ;;
    learning)
        echo -e "${BLUE}📊 Running: ${BOLD}Learning Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/learning/"
        ;;
    models)
        echo -e "${BLUE}📊 Running: ${BOLD}Model Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/models/"
        ;;
    dhw)
        echo -e "${BLUE}📊 Running: ${BOLD}DHW Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/dhw/"
        ;;
    effect)
        echo -e "${BLUE}📊 Running: ${BOLD}Effect Unit Tests${NC}"
        PYTEST_ARGS="tests/unit/effect/"
        ;;
    simulation)
        # Month-long optimization simulation (real DecisionEngine + plant model).
        # Self-test uses 2 synthetic days; see scripts/simulation/README.md.
        echo -e "${BLUE}📊 Running: ${BOLD}Optimization Simulation (self-test)${NC}"
        python scripts/simulation/sim_harness.py --selftest
        exit $?
        ;;
    *)
        echo -e "${RED}✗ Unknown category: $CATEGORY${NC}"
        exit 1
        ;;
esac

echo ""

# Add options
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -v"
else
    PYTEST_ARGS="$PYTEST_ARGS -q"
fi

if [ "$FAIL_FAST" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -x"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m $MARKERS"
fi

if [ "$PARALLEL" = true ]; then
    if command -v pytest-xdist &> /dev/null; then
        PYTEST_ARGS="$PYTEST_ARGS -n auto"
        echo -e "${CYAN}⚡ Parallel execution enabled${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: pytest-xdist not installed, running sequentially${NC}"
        echo -e "${YELLOW}  Install with: pip install pytest-xdist${NC}"
    fi
fi

# Coverage options
if [ "$COVERAGE" = true ]; then
    if command -v pytest-cov &> /dev/null; then
        PYTEST_ARGS="$PYTEST_ARGS --cov=custom_components/effektguard --cov-report=term-missing --cov-report=html"
        echo -e "${CYAN}📈 Coverage reporting enabled${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: pytest-cov not installed, skipping coverage${NC}"
        echo -e "${YELLOW}  Install with: pip install pytest-cov${NC}"
    fi
fi

echo ""

# Run tests
START_TIME=$(date +%s)

# Build final command with optional tail
if [ "$VERBOSE" = true ]; then
    # Verbose mode: show all output
    set +e  # Temporarily disable exit on error
    TEST_OUTPUT=$($PYTEST_CMD $PYTEST_ARGS 2>&1)
    TEST_EXIT=$?
    set -e  # Re-enable
    echo "$TEST_OUTPUT"
else
    # Quiet mode: show last N lines (summary)
    set +e  # Temporarily disable exit on error
    TEST_OUTPUT=$($PYTEST_CMD $PYTEST_ARGS 2>&1)
    TEST_EXIT=$?
    set -e  # Re-enable
    echo "$TEST_OUTPUT" | tail -n $TAIL_LINES
fi

if [ $TEST_EXIT -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}${BOLD}✓ All tests passed!${NC}"
    echo -e "${GREEN}  Execution time: ${DURATION}s${NC}"
    
    if [ "$COVERAGE" = true ] && command -v pytest-cov &> /dev/null; then
        echo ""
        echo -e "${CYAN}📊 Coverage report generated: ${BOLD}htmlcov/index.html${NC}"
    fi
    
    echo ""
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${RED}${BOLD}✗ Tests failed${NC}"
    echo -e "${RED}  Execution time: ${DURATION}s${NC}"
    echo ""
    exit 1
fi
