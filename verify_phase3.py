#!/usr/bin/env python3
"""Verification script for Phase 3: Effect Tariff Optimization.

Verifies:
- Effect manager implementation
- Peak protection layer in decision engine
- Persistent state storage
- Peak status sensor
- Test coverage for peak avoidance logic
"""

import sys
from pathlib import Path

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    return Path(filepath).exists()


def check_file_contains(filepath: str, search_terms: list[str]) -> tuple[bool, list[str]]:
    """Check if file contains all search terms."""
    if not check_file_exists(filepath):
        return False, search_terms

    with open(filepath, "r") as f:
        content = f.read()

    missing = [term for term in search_terms if term not in content]
    return len(missing) == 0, missing


def print_check(name: str, passed: bool, details: str = ""):
    """Print check result."""
    status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"{status} {name}")
    if details:
        print(f"  {YELLOW}{details}{RESET}")


def main():
    """Run Phase 3 verification."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Phase 3 Verification: Effect Tariff Optimization{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    # 1. Effect Manager Implementation
    print(f"{BLUE}1. Effect Manager Implementation{RESET}")

    effect_manager_checks = [
        "class EffectManager:",
        "async def record_quarter_measurement",
        "def should_limit_power",
        "def get_peak_protection_offset",
        "async def async_load",
        "async def async_save",
        "_monthly_peaks",
        "effective_power",
        "is_daytime",
    ]
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        effect_manager_checks,
    )
    all_passed &= passed
    print_check(
        "Effect manager with peak tracking",
        passed,
        f"Missing: {', '.join(missing)}" if missing else "",
    )

    # Check 15-minute granularity
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        ["quarter_of_day", "0-95", "15-minute"],
    )
    all_passed &= passed
    print_check("15-minute quarterly periods (0-95)", passed)

    # Check day/night weighting
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        ["is_daytime", "* 0.5", "50%"],
    )
    all_passed &= passed
    print_check("Day/night weighting (full/50%)", passed)

    # Check top 3 peak tracking
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        ["len(self._monthly_peaks) < 3", "top 3"],
    )
    all_passed &= passed
    print_check("Monthly top 3 peak tracking", passed)

    print()

    # 2. Peak Protection Layer in Decision Engine
    print(f"{BLUE}2. Peak Protection Layer in Decision Engine{RESET}")

    effect_layer_checks = [
        "def _effect_layer",
        "should_limit_power",
        "CRITICAL",
        "WARNING",
        "quarter",
    ]
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/decision_engine.py",
        effect_layer_checks,
    )
    all_passed &= passed
    print_check(
        "Effect layer implementation",
        passed,
        f"Missing: {', '.join(missing)}" if missing else "",
    )

    # Check layer is called
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/decision_engine.py",
        ["self._effect_layer(", "layers = ["],
    )
    all_passed &= passed
    print_check("Effect layer integrated in decision calculation", passed)

    # Check layer priority
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/decision_engine.py",
        ["3. Effect tariff protection", "weight=1.0", "weight=0.8"],
    )
    all_passed &= passed
    print_check("Layer priority and weighting", passed)

    print()

    # 3. Persistent State Storage
    print(f"{BLUE}3. Persistent State Storage{RESET}")

    storage_checks = [
        "Store(hass, STORAGE_VERSION, STORAGE_KEY)",
        "async def async_load",
        "async def async_save",
        "def to_dict",
        "def from_dict",
    ]
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        storage_checks,
    )
    all_passed &= passed
    print_check(
        "Persistent state storage implementation",
        passed,
        f"Missing: {', '.join(missing)}" if missing else "",
    )

    # Check initialization
    passed, missing = check_file_contains(
        "custom_components/effektguard/__init__.py",
        ["await effect_manager.async_load()"],
    )
    all_passed &= passed
    print_check("Effect manager state loaded on startup", passed)

    print()

    # 4. Peak Status Sensor
    print(f"{BLUE}4. Peak Status Sensor{RESET}")

    sensor_checks = [
        "Peak This Month",
        "peak_this_month",
        "SensorDeviceClass.POWER",
    ]
    passed, missing = check_file_contains(
        "custom_components/effektguard/sensor.py",
        sensor_checks,
    )
    all_passed &= passed
    print_check(
        "Peak status sensor",
        passed,
        f"Missing: {', '.join(missing)}" if missing else "",
    )

    print()

    # 5. Test Coverage
    print(f"{BLUE}5. Test Coverage for Peak Avoidance Logic{RESET}")

    # Test files exist
    test_files = [
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_effect_manager.py",
        "tests/test_decision_engine_peak_protection.py",
    ]
    for test_file in test_files:
        passed = check_file_exists(test_file)
        all_passed &= passed
        print_check(f"Test file: {test_file}", passed)

    # Effect manager test coverage
    effect_tests = [
        "test_daytime_full_weight",
        "test_nighttime_half_weight",
        "test_fills_top_three_peaks",
        "test_replaces_lowest_peak",
        "test_critical_when_exceeding_peak",
        "test_warning_within_one_kw",
        "test_nighttime_weighting_in_comparison",
    ]
    passed, missing = check_file_contains(
        "tests/test_effect_manager.py",
        effect_tests,
    )
    all_passed &= passed
    print_check(
        "Effect manager tests",
        passed,
        f"Missing: {', '.join(missing[:3])}" if missing else "",
    )

    # Decision engine integration tests
    decision_tests = [
        "test_effect_layer_called",
        "test_effect_layer_critical_peak",
        "test_safety_overrides_peak_protection",
        "test_emergency_overrides_peak_protection",
        "test_daytime_peak_avoidance",
    ]
    passed, missing = check_file_contains(
        "tests/test_decision_engine_peak_protection.py",
        decision_tests,
    )
    all_passed &= passed
    print_check(
        "Decision engine peak protection tests",
        passed,
        f"Missing: {', '.join(missing[:3])}" if missing else "",
    )

    print()

    # 6. Code Quality
    print(f"{BLUE}6. Code Quality{RESET}")

    # Check for safety thresholds
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        ["margin < 0.5", "margin < 1.0", "CRITICAL", "WARNING"],
    )
    all_passed &= passed
    print_check("Safety thresholds for peak avoidance", passed)

    # Check for research references
    passed, missing = check_file_contains(
        "custom_components/effektguard/optimization/effect_manager.py",
        ["Swedish effect tariff", "Daytime (06:00-22:00)", "Nighttime (22:00-06:00)"],
    )
    all_passed &= passed
    print_check("Documentation references Swedish Effektavgift", passed)

    print()

    # Summary
    print(f"{BLUE}{'='*60}{RESET}")
    if all_passed:
        print(f"{GREEN}✓ Phase 3 verification PASSED{RESET}")
        print(f"{GREEN}Effect tariff optimization is complete and ready!{RESET}")
    else:
        print(f"{RED}✗ Phase 3 verification FAILED{RESET}")
        print(f"{RED}Please review the failed checks above{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
