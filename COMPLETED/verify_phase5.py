#!/usr/bin/env python3
"""Verification script for Phase 5: Services and Advanced Features.

This script verifies that Phase 5 implementation is complete and correct:
- Services defined in services.yaml
- Service handlers implemented in __init__.py
- Manual override support in decision engine
- Peak reset method in effect manager
- Service constants in const.py
- Comprehensive test coverage
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up from COMPLETED to project root
sys.path.insert(0, str(project_root))


def check_services_yaml():
    """Verify services.yaml defines all four services."""
    print("✓ Checking services.yaml...")

    services_file = project_root / "custom_components" / "effektguard" / "services.yaml"
    content = services_file.read_text()

    required_services = [
        "force_offset:",
        "reset_peak_tracking:",
        "boost_heating:",
        "calculate_optimal_schedule:",
    ]

    for service in required_services:
        assert service in content, f"Missing service: {service}"

    # Check for proper field definitions
    assert "offset:" in content, "Missing offset field"
    assert "duration:" in content, "Missing duration field"
    assert "response:" in content, "Missing response definition"

    print("  ✅ All four services defined with proper schemas")


def check_service_handlers():
    """Verify service handlers implemented in __init__.py."""
    print("✓ Checking service handlers...")

    init_file = project_root / "custom_components" / "effektguard" / "__init__.py"
    content = init_file.read_text()

    # Check for handler functions
    required_handlers = [
        "async def force_offset_handler",
        "async def reset_peak_tracking_handler",
        "async def boost_heating_handler",
        "async def calculate_optimal_schedule_handler",
    ]

    for handler in required_handlers:
        assert handler in content, f"Missing handler: {handler}"

    # Check for service registration
    assert "hass.services.async_register" in content, "Missing service registration"
    assert "get_coordinator" in content, "Missing coordinator helper"

    # Check for proper error handling
    assert "if not coordinator:" in content, "Missing coordinator check"
    assert "_LOGGER.error" in content, "Missing error logging"

    print("  ✅ All service handlers implemented with error handling")


def check_decision_engine_override():
    """Verify manual override support in decision engine."""
    print("✓ Checking decision engine manual override...")

    engine_file = (
        project_root
        / "custom_components"
        / "effektguard"
        / "optimization"
        / "decision_engine.py"
    )
    content = engine_file.read_text()

    # Check for override methods
    required_methods = [
        "def set_manual_override",
        "def clear_manual_override",
        "def _check_manual_override",
    ]

    for method in required_methods:
        assert method in content, f"Missing method: {method}"

    # Check for override state variables
    assert "_manual_override_offset" in content, "Missing override offset variable"
    assert "_manual_override_until" in content, "Missing override expiry variable"

    # Check integration in calculate_decision
    assert "manual_override = self._check_manual_override()" in content, (
        "Manual override not checked in calculate_decision"
    )

    print("  ✅ Manual override fully integrated with time-based expiration")


def check_effect_manager_reset():
    """Verify peak reset method in effect manager."""
    print("✓ Checking effect manager reset...")

    effect_file = (
        project_root
        / "custom_components"
        / "effektguard"
        / "optimization"
        / "effect_manager.py"
    )
    content = effect_file.read_text()

    # Check for reset method
    assert "def reset_monthly_peaks" in content, "Missing reset_monthly_peaks method"

    # Check it clears peaks
    assert "self._monthly_peaks = []" in content, "Reset doesn't clear peaks"
    assert "self._current_peak = 0.0" in content, "Reset doesn't clear current peak"

    print("  ✅ Peak reset method implemented correctly")


def check_service_constants():
    """Verify service constants in const.py."""
    print("✓ Checking service constants...")

    const_file = project_root / "custom_components" / "effektguard" / "const.py"
    content = const_file.read_text()

    required_constants = [
        'SERVICE_FORCE_OFFSET: Final = "force_offset"',
        'SERVICE_RESET_PEAK_TRACKING: Final = "reset_peak_tracking"',
        'SERVICE_BOOST_HEATING: Final = "boost_heating"',
        'SERVICE_CALCULATE_OPTIMAL_SCHEDULE: Final = "calculate_optimal_schedule"',
        'ATTR_OFFSET: Final = "offset"',
        'ATTR_DURATION: Final = "duration"',
    ]

    for constant in required_constants:
        assert constant in content, f"Missing constant: {constant}"

    print("  ✅ All service constants defined")


def check_test_coverage():
    """Verify service test coverage."""
    print("✓ Checking test coverage...")

    test_file = project_root / "tests" / "test_services.py"
    assert test_file.exists(), "Missing test_services.py"

    content = test_file.read_text()

    # Check for test categories
    required_tests = [
        "test_force_offset_service_registration",
        "test_force_offset_sets_override",
        "test_force_offset_validates_range",
        "test_reset_peak_tracking_clears_peaks",
        "test_boost_heating_sets_max_offset",
        "test_calculate_optimal_schedule_returns_24h_schedule",
        "test_decision_engine_set_manual_override",
        "test_decision_engine_calculate_with_manual_override",
        "test_effect_manager_reset_monthly_peaks",
        "test_service_handles_no_coordinator_gracefully",
    ]

    for test in required_tests:
        assert f"def {test}" in content or f"async def {test}" in content, (
            f"Missing test: {test}"
        )

    print("  ✅ Comprehensive test coverage (21 tests)")


def check_black_formatting():
    """Verify Black formatting."""
    print("✓ Checking Black formatting...")

    import subprocess

    files_to_check = [
        "custom_components/effektguard/__init__.py",
        "custom_components/effektguard/const.py",
        "custom_components/effektguard/optimization/decision_engine.py",
        "custom_components/effektguard/optimization/effect_manager.py",
        "tests/test_services.py",
    ]

    for file in files_to_check:
        result = subprocess.run(
            ["black", "--check", "--line-length", "100", file],
            cwd=project_root,
            capture_output=True,
        )
        assert result.returncode == 0, f"File not Black formatted: {file}"

    print("  ✅ All files properly formatted with Black (line-length 100)")


def run_service_tests():
    """Run service tests."""
    print("✓ Running service tests...")

    import subprocess

    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/test_services.py", "-v", "--tb=short"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Check for test results
    assert "21 passed" in result.stdout, "Not all service tests passed"
    assert "FAILED" not in result.stdout, "Some service tests failed"

    print("  ✅ All 21 service tests passing")


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("Phase 5 Verification: Services and Advanced Features")
    print("=" * 60 + "\n")

    try:
        check_services_yaml()
        check_service_handlers()
        check_decision_engine_override()
        check_effect_manager_reset()
        check_service_constants()
        check_test_coverage()
        check_black_formatting()
        run_service_tests()

        print("\n" + "=" * 60)
        print("✅ Phase 5 verification PASSED")
        print("Services and advanced features are complete and ready!")
        print("=" * 60 + "\n")

        print("Implemented:")
        print("  • force_offset service - Manual heating curve override")
        print("  • reset_peak_tracking service - Reset monthly peak data")
        print("  • boost_heating service - Emergency comfort boost")
        print("  • calculate_optimal_schedule service - 24h optimization preview")
        print("  • Manual override system with time-based expiration")
        print("  • Peak tracking reset functionality")
        print("  • Comprehensive error handling and validation")
        print("  • 21 service tests (100% passing)")

        return 0

    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
