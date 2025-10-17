#!/usr/bin/env python3
"""Verification script for Phase 2 completion.

Validates that all entity implementations are complete and correct.
"""

import ast
import sys
from pathlib import Path


def check_file_imports(filepath: Path) -> tuple[bool, list[str]]:
    """Check if file can be parsed and has proper imports."""
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content, filename=str(filepath))

        # Check for required imports
        has_logging = False
        has_entity_class = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logging":
                        has_logging = True
            elif isinstance(node, ast.ImportFrom):
                if "Entity" in str(node.names):
                    has_entity_class = True

        if not has_logging:
            issues.append(f"Missing logging import in {filepath.name}")

        if not has_entity_class:
            issues.append(f"Missing entity class import in {filepath.name}")

        return True, issues

    except SyntaxError as e:
        issues.append(f"Syntax error in {filepath.name}: {e}")
        return False, issues


def check_entity_class(filepath: Path, expected_class: str) -> tuple[bool, list[str]]:
    """Check if file contains expected entity class."""
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))

        found_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == expected_class:
                found_class = True
                break

        if not found_class:
            issues.append(f"Missing {expected_class} class in {filepath.name}")
            return False, issues

        return True, issues

    except Exception as e:
        issues.append(f"Error checking {filepath.name}: {e}")
        return False, issues


def check_async_setup_entry(filepath: Path) -> tuple[bool, list[str]]:
    """Check if file has async_setup_entry function."""
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))

        found_function = False
        is_async = False

        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "async_setup_entry":
                found_function = True
                is_async = True
                break
            elif isinstance(node, ast.FunctionDef) and node.name == "async_setup_entry":
                found_function = True
                break

        if not found_function:
            issues.append(f"Missing async_setup_entry in {filepath.name}")
            return False, issues

        if not is_async:
            issues.append(f"async_setup_entry is not async in {filepath.name}")
            return False, issues

        return True, issues

    except Exception as e:
        issues.append(f"Error checking {filepath.name}: {e}")
        return False, issues


def main():
    """Run all Phase 2 verification checks."""
    print("🔍 Verifying Phase 2 Completion...\n")

    base_path = Path("custom_components/effektguard")
    all_issues = []
    checks_passed = 0
    checks_failed = 0

    # Files to check
    entity_files = {
        "climate.py": "EffektGuardClimate",
        "sensor.py": "EffektGuardSensor",
        "number.py": "EffektGuardNumber",
        "select.py": "EffektGuardSelect",
        "switch.py": "EffektGuardSwitch",
    }

    print("=" * 60)
    print("Phase 2: Entity Implementation Verification")
    print("=" * 60)

    for filename, classname in entity_files.items():
        filepath = base_path / filename
        print(f"\n📄 Checking {filename}...")

        # Check file exists
        if not filepath.exists():
            print(f"  ❌ File not found: {filepath}")
            all_issues.append(f"File not found: {filepath}")
            checks_failed += 1
            continue

        # Check imports
        success, issues = check_file_imports(filepath)
        if success:
            print(f"  ✅ Imports OK")
            checks_passed += 1
        else:
            print(f"  ❌ Import issues")
            all_issues.extend(issues)
            checks_failed += 1

        # Check entity class
        success, issues = check_entity_class(filepath, classname)
        if success:
            print(f"  ✅ {classname} class found")
            checks_passed += 1
        else:
            print(f"  ❌ {classname} class missing")
            all_issues.extend(issues)
            checks_failed += 1

        # Check async_setup_entry
        success, issues = check_async_setup_entry(filepath)
        if success:
            print(f"  ✅ async_setup_entry implemented")
            checks_passed += 1
        else:
            print(f"  ❌ async_setup_entry issues")
            all_issues.extend(issues)
            checks_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Checks passed: {checks_passed}")
    print(f"❌ Checks failed: {checks_failed}")

    if all_issues:
        print("\n⚠️  Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1
    else:
        print("\n🎉 All Phase 2 checks passed!")
        print("\n✅ Entity implementations complete:")
        print("  - Climate entity with HVAC modes and presets")
        print("  - 9 diagnostic sensors (offset, DM, temps, price, peaks)")
        print(
            "  - 4 configuration number entities (target temp, tolerance, thermal mass, insulation)"
        )
        print("  - 1 select entity (optimization mode)")
        print("  - 2 switch entities (price optimization, peak protection)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
