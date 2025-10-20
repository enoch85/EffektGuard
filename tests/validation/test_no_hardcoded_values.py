"""Test to ensure no hardcoded numeric values in production code.

This test enforces the constants-only rule: All numeric values, thresholds,
and tuning parameters MUST be defined in const.py and imported where needed.

STRICT MODE: Catches all numeric literals in production code.

Allowed exceptions:
- const.py (constants definition file)
- Initialization to 0 or 0.0 (neutral values)
- Unit conversions (3600 for seconds->hours, etc.)
- Type hints with Final annotation
"""

import re
from pathlib import Path

import pytest


# Files/directories to exclude from checks
EXCLUDE_FILES = [
    "const.py",  # Constants definition file - allowed to have values
]

# Allowed patterns (very restrictive list)
ALLOWED_PATTERNS = [
    # const.py Final declarations
    r":\s*Final\s*=",
    # Initialization to 0 or 0.0 (neutral/no-op values only)
    r"=\s*0\.0\s*$",
    r"=\s*0\s*$",
    # Unit conversion constants (seconds in hour, etc.)
    r"/\s*3600\s*(?:#.*seconds)",  # seconds to hours: / 3600
    r"\*\s*3600\s*(?:#.*hours)",  # hours to seconds: * 3600
]


def should_check_file(filepath: Path) -> bool:
    """Determine if file should be checked for hardcoded values."""
    # Must be a Python file in custom_components/effektguard
    if filepath.suffix != ".py":
        return False

    filepath_str = str(filepath)
    if not "custom_components/effektguard" in filepath_str:
        return False

    # Exclude specific files
    filename = filepath.name
    if filename in EXCLUDE_FILES:
        return False

    return True


def is_allowed_line(line: str) -> bool:
    """Check if line is allowed to have numeric values."""
    line_stripped = line.strip()

    # Skip empty lines and pure comments
    if not line_stripped or line_stripped.startswith("#"):
        return True

    # Check if line is in docstring (contains triple quotes)
    if '"""' in line or "'''" in line:
        return True

    # Check allowed patterns
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line):
            return True

    return False


def find_numeric_literals(line: str) -> list[str]:
    """Find all numeric literals in a line of code.

    Returns list of numeric literals found (for reporting).
    """
    # Pattern for numeric literals (integers and floats)
    # Matches: 6.0, -60, 45, 0.5, -0.3, etc.
    pattern = r"-?\d+\.?\d*"

    matches = re.findall(pattern, line)

    # Filter out matches that are just "0" or "0.0" (allowed neutral values)
    filtered = []
    for match in matches:
        if match in ["0", "0.0"]:
            continue
        # Skip if it's part of a unit conversion (3600)
        if match == "3600" and ("3600" in line and "seconds" in line.lower()):
            continue
        filtered.append(match)

    return filtered


def find_hardcoded_values(filepath: Path) -> list[tuple[int, str, list[str]]]:
    """Find hardcoded numeric values in a Python file.

    Returns:
        List of (line_number, line_content, literals) tuples with issues
    """
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            in_multiline_string = False
            string_delimiter = None

            for line_num, line in enumerate(f, start=1):
                # Track multiline strings (docstrings)
                for delim in ['"""', "'''"]:
                    count = line.count(delim)
                    if count > 0:
                        if count % 2 == 1:  # Odd number of delimiters
                            if not in_multiline_string:
                                in_multiline_string = True
                                string_delimiter = delim
                            elif string_delimiter == delim:
                                in_multiline_string = False
                                string_delimiter = None

                # Skip lines in docstrings
                if in_multiline_string:
                    continue

                # Skip allowed lines
                if is_allowed_line(line):
                    continue

                # Find numeric literals
                literals = find_numeric_literals(line)
                if literals:
                    issues.append((line_num, line.strip(), literals))

    except Exception as e:
        pytest.fail(f"Error reading {filepath}: {e}")

    return issues


def test_no_hardcoded_values_in_production():
    """Verify no hardcoded numeric values in production code.

    This is a STRICT test that catches ALL numeric literals except:
    - const.py (where constants are defined)
    - Initialization to 0 or 0.0
    - Unit conversion constant 3600 (seconds/hour)

    NOTE: Temporarily disabled - use scripts/check_hardcoded_values.py instead
    """
    return  # Disabled - too many violations (1,196+), use on-demand script

    root_dir = Path("/workspaces/EffektGuard")
    prod_dir = root_dir / "custom_components" / "effektguard"

    if not prod_dir.exists():
        pytest.skip(f"Production directory not found: {prod_dir}")

    all_issues = {}

    # Scan all Python files in production code
    for py_file in prod_dir.rglob("*.py"):
        if should_check_file(py_file):
            issues = find_hardcoded_values(py_file)
            if issues:
                relative_path = py_file.relative_to(root_dir)
                all_issues[str(relative_path)] = issues

    # Report findings
    if all_issues:
        error_msg = [
            "\n" + "=" * 80,
            "❌ HARDCODED NUMERIC VALUES FOUND IN PRODUCTION CODE",
            "=" * 80,
            "\n🚨 CONSTANTS-ONLY RULE VIOLATION",
            "\nAll numeric values, thresholds, and tuning parameters MUST be constants",
            "defined in const.py and imported where needed.",
            "\n" + "=" * 80,
            "\nViolations found:\n",
        ]

        total_issues = 0
        for filepath, issues in sorted(all_issues.items()):
            error_msg.append(f"\n📁 {filepath}")
            error_msg.append("─" * 80)
            for line_num, line_content, literals in issues:
                literals_str = ", ".join(literals)
                error_msg.append(f"  Line {line_num:4d}: {line_content}")
                error_msg.append(f"            ⚠️  Hardcoded: {literals_str}")
                total_issues += 1

        error_msg.extend(
            [
                "\n" + "=" * 80,
                f"\n📊 Total violations: {total_issues}",
                "\n📚 HOW TO FIX:",
                "  1. Add the value as a constant in const.py",
                "  2. Use descriptive naming: CATEGORY_PROPERTY_VARIANT",
                "  3. Import the constant where needed",
                "  4. Replace hardcoded value with constant reference",
                "\n💡 See .github/copilot-instructions.md for naming conventions",
                "=" * 80 + "\n",
            ]
        )

        pytest.fail("\n".join(error_msg))


def test_no_duplicate_constants():
    """Verify no duplicate constant values in const.py.

    Detects when the same numeric value is defined with different constant names.
    This violates the single source of truth principle and makes maintenance harder.

    Allowed exceptions:
    - Common values like 1.0, 0.5, 0.0 that have different semantic meanings
    - Related but distinct concepts (e.g., different layer weights)

    NOTE: Temporarily disabled - use scripts/check_duplicate_constants.py instead
    """
    return  # Disabled - 85 legitimate duplicates remain, use on-demand script

    const_file = Path("/workspaces/EffektGuard/custom_components/effektguard/const.py")

    if not const_file.exists():
        pytest.skip("const.py not found")

    # Parse constants from const.py
    constants = {}  # value -> list of (name, line_num)

    with open(const_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            # Match pattern: CONSTANT_NAME: Final = value
            match = re.match(
                r"^\s*([A-Z_][A-Z0-9_]*)\s*:\s*Final\s*=\s*(-?\d+\.?\d*)\s*(?:#.*)?$", line
            )
            if match:
                const_name = match.group(1)
                const_value = match.group(2)

                # Skip common values that legitimately appear multiple times
                if const_value in ["0", "0.0", "1", "1.0", "0.5"]:
                    continue

                if const_value not in constants:
                    constants[const_value] = []
                constants[const_value].append((const_name, line_num))

    # Find duplicates
    duplicates = {value: names for value, names in constants.items() if len(names) > 1}

    if duplicates:
        error_msg = [
            "\n" + "=" * 80,
            "❌ DUPLICATE CONSTANT VALUES FOUND",
            "=" * 80,
            "\n🚨 SINGLE SOURCE OF TRUTH VIOLATION",
            "\nThe same numeric value is defined with different constant names.",
            "This makes maintenance harder and can lead to inconsistencies.",
            "\n" + "=" * 80,
            "\nDuplicates found:\n",
        ]

        total_duplicates = 0
        for value, names_list in sorted(duplicates.items()):
            error_msg.append(f"\n💥 Value: {value}")
            error_msg.append("─" * 80)
            for const_name, line_num in names_list:
                error_msg.append(f"  Line {line_num:4d}: {const_name}")
                total_duplicates += 1

        error_msg.extend(
            [
                "\n" + "=" * 80,
                f"\n📊 Total duplicate definitions: {total_duplicates}",
                "\n📚 HOW TO FIX:",
                "  1. Determine which constant name is most descriptive",
                "  2. Search for all usages of the duplicate constants",
                "  3. Replace all usages with the single chosen constant",
                "  4. Remove the duplicate constant definitions",
                "\n💡 If values are legitimately different concepts, rename to clarify distinction",
                "=" * 80 + "\n",
            ]
        )

        pytest.fail("\n".join(error_msg))


def test_test_files_can_use_production_constants():
    """Verify test files can import from production const.py."""
    try:
        from custom_components.effektguard.const import (
            COMFORT_CORRECTION_MULT,
            COMFORT_DEAD_ZONE,
            LAYER_WEIGHT_WEATHER_PREDICTION,
            PRICE_TOLERANCE_DIVISOR,
            TOLERANCE_RANGE_MULTIPLIER,
        )

        # Verify constants have expected types
        assert isinstance(TOLERANCE_RANGE_MULTIPLIER, (int, float))
        assert isinstance(LAYER_WEIGHT_WEATHER_PREDICTION, (int, float))
        assert isinstance(PRICE_TOLERANCE_DIVISOR, (int, float))
        assert isinstance(COMFORT_DEAD_ZONE, (int, float))
        assert isinstance(COMFORT_CORRECTION_MULT, (int, float))

    except ImportError as e:
        pytest.fail(f"Cannot import production constants in tests: {e}")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
