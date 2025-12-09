#!/usr/bin/env python3
"""Find duplicate and unused constants in const.py.

This script analyzes const.py for:
1. Duplicate values (same number assigned to different constants)
2. Similar naming patterns that might indicate redundancy
3. Unused constants (defined but never imported in production code)

Usage:
    python scripts/find_duplicate_constants.py
    python scripts/find_duplicate_constants.py --remove-unused  # Show removal commands
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def parse_constants(const_file: Path) -> dict[str, tuple[any, int]]:
    """Parse const.py and extract all constant definitions.
    
    Returns dict of {constant_name: (value, line_number)}
    """
    constants = {}
    
    with open(const_file, "r") as f:
        content = f.read()
        lines = content.split("\n")
    
    # Pattern for constants: NAME: Final = value
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*Final\s*=\s*(.+?)(?:\s*#.*)?$")
    
    for line_num, line in enumerate(lines, 1):
        match = pattern.match(line.strip())
        if match:
            name = match.group(1)
            value_str = match.group(2).strip()
            
            # Try to evaluate the value
            try:
                # Handle references to other constants
                if value_str in constants:
                    value = constants[value_str][0]
                else:
                    value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Keep as string if can't evaluate
                value = value_str
            
            constants[name] = (value, line_num)
    
    return constants


def find_duplicate_values(constants: dict[str, tuple[any, int]]) -> dict[any, list[str]]:
    """Find constants with identical values."""
    value_to_names = defaultdict(list)
    
    for name, (value, _) in constants.items():
        # Only check numeric values (most likely to be duplicated)
        if isinstance(value, (int, float)):
            value_to_names[value].append(name)
    
    # Filter to only duplicates
    return {v: names for v, names in value_to_names.items() if len(names) > 1}


def find_similar_names(constants: dict[str, tuple[any, int]]) -> list[tuple[str, str, float]]:
    """Find constants with similar names that might be duplicates."""
    from difflib import SequenceMatcher
    
    similar = []
    names = list(constants.keys())
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            # Skip if same prefix group (e.g., PROACTIVE_ZONE1 vs PROACTIVE_ZONE2)
            # These are intentionally different
            if _same_prefix_group(name1, name2):
                continue
                
            ratio = SequenceMatcher(None, name1, name2).ratio()
            if ratio > 0.7:  # 70% similar
                similar.append((name1, name2, ratio))
    
    return sorted(similar, key=lambda x: -x[2])  # Sort by similarity


def _same_prefix_group(name1: str, name2: str) -> bool:
    """Check if two names are in the same numbered group (e.g., ZONE1, ZONE2)."""
    # Remove trailing numbers and compare
    pattern = re.compile(r"^(.+?)(\d+)(.*)$")
    m1 = pattern.match(name1)
    m2 = pattern.match(name2)
    
    if m1 and m2:
        # Same prefix and suffix, different number
        if m1.group(1) == m2.group(1) and m1.group(3) == m2.group(3):
            return True
    
    return False


def find_unused_constants(
    constants: dict[str, tuple[any, int]], 
    project_root: Path,
    include_test_usage: bool = True
) -> list[tuple[str, int]]:
    """Find constants that are never imported in production code.
    
    Args:
        constants: Dict of constant names to (value, line_number)
        project_root: Project root path
        include_test_usage: If True, also check tests/scripts for usage
    """
    unused = []
    
    # Get all Python files in production code (not tests, not scripts)
    prod_files = list((project_root / "custom_components" / "effektguard").rglob("*.py"))
    const_file = project_root / "custom_components" / "effektguard" / "const.py"
    prod_files = [f for f in prod_files if f != const_file]
    
    # Read all production code
    all_code = ""
    for file in prod_files:
        try:
            with open(file, "r") as f:
                all_code += f.read() + "\n"
        except Exception:
            # Skip files that can't be read (permissions, encoding issues)
            pass
    
    # Optionally include tests and scripts
    if include_test_usage:
        test_files = list((project_root / "tests").rglob("*.py"))
        script_files = list((project_root / "scripts").rglob("*.py"))
        for file in test_files + script_files:
            try:
                with open(file, "r") as f:
                    all_code += f.read() + "\n"
            except Exception:
                # Skip files that can't be read (permissions, encoding issues)
                pass
    
    # Read const.py to check for building block usage (constants used to derive others)
    const_code = ""
    try:
        with open(const_file, "r") as f:
            const_code = f.read()
    except Exception:
        # Skip if const.py can't be read
        pass
    
    # Check each constant
    for name, (_, line_num) in constants.items():
        # Skip configuration keys (CONF_*) - these are used dynamically
        if name.startswith("CONF_"):
            continue
        # Skip attribute names (ATTR_*) - these are used dynamically
        if name.startswith("ATTR_"):
            continue
        # Skip service names (SERVICE_*) - these are used in services.yaml
        if name.startswith("SERVICE_"):
            continue
        # Skip storage keys - used dynamically
        if name.startswith("STORAGE_"):
            continue
        # Skip domain - always used
        if name == "DOMAIN":
            continue
        
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        
        # Check if used in production/test code
        if pattern.search(all_code):
            continue
            
        # Check if used as building block in const.py (more than just its definition)
        # Count occurrences - if > 1, it's used somewhere else in const.py
        matches = list(pattern.finditer(const_code))
        if len(matches) > 1:
            continue  # Used as building block
            
        unused.append((name, line_num))
    
    return sorted(unused, key=lambda x: x[1])  # Sort by line number


def find_unused_imports(project_root: Path) -> list[tuple[Path, str, int]]:
    """Find unused imports across all Python files using ruff.
    
    Returns list of (file_path, message, line_number) tuples.
    """
    import subprocess
    
    unused_imports = []
    
    # Check production code and tests
    dirs_to_check = [
        project_root / "custom_components" / "effektguard",
        project_root / "tests",
        project_root / "scripts",
    ]
    
    for check_dir in dirs_to_check:
        if not check_dir.exists():
            continue
            
        try:
            result = subprocess.run(
                ["ruff", "check", str(check_dir), "--select", "F401", "--output-format", "text"],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            
            # Parse ruff output: file:line:col: F401 message
            for line in result.stdout.strip().split("\n"):
                if not line or "F401" not in line:
                    continue
                # Example: tests/unit/test.py:34:5: F401 `module.name` imported but unused
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    file_path = project_root / parts[0]
                    line_num = int(parts[1]) if parts[1].isdigit() else 0
                    message = parts[3].strip() if len(parts) > 3 else line
                    unused_imports.append((file_path, message, line_num))
        except FileNotFoundError:
            # ruff not installed
            pass
        except Exception:
            pass
    
    return unused_imports


def find_semantic_duplicates(constants: dict[str, tuple[any, int]]) -> list[tuple[str, str, str]]:
    """Find constants that might be semantically equivalent.
    
    Looks for patterns like:
    - EFFECT_MARGIN_WARNING vs EFFECT_PEAK_MARGIN_WARNING
    - FOO_THRESHOLD vs FOO_LIMIT
    """
    duplicates = []
    
    # Group by base name patterns
    patterns = [
        (r"_THRESHOLD$", r"_LIMIT$"),
        (r"_MAX$", r"_MAXIMUM$"),
        (r"_MIN$", r"_MINIMUM$"),
        (r"^EFFECT_", r"^EFFECT_PEAK_"),
    ]
    
    names = list(constants.keys())
    
    for name1 in names:
        for pattern1, pattern2 in patterns:
            if re.search(pattern1, name1):
                # Look for corresponding name with other pattern
                base = re.sub(pattern1, "", name1)
                for name2 in names:
                    if name1 != name2 and re.search(pattern2, name2):
                        base2 = re.sub(pattern2, "", name2)
                        if base == base2:
                            # Same value = definitely duplicate
                            val1 = constants[name1][0]
                            val2 = constants[name2][0]
                            if val1 == val2:
                                duplicates.append((name1, name2, f"Same value: {val1}"))
                            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                if abs(val1 - val2) < 0.1:  # Very similar values
                                    duplicates.append((name1, name2, f"Similar values: {val1} vs {val2}"))
    
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Find duplicate and unused constants")
    parser.add_argument("--remove-unused", action="store_true", 
                       help="Show commands to remove unused constants")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    parser.add_argument("--prod-only", action="store_true",
                       help="Only check production code (exclude tests/scripts)")
    args = parser.parse_args()
    
    project_root = get_project_root()
    const_file = project_root / "custom_components" / "effektguard" / "const.py"
    
    if not const_file.exists():
        print(f"Error: {const_file} not found")
        sys.exit(1)
    
    print("=" * 70)
    print("CONSTANT ANALYSIS REPORT")
    print("=" * 70)
    
    # Parse constants
    constants = parse_constants(const_file)
    print(f"\nTotal constants defined: {len(constants)}")
    
    # 1. Find duplicate values
    print("\n" + "-" * 70)
    print("1. DUPLICATE VALUES (same number, different names)")
    print("-" * 70)
    duplicates = find_duplicate_values(constants)
    if duplicates:
        for value, names in sorted(duplicates.items()):
            # Skip trivial values (0, 1, 0.0, 1.0)
            if value in (0, 1, 0.0, 1.0, -1, -1.0):
                if args.verbose:
                    print(f"  [SKIPPED] Value {value}: {', '.join(names)}")
                continue
            print(f"  Value {value}:")
            for name in names:
                line = constants[name][1]
                print(f"    - {name} (line {line})")
    else:
        print("  No non-trivial duplicate values found.")
    
    # 2. Find similar names
    print("\n" + "-" * 70)
    print("2. SIMILAR NAMES (>70% string similarity)")
    print("-" * 70)
    similar = find_similar_names(constants)
    if similar:
        for name1, name2, ratio in similar[:20]:  # Top 20
            val1 = constants[name1][0]
            val2 = constants[name2][0]
            same = "✓ SAME VALUE" if val1 == val2 else ""
            print(f"  {ratio*100:.0f}% similar: {name1} vs {name2} {same}")
            if args.verbose:
                print(f"       Values: {val1} vs {val2}")
    else:
        print("  No highly similar names found.")
    
    # 3. Find semantic duplicates
    print("\n" + "-" * 70)
    print("3. SEMANTIC DUPLICATES (pattern matching)")
    print("-" * 70)
    semantic = find_semantic_duplicates(constants)
    if semantic:
        for name1, name2, reason in semantic:
            print(f"  {name1} <-> {name2}")
            print(f"    Reason: {reason}")
    else:
        print("  No semantic duplicates found.")
    
    # 4. Find unused constants
    print("\n" + "-" * 70)
    print("4. UNUSED CONSTANTS (not imported in production code)")
    if not args.prod_only:
        print("   (also checking tests/scripts - use --prod-only to exclude)")
    print("-" * 70)
    unused = find_unused_constants(constants, project_root, include_test_usage=not args.prod_only)
    if unused:
        print(f"  Found {len(unused)} unused constants:")
        for name, line in unused:
            value = constants[name][0]
            print(f"    Line {line:4d}: {name} = {value}")
        
        if args.remove_unused:
            print("\n  To remove these, delete the following lines from const.py:")
            for name, line in unused:
                print(f"    Line {line}: {name}")
    else:
        print("  All constants are used!")
    
    # 5. Find unused imports across all files
    print("\n" + "-" * 70)
    print("5. UNUSED IMPORTS (imported but never used in file)")
    print("-" * 70)
    unused_imports = find_unused_imports(project_root)
    if unused_imports:
        print(f"  Found {len(unused_imports)} unused imports:")
        for file_path, message, line_num in unused_imports:
            rel_path = file_path.relative_to(project_root)
            print(f"    {rel_path}:{line_num}: {message}")
        print("\n  Fix with: ruff check --select F401 --fix .")
    else:
        print("  All imports are used!")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total constants: {len(constants)}")
    print(f"  Duplicate value groups: {len([d for d in duplicates.values() if duplicates])}")
    print(f"  Similar name pairs: {len(similar)}")
    print(f"  Semantic duplicates: {len(semantic)}")
    print(f"  Unused constants: {len(unused)}")
    print(f"  Unused imports: {len(unused_imports)}")
    
    has_issues = unused or unused_imports
    if has_issues:
        if unused:
            print(f"\n  ⚠️  {len(unused)} constants can be safely removed!")
        if unused_imports:
            print(f"  ⚠️  {len(unused_imports)} unused imports found!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
