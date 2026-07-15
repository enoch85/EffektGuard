#!/usr/bin/env python3
"""Detect hardcoded numeric values (magic numbers) in production code.

The constants-only rule is the repository's most-emphasised convention: every numeric
threshold, tunable, physical constant, interval and safety limit belongs in const.py,
documented and reused. It is also the rule that has caused the most damage when broken -
a hardcoded `weight >= 0.85` gate silently stopped matching DM_CRITICAL_T2_WEIGHT after
that constant was retuned to 0.81, which let a cost layer override thermal-debt recovery.

The previous enforcement was a regex that flagged EVERY numeric literal - array indices,
loop bounds, `/ 60`, everything. It produced ~1,196 hits, so it was disabled with a bare
`return` and the rule became unenforced.

This checker is AST-based and deliberately high-signal. It reports a literal only when it
is used as a VALUE with meaning, and it ignores the structurally benign cases that made
the regex useless.

Usage:
    python scripts/check_hardcoded_values.py                 # report violations
    python scripts/check_hardcoded_values.py --baseline      # rewrite the baseline file
    python scripts/check_hardcoded_values.py --check         # fail if any file regressed
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_DIR = REPO_ROOT / "custom_components" / "effektguard"
BASELINE_PATH = REPO_ROOT / "tests" / "validation" / "hardcoded_values_baseline.json"

# const.py is where numbers are SUPPOSED to live.
EXCLUDED_FILENAMES = {"const.py"}

# Values that carry no tuning meaning: identity/neutral elements, and the handful of
# universal unit conversions. A literal equal to one of these is never a "magic number".
BENIGN_VALUES: frozenset[float] = frozenset(
    {
        0,
        1,
        2,  # identity / neutral / pairs
        -1,
        -2,
        0.0,
        1.0,
        2.0,
        -1.0,
        60,  # seconds per minute, minutes per hour
        60.0,
        100,  # percent
        100.0,
        1000,  # milli / kilo
        1000.0,
        3600,  # seconds per hour
        3600.0,
        24,  # hours per day
        24.0,
    }
)


class MagicNumberVisitor(ast.NodeVisitor):
    """Collect numeric literals that are used as meaningful values."""

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []
        self._skip: set[int] = set()  # id() of nodes to ignore

    # --- contexts where a literal is structurally benign ---------------------------

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # x[3], x[:48], x[i - 1] - indices and slices are structure, not tuning.
        for child in ast.walk(node.slice):
            self._skip.add(id(child))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # range(96), enumerate(x, 1), round(x, 2) - argument positions are structural.
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in {"range", "enumerate", "round", "zip"}:
            for arg in node.args:
                for child in ast.walk(arg):
                    self._skip.add(id(child))

        self.generic_visit(node)

    # --- the actual check ----------------------------------------------------------

    def visit_Constant(self, node: ast.Constant) -> None:
        if id(node) in self._skip:
            return
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return
        if float(node.value) in BENIGN_VALUES:
            return

        self.violations.append((node.lineno, repr(node.value)))


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, literal)] for one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as err:  # pragma: no cover - a broken file is a bigger problem
        print(f"error: cannot parse {path}: {err}", file=sys.stderr)
        return []

    visitor = MagicNumberVisitor()
    visitor.visit(tree)
    return sorted(visitor.violations)


def scan_production() -> dict[str, list[tuple[int, str]]]:
    """Return {relative_path: [(lineno, literal)]} for all production files."""
    results: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(PROD_DIR.rglob("*.py")):
        if path.name in EXCLUDED_FILENAMES:
            continue
        violations = scan_file(path)
        if violations:
            results[str(path.relative_to(REPO_ROOT))] = violations
    return results


def counts(results: dict[str, list[tuple[int, str]]]) -> dict[str, int]:
    return {path: len(violations) for path, violations in results.items()}


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def write_baseline(current: dict[str, int]) -> None:
    BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_against_baseline(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Return human-readable regressions. Empty list means no regression."""
    regressions: list[str] = []
    for path, count in sorted(current.items()):
        allowed = baseline.get(path, 0)
        if count > allowed:
            regressions.append(
                f"{path}: {count} hardcoded values (baseline allows {allowed}) "
                f"- {count - allowed} new"
            )
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true", help="rewrite the baseline to the current state"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if any file exceeds its baseline"
    )
    args = parser.parse_args()

    current = scan_production()

    if args.baseline:
        write_baseline(counts(current))
        total = sum(counts(current).values())
        print(f"Baseline written: {len(current)} files, {total} hardcoded values.")
        return 0

    if args.check:
        regressions = check_against_baseline(counts(current), load_baseline())
        if regressions:
            print("NEW hardcoded values introduced (constants-only rule):\n")
            for line in regressions:
                print(f"  {line}")
            print("\nMove these into const.py, or regenerate the baseline deliberately.")
            return 1
        print("No new hardcoded values.")
        return 0

    # Default: report everything, grouped by file.
    total = 0
    for path, violations in current.items():
        print(f"\n{path}  ({len(violations)})")
        for lineno, literal in violations:
            print(f"  {lineno}: {literal}")
        total += len(violations)
    print(f"\nTotal: {total} hardcoded values across {len(current)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
