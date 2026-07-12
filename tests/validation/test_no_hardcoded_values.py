"""Enforce the constants-only rule: no NEW hardcoded numeric values in production code.

The rule (.github/copilot-instructions.md, rules 3 and 4) is the repository's most-emphasised
convention: every numeric threshold, tunable, physical constant, interval and safety limit
belongs in const.py, documented and reused.

It is also the rule whose breach has done the most damage. A hardcoded `weight >= 0.85` gate
in the decision engine silently stopped matching DM_CRITICAL_T2_WEIGHT once that constant was
retuned to 0.81 - which let a cost layer override thermal-debt recovery and command a heat
REDUCTION at deep thermal debt. The constant moved; the magic number did not.

WHY THIS FILE WAS REWRITTEN
---------------------------
The previous version enforced nothing at all. Both checks began with a bare `return`:

    def test_no_hardcoded_values_in_production():
        return  # Disabled - too many violations (1,196+), use on-demand script
        root_dir = Path("/workspaces/EffektGuard")     # unreachable, and the wrong path

Three compounding failures: the `return` made them no-ops that reported PASSED; the path
(`/workspaces/EffektGuard`) does not exist (the repo is at `/workspace`), so even without the
`return` they would have SKIPPED; and the two scripts they deferred to
(`scripts/check_hardcoded_values.py`, `scripts/check_duplicate_constants.py`) did not exist.
The rule was enforced by nothing, anywhere, while reporting 3/3 green.

The root cause of the "1,196 violations" was the detector, not the code: a regex that flagged
EVERY numeric literal, including array indices, loop bounds and `/ 60`. It was unusable, so it
was switched off.

THE APPROACH: A RATCHET
-----------------------
`scripts/check_hardcoded_values.py` is AST-based and high-signal (504 real hits, and it does
catch every magic number the audit proved harmful). 504 is still too many to fix in one go, so
this is a ratchet rather than a gate:

  - `tests/validation/hardcoded_values_baseline.json` records the accepted count PER FILE.
  - Adding a magic number to any file makes that file exceed its baseline -> the test FAILS.
  - Removing magic numbers is always allowed; lower the baseline when you do.

This stops the debt growing while it is paid down, which is the only way a rule with 500
existing violations ever becomes enforceable again.

To regenerate the baseline deliberately (e.g. after moving values into const.py):
    python scripts/check_hardcoded_values.py --baseline
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_hardcoded_values import (  # noqa: E402
    BASELINE_PATH,
    check_against_baseline,
    counts,
    load_baseline,
    scan_production,
)


def test_no_new_hardcoded_values_in_production():
    """No file may contain MORE hardcoded numeric values than its baseline allows.

    This is the live enforcement of the constants-only rule. If it fails, you added a magic
    number: move it into const.py with a descriptive name and a comment explaining where the
    value comes from.
    """
    regressions = check_against_baseline(counts(scan_production()), load_baseline())

    if regressions:
        pytest.fail(
            "New hardcoded numeric values were introduced (constants-only rule):\n\n"
            + "\n".join(f"  {line}" for line in regressions)
            + "\n\nMove each value into const.py with a descriptive name "
            "(CATEGORY_PROPERTY_VARIANT) and a comment recording its source.\n"
            "Run `python scripts/check_hardcoded_values.py` to list them.\n"
            "If you genuinely intend to accept them, regenerate the baseline with\n"
            "`python scripts/check_hardcoded_values.py --baseline` and say why in the commit."
        )


def test_baseline_is_present_and_honest():
    """The baseline must exist and must not silently drift above the recorded debt.

    Guards the guard: a baseline that has been regenerated upward without anyone noticing
    would quietly re-disable the rule, which is exactly how the previous version died.
    """
    assert BASELINE_PATH.exists(), (
        f"{BASELINE_PATH} is missing. Generate it with "
        "`python scripts/check_hardcoded_values.py --baseline`."
    )

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline, "The baseline is empty - the checker is probably not scanning anything."

    total = sum(baseline.values())
    # A tripwire, not a target. If the debt grows past this, someone regenerated the baseline
    # to make a failure go away. Lower it as the debt is paid down; do not raise it.
    max_accepted_debt = 510
    assert total <= max_accepted_debt, (
        f"Recorded hardcoded-value debt is {total}, above the accepted ceiling of "
        f"{max_accepted_debt}. The baseline was regenerated upward instead of the values "
        "being moved into const.py."
    )


def test_the_checker_actually_detects_a_magic_number():
    """Guard against the checker silently becoming a no-op.

    The previous enforcement reported PASSED while detecting nothing. A checker that finds
    zero violations in a codebase with known debt is broken, not clean.
    """
    results = scan_production()

    assert results, (
        "The hardcoded-value checker found NOTHING in production code. It is almost certainly "
        "broken (wrong path, or an exception swallowed). It must not pass vacuously."
    )

    # The DM/°C slope in climate_zones.py is the single most load-bearing magic number in the
    # system - every climate-aware DM threshold scales with it. If the checker cannot see that
    # one, it is not doing its job.
    climate_zones = "custom_components/effektguard/optimization/climate_zones.py"
    assert climate_zones in results, (
        f"The checker no longer detects the known magic numbers in {climate_zones}. "
        "Its detection logic has regressed."
    )


def test_test_files_can_use_production_constants():
    """Verify test files can import from production const.py."""
    from custom_components.effektguard.const import (
        COMFORT_CORRECTION_MULT,
        COMFORT_DEAD_ZONE,
        LAYER_WEIGHT_WEATHER_PREDICTION,
        PRICE_TOLERANCE_FACTOR_MAX,
        PRICE_TOLERANCE_FACTOR_MIN,
        PRICE_TOLERANCE_MAX,
        PRICE_TOLERANCE_MIN,
        TOLERANCE_RANGE_MULTIPLIER,
    )

    # These must be real numbers, not accidentally-zeroed placeholders.
    assert 0 < TOLERANCE_RANGE_MULTIPLIER <= 1
    assert 0 < LAYER_WEIGHT_WEATHER_PREDICTION <= 1
    assert PRICE_TOLERANCE_MIN < PRICE_TOLERANCE_MAX
    assert PRICE_TOLERANCE_FACTOR_MIN < PRICE_TOLERANCE_FACTOR_MAX
    assert COMFORT_DEAD_ZONE > 0
    assert COMFORT_CORRECTION_MULT > 0
