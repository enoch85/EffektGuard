"""Home Assistant works in aware UTC. `datetime.now()` returns a naive local time.

Mix the two and Python does not quietly do the wrong thing - it refuses:

    aware - naive  ->  TypeError: can't subtract offset-naive and offset-aware datetimes

And if it did not refuse, it would be worse: the box runs UTC while `datetime.now()` returns local
time, so every interval would be wrong by the UTC offset - two hours in a Swedish summer.

A grep is the right shape of test here: the rule is categorical, it costs nothing to hold, and the
next naive datetime someone adds will be in a file nobody has thought about.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PRODUCTION = pathlib.Path("custom_components/effektguard")

# `dt_util.now()` and `dt_util.utcnow()` are the correct calls and are NOT what this looks for -
# only a bare `datetime.now()` / `datetime.utcnow()`.
NAIVE = {"now", "utcnow"}


def _naive_calls(path: pathlib.Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in NAIVE:
            continue
        value = node.func.value
        # `datetime.now()` - the class, not dt_util
        if isinstance(value, ast.Name) and value.id == "datetime":
            found.append((node.lineno, f"datetime.{node.func.attr}()"))
    return found


@pytest.mark.parametrize(
    "path", sorted(PRODUCTION.rglob("*.py")), ids=lambda p: str(p.relative_to(PRODUCTION))
)
def test_no_production_file_calls_datetime_now(path):
    naive = _naive_calls(path)

    assert not naive, (
        f"{path} calls "
        + ", ".join(f"{call} at line {line}" for line, call in naive)
        + ". Home Assistant works in aware UTC: a naive datetime cannot be compared with an aware "
        "one at all (TypeError), and if it could, this box runs UTC while datetime.now() returns "
        "local time - so the interval would be wrong by the UTC offset, two hours in a Swedish "
        "summer. Use `dt_util.utcnow()`."
    )


def test_the_rule_can_actually_catch_something(tmp_path):
    """The guard on the guard: an AST walker that matches nothing is not a test."""
    offender = tmp_path / "offender.py"
    offender.write_text("from datetime import datetime\n\nx = datetime.now()\n")

    assert _naive_calls(offender) == [(3, "datetime.now()")]


def test_dt_util_is_not_mistaken_for_the_naive_call(tmp_path):
    """`dt_util.utcnow()` is the CORRECT call and must never be flagged."""
    good = tmp_path / "good.py"
    good.write_text("from homeassistant.util import dt as dt_util\n\nx = dt_util.utcnow()\n")

    assert _naive_calls(good) == []
