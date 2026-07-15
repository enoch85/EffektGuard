"""A test that reads the clock when pytest COLLECTS it is measuring the gap between two clocks.

    NOW = dt_util.utcnow()          # <- evaluated at import, i.e. at collection

    async def test_something(...):
        entity = _weather_entity_with_forecast_from(NOW)   # built against the collection clock
        data = await adapter.get_forecast()               # adapter reads the clock again, NOW

Those two clocks agree only while nothing moves the clock between collection and the test running.
Freeze the wall clock at a daylight-saving transition, or collect at 23:59:58, and they diverge -
so the fragility is invisible on an ordinary run.

The rule is narrow on purpose: read the clock INSIDE the test (a fixture is the tidy way), never at
module scope. Constants that are plain literals - a fixed January date used as a label, say - are
fine and are not what this looks for.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS = pathlib.Path("tests")

# The calls that read the real clock. `datetime.now()` is already banned in production by
# test_no_production_code_uses_a_naive_datetime; here it is banned at test-module SCOPE too.
CLOCK_READS = {
    ("dt_util", "now"),
    ("dt_util", "utcnow"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
}


def _module_level_clock_reads(path: pathlib.Path) -> list[tuple[int, str]]:
    """Clock reads evaluated when the module is imported, not when a test runs."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Only statements that RUN AT IMPORT. A def or a class is not one of them - its body runs when
    # the test runs, which is exactly where reading the clock is correct, so we do not descend into
    # them.
    at_import = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    found = []
    for node in at_import:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                continue
            value = child.func.value
            if not isinstance(value, ast.Name):
                continue
            if (value.id, child.func.attr) in CLOCK_READS:
                found.append((child.lineno, f"{value.id}.{child.func.attr}()"))
    return found


@pytest.mark.parametrize(
    "path", sorted(TESTS.rglob("test_*.py")), ids=lambda p: str(p.relative_to(TESTS))
)
def test_the_clock_is_read_when_the_test_runs_not_when_it_is_collected(path):
    reads = _module_level_clock_reads(path)

    assert not reads, (
        f"{path} reads the clock at module scope: "
        + ", ".join(f"{call} on line {line}" for line, call in reads)
        + ". That value is captured when pytest COLLECTS the file, while the code under test reads "
        "the clock when the test RUNS. The two agree only while nothing moves the clock - freeze it "
        "at a daylight-saving transition, or collect at 23:59:58, and they diverge, and the test is "
        "then measuring the gap between two clocks rather than the behaviour it is named for. Read "
        "the clock inside the test; a fixture is the tidy way."
    )


class TestTheRuleCanActuallyCatchSomething:
    """A walker that matches nothing is not a guard."""

    def test_a_module_level_capture_is_caught(self, tmp_path):
        bad = tmp_path / "test_bad.py"
        bad.write_text("from homeassistant.util import dt as dt_util\n\nNOW = dt_util.utcnow()\n")

        assert _module_level_clock_reads(bad) == [(3, "dt_util.utcnow()")]

    def test_a_read_inside_a_test_is_allowed(self, tmp_path):
        good = tmp_path / "test_good.py"
        good.write_text(
            "from homeassistant.util import dt as dt_util\n\n\n"
            "def test_thing():\n    now = dt_util.utcnow()\n    assert now\n"
        )

        assert _module_level_clock_reads(good) == []

    def test_a_read_inside_a_fixture_is_allowed(self, tmp_path):
        good = tmp_path / "test_fixture.py"
        good.write_text(
            "import pytest\nfrom homeassistant.util import dt as dt_util\n\n\n"
            "@pytest.fixture\ndef now():\n    return dt_util.utcnow()\n"
        )

        assert _module_level_clock_reads(good) == []
