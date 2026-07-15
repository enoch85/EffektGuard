"""`integer_offset_for` applies whole demand-backed degrees: truncate, deadband, clamp.

Truncation (int(), main's original design) is deliberate: the caller recomputes pending
demand every cycle as (calculated - register), so only the WHOLE degrees the demand covers
are applied and the fraction stays pending - never lost, never over-applied. round() would
write up to 0.5 C the engine did not ask for, then oscillate back when the recomputed demand
reversed sign. The 1 C deadband is hysteresis for a rate-limited register; the result is
clamped to the register's range. Shared with the simulation harness.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    MAX_OFFSET,
    MIN_OFFSET,
    NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD,
)
from custom_components.effektguard.utils.offset import integer_offset_for


class TestWholeDegreesOnly:
    """Only the integer part of the demand reaches the register; the fraction stays pending."""

    @pytest.mark.parametrize(
        ("demand", "expected"),
        [
            (-1.9, -1),  # the 0.9 stays pending, re-derived next cycle
            (1.9, 1),
            (-2.6, -2),
            (2.6, 2),
            (1.0, 1),  # exactly one degree is fully backed
            (-1.0, -1),
        ],
    )
    def test_truncation_applies_only_backed_degrees(self, demand, expected):
        assert integer_offset_for(calculated=demand, current=0) == expected

    def test_truncation_is_symmetric_toward_zero(self):
        assert integer_offset_for(1.9, 0) == -integer_offset_for(-1.9, 0)

    def test_the_register_is_never_pushed_past_the_demand(self):
        """The property truncation buys: the pump never does MORE than the engine asked."""
        for tenths in range(-100, 101):
            calculated = tenths / 10.0
            written = integer_offset_for(calculated, current=0)
            assert abs(written) <= abs(calculated) + 1e-9, (
                f"demand {calculated:+.1f} wrote {written:+d} - the register got more than "
                f"the engine asked for, which truncation exists to prevent"
            )

    def test_the_pending_fraction_is_not_lost(self):
        """Held demand converges as the fraction is re-derived against the updated register.

        Cycle 1 at +2.6 from 0 writes +2 (0.6 pending). If the engine's demand grows to
        +3.1, the recomputed pending demand (1.1) crosses a whole degree and writes +3.
        Nothing was truncated away permanently.
        """
        first = integer_offset_for(2.6, current=0)
        assert first == 2

        second = integer_offset_for(3.1, current=first)
        assert second == 3


class TestTheDeadbandIsDeliberate:
    def test_a_demand_that_has_barely_moved_does_not_rewrite_the_register(self):
        assert integer_offset_for(2.4, current=2) == 2

    def test_the_threshold_is_a_whole_degree(self):
        assert NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD == 1.0

    def test_it_settles_rather_than_oscillating(self):
        """Demand wandering inside the deadband around the held value writes nothing."""
        current = 2
        for calculated in (2.3, 1.7, 2.4, 1.6, 2.0):
            assert integer_offset_for(calculated, current) == current


class TestTheClamp:
    def test_the_offset_is_clamped_to_what_the_register_can_hold(self):
        assert integer_offset_for(25.0, current=0) == MAX_OFFSET
        assert integer_offset_for(-25.0, current=0) == MIN_OFFSET
