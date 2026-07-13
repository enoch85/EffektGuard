"""`int(-1.9)` is `-1`. Every offset came out smaller than the engine asked for.

NIBE's curve-offset register is integer-only and the decision engine calculates fractional offsets,
so something has to bridge the two. That something is the LAST thing to touch the number before it
reaches the heat pump - which makes a bias there invisible and universal. It attenuates every
decision the engine makes, and every constant anyone has ever tuned.

It truncated toward zero:

    accumulated_adjustment = int(self._fractional_accumulator)

Python's `int()` rounds toward zero, so the error was never random. It was always in the direction
of doing LESS:

    engine wants -1.9 C  ->  pump got -1   (0.9 C short)
    engine wants +2.7 C  ->  pump got +2   (0.7 C short)

and the residual was never re-applied, so the shortfall was permanent.

The month-long simulation says this costs no measurable money (5113 SEK truncating vs 5121 SEK
rounding, across five houses) - so the fix is made for correctness, not for savings, and the claim
that it "makes you pay more in expensive quarters" is not supported and is not repeated here. What
it does buy is that the pump does what the engine computed.

The simulation harness used to carry its OWN transcription of this arithmetic, truncation and all,
which is precisely how a plant model and the code it is meant to be testing drift apart without
anyone noticing. Both now call `integer_offset_for`.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    MAX_OFFSET,
    MIN_OFFSET,
    NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD,
)
from custom_components.effektguard.utils.offset import integer_offset_for


class TestTheBiasIsGone:
    """The whole point: the error must not always point the same way."""

    @pytest.mark.parametrize(
        ("demand", "expected"),
        [
            (-1.9, -2),  # int() gave -1
            (-2.7, -3),  # int() gave -2
            (+1.9, +2),  # int() gave +1
            (+2.7, +3),  # int() gave +2
            (-1.4, -1),
            (+1.4, +1),
        ],
    )
    def test_the_offset_is_rounded_not_truncated(self, demand, expected):
        applied = integer_offset_for(demand, current=0)

        assert applied == expected, (
            f"The engine asked for {demand:+.1f} C and the pump was given {applied:+d} C. "
            f"int({demand}) is {int(demand)} - Python truncates toward zero - so the pump always "
            f"did LESS than it was told, in the same direction, permanently."
        )

    def test_the_error_is_symmetric_around_zero(self):
        """A biased quantiser silently retunes every constant in const.py."""
        for magnitude in (1.1, 1.5, 1.9, 2.3, 2.5, 2.9, 3.4):
            up = integer_offset_for(+magnitude, current=0)
            down = integer_offset_for(-magnitude, current=0)
            assert up == -down, (
                f"A demand of +{magnitude} became {up:+d} but -{magnitude} became {down:+d}. "
                f"The quantiser must not prefer one direction."
            )

    def test_the_residual_error_never_exceeds_half_a_degree(self):
        """The best an integer register can do. Truncation gave up to a full degree."""
        for demand in [x / 10 for x in range(-100, 101)]:
            applied = integer_offset_for(demand, current=0)
            if applied != 0:  # outside the deadband
                assert abs(demand - applied) <= 0.5 + 1e-9, (
                    f"demand {demand:+.1f} -> {applied:+d}, an error of "
                    f"{abs(demand - applied):.2f} C"
                )


class TestTheDeadbandIsDeliberate:
    """Hysteresis, not arithmetic. It stops the register churning; do not remove it by accident."""

    def test_a_demand_that_has_barely_moved_does_not_rewrite_the_register(self):
        assert integer_offset_for(-2.4, current=-2) == -2
        assert integer_offset_for(+0.9, current=0) == 0

    def test_the_threshold_is_a_whole_degree(self):
        assert NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD == 1.0
        assert integer_offset_for(-0.99, current=0) == 0
        assert integer_offset_for(-1.0, current=0) == -1

    def test_it_settles_rather_than_oscillating(self):
        """Apply the same demand repeatedly: the register must reach a value and stay there."""
        demand = -1.9
        current = 0
        seen = []
        for _ in range(10):
            current = integer_offset_for(demand, current)
            seen.append(current)

        assert seen[-3:] == [-2, -2, -2], f"the register never settled: {seen}"


class TestTheRegisterCannotBeOverrun:
    def test_the_offset_is_clamped_to_what_the_register_can_hold(self):
        assert integer_offset_for(-50.0, current=0) == MIN_OFFSET
        assert integer_offset_for(+50.0, current=0) == MAX_OFFSET
