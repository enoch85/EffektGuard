"""`integer_offset_for` must ROUND, not truncate, when bridging fractional offsets to NIBE's
integer register.

`int(-1.9)` is `-1`: truncation toward zero made every offset come out smaller than the engine
asked, always in the same direction, permanently (the residual was never re-applied). Since this is
the last thing to touch the number before the pump, that bias silently attenuates every decision and
every tuned constant. Rounding bounds the error at 0.5 C and makes it unbiased. The 1 C deadband is
deliberate hysteresis (MyUplink is rate-limited), and the value is clamped to the register range.
Shared with the simulation harness so the plant model and the code cannot drift.
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
