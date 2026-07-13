"""Turning a fractional curve offset into the integer NIBE's register can hold.

NIBE's heating-curve offset register (47011 on the F-series) is integer-only, and the decision
engine calculates fractional offsets. Something has to bridge the two, and it is the last thing
that touches the number before it reaches the heat pump - so a bias here silently attenuates every
decision the engine makes and every constant anyone has ever tuned.

IT USED TO TRUNCATE TOWARD ZERO.

    accumulated_adjustment = int(self._fractional_accumulator)

`int(-1.9)` is `-1`, not `-2`. Python's `int()` truncates toward zero, so the error was never
random: it was always in the direction of doing LESS than the engine asked for.

    engine wants -1.9 C  ->  pump got -1   (0.9 C short)
    engine wants +2.7 C  ->  pump got +2   (0.7 C short)

and the residual was never re-applied, so the shortfall was permanent. Rounding to nearest bounds
the error at 0.5 C and, more importantly, makes it unbiased.

THE DEADBAND IS DELIBERATE, AND IT IS NOT ROUNDING.

A write only happens once the demand differs from what the pump currently holds by a whole degree.
That is hysteresis, not arithmetic: it stops the register being rewritten every five minutes as the
demand wanders across a rounding boundary, and MyUplink's API is rate-limited. The cost is that a
demand which settles at less than 1 C from the current value is not expressed at all.

This module is shared by the adapter and by the simulation harness. The harness used to carry its
own copy of this logic, which is exactly how a plant model and the code it is supposed to be
testing drift apart without anyone noticing.
"""

from ..const import (
    MAX_OFFSET,
    MIN_OFFSET,
    NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD,
)


def integer_offset_for(calculated: float, current: int) -> int:
    """The integer the pump's offset register should hold.

    Args:
        calculated: The fractional offset the decision engine asked for (°C).
        current: What the register holds right now (°C).

    Returns:
        The integer to write, clamped to the register's range. Equal to ``current`` when the
        demand has not moved far enough to be worth a write.
    """
    demand = calculated - current
    if abs(demand) < NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD:
        return current

    # round(), not int(). See the module docstring: int() truncates toward zero, so every offset
    # came out smaller than the engine asked for, always in the same direction.
    target = current + round(demand)
    return int(max(MIN_OFFSET, min(target, MAX_OFFSET)))
