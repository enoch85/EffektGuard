"""Turning a fractional curve offset into the integer NIBE's register (47011) can hold.

The last thing to touch the number before the heat pump, so a bias here silently attenuates every
decision the engine makes. Two invariants:

  * ROUND, never truncate. `int(-1.9)` is -1: Python truncates toward zero, so `int()` always did
    LESS than the engine asked (residual never re-applied) - a permanent one-directional shortfall.
    round() bounds the error at 0.5 C and makes it unbiased.
  * The sub-degree DEADBAND is hysteresis, not rounding: it stops MyUplink's rate-limited register
    being rewritten as demand wanders across a boundary. Cost: a demand settling <1 C from current
    is not expressed.

Shared by the adapter and the simulation harness so the two cannot drift apart.

tests/unit/utils/test_the_pump_does_what_the_engine_asked.py
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
