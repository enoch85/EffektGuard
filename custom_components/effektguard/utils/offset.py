"""Turning a fractional curve offset into the integer NIBE's register (47011) can hold.

TRUNCATE (int()), never round() - main's original design, and it is deliberate. The caller
recomputes the pending demand every cycle as (calculated - register), so truncation applies
only the WHOLE degrees the demand actually covers and leaves the fraction pending for the
next cycle: nothing is lost, and the register never receives tenths the engine did not ask
for. round() would over-apply by up to 0.5 C and then oscillate back as the recomputed
demand reverses sign.

The sub-degree DEADBAND is hysteresis, not rounding: it stops MyUplink's rate-limited
register being rewritten as demand wanders across a boundary.

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
        demand has not crossed a whole degree yet - the fraction stays pending and is
        re-derived next cycle.
    """
    demand = calculated - current
    if abs(demand) < NIBE_FRACTIONAL_ACCUMULATOR_THRESHOLD:
        return current

    # int(), not round(): apply only the whole degrees the demand covers. See module docstring.
    target = current + int(demand)
    return int(max(MIN_OFFSET, min(target, MAX_OFFSET)))
