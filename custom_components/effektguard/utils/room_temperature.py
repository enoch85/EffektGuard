"""One indoor temperature from several room sensors.

Pure functions, shared by the NIBE adapter and the analysis scripts, so the number the engine sees
is the number a replay of production computes.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..const import (
    INDOOR_SENSOR_TRIM_FRACTION,
    INDOOR_TEMP_METHOD_AVERAGE,
    INDOOR_TEMP_METHOD_MEDIAN,
)


def interquartile_mean(temps: Sequence[float]) -> float:
    """The mean of the middle sensors, after dropping a quarter (rounded down) at each end.

    Every sensor that is kept counts equally, so one sensor moving changes the reading by at most
    its own change divided by the number kept, and one that crosses the others swaps places
    without a jump. With fewer than four sensors nothing is dropped: it is their mean.
    """
    ordered = sorted(temps)
    trimmed = int(len(ordered) * INDOOR_SENSOR_TRIM_FRACTION)
    middle = ordered[trimmed : len(ordered) - trimmed]
    return sum(middle) / len(middle)


def median(temps: Sequence[float]) -> float:
    """The middle sensor, or the mean of the two middle ones."""
    ordered = sorted(temps)
    half = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[half]
    return (ordered[half - 1] + ordered[half]) / 2


def combine_room_temperatures(temps: Sequence[float], method: str) -> float:
    """One indoor temperature from the readings, by the configured method.

    Args:
        temps: At least one reading, in °C.
        method: One of INDOOR_TEMP_METHODS. Anything else is the interquartile mean.

    Returns:
        The combined temperature in °C.
    """
    if method == INDOOR_TEMP_METHOD_AVERAGE:
        return sum(temps) / len(temps)
    if method == INDOOR_TEMP_METHOD_MEDIAN:
        return median(temps)
    return interquartile_mean(temps)
