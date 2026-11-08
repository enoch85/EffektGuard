"""Conditional room forecasts, scored on later observations at the requested horizon.

The intercept includes native heating and unobserved gains; offset zero never means zero heat.
These coefficients predict temperature change, not physical heat loss, capacitance or savings.
The earlier training half and later validation half are disjoint, including target timestamps.
Least squares uses SVD rank checking: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..const import (
    LEARNING_MIN_OBSERVATIONS,
    LEARNING_OBSERVATION_INTERVAL_MINUTES,
    MINUTES_PER_HOUR,
    PREDICTION_MAX_SAMPLE_GAP_MINUTES,
    THERMAL_FORECAST_MIN_VALIDATION_ORIGINS,
    THERMAL_FORECAST_RANK_TOLERANCE,
    THERMAL_FORECAST_SENSOR_QUANTUM,
    THERMAL_FORECAST_TRAIN_FRACTION,
)
from .learning_types import AdaptiveThermalObservation, ThermalForecast


def fit_forecast(
    observations: Sequence[AdaptiveThermalObservation], hours: int
) -> ThermalForecast | None:
    """Fit one horizon; a skill score is improvement in squared error over persistence.

    The time span and cadence are checked in absolute time. A five-minute data burst cannot stand
    in for days of learning, and duplicate timestamps cannot contribute training or validation.
    Only the contiguous hourly suffix is used. Ill-conditioned, flat, sign-reversed and unstable
    fits abstain rather than supplying an apparently plausible heating response.
    """
    if hours <= 0 or len(observations) < LEARNING_MIN_OBSERVATIONS:
        return None
    interval = LEARNING_OBSERVATION_INTERVAL_MINUTES * MINUTES_PER_HOUR
    jitter = PREDICTION_MAX_SAMPLE_GAP_MINUTES * MINUTES_PER_HOUR
    if observations[-1].timestamp.tzinfo is None:
        return None
    suffix = [observations[-1]]
    for previous in reversed(observations[:-1]):
        if previous.timestamp.tzinfo is None:
            break
        elapsed = suffix[-1].timestamp.timestamp() - previous.timestamp.timestamp()
        if elapsed < interval - jitter:
            continue
        if elapsed > interval + jitter:
            break
        suffix.append(previous)
    suffix.reverse()
    if len(suffix) < LEARNING_MIN_OBSERVATIONS:
        return None
    if (
        suffix[-1].timestamp.timestamp() - suffix[0].timestamp.timestamp()
        < (LEARNING_MIN_OBSERVATIONS - 1) * interval
    ):
        return None
    values = np.asarray(
        [[o.indoor_temp, o.outdoor_temp, o.heating_offset] for o in suffix], dtype=float
    )
    if not np.isfinite(values).all():
        return None
    split = int(len(values) * THERMAL_FORECAST_TRAIN_FRACTION)
    count = len(values) - hours
    if count - split < THERMAL_FORECAST_MIN_VALIDATION_ORIGINS or split <= hours:
        return None

    # Conditional future inputs are averaged over the interval, while the target room is never
    # used as an input. Training origins end before their targets can reach the validation split.
    sums = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0)))
    means = (sums[hours : hours + count] - sums[:count]) / hours
    features = np.column_stack((np.ones(count), values[:count, 0], means[:, 1:]))
    targets = values[hours:, 0] - values[:count, 0]
    training = split - hours
    center = features[:training, 1:].mean(axis=0)
    scale = features[:training, 1:].std(axis=0)
    if (scale <= THERMAL_FORECAST_RANK_TOLERANCE).any():
        return None
    normalized = features.copy()
    normalized[:, 1:] = (normalized[:, 1:] - center) / scale
    try:
        fitted, _, rank, _ = np.linalg.lstsq(
            normalized[:training], targets[:training], rcond=THERMAL_FORECAST_RANK_TOLERANCE
        )
    except np.linalg.LinAlgError:
        return None
    if rank != features.shape[1]:
        return None
    coefficients = fitted.copy()
    coefficients[1:] /= scale
    coefficients[0] -= float(np.dot(coefficients[1:], center))
    # The room response must decay without reversing sign; extra offset cannot predict less heat.
    if not -1.0 <= coefficients[1] <= 0.0 or coefficients[-1] <= 0.0:
        return None

    errors = normalized[split:] @ fitted - targets[split:]
    persistence = targets[split:]
    baseline_mse = float(np.mean(persistence**2))
    if baseline_mse <= THERMAL_FORECAST_SENSOR_QUANTUM**2:
        skill = 0.0
    else:
        skill = max(0.0, 1.0 - float(np.mean(errors**2)) / baseline_mse)
    # Input domains come only from training, not from the untouched validation measurements.
    domains = [
        (float(values[:split, i].min()), float(values[:split, i].max()))
        for i in range(values.shape[1])
    ]
    return ThermalForecast(
        hours=hours,
        coefficients=tuple(float(c) for c in coefficients),
        skill=skill,
        validation_mae=float(np.mean(np.abs(errors))),
        persistence_mae=float(np.mean(np.abs(persistence))),
        max_validation_error=float(np.max(np.abs(errors))),
        room_range=domains[0],
        outdoor_range=domains[1],
        offset_range=domains[2],
    )
