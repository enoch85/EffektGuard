"""A sensor that tells us nothing must not be the one we trust most.

`_calculate_confidence` decides whether the learned thermal parameters are good enough to drive the
heat pump. Learning engages at LEARNING_CONFIDENCE_THRESHOLD (0.7), and one of its three terms is:

    consistency = 1.0 - min( std(rates) / max(mean(rates), 0.1), 1.0 )

where each rate is `temp_change / time_delta_hours` for an observation taken under heating.

The `max(mean, 0.1)` is there to stop a divide-by-zero. What it actually does is turn *no signal*
into *a perfect signal*. If every rate is identical - which is what a 0.1 C indoor sensor reports
when it is sampled every five minutes and the house is holding steady, and what a FAILED sensor
reports always - then std is 0, and:

    consistency = 1.0 - min(0 / 0.1, 1.0) = 1.0        PERFECT

A house that is genuinely, consistently heating (rates around 0.30 C/h, std 0.03) scores 0.912 -
LESS than the house that reported nothing at all. The metric is inverted at its degenerate limit:
it rewards the absence of information.

That is not theoretical. With consistency at 1.0 the total confidence reaches

    0.4 (observations, full deque) + 0.4 (consistency) + 0.067 (time span) = 0.867  >  0.7

so learning ENGAGES, and feeds heating_efficiency and thermal_decay_rate - computed from that same
all-zero data - into the pre-heat layer at weight 0.65. Simulated over 90 days at the coordinator's
real 5-minute cadence, it switched itself on at day 4, off by day 7, on again at day 60. It does not
converge; it flickers, and it flickers ON exactly when the data has gone degenerate.

The fix is strictly one-directional: a signal too weak to carry information scores ZERO, never one.
Nothing that scored below the threshold can rise above it, so this cannot switch learning on
anywhere it was not already on. It can only stop it engaging on nothing.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from custom_components.effektguard.const import (
    LEARNING_CONFIDENCE_THRESHOLD,
    LEARNING_MIN_OBSERVATIONS,
    LEARNING_OBSERVATION_WINDOW,
)
from custom_components.effektguard.optimization.adaptive_learning import AdaptiveThermalModel

START = datetime(2026, 1, 1)
CADENCE_MIN = 5  # the coordinator's real aligned-refresh interval

# A full deque. The defect only shows at full strength once the observation term has maxed out,
# which is exactly the state a real installation reaches after 56 hours and stays in forever.
FULL = LEARNING_OBSERVATION_WINDOW


def _model_from(
    temps: list[float], offset: float = 2.0, cadence_min: int = CADENCE_MIN
) -> AdaptiveThermalModel:
    """Feed a model a run of indoor readings under active heating."""
    model = AdaptiveThermalModel()
    for i, indoor in enumerate(temps):
        model.record_observation(
            timestamp=START + timedelta(minutes=i * cadence_min),
            indoor_temp=indoor,
            outdoor_temp=-5.0,
            heating_offset=offset,
        )
    return model


def test_a_flatlined_sensor_earns_no_confidence():
    """The degenerate case, stated plainly: an unchanging reading teaches us nothing.

    This is also exactly what a FAILED indoor sensor looks like - one value, forever.
    """
    flatlined = _model_from([21.0] * FULL)

    params = flatlined.get_parameters()
    assert params is not None, "precondition: enough observations to attempt learning"

    assert params.confidence < LEARNING_CONFIDENCE_THRESHOLD, (
        f"An indoor sensor that reported exactly 21.0 C for {FULL} consecutive samples - a house "
        f"that showed no measurable response to heating at all, or a sensor that has failed - "
        f"scored {params.confidence:.3f} against a {LEARNING_CONFIDENCE_THRESHOLD} threshold. "
        f"Learning ENGAGES, and drives the heat pump with parameters derived from that flat line."
    )


def test_a_house_that_teaches_us_something_beats_one_that_teaches_us_nothing():
    """Ordering, not just values. Confidence must rank real signal above no signal.

    Sampled HOURLY, where a 0.1 C sensor can actually resolve a building's response. At the
    coordinator's real 5-minute cadence neither house is distinguishable - a 0.30 C/h climb and a
    dead flat line both quantise to the same run of 0.0 and 0.1 ticks - and both now score zero,
    which is the honest answer. The ordering property has to be checked where the signal exists at
    all; that it does NOT exist at 5 minutes is the other half of F-132, and the owner's call.
    """
    rng = np.random.default_rng(3)
    hourly = 60

    # A house genuinely responding to heat: a real climb, with the ordinary variation of a real
    # building. Consistent, but not a straight line - nothing physical ever is.
    indoor, temps = 21.0, []
    for _ in range(FULL):
        indoor += 0.30 + rng.normal(0, 0.02)
        temps.append(round(indoor, 1))
    climbing = _model_from(temps, cadence_min=hourly)

    flatlined = _model_from([21.0] * FULL, cadence_min=hourly)

    real = climbing.get_parameters().confidence
    silent = flatlined.get_parameters().confidence

    assert real > silent, (
        f"A house that responded to heat with a steady, measurable climb scored {real:.3f}, and a "
        f"house whose sensor never moved scored {silent:.3f}. Confidence is meant to say how well "
        f"we know the building. It is ranking silence at or above evidence."
    )


def test_confidence_does_not_flicker_across_the_threshold():
    """It engaged on day 4, disengaged by day 7, and engaged again on day 60.

    A learned model that switches itself on and off as the noise in a rolling 56-hour window
    happens to fall is not learning. Whatever confidence means, it must not cross the threshold on
    a coin flip - so a stable house, observed for three months, must give one stable answer.
    """
    rng = np.random.default_rng(7)
    model = AdaptiveThermalModel()
    indoor = 21.0
    verdicts = set()

    for i in range(90 * 24 * 60 // CADENCE_MIN):
        hour = (i * CADENCE_MIN / 60) % 24
        offset = float(rng.integers(-3, 3))
        indoor += (0.02 * offset - 0.004 * (indoor - 21.0)) * (CADENCE_MIN / 60) + rng.normal(
            0, 0.002
        )
        model.record_observation(
            timestamp=START + timedelta(minutes=i * CADENCE_MIN),
            indoor_temp=round(indoor, 1),  # BT1 reports to 0.1 C
            outdoor_temp=round(-5.0 + 6.0 * float(np.sin(hour / 24 * 2 * np.pi)), 1),
            heating_offset=offset,
        )
        if (i * CADENCE_MIN) % (60 * 24) == 0 and i > 0:
            params = model.get_parameters()
            if params is not None:
                verdicts.add(params.confidence >= LEARNING_CONFIDENCE_THRESHOLD)

    assert verdicts != {True, False}, (
        "Over 90 days of one unchanging house, learning both engaged and disengaged. The 672-entry "
        "deque spans only 56 hours at the 5-minute observation cadence, so the model re-decides "
        "from scratch every two days on whatever noise it happens to hold - and it engages when "
        "the 0.1 C sensor's deltas collapse to a constant. Day 4: on. Day 7: off. Day 60: on."
    )


@pytest.mark.parametrize("samples", [3, 8, 10])
def test_too_few_samples_is_no_evidence_not_half_evidence(samples):
    """`else: consistency = 0.5` hands out half marks for having said nothing yet."""
    model = _model_from([21.0 + 0.05 * i for i in range(LEARNING_MIN_OBSERVATIONS * 2)], offset=2.0)

    # Rewrite history so only `samples` observations were taken under heating; the rest coast.
    for i, obs in enumerate(model.observations):
        obs.heating_offset = 2.0 if i < samples else 0.0

    params = model.get_parameters()

    assert params.confidence < LEARNING_CONFIDENCE_THRESHOLD, (
        f"With only {samples} observations taken under active heating, the consistency term fell "
        f"through to a hardcoded 0.5 - half confidence, awarded for an absence of data - and the "
        f"total reached {params.confidence:.3f}. Too little evidence is not half the evidence."
    )
