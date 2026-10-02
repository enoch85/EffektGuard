"""Thermal-debt deferral must SCALE the owner's weather weight, and keep doing so.

When degree minutes go negative, the weather-compensation layer stands down so that DM,
comfort and the proactive zones decide instead. The reduction used to be written as

    defer_factor = WEATHER_COMP_DEFER_WEIGHT_CRITICAL / DEFAULT_WEATHER_COMPENSATION_WEIGHT
    final_weight = (dynamic_weight * self.weather_comp_weight) * defer_factor

A fraction reconstructed by dividing one tuning constant by another, inside a layer that
then multiplies it by a third. THAT IS NOT A BUG - the owner's weight is multiplied, not
cancelled - but it reads like one: two readers of this code concluded in turn that the
0.49 in the numerator cancelled the 0.49 the owner's weight contributes, which would make
the deferral an override rather than a reduction. It took running it to settle.

So the constants state the retained fraction directly, derived from the per-tier weights a
default install was tuned to produce, which keeps every existing install bit-identical.

These tests are therefore GUARDS, not regressions: they pin the proportionality that was
already correct, so the next person to touch this does not have to run it to find out.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.const import (
    DEFAULT_WEATHER_COMPENSATION_WEIGHT,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    WEATHER_COMP_DEFER_DM_LIGHT,
    WEATHER_COMP_DEFER_DM_MODERATE,
    WEATHER_COMP_DEFER_DM_SIGNIFICANT,
    WEATHER_COMP_DEFER_RETAIN_CRITICAL,
    WEATHER_COMP_DEFER_RETAIN_LIGHT,
    WEATHER_COMP_DEFER_RETAIN_MODERATE,
    WEATHER_COMP_DEFER_RETAIN_SIGNIFICANT,
)
from custom_components.effektguard.optimization.weather_layer import (
    AdaptiveClimateSystem,
    WeatherCompensationCalculator,
    WeatherCompensationLayer,
)

TZ = timezone(timedelta(hours=1))


class _Forecast:
    def __init__(self, temp):
        self.datetime = datetime(2026, 1, 15, 12, 0, tzinfo=TZ)
        self.temperature = temp


class _Weather:
    def __init__(self):
        self.current_temp = -5.0
        self.forecast_hours = [_Forecast(-5.0) for _ in range(24)]


class _State:
    def __init__(self, dm):
        self.degree_minutes = dm
        self.outdoor_temp = -5.0
        self.indoor_temp = 21.0
        self.flow_temp = 34.0
        self.supply_temp = 34.0
        self.current_offset = 0.0
        self.timestamp = datetime(2026, 1, 15, 12, 0, tzinfo=TZ)


def _layer(owner_weight: float) -> WeatherCompensationLayer:
    return WeatherCompensationLayer(
        weather_comp=WeatherCompensationCalculator(heating_type="radiator"),
        climate_system=AdaptiveClimateSystem(latitude=59.33),
        weather_learner=None,
        weather_comp_weight=owner_weight,
    )


def _weight_at(owner_weight: float, dm: float) -> float:
    decision = _layer(owner_weight).evaluate_layer(
        nibe_state=_State(dm),
        weather_data=_Weather(),
        target_temp=21.0,
        enable_weather_compensation=True,
    )
    return decision.weight


TIERS = [
    (WEATHER_COMP_DEFER_DM_LIGHT, WEATHER_COMP_DEFER_RETAIN_LIGHT, "light"),
    (WEATHER_COMP_DEFER_DM_MODERATE, WEATHER_COMP_DEFER_RETAIN_MODERATE, "moderate"),
    (WEATHER_COMP_DEFER_DM_SIGNIFICANT, WEATHER_COMP_DEFER_RETAIN_SIGNIFICANT, "significant"),
    (WEATHER_COMP_DEFER_DM_CRITICAL, WEATHER_COMP_DEFER_RETAIN_CRITICAL, "critical"),
]


@pytest.mark.parametrize("threshold,retained,name", TIERS)
def test_a_raised_weather_weight_is_scaled_not_discarded(threshold, retained, name):
    """Doubling the owner's weight must roughly double the deferred weight."""
    dm = threshold - 10  # just past the tier boundary
    low = _weight_at(0.4, dm)
    high = _weight_at(0.8, dm)
    assert high > low, (
        f"{name} debt: an owner who set weather_compensation_weight to 0.8 got the same "
        f"weight ({high:.3f}) as one who set 0.4 ({low:.3f}) - the deferral has stopped "
        "being a reduction of their setting and become a replacement for it."
    )
    assert high == pytest.approx(low * 2.0, rel=1e-6), (
        f"{name} debt: the deferral must scale the owner's weight, so 0.8 should give "
        f"exactly twice 0.4. Got {high:.4f} vs {low:.4f}."
    )


@pytest.mark.parametrize("threshold,retained,name", TIERS)
def test_the_default_owner_sees_the_same_weight_as_before(threshold, retained, name):
    """Restating the constants must not retune anybody's install.

    The RETAIN_* fractions are derived from the per-tier default weights rather than
    hand-rounded, so a default owner lands on exactly 0.45/0.41/0.36/0.30 as before. A
    rounded 0.61 would have shifted every install by about 0.2%.
    """
    dm = threshold - 10
    undeferred = _weight_at(DEFAULT_WEATHER_COMPENSATION_WEIGHT, 0.0)
    deferred = _weight_at(DEFAULT_WEATHER_COMPENSATION_WEIGHT, dm)
    assert deferred == pytest.approx(undeferred * retained, rel=1e-6), (
        f"{name} debt: expected the undeferred weight {undeferred:.4f} scaled by "
        f"{retained}, got {deferred:.4f}"
    )


def test_deeper_debt_defers_harder():
    """Monotonic: the deeper the debt, the less say the outdoor temperature has."""
    weights = [
        _weight_at(DEFAULT_WEATHER_COMPENSATION_WEIGHT, threshold - 10)
        for threshold, _, _ in TIERS
    ]
    assert weights == sorted(weights, reverse=True), (
        f"deferral is not monotonic across the four tiers: {weights}"
    )


def test_no_debt_means_no_deferral():
    """At DM 0 the layer keeps the owner's full configured influence."""
    for owner_weight in (0.3, 0.49, 0.8):
        decision = _layer(owner_weight).evaluate_layer(
            nibe_state=_State(0.0),
            weather_data=_Weather(),
            target_temp=21.0,
            enable_weather_compensation=True,
        )
        assert decision.defer_factor == 1.0


def test_the_retained_fractions_are_fractions():
    """A retained fraction above 1.0 would AMPLIFY the layer during thermal debt."""
    for _, retained, name in TIERS:
        assert 0.0 < retained < 1.0, f"{name}: {retained} is not a reduction"
