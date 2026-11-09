"""The predictive peak branch judges a period as the bill does, as the immediate check judges it.

should_limit_power weights a period by its tariff (night at half) and compares it with the lowest of
the month's top three peaks, the threshold a period must stay under. The predictive branch beside it
subtracted the unweighted predicted power from the month's highest peak, so at night it voted for a
period the bill halves, and with three peaks it measured against the wrong one.
"""

from datetime import datetime, timedelta

import pytest

from custom_components.effektguard.const import (
    DAYTIME_START_HOUR,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD,
    THERMAL_CHANGE_MODERATE_COOLING,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

DAYTIME_PERIOD = (DAYTIME_START_HOUR + 1) * 4  # 07:00 quarter - avoids night weighting
NIGHT_PERIOD = (DAYTIME_START_HOUR - 1) * 4  # 05:00 quarter - billed at half
OCTOBER = datetime(2025, 10, 20, 7, 0)

# Cooling between the rapid and the moderate thresholds: the smaller predicted increase.
MODERATE_COOLING_TREND = {
    "trend": "cooling",
    "rate_per_hour": (EFFECT_PREDICTIVE_RAPID_COOLING_THRESHOLD + THERMAL_CHANGE_MODERATE_COOLING)
    / 2,
    "confidence": 1.0,
}
BILLING_PERIOD_NOW = (
    "custom_components.effektguard.optimization.effect_layer.get_current_billing_period"
)


@pytest.mark.parametrize(
    ("period", "peaks", "votes"),
    [
        pytest.param(DAYTIME_PERIOD, [4.0], True, id="day"),
        pytest.param(NIGHT_PERIOD, [4.0], False, id="night"),
        pytest.param(DAYTIME_PERIOD, [6.0, 5.0, 4.0], True, id="the lowest of three peaks"),
    ],
)
@pytest.mark.asyncio
async def test_the_prediction_is_billed_as_its_period_is(hass, monkeypatch, period, peaks, votes):
    """3.0 kW heading for 3.5 kW is billed 3.5 by day and 1.75 by night, against the 4.0 kW a
    period must stay under. Compared unweighted, the night period voted -1.5 at weight 0.85 too;
    compared with the month's highest peak of three, the day period heading within a kilowatt of
    the lowest did not vote. The vote names the peak it was measured against."""
    effect = EffectManager(hass)
    for day, peak in enumerate(peaks):
        await effect.record_period_measurement(peak, DAYTIME_PERIOD, OCTOBER + timedelta(days=day))
    monkeypatch.setattr(BILLING_PERIOD_NOW, lambda: period)

    decision = effect.evaluate_layer(
        current_peak=max(peaks),
        current_power=3.0,
        thermal_trend=MODERATE_COOLING_TREND,
        enable_peak_protection=True,
    )

    assert (decision.offset == pytest.approx(EFFECT_OFFSET_PREDICTIVE)) == votes, decision
    if votes:
        assert "heading for the 4.0 kW peak" in decision.reason, decision.reason
