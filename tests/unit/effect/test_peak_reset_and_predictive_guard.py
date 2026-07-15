"""Three effect-tariff invariants, each a case of acting on a number that did not mean what it said.

* Monthly peaks must reset on a month boundary, not only at startup: `_clean_old_peaks()` was
  reachable only from `async_load()`, so an instance up across 1 November carried October's top-3
  into November (protection threshold and sensors a month stale).
* `peak_this_month` must track the HIGHEST peak, not the latest: `record_period_measurement()`
  returns a PeakEvent for any entry while the top-3 fills, so assigning its power dropped a 6.0 kW
  peak to a later 2.0 kW one.
* The predictive branch must ABSTAIN with no peak history: current_peak 0.0 makes
  `predicted_margin` always negative, which voted -1.5 C at weight 0.85 - above T1 (0.65) and
  T2 (0.81) thermal-debt recovery.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    DAYTIME_START_HOUR,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_WEIGHT_PREDICTIVE,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

DAYTIME_HOUR = DAYTIME_START_HOUR + 1  # 07:00 - avoids the 50% night weighting

OCTOBER = datetime(2025, 10, 20, 7, 0)
NOVEMBER = datetime(2025, 11, 3, 7, 0)

# A house cooling fast enough to trigger the predictive power-increase branch.
COOLING_TREND = {"trend": "cooling", "rate_per_hour": -0.5, "confidence": 1.0}


class TestMonthlyPeaksReset:
    @pytest.mark.asyncio
    async def test_last_months_peaks_do_not_survive_into_this_month(self, hass, monkeypatch):
        """An instance up across a month boundary carried October into November."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(6.0, DAYTIME_HOUR, OCTOBER)
        assert effect.get_monthly_peak_summary()["count"] == 1

        # Time moves into November. This is what the coordinator now calls on month change.
        monkeypatch.setattr(
            "custom_components.effektguard.optimization.effect_layer.dt_util.now",
            lambda: NOVEMBER,
        )
        effect.prune_peaks_for_current_month()

        summary = effect.get_monthly_peak_summary()
        assert summary["count"] == 0, (
            "October's peaks survived into November. The effect tariff bills a MONTHLY peak, "
            "so the protection threshold and the peak sensor would be a month stale."
        )
        assert summary["highest"] == 0.0

    @pytest.mark.asyncio
    async def test_this_months_peaks_are_kept(self, hass, monkeypatch):
        """Do not over-correct: pruning must not eat the current month."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(6.0, DAYTIME_HOUR, NOVEMBER)

        monkeypatch.setattr(
            "custom_components.effektguard.optimization.effect_layer.dt_util.now",
            lambda: NOVEMBER,
        )
        effect.prune_peaks_for_current_month()

        assert effect.get_monthly_peak_summary()["count"] == 1


class TestMonthlyPeakIsTheHighest:
    @pytest.mark.asyncio
    async def test_summary_reports_the_highest_not_the_latest(self, hass):
        """The coordinator must read `highest`, not the returned PeakEvent."""
        effect = EffectManager(hass)

        await effect.record_period_measurement(6.0, DAYTIME_HOUR, OCTOBER)
        event = await effect.record_period_measurement(
            2.0, DAYTIME_HOUR + 4, OCTOBER + timedelta(days=1)
        )

        # The second, SMALLER hour (on its own day) still returns a PeakEvent (top-3 not full).
        assert event is not None
        assert event.effective_power == pytest.approx(2.0)

        # Which is exactly why assigning it to peak_this_month was wrong.
        assert effect.get_monthly_peak_summary()["highest"] == pytest.approx(6.0)

    def test_coordinator_reads_the_summary_not_the_event(self):
        """Regression guard on the coordinator's assignment."""
        import inspect

        from custom_components.effektguard.coordinator import EffektGuardCoordinator

        src = inspect.getsource(EffektGuardCoordinator._update_peak_tracking)

        assert "self.peak_this_month = peak_event.effective_power" not in src, (
            "peak_this_month is being set to the LATEST peak. A 6.0 kW peak followed by a "
            "2.0 kW quarter would silently drop the monthly peak to 2.0 kW."
        )
        assert 'get_monthly_peak_summary()["highest"]' in src


class TestPredictiveBranchNeedsAPeakHistory:
    def test_no_peak_history_means_no_heat_reducing_vote(self, hass):
        """On a fresh install the layer must ABSTAIN, not vote -1.5 @ 0.85."""
        effect = EffectManager(hass)  # no peaks recorded at all

        decision = effect.evaluate_layer(
            current_peak=0.0,
            current_power=2.0,
            thermal_trend=COOLING_TREND,
            enable_peak_protection=True,
        )

        assert decision.offset != pytest.approx(EFFECT_OFFSET_PREDICTIVE), (
            f"With no peak history the effect layer voted {decision.offset:+.1f} C at weight "
            f"{decision.weight} - because current_peak is 0.0, so predicted_margin is always "
            "negative. Weight 0.85 outranks T1 (0.65) and T2 (0.81) thermal-debt recovery."
        )
        assert decision.weight < EFFECT_WEIGHT_PREDICTIVE
        assert decision.offset >= 0.0, "Missing input must never produce a heat-reducing vote"

    @pytest.mark.asyncio
    async def test_predictive_still_fires_once_a_peak_exists(self, hass):
        """Do not over-correct: with real history the predictive branch must still work."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(3.0, DAYTIME_HOUR, OCTOBER)

        decision = effect.evaluate_layer(
            current_peak=3.0,
            current_power=2.5,  # +1.5 kW predicted increase -> margin < 1.0 kW
            thermal_trend=COOLING_TREND,
            enable_peak_protection=True,
        )

        assert decision.offset == pytest.approx(EFFECT_OFFSET_PREDICTIVE)
        assert decision.weight == pytest.approx(EFFECT_WEIGHT_PREDICTIVE)
