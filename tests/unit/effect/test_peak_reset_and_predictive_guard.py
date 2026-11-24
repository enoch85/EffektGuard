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
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    CONF_POWER_SENSOR_ENTITY,
    DAYTIME_START_HOUR,
    DOMAIN,
    EFFECT_OFFSET_PREDICTIVE,
    EFFECT_WEIGHT_PREDICTIVE,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

DAYTIME_PERIOD = (DAYTIME_START_HOUR + 1) * 4  # 07:00 quarter - avoids night weighting

OCTOBER = datetime(2025, 10, 20, 7, 0)
NOVEMBER = datetime(2025, 11, 3, 7, 0)

STOCKHOLM = "Europe/Stockholm"
METER = "sensor.house_power"
# The first quarter observed is partial and never recorded, so the evening starts a quarter early
OCTOBER_EVENING = datetime(2026, 10, 31, 23, 30, 10, tzinfo=ZoneInfo(STOCKHOLM))
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
CYCLES_TO_CLOSE_NOVEMBERS_FIRST_QUARTER = 10  # 23:30:10 to 00:15:10
OCTOBER_EVENING_KW = 20.0
NIGHT_KW = 2.0


def _pump(now: datetime) -> NibeState:
    return NibeState(
        outdoor_temp=0.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=now,
    )


# A house cooling fast enough to trigger the predictive power-increase branch.
COOLING_TREND = {"trend": "cooling", "rate_per_hour": -0.5, "confidence": 1.0}
BILLING_PERIOD_NOW = (
    "custom_components.effektguard.optimization.effect_layer.get_current_billing_period"
)


class TestMonthlyPeaksReset:
    @pytest.mark.asyncio
    async def test_last_months_peaks_do_not_survive_into_this_month(self, hass, monkeypatch):
        """An instance up across a month boundary carried October into November."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(6.0, DAYTIME_PERIOD, OCTOBER)
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

    async def test_last_months_final_quarter_stays_out_of_this_month(self, hass, freezer):
        """A month's last quarter closes at the first reading of the next, after the rollover.

        Recorded then, October's 23:45 quarter (20 kW, 10.0 effective at night) was November's
        highest peak, over November's own first quarter at 1.0, until the next restart. Real
        control cycles across midnight, through the house as Home Assistant builds it.
        """
        await hass.config.async_set_time_zone(STOCKHOLM)
        freezer.move_to(OCTOBER_EVENING)
        entry = MockConfigEntry(domain=DOMAIN, data={CONF_POWER_SENSOR_ENTITY: METER}, options={})
        entry.add_to_hass(hass)
        coordinator = await _create_coordinator(hass, entry)
        coordinator.gespot.get_prices = AsyncMock(return_value=None)
        coordinator.weather.get_forecast = AsyncMock(return_value=None)
        coordinator._schedule_aligned_refresh = MagicMock()

        for cycle in range(CYCLES_TO_CLOSE_NOVEMBERS_FIRST_QUARTER):
            now = OCTOBER_EVENING + cycle * CYCLE
            freezer.move_to(now)
            power = OCTOBER_EVENING_KW if dt_util.as_local(now).month == OCTOBER.month else NIGHT_KW
            hass.states.async_set(METER, str(power), {"unit_of_measurement": "kW"})
            coordinator.nibe.get_current_state = AsyncMock(return_value=_pump(now))
            await coordinator._async_update_data()

        summary = coordinator.effect.get_monthly_peak_summary()
        months = {dt_util.parse_datetime(peak["timestamp"]).month for peak in summary["peaks"]}
        assert months == {NOVEMBER.month}, summary

    @pytest.mark.asyncio
    async def test_this_months_peaks_are_kept(self, hass, monkeypatch):
        """Do not over-correct: pruning must not eat the current month."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(6.0, DAYTIME_PERIOD, NOVEMBER)

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

        await effect.record_period_measurement(6.0, DAYTIME_PERIOD, OCTOBER)
        event = await effect.record_period_measurement(
            2.0,
            DAYTIME_PERIOD + 4,  # one hour later, still daytime
            OCTOBER + timedelta(days=1),
        )

        # The second, SMALLER hour (on its own day) still returns a PeakEvent (top-3 not full).
        assert event is not None
        assert event.effective_power == pytest.approx(2.0)

        # Which is exactly why assigning it to peak_this_month was wrong.
        assert effect.get_monthly_peak_summary()["highest"] == pytest.approx(6.0)


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
    async def test_predictive_still_fires_once_a_peak_exists(self, hass, monkeypatch):
        """Do not over-correct: with real history the predictive branch must still work."""
        effect = EffectManager(hass)
        await effect.record_period_measurement(3.0, DAYTIME_PERIOD, OCTOBER)
        monkeypatch.setattr(BILLING_PERIOD_NOW, lambda: DAYTIME_PERIOD)

        decision = effect.evaluate_layer(
            current_peak=3.0,
            current_power=2.5,  # +1.5 kW predicted increase -> margin < 1.0 kW
            thermal_trend=COOLING_TREND,
            enable_peak_protection=True,
        )

        assert decision.offset == pytest.approx(EFFECT_OFFSET_PREDICTIVE)
        assert decision.weight == pytest.approx(EFFECT_WEIGHT_PREDICTIVE)
