"""The effect layer must receive INSTANTANEOUS power, never the daily peak.

`peak_today` is a daily high-water mark: it only ratchets upward until the midnight reset
(coordinator._update_peak_tracking). Feeding it to the decision engine as "current power"
meant one unrelated household spike - an oven, a kettle, an EV charger - pinned the effect
layer to CRITICAL (weight 1.0, offset -3.0 C) for the remainder of the day, regardless of
what the heat pump was actually drawing.

That is harmful on its own (it suppresses heating for up to 15 hours on a false premise),
and it was the trigger for the safety-priority inversion: a permanently-critical cost
layer is what crushed T1/T2 thermal-debt recovery.

Note on determinism: EffectManager weights night-time power at 50% (Swedish effect tariff)
and derives its threshold from the recorded monthly peaks - not from the `current_peak`
argument. Both tests below therefore pin an explicit DAYTIME quarter and seed the peak
list, rather than depending on the wall clock.
"""

import inspect
from datetime import datetime

import pytest

from custom_components.effektguard.const import DAYTIME_START_HOUR
from custom_components.effektguard.optimization.effect_layer import EffectManager

# A quarter safely inside the daytime band, so the 50% night weighting never applies.
DAYTIME_HOUR = DAYTIME_START_HOUR + 1  # 07:00

# Fixed instant - the effect layer's night/day weighting is wall-clock sensitive, so the
# test must never read the real clock.
FIXED_TIME = datetime(2026, 1, 15, 7, 0)

MONTHLY_PEAK_KW = 5.0
SPIKE_KW = 5.5  # oven + pump: exceeds the monthly peak
IDLE_KW = 0.3  # heat pump idling later the same day


async def _seeded_effect_manager(hass) -> EffectManager:
    """EffectManager with one recorded monthly peak of MONTHLY_PEAK_KW."""
    effect = EffectManager(hass)
    await effect.record_period_measurement(MONTHLY_PEAK_KW, DAYTIME_HOUR, FIXED_TIME)
    return effect


class TestEffectSeverityTracksInstantaneousPower:
    """A spent daily peak must not keep the effect layer critical.

    CHARACTERIZATION, not regression: these pass both before and after the coordinator fix,
    because EffectManager itself was always correct - it relaxes properly when handed real
    power. They exist to show WHY feeding it `peak_today` was harmful. The actual
    regression guard is TestCoordinatorPowerContract below, which fails on the old code.
    """

    @pytest.mark.asyncio
    async def test_idle_pump_after_a_morning_spike_is_not_critical(self, hass):
        """The F-047 scenario.

        07:00 an oven pushes the house to 5.5 kW against a 5.0 kW monthly peak -> CRITICAL.
        By 11:00 the pump idles at 0.3 kW.

        Fed `peak_today` (5.5) the effect layer stays CRITICAL all day.
        Fed instantaneous power (0.3) it must relax.
        """
        effect = await _seeded_effect_manager(hass)

        spike = effect.should_limit_power(SPIKE_KW, DAYTIME_HOUR)
        assert spike.severity == "CRITICAL", "5.5 kW against a 5.0 kW peak must be critical"

        idle = effect.should_limit_power(IDLE_KW, DAYTIME_HOUR)

        assert idle.severity == "OK", (
            f"Effect layer still {idle.severity} at {IDLE_KW} kW ({idle.reason}). "
            "It would be reacting to a spent daily maximum, not real consumption."
        )
        assert not idle.should_limit
        assert idle.recommended_offset == 0.0

    @pytest.mark.asyncio
    async def test_protection_returns_when_power_actually_rises(self, hass):
        """Relaxing on idle must not disable protection when demand genuinely returns."""
        effect = await _seeded_effect_manager(hass)

        assert effect.should_limit_power(IDLE_KW, DAYTIME_HOUR).severity == "OK"

        back_at_peak = effect.should_limit_power(SPIKE_KW, DAYTIME_HOUR)
        assert back_at_peak.severity == "CRITICAL"
        assert back_at_peak.should_limit


class TestCoordinatorPowerContract:
    """The coordinator must feed the engine live power, not the daily maximum."""

    def test_decision_path_does_not_consume_peak_today(self):
        """Guards against a future re-merge of the two concepts.

        `peak_today` is a daily maximum for display/diagnostics; `current_power_kw` is the
        live reading the effect layer consumes. They are different quantities and must not
        be aliased.
        """
        from custom_components.effektguard.coordinator import EffektGuardCoordinator

        update_src = inspect.getsource(EffektGuardCoordinator._read_and_decide)

        assert "current_power_for_decision = self.peak_today" not in update_src, (
            "The decision engine is being fed peak_today (a daily MAXIMUM) as current power. "
            "One morning spike would pin the effect layer to CRITICAL until midnight."
        )
        assert "projected_hour_mean" in update_src and "self.current_power_kw" in update_src, (
            "The decision engine must be fed the live reading PROJECTED over the billing hour "
            "- the monthly record it is compared against is an hourly mean, so an instantaneous "
            "spike is not the same quantity. See "
            "tests/unit/optimization/test_peak_protection_compares_like_with_like.py."
        )
