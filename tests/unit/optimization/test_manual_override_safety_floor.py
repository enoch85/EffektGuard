"""A user command is authoritative - but not below the absolute safety floor.

`force_offset` and `boost_heating` previously returned from calculate_decision BEFORE the
safety layer, the emergency thermal-debt layer, and the anti-windup flag were computed, and
the coordinator explicitly bypassed the offset-volatility blocker for manual decisions. So
`force_offset(-10)` for 6 hours would hold maximum heat REDUCTION while the house fell below
MIN_TEMP_LIMIT, or while DM sat past DM_THRESHOLD_AUX_LIMIT with the immersion heater running.

The fix applies the floor as a FLOOR, not a replacement: a user asking for MORE heat than
safety requires is passed through untouched (boost_heating(+10) still boosts); a command that
would leave the system below the safety floor is raised to it.
"""

from dataclasses import replace
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DM_THRESHOLD_AUX_LIMIT,
    MIN_OFFSET,
    MIN_TEMP_LIMIT,
    SAFETY_EMERGENCY_OFFSET,
    UPDATE_INTERVAL_MINUTES,
)
from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    OptimizationDecision,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.utils.volatile_helpers import (
    SECONDS_PER_MINUTE,
    OffsetVolatilityTracker,
    gate_offset,
)

STOCKHOLM_LATITUDE = 59.33
# A PEAK-layer cut on the register, and the floor the engine lifts it to at the aux limit
# (TestAbsoluteSafetyFloor): a reversal of 6 C toward zero, which the gate holds for 45 minutes
# unless the decision is flagged.
PEAK_CUT = -6.0
SAFETY_FLOOR_AT_AUX_LIMIT = 0.0


@pytest.fixture
def engine():
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
        config={
            "target_indoor_temp": 21.0,
            "tolerance": 0.5,
            "latitude": STOCKHOLM_LATITUDE,
        },
    )


def state(indoor_temp: float = 21.0, degree_minutes: float = -100.0) -> NibeState:
    return NibeState(
        outdoor_temp=-10.0,
        indoor_temp=indoor_temp,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 6, 0),
    )


def decide(engine: DecisionEngine, nibe_state: NibeState):
    """calculate_decision on the manual-override path.

    The override branch returns before any price/weather layer runs, so None inputs are
    safe here and keep the test deterministic.
    """
    return engine.calculate_decision(
        nibe_state=nibe_state,
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=0.0,
    )


class TestManualOverrideRespectsAbsoluteSafetyFloor:
    def test_force_offset_cannot_hold_the_house_below_min_temp_limit(self, engine):
        """force_offset(-10) while indoor is below MIN_TEMP_LIMIT must be raised."""
        engine.set_manual_override(MIN_OFFSET, duration_minutes=360)

        decision = decide(engine, state(indoor_temp=MIN_TEMP_LIMIT - 1.0))

        assert decision.offset == pytest.approx(SAFETY_EMERGENCY_OFFSET), (
            f"Manual override held {decision.offset:+.1f}°C while indoor was below "
            f"{MIN_TEMP_LIMIT}°C. The safety floor must outrank a user command."
        )
        assert decision.is_emergency is True

    def test_force_offset_cannot_hold_dm_past_the_aux_limit(self, engine):
        """force_offset(-10) while DM is past the aux limit is raised to the no-cut floor, 0."""
        engine.set_manual_override(MIN_OFFSET, duration_minutes=360)

        decision = decide(engine, state(degree_minutes=DM_THRESHOLD_AUX_LIMIT - 20))

        assert decision.offset == pytest.approx(0.0), (
            f"Manual override held {decision.offset:+.1f}°C at DM "
            f"{DM_THRESHOLD_AUX_LIMIT - 20}. Past the aux limit the immersion heater is "
            "running; reducing heat deepens the debt."
        )
        assert decision.is_emergency is True

    def test_boost_heating_is_passed_through_untouched(self, engine):
        """The floor only ever RAISES. A user asking for more heat still gets it."""
        engine.set_manual_override(SAFETY_EMERGENCY_OFFSET, duration_minutes=360)

        decision = decide(engine, state())

        assert decision.offset == pytest.approx(SAFETY_EMERGENCY_OFFSET)
        assert decision.is_manual_override is True
        assert decision.is_emergency is False

    def test_normal_manual_reduction_is_honoured_when_safe(self, engine):
        """With the house warm and DM healthy, a user reduction is a preference, not a fault."""
        engine.set_manual_override(-3.0, duration_minutes=60)

        decision = decide(engine, state(indoor_temp=22.0, degree_minutes=-100.0))

        assert decision.offset == pytest.approx(-3.0)
        assert decision.is_manual_override is True
        assert decision.is_emergency is False


class TestAbsoluteSafetyFloor:
    def test_floor_is_none_under_normal_conditions(self, engine):
        assert engine._absolute_safety_floor(state()) is None

    def test_floor_engages_below_min_temp_limit(self, engine):
        floor = engine._absolute_safety_floor(state(indoor_temp=MIN_TEMP_LIMIT - 0.1))
        assert floor == pytest.approx(SAFETY_EMERGENCY_OFFSET)

    def test_floor_engages_at_the_aux_limit(self, engine):
        floor = engine._absolute_safety_floor(state(degree_minutes=DM_THRESHOLD_AUX_LIMIT))
        assert floor == pytest.approx(0.0)


class TestVolatilityBlockerBypassesEmergency:
    """The coordinator must not defer an emergency for 45 minutes."""

    def test_an_emergency_goes_past_the_tracker_inside_its_window(self):
        """Regression guard for the offset-volatility blocker.

        Pre-fix the blocker bypassed only `is_manual_override` and `anti_windup_active`. An
        aux-limit emergency sets neither, so lifting a -6.0 PEAK offset there was a "volatile
        reversal" and got deferred for up to 45 minutes while DM kept falling. The policy is
        `gate_offset`, which the coordinator and the simulator both run; the same lift without the
        flag is held, which is what makes the flag observable.
        """
        clock = {"now": 0.0}
        tracker = OffsetVolatilityTracker(time_source=lambda: clock["now"])
        tracker.record_change(PEAK_CUT, "peak")
        clock["now"] += UPDATE_INTERVAL_MINUTES * SECONDS_PER_MINUTE

        lift = OptimizationDecision(offset=SAFETY_FLOOR_AT_AUX_LIMIT, reasoning="lift")
        assert gate_offset(tracker, lift).blocked, "precondition: an automatic lift is held"
        assert tracker.last_offset == PEAK_CUT

        gated = gate_offset(tracker, replace(lift, is_emergency=True))

        assert not gated.blocked, "an aux-limit recovery was deferred behind a price cut"
        assert gated.offset == pytest.approx(SAFETY_FLOOR_AT_AUX_LIMIT)
        assert tracker.last_offset == pytest.approx(
            SAFETY_FLOOR_AT_AUX_LIMIT
        ), "the recovery must become the tracker's baseline, or the next cycle re-fights it"
