"""Five tests asserted a literal against itself and called no production code at all.

    max_offset_change_per_update = 3.0  # °C
    assert max_offset_change_per_update <= 3.0

    min_write_interval_seconds = 300  # 5 minutes minimum
    assert min_write_interval_seconds >= 300

    startup_delay_seconds = 10
    assert startup_delay_seconds >= 10

    required_forecast_hours = 12
    assert required_forecast_hours >= 12

    max_offset_change = 3.0
    assert max_offset_change <= 3.0

Every one of them binds a local to a number and then asserts that number against itself. They
cannot fail. One of them even says so out loud - "This limit is enforced by decision engine
aggregation" - and then tests nothing at all. They are named for real safety properties: thermal
shock, compressor wear, NIBE controller wear, startup ordering.

AND THE PROPERTY TWO OF THEM CLAIM IS FALSE. The engine does NOT bound its offset change to 3.0 °C
per update. Measured across five houses and 31 days of real weather and real prices, the largest
jump between consecutive decisions is 4.41 °C - and the trace is sampled every 30 minutes, so the
true 5-minute jump is larger still.

THAT IS NOT A BUG, AND IT MUST NOT BE "FIXED". A per-update magnitude limit would rate-limit the
EMERGENCY response, which has to be able to go from 0 to +10 in a single cycle when degree minutes
reach the auxiliary-heat limit. Deferring that for even one cycle is the death spiral the
anti-windup work exists to prevent. So this file asserts what is ACTUALLY true and load-bearing -
the register bounds, the write rate limit, and that the emergency path is deliberately exempt -
rather than inventing a limit that would make the pump less safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    LAYER_WEIGHT_SAFETY,
    MAX_OFFSET,
    MIN_OFFSET,
    SAFETY_EMERGENCY_OFFSET,
    SERVICE_RATE_LIMIT_MINUTES,
    UPDATE_INTERVAL_MINUTES,
    WEATHER_FORECAST_HORIZON,
)
from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
    SAFETY_LAYER_NAME,
)


def _adapter() -> NibeAdapter:
    hass = MagicMock()
    state = MagicMock()
    state.state = "0"
    state.attributes = {}
    hass.states.get.return_value = state
    hass.services.async_call = AsyncMock()

    adapter = NibeAdapter(hass, {"nibe_entity": "number.offset"})
    adapter._entity_cache = {"offset": "number.offset"}
    return adapter


def _engine() -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={"target_indoor_temp": 21.0, "tolerance": 0.5},
    )


class TestTheWriteRateLimitActuallyRefuses:
    """The old test asserted `300 >= 300` and never touched the adapter."""

    @pytest.mark.asyncio
    async def test_a_second_write_inside_the_cooldown_is_refused(self):
        adapter = _adapter()

        first = await adapter.set_curve_offset(-3.0)
        immediately_after = await adapter.set_curve_offset(3.0)

        assert first == -3, "precondition: the first write must land"
        assert immediately_after is None, (
            f"A second write was accepted immediately after the first. The cooldown is "
            f"SERVICE_RATE_LIMIT_MINUTES ({SERVICE_RATE_LIMIT_MINUTES} min), and it exists to stop "
            f"the NIBE controller being rewritten every cycle. The test that used to guard this "
            f"asserted `300 >= 300` and never called the adapter."
        )

    @pytest.mark.asyncio
    async def test_a_write_after_the_cooldown_is_accepted(self):
        """The regression guard on the guard: the rate limit must not become a permanent block."""
        adapter = _adapter()

        assert await adapter.set_curve_offset(-3.0) == -3

        adapter._last_write = adapter._last_write - timedelta(
            minutes=SERVICE_RATE_LIMIT_MINUTES + 1
        )

        assert await adapter.set_curve_offset(3.0) == 3

    def test_the_cooldown_is_at_least_one_update_cycle(self):
        """A cooldown shorter than the update interval would not rate-limit anything."""
        assert SERVICE_RATE_LIMIT_MINUTES >= UPDATE_INTERVAL_MINUTES, (
            f"The write cooldown ({SERVICE_RATE_LIMIT_MINUTES} min) is shorter than the coordinator's "
            f"own update interval ({UPDATE_INTERVAL_MINUTES} min), so it can never actually refuse a "
            f"scheduled write and the wear protection is decorative."
        )


class TestTheRegisterBoundsAreTheRealLimit:
    """There is no per-update magnitude limit, and there must not be. This is what bounds it."""

    @pytest.mark.parametrize("wild", [-99.0, -10.5, 10.5, 99.0])
    def test_no_layer_can_drive_the_offset_outside_the_register(self, wild):
        engine = _engine()
        layers = [LayerDecision(name="Rogue", offset=wild, weight=1.0, reason="")]

        offset = engine._aggregate_layers(layers)

        assert MIN_OFFSET <= offset <= MAX_OFFSET, (
            f"A layer voting {wild:+.1f} produced a final offset of {offset:+.1f}, outside the "
            f"[{MIN_OFFSET}, {MAX_OFFSET}] the NIBE register can hold."
        )


class TestTheEmergencyPathIsDeliberatelyExemptFromSmoothing:
    """Why no per-update magnitude limit exists. Do not add one.

    The two deleted tests asserted the engine never moves more than 3.0 °C in one update. It does -
    4.41 °C measured over 31 days. Enforcing 3.0 would rate-limit the response below, and degree
    minutes at the auxiliary-heat limit cannot wait three cycles for full heat.
    """

    def test_the_safety_layer_reaches_full_heat_in_a_single_update(self):
        engine = _engine()
        layers = [
            LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=SAFETY_EMERGENCY_OFFSET,
                weight=LAYER_WEIGHT_SAFETY,
                reason="Indoor below the floor",
            ),
        ]

        offset = engine._aggregate_layers(layers)

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET), (
            f"The safety layer asked for {SAFETY_EMERGENCY_OFFSET:+.1f} and the engine emitted "
            f"{offset:+.1f}. A per-update magnitude limit would throttle exactly this - the house "
            f"is below its absolute floor, and it cannot wait three cycles for full heat."
        )

    def test_a_jump_from_zero_to_full_heat_is_more_than_the_deleted_tests_allowed(self):
        """Stated explicitly so nobody 'restores' the 3.0 limit and breaks the emergency path."""
        jump = abs(SAFETY_EMERGENCY_OFFSET - 0.0)

        assert jump > 3.0, (
            f"The emergency response is a {jump:.0f} °C jump in one update. The deleted tests "
            f"asserted the engine never moves more than 3.0 °C per update. Both cannot be true, "
            f"and it is the emergency response that has to win."
        )


def test_the_forecast_horizon_is_a_real_constant_not_a_number_in_a_test():
    """The old test bound `required_forecast_hours = 12` and asserted `12 >= 12`."""
    assert WEATHER_FORECAST_HORIZON >= 12.0, (
        f"The forecast horizon is {WEATHER_FORECAST_HORIZON} h. Pre-heating decisions need at "
        f"least 12 h of look-ahead; below that the pre-heat cannot see the cold coming."
    )
