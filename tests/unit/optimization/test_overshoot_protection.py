"""Overshoot protection: the ComfortLayer coasts a warm house, never a cold one, and never in a jump.

Drives the real ComfortLayer.evaluate_layer. Above its band the vote ramps continuously from the
in-band correction at the band edge to OVERSHOOT_PROTECTION_OFFSET_MAX at
LAYER_WEIGHT_COMFORT_CRITICAL at OVERSHOOT_PROTECTION_FULL, whatever the prices ahead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    COMFORT_CORRECTION_MULT,
    DEFAULT_TOLERANCE,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_MIN,
    LAYER_WEIGHT_SAFETY,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OPTIMIZATION_MODE_COMFORT,
    OPTIMIZATION_MODE_SAVINGS,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
    PRICE_TOLERANCE_MAX,
    PRICE_TOLERANCE_MIN,
    TOLERANCE_RANGE_MULTIPLIER,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TARGET = 21.0
STEP = 0.01
NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
# Every mode at the slider's minimum (the default), the owner's 1.2 and the slider's maximum.
CASES = [
    (mode, slider)
    for mode in (OPTIMIZATION_MODE_COMFORT, OPTIMIZATION_MODE_BALANCED, OPTIMIZATION_MODE_SAVINGS)
    for slider in (PRICE_TOLERANCE_MIN, 1.2, PRICE_TOLERANCE_MAX)
]


def _house_at(indoor: float) -> NibeState:
    return NibeState(
        outdoor_temp=0.0,
        indoor_temp=indoor,
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=-50.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=NOW,
        indoor_temp_valid=True,
    )


def _decide(
    overshoot: float, mode: str = OPTIMIZATION_MODE_BALANCED, slider: float = DEFAULT_TOLERANCE
):
    """Drive the REAL layer with the house `overshoot` degrees above target."""
    layer = ComfortLayer(
        target_temp=TARGET,
        tolerance_range=slider * TOLERANCE_RANGE_MULTIPLIER,
        mode_config=MODE_CONFIGS[mode],
    )
    return layer.evaluate_layer(_house_at(TARGET + overshoot))


def _ramp(mode: str, slider: float) -> tuple[float, float, float]:
    """Where the ramp starts (the band edge, or a wider dead zone), and its vote there."""
    edge = max(slider * TOLERANCE_RANGE_MULTIPLIER, MODE_CONFIGS[mode].dead_zone)
    in_band_weight = LAYER_WEIGHT_COMFORT_MIN * MODE_CONFIGS[mode].comfort_weight_multiplier
    return edge, -edge * COMFORT_CORRECTION_MULT, in_band_weight


class TestTheRampHasNoCliff:
    """The vote used to jump to -7 C at weight 0.7 at 0.6 C above target."""

    @pytest.mark.parametrize("mode, slider", CASES)
    def test_the_ramp_starts_from_the_in_band_vote(self, mode, slider):
        edge, offset, weight = _ramp(mode, slider)

        vote = _decide(edge + 1e-6, mode, slider)  # just past it: 21.0 + 0.2 - 21.0 < 0.2

        assert (vote.offset, vote.weight) == pytest.approx((offset, weight), abs=1e-4), vote.reason

    @pytest.mark.parametrize("mode", list(MODE_CONFIGS))
    def test_a_house_exactly_on_the_band_edge_is_on_the_ramp(self, mode):
        """It used to fall through to "Too cold". 2.5 x 0.4 is exactly 1.0, as is 22.0 - 21.0."""
        edge, offset, weight = _ramp(mode, 2.5)

        vote = _decide(edge, mode, 2.5)

        assert (vote.offset, vote.weight) == pytest.approx((offset, weight)), vote.reason

    @pytest.mark.parametrize("mode, slider", CASES)
    def test_no_step_past_the_dead_zone_is_steeper_than_the_ramp(self, mode, slider):
        edge, offset, weight = _ramp(mode, slider)
        span = OVERSHOOT_PROTECTION_FULL - edge
        max_offset_step = (offset - OVERSHOOT_PROTECTION_OFFSET_MAX) * STEP / span + 1e-9
        max_weight_step = (LAYER_WEIGHT_COMFORT_CRITICAL - weight) * STEP / span + 1e-9
        start = MODE_CONFIGS[mode].dead_zone + STEP
        steps = round((OVERSHOOT_PROTECTION_FULL + 0.1 - start) / STEP)
        votes = [_decide(round(start + i * STEP, 2), mode, slider) for i in range(steps + 1)]

        for a, b in zip(votes, votes[1:]):
            assert abs(b.offset - a.offset) <= max_offset_step, f"{a.reason!r} -> {b.reason!r}"
            assert abs(b.weight - a.weight) <= max_weight_step, f"{a.reason!r} -> {b.reason!r}"

    @pytest.mark.parametrize("mode, slider", CASES)
    def test_the_response_is_monotonic_across_the_ramp(self, mode, slider):
        """More overshoot must never mean less coasting."""
        edge = _ramp(mode, slider)[0]
        votes = [_decide(edge + i * 0.1, mode, slider) for i in range(11)]

        offsets = [v.offset for v in votes]
        weights = [v.weight for v in votes]

        assert offsets == sorted(offsets, reverse=True), f"offsets not monotonic: {offsets}"
        assert weights == sorted(weights), f"weights not monotonic: {weights}"

    @pytest.mark.parametrize("mode, slider", CASES)
    def test_the_full_coast_wins_outright(self, mode, slider):
        """At and past FULL the vote is the full coast at a weight the aggregator obeys outright."""
        for overshoot in (OVERSHOOT_PROTECTION_FULL, OVERSHOOT_PROTECTION_FULL + 4.0):
            vote = _decide(overshoot, mode, slider)

            assert vote.offset == OVERSHOOT_PROTECTION_OFFSET_MAX, vote.reason
            assert vote.weight >= LAYER_WEIGHT_SAFETY, vote.reason


class TestTheVoteDoesNotReadPrices:
    """A pricier quarter in the look-ahead used to swap -8 C for a near-neutral vote."""

    @staticmethod
    def _prices(ahead: float) -> PriceData:
        """Today's quarters at 1.0, the two after the current one at `ahead`."""
        midnight = NOW.replace(hour=0)
        current = (NOW - midnight) // timedelta(minutes=15)
        return PriceData(
            today=[
                QuarterPeriod(
                    start_time=midnight + q * timedelta(minutes=15),
                    price=ahead if current < q <= current + 2 else 1.0,
                )
                for q in range(96)
            ],
            tomorrow=[],
            has_tomorrow=False,
        )

    def test_a_pricier_quarter_ahead_does_not_change_the_comfort_vote(self, monkeypatch):
        monkeypatch.setattr(dt_util, "now", lambda time_zone=None: NOW)
        engine = DecisionEngine(
            price_analyzer=PriceAnalyzer(),
            effect_manager=EffectManager(MagicMock()),
            thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
            config={
                "target_indoor_temp": TARGET,
                "tolerance": DEFAULT_TOLERANCE,
                "latitude": 59.33,
            },
        )
        house = _house_at(TARGET + 1.0)

        def comfort_vote(prices: PriceData):
            decision = engine.calculate_decision(
                nibe_state=house,
                price_data=prices,
                weather_data=None,
                current_peak=0.0,
                current_power=2.0,
            )
            vote = next(layer for layer in decision.layers if layer.name == "Comfort")
            return vote.offset, vote.weight

        flat = comfort_vote(self._prices(1.0))
        spike = comfort_vote(self._prices(2.0 * PRICE_FORECAST_EXPENSIVE_THRESHOLD))

        assert spike == flat, (
            f"A house 1.0 C above target got comfort vote {flat} with flat prices and {spike} "
            "with a pricier quarter ahead: a vote that changes with the look-ahead is a cliff."
        )


class TestOvershootProtectionOnlyFiresUpwards:
    """The regression guard. It coasts a warm house; it must never coast a cold one."""

    def test_a_cold_house_is_never_coasted(self):
        decision = _decide(-1.0)

        assert (
            decision.offset >= 0.0
        ), f"A house 1.0 C BELOW target was told to coast ({decision.offset:+.2f} C)."

    def test_a_house_on_target_is_left_alone(self):
        decision = _decide(0.0)

        assert decision.offset == 0.0
        assert decision.weight == 0.0
