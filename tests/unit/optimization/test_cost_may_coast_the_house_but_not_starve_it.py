"""A cost layer may coast the house within its comfort band. It may not coast it out.

Using the band is the whole point of the integration - that is the thermal battery. But step 4 of
`_aggregate_layers` takes the critical layer's vote ALONE:

    critical_layers = [layer for layer in layers if layer.weight >= LAYER_WEIGHT_SAFETY]
    chosen = max_offset if abs(max_offset) >= abs(min_offset) else min_offset
    return self._clamp_offset(chosen)

With a price layer at PEAK (weight 1.0, offset -10.0) the price layer is BOTH the max and the min,
so the comfort layer never enters the sum at all - at any indoor temperature. Cost kept cutting heat
into a house that was already too cold, and nothing objected until the hard 18 C floor fired, three
degrees later.

NOTHING ELSE CAN SEE THIS. Degree minutes are blind to it by construction: DM = integral(BT25 - S1),
so lowering the curve lowers S1 and DM *improves* as the house gets colder. In the month-long
simulation the house sat 1.1 C below target with DM at -45 - a perfectly healthy number - while the
price layer held -3.0.

The month-long simulation, against a physically honest plant, put a number on it. Across five houses
the optimiser spent between 4 000 and 33 000 minutes below the comfort band, and a DO-NOTHING
controller held target on every one of them. `main` was worse than this branch on all five, so this
is long-standing, not a regression - but it means the optimiser was making the house colder than
switching it off would have.

So a cost layer's heat reduction is floored once the house is outside the band. The comfort layer's
own demand is the floor: it is already graduated by how far out the house is, and it is the only
layer that can see the problem at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    LAYER_WEIGHT_SAFETY,
    PRICE_OFFSET_PEAK,
    SAFETY_EMERGENCY_OFFSET,
)
from custom_components.effektguard.optimization.decision_engine import (
    COMFORT_LAYER_NAME,
    DecisionEngine,
    LayerDecision,
    SAFETY_LAYER_NAME,
)


def _engine() -> DecisionEngine:
    from unittest.mock import MagicMock

    return DecisionEngine(
        price_analyzer=MagicMock(),
        effect_manager=MagicMock(),
        thermal_model=MagicMock(),
        config={"target_indoor_temp": 21.0, "tolerance": 0.5},
    )


def _price_at_peak() -> LayerDecision:
    return LayerDecision(
        name="Spot Price",
        offset=PRICE_OFFSET_PEAK,
        weight=LAYER_WEIGHT_SAFETY,
        reason="PEAK quarter",
        is_cost_layer=True,
    )


def _comfort_wanting_heat(offset: float = 0.9) -> LayerDecision:
    return LayerDecision(
        name=COMFORT_LAYER_NAME,
        offset=offset,
        weight=0.5,
        reason="Too cold",
    )


class TestInsideTheBandCostIsFree:
    """The thermal battery. Do not break it while fixing the starvation."""

    def test_a_peak_quarter_may_coast_a_house_that_is_at_target(self):
        engine = _engine()
        layers = [
            _price_at_peak(),
            LayerDecision(name=COMFORT_LAYER_NAME, offset=0.0, weight=0.0, reason="At target"),
        ]

        offset = engine._aggregate_layers(layers, starvation=0.0)

        assert offset == pytest.approx(PRICE_OFFSET_PEAK), (
            f"A PEAK quarter with the house at target commanded {offset:+.2f} instead of "
            f"{PRICE_OFFSET_PEAK:+.2f}. Coasting a house that is AT target is the entire point of "
            f"the integration - the fix for starvation must not disable it."
        )

    def test_a_peak_quarter_may_coast_a_house_drifting_inside_the_band(self):
        """0.3 C below target with a 0.5 C tolerance: still inside the band. Cost may use it."""
        engine = _engine()
        layers = [_price_at_peak(), _comfort_wanting_heat(offset=0.2)]

        offset = engine._aggregate_layers(layers, starvation=0.0)

        assert offset == pytest.approx(PRICE_OFFSET_PEAK)


class TestOutsideTheBandCostMustYield:
    """The house is colder than the owner asked for. Money stops being the priority."""

    def test_a_peak_quarter_may_not_starve_a_house_below_its_band(self):
        engine = _engine()
        comfort = _comfort_wanting_heat(offset=0.9)
        layers = [_price_at_peak(), comfort]

        offset = engine._aggregate_layers(layers, starvation=1.0)

        assert offset >= comfort.offset, (
            f"The house is below its comfort band and the price layer commanded {offset:+.2f} C - "
            f"maximum heat reduction - while the comfort layer asked for {comfort.offset:+.2f} C. "
            f"Comfort never entered the sum: step 4 takes the critical layer's vote alone. Degree "
            f"minutes cannot object either, because lowering the curve makes DM look BETTER as the "
            f"house gets colder. Nothing would have stopped this until the 18 C floor."
        )

    @pytest.mark.parametrize("comfort_demand", [0.3, 0.9, 1.5, 3.0])
    def test_the_floor_is_the_comfort_layers_own_graduated_demand(self, comfort_demand):
        """Not a fixed number: the colder the house, the higher the floor."""
        engine = _engine()
        layers = [_price_at_peak(), _comfort_wanting_heat(offset=comfort_demand)]

        offset = engine._aggregate_layers(layers, starvation=1.0)

        assert offset == pytest.approx(comfort_demand)

    def test_cost_is_still_allowed_to_reduce_heat_below_what_comfort_asked_for_it_just_cannot_cut(
        self,
    ):
        """The floor never ADDS heat beyond comfort's request - it only stops the cut."""
        engine = _engine()
        layers = [_price_at_peak(), _comfort_wanting_heat(offset=0.9)]

        offset = engine._aggregate_layers(layers, starvation=1.0)

        assert offset <= 0.9, "the floor must not become a heat SOURCE"


class TestTheFloorNeverWeakensSafety:
    """It exists to bound COST. It must not touch a safety or physics vote."""

    def test_a_critical_safety_vote_is_untouched(self):
        """Safety at +10 must still win outright - the floor must not reduce it to comfort's ask."""
        engine = _engine()
        layers = [
            LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=SAFETY_EMERGENCY_OFFSET,
                weight=LAYER_WEIGHT_SAFETY,
                reason="Below floor",
            ),
            _comfort_wanting_heat(offset=0.9),
        ]

        offset = engine._aggregate_layers(layers, starvation=1.0)

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET)

    def test_a_safety_vote_alongside_a_cost_vote_still_wins_the_tie_break(self):
        """Safety +10 vs price -10 ties by construction; the safety-biased tie-break must hold."""
        engine = _engine()
        layers = [
            LayerDecision(
                name=SAFETY_LAYER_NAME,
                offset=SAFETY_EMERGENCY_OFFSET,
                weight=LAYER_WEIGHT_SAFETY,
                reason="Below floor",
            ),
            _price_at_peak(),
            _comfort_wanting_heat(offset=0.9),
        ]

        offset = engine._aggregate_layers(layers, starvation=1.0)

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET), (
            "With a non-cost layer also voting at critical weight, the tie-break already had a "
            "safety opinion to weigh and the comfort floor must not interfere with it."
        )

    def test_the_floor_only_engages_when_every_critical_layer_is_a_cost_layer(self):
        engine = _engine()
        assert engine._all_critical_are_cost([_price_at_peak()]) is True
        assert (
            engine._all_critical_are_cost(
                [
                    _price_at_peak(),
                    LayerDecision(
                        name=SAFETY_LAYER_NAME,
                        offset=10.0,
                        weight=LAYER_WEIGHT_SAFETY,
                        reason="",
                    ),
                ]
            )
            is False
        )


class TestTheFloorIsRampedNotSwitched:
    """A boolean on a temperature threshold is a bang-bang controller, and I shipped one.

    The first version of this floor was `below_comfort_band: bool`, evaluated at
    `target - tolerance_range`. AT that boundary the comfort layer is asking for nothing, so
    "floor at comfort" meant "jump to zero", and driving the real engine across it gave:

        indoor 20.80 C  ->  offset -10.00      cost layer free
        indoor 20.79 C  ->  offset  +0.01      floored at comfort

    A hundredth of a degree flipping the command by ten degrees. Real indoor sensors dither by more
    than that, so the house would sit on the boundary flipping the curve between its extremes -
    and every flip is a Modbus write to a real heat pump.

    The ramp fixes it by construction: at the boundary the floor IS the cost layer's own vote, so
    nothing moves, and it climbs to the comfort layer's demand as the house leaves the band the
    owner actually asked for.
    """

    def _sweep(self, engine, comfort_offset: float = 0.2):
        """The final offset as the house cools through the band, driving the real engine."""
        results = []
        for indoor in [21.00 - i * 0.01 for i in range(0, 61)]:
            nibe = MagicMock()
            nibe.indoor_temp = indoor
            nibe.indoor_temp_valid = True
            layers = [_price_at_peak(), _comfort_wanting_heat(offset=comfort_offset)]
            starvation = engine._starvation_fraction(nibe)
            results.append((indoor, engine._aggregate_layers(layers, starvation=starvation)))
        return results

    def test_no_hundredth_of_a_degree_moves_the_command_by_more_than_a_degree(self):
        """The defect, stated as the invariant it violates. It moved it by ten."""
        engine = _engine()
        sweep = self._sweep(engine)

        jumps = [
            (a_temp, b_temp, abs(b_off - a_off))
            for (a_temp, a_off), (b_temp, b_off) in zip(sweep, sweep[1:])
            if abs(b_off - a_off) > 1.0
        ]

        assert not jumps, (
            "The control law is discontinuous. A 0.01 C step in indoor temperature moves the "
            "commanded curve offset by: "
            + ", ".join(f"{d:.2f} C between {a:.2f} and {b:.2f}" for a, b, d in jumps)
            + ". A real indoor sensor dithers by more than 0.01 C, so the house sits on that "
            "boundary flipping the curve between its extremes, writing to the pump every cycle."
        )

    def test_at_the_inner_edge_the_cost_layer_is_still_free(self):
        """The thermal battery must not be narrowed by the ramp. Above the inner band, nothing."""
        engine = _engine()
        nibe = MagicMock()
        nibe.indoor_temp = engine.target_temp - engine.tolerance_range  # exactly the inner edge
        nibe.indoor_temp_valid = True

        assert engine._starvation_fraction(nibe) == 0.0
        assert engine._aggregate_layers(
            [_price_at_peak(), _comfort_wanting_heat(offset=0.2)],
            starvation=engine._starvation_fraction(nibe),
        ) == pytest.approx(PRICE_OFFSET_PEAK)

    def test_at_the_band_the_owner_asked_for_the_comfort_layer_has_the_floor(self):
        """The other end of the ramp. `tolerance` is the owner's own limit and it is honoured."""
        engine = _engine()
        nibe = MagicMock()
        nibe.indoor_temp = engine.target_temp - engine.tolerance  # 20.5 at the defaults
        nibe.indoor_temp_valid = True

        assert engine._starvation_fraction(nibe) == 1.0
        assert engine._aggregate_layers(
            [_price_at_peak(), _comfort_wanting_heat(offset=0.4)],
            starvation=engine._starvation_fraction(nibe),
        ) == pytest.approx(0.4)

    def test_the_ramp_is_monotone(self):
        """Colder house, higher floor. Never the reverse."""
        offsets = [offset for _, offset in self._sweep(_engine())]

        assert offsets == sorted(offsets), (
            "The floor must rise monotonically as the house cools. It does not: "
            f"{[round(o, 2) for o in offsets]}"
        )

    def test_an_invalid_indoor_reading_abstains(self):
        """Without a reading this cannot be measured, and nothing else can see it either."""
        engine = _engine()
        nibe = MagicMock()
        nibe.indoor_temp = 15.0  # would be deeply starved, if it were believable
        nibe.indoor_temp_valid = False

        assert engine._starvation_fraction(nibe) == 0.0
