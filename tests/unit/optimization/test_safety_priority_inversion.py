"""Safety-priority regression tests: cost must never override thermal-debt safety.

These tests encode the single most important invariant in EffektGuard:

    A cost layer (spot price, effect tariff) MUST NEVER be able to reduce heating
    while the emergency thermal-debt layer is actively recovering.

Every test here was written to FAIL against the pre-fix implementation, where the
decision aggregator reconstructed the emergency tier from layer *weights* and
*offset magnitudes* instead of reading the `tier` field it already carries. That
inference broke in four independent ways, each of which let cost win:

  1. DM <= DM_THRESHOLD_AUX_LIMIT emitted +10.0 at weight 1.0, but the aggregator's
     absolute-priority check only inspected the Safety layer, so the EMERGENCY tier
     fell through to the peak-aware compromise (+1.0) or the critical tie-break.
  2. The critical tie-break `abs(max) > abs(min)` returns `min` on an exact tie, and
     SAFETY_EMERGENCY_OFFSET (+10.0) vs PRICE_OFFSET_PEAK (-10.0) tie by construction
     -> maximum heat REDUCTION at the aux-heat limit.
  3. The peak-aware gate required weight >= 0.85 while DM_CRITICAL_T2_WEIGHT is 0.81,
     so a T2 recovery was crushed by a critical effect peak (-3.0).
  4. The tier was inferred from the POST-damping offset, so a damped T3 (floored at
     THERMAL_RECOVERY_T3_MIN_OFFSET) was misread as T1 and got T1's minimal offset.

Also covered: the DM_THRESHOLD_AUX_LIMIT hard limit must be enforced *before* the
anti-windup and "too warm" early returns in EmergencyLayer.evaluate_layer.

Physical basis: DM_THRESHOLD_AUX_LIMIT (-1500) is the point at which NIBE engages the
auxiliary immersion heater. Throttling recovery there does not stop DM falling - it
guarantees the aux heater runs, which draws several kW and creates a LARGER power peak
than the compressor would have. Cost-driven suppression at that threshold is both
unsafe and self-defeating.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DM_CRITICAL_T1_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T2_OFFSET,
    DM_CRITICAL_T2_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T2_WEIGHT,
    DM_CRITICAL_T3_OFFSET,
    DM_CRITICAL_T3_PEAK_AWARE_OFFSET,
    DM_CRITICAL_T3_WEIGHT,
    DM_THRESHOLD_AUX_LIMIT,
    EFFECT_OFFSET_CRITICAL,
    EFFECT_WEIGHT_CRITICAL,
    LAYER_WEIGHT_SAFETY,
    MAX_OFFSET,
    MIN_OFFSET,
    PRICE_OFFSET_PEAK,
    SAFETY_EMERGENCY_OFFSET,
    THERMAL_RECOVERY_T3_MIN_OFFSET,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import (
    DecisionEngine,
    LayerDecision,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import (
    EmergencyLayer,
    EmergencyLayerDecision,
    ThermalModel,
)

# Stockholm - the reference climate zone used throughout the project docs.
STOCKHOLM_LATITUDE = 59.33


@pytest.fixture
def engine():
    """DecisionEngine with the CONFIG KEYS THE ENGINE ACTUALLY READS.

    Note `target_indoor_temp` (not `target_temperature`) and the production default
    tolerance of 0.5. Several existing test fixtures pass `target_temperature` and
    `tolerance: 5.0`; the engine reads neither, which widens the emergency layer's
    "too warm" gate by 10x and hides real defects.
    """
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


def build_layers(
    emergency: EmergencyLayerDecision,
    effect: LayerDecision | None = None,
    price: LayerDecision | None = None,
) -> list[LayerDecision]:
    """Build the 9-layer list in the exact order DecisionEngine.calculate_decision uses.

    Layers not under test are neutral (weight 0.0) so they cannot influence the result.
    """
    neutral = lambda name: LayerDecision(name=name, offset=0.0, weight=0.0, reason="n/a")
    return [
        neutral("Safety"),
        emergency,
        neutral("Proactive"),
        effect or neutral("Peak Protection"),
        neutral("Learned Pre-heat"),
        neutral("Math WC"),
        neutral("Weather"),
        price or neutral("Spot Price"),
        neutral("Comfort"),
    ]


def emergency_at_aux_limit() -> EmergencyLayerDecision:
    """The EMERGENCY tier exactly as thermal_layer emits it at DM <= -1500."""
    return EmergencyLayerDecision(
        name="Thermal Debt",
        offset=SAFETY_EMERGENCY_OFFSET,
        weight=LAYER_WEIGHT_SAFETY,
        reason="EMERGENCY: DM at aux limit",
        tier="EMERGENCY",
        degree_minutes=DM_THRESHOLD_AUX_LIMIT - 20,
    )


def critical_effect_peak() -> LayerDecision:
    """Effect layer at CRITICAL: already at/above the monthly peak.

    `is_cost_layer` mirrors how DecisionEngine.calculate_decision wraps the effect
    layer - the effect tariff optimizes cost, not comfort or safety.
    """
    return LayerDecision(
        name="Peak Protection",
        offset=EFFECT_OFFSET_CRITICAL,
        weight=EFFECT_WEIGHT_CRITICAL,
        reason="At monthly peak",
        is_cost_layer=True,
    )


def price_peak() -> LayerDecision:
    """Price layer at PEAK. price_layer.py promotes itself to weight 1.0 here."""
    return LayerDecision(
        name="Spot Price",
        offset=PRICE_OFFSET_PEAK,
        weight=LAYER_WEIGHT_SAFETY,
        reason="PEAK quarter",
        is_cost_layer=True,
    )


class TestAuxLimitIsAbsolute:
    """DM <= DM_THRESHOLD_AUX_LIMIT must dominate every cost layer, unconditionally."""

    def test_price_peak_cannot_override_aux_limit_emergency(self, engine):
        """Price PEAK (-10.0 @ 1.0) must NOT beat the aux-limit emergency (+10.0 @ 1.0).

        Pre-fix: the tie-break `abs(max) > abs(min)` is False on the exact 10.0/-10.0 tie,
        so it returned min_offset = -10.0 - MAXIMUM HEAT REDUCTION at the aux-heat limit.
        """
        offset = engine._aggregate_layers(
            build_layers(emergency_at_aux_limit(), price=price_peak())
        )

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET), (
            f"Cost overrode the absolute DM safety limit: got {offset:+.1f}. "
            f"At DM <= {DM_THRESHOLD_AUX_LIMIT} the aux immersion heater engages; "
            f"reducing heat here deepens the debt AND creates a larger peak."
        )

    def test_critical_effect_peak_cannot_throttle_aux_limit_emergency(self, engine):
        """A critical effect peak must not throttle the aux-limit emergency to +1.0.

        Pre-fix: the peak-aware compromise fired for the EMERGENCY tier and replaced
        +10.0 with DM_CRITICAL_T3_PEAK_AWARE_OFFSET (+1.0).
        """
        offset = engine._aggregate_layers(
            build_layers(emergency_at_aux_limit(), effect=critical_effect_peak())
        )

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET), (
            f"Effect-tariff protection throttled the absolute DM limit to {offset:+.1f}. "
            "Peak protection must never suppress aux-limit recovery."
        )

    def test_both_cost_layers_together_cannot_override_aux_limit(self, engine):
        """Price PEAK and a critical effect peak together still must not win."""
        offset = engine._aggregate_layers(
            build_layers(
                emergency_at_aux_limit(),
                effect=critical_effect_peak(),
                price=price_peak(),
            )
        )

        assert offset == pytest.approx(SAFETY_EMERGENCY_OFFSET)


class TestRecoveryTiersSurviveCostLayers:
    """T1/T2/T3 recovery must never be driven NEGATIVE by a cost layer."""

    def test_t2_recovery_is_not_crushed_by_critical_effect_peak(self, engine):
        """T2 (weight 0.81) + critical effect peak must not yield a heat REDUCTION.

        Pre-fix: the peak-aware gate was a hardcoded `weight >= 0.85`, but
        DM_CRITICAL_T2_WEIGHT is 0.81, so T2 fell through to the critical-override
        branch and returned the effect layer's -3.0 while in deep thermal debt.
        """
        t2 = EmergencyLayerDecision(
            name="T2",
            offset=DM_CRITICAL_T2_OFFSET,
            weight=DM_CRITICAL_T2_WEIGHT,
            reason="T2 recovery",
            tier="T2",
            degree_minutes=-900,
        )

        offset = engine._aggregate_layers(build_layers(t2, effect=critical_effect_peak()))

        assert offset == pytest.approx(DM_CRITICAL_T2_PEAK_AWARE_OFFSET), (
            f"T2 thermal-debt recovery returned {offset:+.1f}. A negative offset here "
            "actively deepens the debt toward the aux limit."
        )
        assert offset > 0, "Recovery must never be negative while in thermal debt"

    def test_price_peak_cannot_crush_t3_recovery(self, engine):
        """Price PEAK (weight 1.0) must not outvote a T3 recovery (weight 0.91).

        Pre-fix: price_layer promotes itself to weight 1.0 on any PEAK quarter, entering
        the critical-override branch that emergency tiers (max 0.91) cannot reach.
        Result: -10.0 while DM is ~50 from the aux limit.
        """
        t3 = EmergencyLayerDecision(
            name="T3",
            offset=DM_CRITICAL_T3_OFFSET,
            weight=DM_CRITICAL_T3_WEIGHT,
            reason="T3 recovery",
            tier="T3",
            degree_minutes=-1400,
        )

        offset = engine._aggregate_layers(build_layers(t3, price=price_peak()))

        assert offset > 0, (
            f"Spot price outvoted T3 emergency recovery: got {offset:+.1f}. "
            "A cost layer must never reduce heat during thermal-debt recovery."
        )

    def test_damped_t3_still_gets_the_t3_peak_aware_offset(self, engine):
        """A DAMPED T3 must be treated as T3, not misread as T1.

        Pre-fix: the tier was inferred by comparing the emergency layer's offset against
        DM_CRITICAL_T3_OFFSET (8.5) / DM_CRITICAL_T2_OFFSET (7.0). But that offset has
        already been through thermal-recovery damping and bottoms out at
        THERMAL_RECOVERY_T3_MIN_OFFSET (2.0), so it fell through to the T1 branch and a
        genuine T3 emergency received T1's minimal offset.
        """
        damped_t3 = EmergencyLayerDecision(
            name="T3",
            offset=THERMAL_RECOVERY_T3_MIN_OFFSET,  # damped from 8.5 by solar gain
            weight=DM_CRITICAL_T3_WEIGHT,
            reason="T3 recovery [damped: warming]",
            tier="T3",
            degree_minutes=-1400,
        )

        offset = engine._aggregate_layers(build_layers(damped_t3, effect=critical_effect_peak()))

        assert offset == pytest.approx(DM_CRITICAL_T3_PEAK_AWARE_OFFSET), (
            f"Damped T3 got {offset:+.1f}; expected the T3 peak-aware offset "
            f"({DM_CRITICAL_T3_PEAK_AWARE_OFFSET}). Tier must come from the `tier` field, "
            "not from the post-damping offset magnitude."
        )
        assert offset != pytest.approx(
            DM_CRITICAL_T1_PEAK_AWARE_OFFSET
        ), "Damped T3 was misclassified as T1"


class TestAggregateOutputIsBounded:
    """The aggregator must never emit an offset outside the pump's valid range."""

    def test_aggregate_never_exceeds_offset_bounds(self, engine):
        """Even with extreme layer votes, the result stays within [MIN_OFFSET, MAX_OFFSET]."""
        extreme = EmergencyLayerDecision(
            name="T3",
            offset=999.0,
            weight=DM_CRITICAL_T3_WEIGHT,
            reason="pathological",
            tier="T3",
            degree_minutes=-1400,
        )

        offset = engine._aggregate_layers(build_layers(extreme))

        assert MIN_OFFSET <= offset <= MAX_OFFSET


class TestAuxLimitEnforcedBeforeEarlyReturns:
    """thermal_layer must check the aux limit BEFORE its early-return branches."""

    @staticmethod
    def _layer() -> EmergencyLayer:
        return EmergencyLayer(
            climate_detector=ClimateZoneDetector(STOCKHOLM_LATITUDE),
            heating_type="radiator",
        )

    @staticmethod
    def _state(degree_minutes: float, indoor_temp: float, current_offset: float = 0.0):
        return NibeState(
            outdoor_temp=-15.0,
            indoor_temp=indoor_temp,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=degree_minutes,
            current_offset=current_offset,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime(2026, 1, 15, 6, 0),
        )

    def test_aux_limit_enforced_even_when_house_is_too_warm(self):
        """DM past the aux limit must fire EMERGENCY even if indoor is above tolerance.

        Pre-fix: Case 1 ("too warm") returned offset 0.0 / weight 0.0 with NO aux-limit
        guard, while the neighbouring Case 2 DID guard on `dm > DM_THRESHOLD_AUX_LIMIT`.
        That asymmetry meant a solar-gain morning during a debt spiral silently disabled
        the hard limit: the immersion heater engages while EffektGuard says "let cool
        naturally".

        With the production default tolerance (0.5 -> tolerance_range 0.2 C), an indoor
        temp just 0.3 C over target is enough to trigger Case 1.
        """
        decision = self._layer().evaluate_layer(
            nibe_state=self._state(degree_minutes=DM_THRESHOLD_AUX_LIMIT - 50, indoor_temp=21.3),
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.2,  # production default: tolerance 0.5 * 0.4
        )

        assert decision.tier == "EMERGENCY", (
            f"Aux limit not enforced when too warm - got tier={decision.tier!r}, "
            f"offset={decision.offset:+.1f}, weight={decision.weight}. "
            "The DM -1500 hard limit must outrank the 'too warm' early return."
        )
        assert decision.weight == pytest.approx(LAYER_WEIGHT_SAFETY)
        assert decision.offset == pytest.approx(SAFETY_EMERGENCY_OFFSET)

    def test_aux_limit_enforced_during_anti_windup_cooldown(self):
        """DM past the aux limit must fire EMERGENCY even inside the anti-windup cooldown.

        Pre-fix: the cooldown branch returned early with weight 0.7 and the pump's current
        offset, so for up to ANTI_WINDUP_COOLDOWN_MINUTES the aux limit was not enforced
        at all.
        """
        layer = self._layer()
        now = datetime(2026, 1, 15, 6, 0)
        layer._anti_windup_cooldown_until = now + timedelta(minutes=20)

        decision = layer.evaluate_layer(
            nibe_state=self._state(
                degree_minutes=DM_THRESHOLD_AUX_LIMIT - 50,
                indoor_temp=20.5,
                current_offset=1.0,
            ),
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.2,
        )

        assert decision.tier == "EMERGENCY", (
            f"Aux limit not enforced during anti-windup cooldown - got tier={decision.tier!r}, "
            f"offset={decision.offset:+.1f}. The hard limit must outrank the cooldown."
        )
        assert decision.offset == pytest.approx(SAFETY_EMERGENCY_OFFSET)
