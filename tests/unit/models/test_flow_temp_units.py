"""Regression tests for the flow-temperature formula unit conversion.

The Kuehne formula operates on heat demand in kW: profiles previously fed the
heat-loss coefficient in W/K without converting (weather_layer.py converts),
producing astronomically large values that were always masked by the
efficiency clamp. With HLC=180 W/K and dT=31 K the formula term must be
~30.7 C, not ~2200 C.
"""

from custom_components.effektguard.const import (
    KUEHNE_COEFFICIENT,
    KUEHNE_POWER,
    WATTS_PER_KILOWATT,
)
from custom_components.effektguard.models.nibe import (
    NibeF750Profile,
    NibeF1155Profile,
    NibeS1155Profile,
)


def kuehne_reference(hlc_w_per_k: float, temp_diff: float, indoor: float) -> float:
    """The shared formula's correct form (kW basis, weather_layer parity)."""
    return (
        KUEHNE_COEFFICIENT * (hlc_w_per_k / WATTS_PER_KILOWATT * temp_diff) ** KUEHNE_POWER + indoor
    )


class TestProfileFlowTempUnits:
    def test_f750_formula_matches_kw_basis(self):
        """At mild outdoor temp the formula (not the clamp) must decide."""
        profile = NibeF750Profile()
        result = profile.calculate_optimal_flow_temp(
            outdoor_temp=5.0, indoor_target=21.0, heat_demand_kw=3.0
        )
        # Correct formula: 2.55*(180/1000*16)**0.78+21 = ~26.8, below the
        # efficiency clamp (5+30+3=38). The old W-basis value (~1300) always
        # hit the clamp, masking the unit error.
        reference = kuehne_reference(180.0, 16.0, 21.0)
        assert result == min(
            reference, 5.0 + profile.optimal_flow_delta + 3.0
        ), "formula must be computed on a kW basis"
        assert result < 30.0

    def test_gshp_profiles_share_correct_basis(self):
        """S1155 and the inheriting F1155 use the same corrected formula."""
        for profile in (NibeS1155Profile(), NibeF1155Profile()):
            result = profile.calculate_optimal_flow_temp(
                outdoor_temp=5.0, indoor_target=21.0, heat_demand_kw=3.0
            )
            reference = kuehne_reference(180.0, 16.0, 21.0)
            expected = max(
                profile.min_flow_temp,
                min(min(reference, 5.0 + profile.optimal_flow_delta + 3.0), profile.max_flow_temp),
            )
            assert result == expected

    def test_formula_reference_value(self):
        """HLC=180 W/K, dT=31 K: the reviewer's reference case ~30.7 C."""
        value = kuehne_reference(180.0, 31.0, 21.0)
        assert 29.0 < value < 32.0
