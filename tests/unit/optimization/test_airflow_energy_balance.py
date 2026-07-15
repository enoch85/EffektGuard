"""Enhanced airflow must obey the energy balance, and it does not pay in a Swedish winter.

The extra heat an exhaust-air pump extracts and its "improved COP" are the same joules: in steady
state Q_cond = P_el + Q_evap, so at constant electrical input d(Q_cond) = d(Q_evap) = P_el*d(COP).
`calculate_net_thermal_gain` must count that heat once - extra extraction minus the ventilation
penalty - never adding a separate COP term.

Consequence: enhancement pays only above an outdoor temperature of
(indoor - AIRFLOW_EVAPORATOR_TEMP_DROP), around +9 C. Across the whole heating season it is a net
thermal LOSS, because the evaporator recovers only that drop while the building reheats every extra
cubic metre from outdoor all the way to indoor.
"""

import pytest

from custom_components.effektguard.const import (
    AIRFLOW_DEFAULT_ENHANCED,
    AIRFLOW_DEFAULT_STANDARD,
    AIRFLOW_EVAPORATOR_TEMP_DROP,
)
from custom_components.effektguard.optimization.airflow_optimizer import (
    calculate_net_thermal_gain,
    evaporator_heat_extraction,
    ventilation_heat_loss,
)

INDOOR = 21.0

# Above this outdoor temperature the building has to reheat the extra air by less than the
# evaporator takes out of it, so enhancing pays. Below it, it cannot.
BREAK_EVEN_OUTDOOR = INDOOR - AIRFLOW_EVAPORATOR_TEMP_DROP


def _net(outdoor: float) -> float:
    return calculate_net_thermal_gain(
        flow_standard=AIRFLOW_DEFAULT_STANDARD,
        flow_enhanced=AIRFLOW_DEFAULT_ENHANCED,
        temp_indoor=INDOOR,
        temp_outdoor=outdoor,
    )


def test_the_extra_heat_is_not_counted_twice():
    """Net gain must be the extra extraction minus the ventilation penalty. Nothing else.

    Both extra terms describe the same joules: heat that entered the refrigerant at the
    evaporator and left it at the condenser.
    """
    outdoor = 0.0

    extraction = evaporator_heat_extraction(AIRFLOW_DEFAULT_ENHANCED) - evaporator_heat_extraction(
        AIRFLOW_DEFAULT_STANDARD
    )
    penalty = ventilation_heat_loss(
        AIRFLOW_DEFAULT_ENHANCED, INDOOR, outdoor
    ) - ventilation_heat_loss(AIRFLOW_DEFAULT_STANDARD, INDOOR, outdoor)

    assert _net(outdoor) == pytest.approx(extraction - penalty, abs=0.01), (
        f"Net gain at {outdoor:.0f} C is {_net(outdoor):.3f} kW, but the energy balance allows "
        f"only extraction ({extraction:.3f}) minus penalty ({penalty:.3f}) = "
        f"{extraction - penalty:.3f} kW. The COP term is the extraction term again."
    )


@pytest.mark.parametrize("outdoor", [8.0, 5.0, 0.0, -5.0, -10.0, -15.0])
def test_enhancing_is_a_thermal_loss_all_winter(outdoor):
    """Across the whole Swedish heating season, pulling more air through the house costs heat.

    Break-even is +9 C. Every one of these is a heating-season temperature and every one of them
    is below it.
    """
    net = _net(outdoor)

    assert net < 0.0, (
        f"At {outdoor:+.0f} C outdoor the model says enhanced airflow GAINS {net:.3f} kW. The "
        f"evaporator takes only {AIRFLOW_EVAPORATOR_TEMP_DROP:.0f} C out of the extra air while "
        f"the building must reheat it from {outdoor:+.0f} C to {INDOOR:.0f} C."
    )


def test_break_even_is_where_the_physics_puts_it():
    """Break-even is indoor minus the evaporator's temperature drop - about +9 C, not -15 C."""
    assert _net(BREAK_EVEN_OUTDOOR + 2.0) > 0.0, "above break-even, enhancing should pay"
    assert _net(BREAK_EVEN_OUTDOOR - 2.0) < 0.0, "below break-even, it cannot"


def test_the_loss_deepens_as_it_gets_colder():
    """Colder outdoor air means a bigger reheat bill for the same extra cubic metres."""
    losses = [_net(t) for t in (10.0, 0.0, -10.0, -20.0)]

    for warmer, colder in zip(losses, losses[1:]):
        assert colder < warmer, f"net gain must fall as it gets colder, got {losses}"
