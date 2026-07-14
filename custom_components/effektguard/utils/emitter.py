"""The EN 442 emitter law: the flow temperature a heating system needs.

The single source of truth for "how hot must the water be right now".

    phi     = (T_room - T_out) / (T_room - T_out_design)     relative load   [EN 12831]
    dT      = dT_design * phi ** (1 / n)                     emitter law     [EN 442-1 3.31]
    T_flow  = T_room + dT + spread_design / 2

EN 12831 makes heat loss linear in the indoor/outdoor difference, so relative load `phi` is a
ratio of temperature differences. EN 442-1 3.31 gives output as (dT / dT_N) ** n; setting output
equal to load and inverting yields the 1/n exponent.

THE SPREAD IS CONSTANT - never scale it by phi. A fixed-speed circulator gives constant mass flow
and a load-proportional spread (a wet boiler); a heat pump MODULATES its circulator to hold the
commissioned spread and varies the flow RATE. Scaling it runs the curve too COOL in mild weather
and too HOT in cold, pivoting invisibly on the design point where the error is zero.

INTERNAL GAINS ARE WATTS, NEVER A CURVE FIT. Demand is linear in (balance - T_out), not
(T_room - T_out), so the caller passes a balance point DERIVED from watts
(`indoor - gains_W / heat_loss_W_per_K`). It must never be fitted: the constant-spread term and
the balance-point term are the same basis function with opposite signs, so any assumed spread
manufactures a matching "gains" figure - even out of a curve with provably zero gains. Fitting it
against NIBE's curve once produced exactly that spurious number and read it back as evidence.

This is OpenEnergyMonitor's method (github.com/openenergymonitor/tools, weathercomp.js), not an
invention; see docs/research/02_emitter_law.md and
tests/validation/test_emitter_law_matches_openenergymonitor.py.
"""

import logging

_LOGGER = logging.getLogger(__name__)


def en442_flow_temp(
    indoor_setpoint: float,
    outdoor_temp: float,
    design_outdoor_temp: float,
    design_flow_temp: float,
    design_spread: float,
    emitter_exponent: float,
    balance_point_temp: float | None = None,
) -> float:
    """Flow temperature the emitters need to hold ``indoor_setpoint`` at ``outdoor_temp``.

    Args:
        indoor_setpoint: Target indoor temperature (C).
        outdoor_temp: Current outdoor temperature (C).
        design_outdoor_temp: Dimensioning outdoor temperature the emitters were sized for (C).
        design_flow_temp: Supply temperature the system needs at ``design_outdoor_temp`` (C).
        design_spread: Flow-return spread the circulator holds (C). NOT scaled by load.
        emitter_exponent: EN 442 exponent n (1.3 radiators, 1.1 underfloor).
        balance_point_temp: Outdoor temperature at which the house needs no heat at all, because
            bodies, appliances and the sun already supply its losses. DERIVE it from watts -
            ``indoor - internal_gains_W / heat_loss_W_per_K`` - and never fit it against a heating
            curve; see the module docstring for why that fit cannot work. Defaults to
            ``indoor_setpoint``, i.e. no gains, which is what OpenEnergyMonitor's tool assumes and
            what a bare emitter law implies.

    Returns:
        Required flow temperature (C). Never below ``indoor_setpoint``: water colder than the
        room removes heat from it.
    """
    balance = indoor_setpoint if balance_point_temp is None else balance_point_temp

    load = balance - outdoor_temp
    if load <= 0:
        # Warmer than the balance point: the house is heating itself, so the emitters need no
        # excess over the room at all. Return the CONTINUOUS limit of the expression below as
        # load -> 0 (excess -> 0), not the bare setpoint: dropping the spread term here would put
        # a spread/2 cliff - 2.5 C on the defaults - at the balance point, and the balance point
        # sits in the middle of the Swedish shoulder season, where the outdoor temperature crosses
        # it back and forth all day. A step there is chatter.
        return indoor_setpoint + (design_spread / 2.0)

    design_load = balance - design_outdoor_temp
    if design_load <= 0 or emitter_exponent <= 0:
        # A design point that cannot be extrapolated from. Return the setpoint rather than
        # fabricate a temperature; weather compensation then commands no change.
        _LOGGER.warning(
            "Cannot size emitters: design outdoor %.1f C is not below the %.1f C setpoint "
            "(exponent %.2f)",
            design_outdoor_temp,
            indoor_setpoint,
            emitter_exponent,
        )
        return indoor_setpoint

    # Mean water temperature above the room at the design point, implied by the design point
    # rather than configured separately.
    design_excess = design_flow_temp - (design_spread / 2.0) - indoor_setpoint
    if design_excess <= 0:
        _LOGGER.warning(
            "Cannot size emitters: design flow %.1f C with spread %.1f C does not exceed the "
            "%.1f C setpoint",
            design_flow_temp,
            design_spread,
            indoor_setpoint,
        )
        return indoor_setpoint

    # Exceeds 1.0 below the design temperature, which must ask for MORE than the design flow.
    # Do not clamp it: that silently under-heats exactly when the house can least afford it.
    phi = load / design_load

    excess = design_excess * (phi ** (1.0 / emitter_exponent))

    # The spread does NOT scale with load - a heat pump modulates its circulator to hold it.
    # See the module docstring: scaling it made the curve too cool when mild and too hot when
    # cold, pivoting invisibly on the design point.
    return indoor_setpoint + excess + (design_spread / 2.0)
