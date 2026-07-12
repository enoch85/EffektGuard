"""The EN 442 emitter law: the flow temperature a heating system needs.

The single source of truth for "how hot must the water be right now".

    phi     = (T_room - T_out) / (T_room - T_out_design)     relative load   [EN 12831]
    dT      = dT_design * phi ** (1 / n)                     emitter law     [EN 442-1 3.31]
    spread  = spread_design * phi                            constant mass flow
    T_flow  = T_room + dT + spread / 2

EN 12831 makes a building's heat loss linear in the indoor/outdoor difference, so the relative
load `phi` is a ratio of temperature differences. EN 442-1 3.31 gives the emitter's output as
`Phi / Phi_N = (dT / dT_N) ** n`; setting output equal to load and inverting it yields the 1/n
exponent. Constant mass flow makes the flow-return spread linear in load.
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
) -> float:
    """Flow temperature the emitters need to hold ``indoor_setpoint`` at ``outdoor_temp``.

    Args:
        indoor_setpoint: Target indoor temperature (C).
        outdoor_temp: Current outdoor temperature (C).
        design_outdoor_temp: Dimensioning outdoor temperature the emitters were sized for (C).
        design_flow_temp: Supply temperature the system needs at ``design_outdoor_temp`` (C).
        design_spread: Flow-return spread at the design load (C).
        emitter_exponent: EN 442 exponent n (1.3 radiators, 1.1 underfloor).

    Returns:
        Required flow temperature (C). Never below ``indoor_setpoint``: water colder than the
        room removes heat from it.
    """
    load = indoor_setpoint - outdoor_temp
    if load <= 0:
        return indoor_setpoint

    design_load = indoor_setpoint - design_outdoor_temp
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
    spread = design_spread * phi

    return indoor_setpoint + excess + (spread / 2.0)
