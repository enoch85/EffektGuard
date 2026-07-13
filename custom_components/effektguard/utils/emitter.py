"""The EN 442 emitter law: the flow temperature a heating system needs.

The single source of truth for "how hot must the water be right now".

    phi     = (T_room - T_out) / (T_room - T_out_design)     relative load   [EN 12831]
    dT      = dT_design * phi ** (1 / n)                     emitter law     [EN 442-1 3.31]
    T_flow  = T_room + dT + spread_design / 2

EN 12831 makes a building's heat loss linear in the indoor/outdoor difference, so the relative
load `phi` is a ratio of temperature differences. EN 442-1 3.31 gives the emitter's output as
`Phi / Phi_N = (dT / dT_N) ** n`; setting output equal to load and inverting it yields the 1/n
exponent.

THE SPREAD IS CONSTANT, AND THIS FILE USED TO SCALE IT.

    spread = spread_design * phi        # "constant mass flow"

A fixed-speed circulator gives constant mass flow, and then the flow-return spread really is
proportional to the heat being carried. That is a wet boiler. A heat pump MODULATES its
circulator to hold the spread at its commissioned value - typically 5 K - and varies the flow
RATE instead. Scaling the spread models the wrong machine.

It is not a rounding error, and it is not symmetric: the mistake pivots on the design point, so
the flow temperature comes out too COOL in mild weather and too HOT in cold weather. Measured
against OpenEnergyMonitor's own weather-compensation tool (their defaults: 3 kW loss, 15 kW of
emitters rated at dT50, room 20 C, design -3 C, spread 5 K):

    outdoor    OEM tool    scaled spread    error
      +12 C      28.93         27.30        -1.63
       +5 C      32.94         32.07        -0.87
       -3 C      37.00         37.00        +0.00      <- the design point, where it hides
      -12 C      41.19         42.17        +0.98

With the spread held constant the two agree to 0.00 C at every outdoor temperature.

Sources - this is OpenEnergyMonitor's method, not an invention:
  - github.com/openenergymonitor/tools  www/tools/weathercomp/weathercomp.js
        heat_demand = HTC * (room_temperature - outsideT)
        DT    = (heat_demand / rated_emitter_output_dt50) ** (1/1.3) * 50
        flowT = room_temperature + DT + systemDT * 0.5      <- systemDT, not systemDT * phi
  - docs.openenergymonitor.org/heatpumps/basics.html
        "Heat_output = Rated_Heat_Output x (Delta_T / Rated_Delta_T) ^ 1.3"
        "Delta_T = (Heat_output / Rated_Heat_Output)^(1/1.3) x Rated_Delta_T"
  - Andre Kuhne's reverse-engineering of Vaillant's heat curve is the SAME law wearing a
    different hat: TFlow = 2.55 * (HC * (Tset - Tout))**0.78 + Tset, and 1/0.78 = 1.28 ~ 1.3,
    the radiator exponent. He fitted it to Vaillant's published curves and validated it against
    eBus readings from his own AroTherm to within 0.07 C.

Also note what is NOT here: there is no internal-gains term. Heat demand is linear in
(T_room - T_out), full stop. OEM's tool does the same. Free heat from bodies and appliances is
real, but their own heat-loss guidance finds it roughly cancels the domestic hot water draw over
a heating season, and a heat pump's own minimum modulation - not the gains - is what sets the
outdoor temperature at which it stops being able to turn down.
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
            bodies, appliances and the sun are already supplying its losses. Defaults to
            ``indoor_setpoint``, i.e. no internal gains - which is what OpenEnergyMonitor's tool
            assumes, and which is WRONG for a real house. See below.

    Returns:
        Required flow temperature (C). Never below ``indoor_setpoint``: water colder than the
        room removes heat from it.

    THE BALANCE POINT, AND WHY IT IS NOT THE ROOM TEMPERATURE.

    A house does not start needing heat the moment it is one degree cooler outside than in. Bodies,
    appliances and the sun supply several hundred watts, so demand only reaches zero somewhere
    around 17 C outdoors. Modelling demand as linear in (indoor - outdoor) therefore over-predicts
    the flow temperature in mild weather - by up to 2.7 C - and asks the pump to run hot in exactly
    the conditions where a heat pump is most efficient and has the most to lose.

    Fitted against NIBE's OWN published heating curve 9 (52.6 C at -15 C; 41.0 C at 0 C):

        balance    -15C    -10C     -5C     +0C     +5C    +10C |  RMS
          21.0    +0.00   +0.84   +1.26   +1.72   +2.19   +2.69 |  1.70   <- no gains
          17.0    +0.00   +0.43   +0.41   +0.39   +0.28   +0.04 |  0.31   <- best fit

    17 C. And independently, from UK field data: "the heating demand is typically zero or negative
    until the external temperature falls below about 17 C" (Protons for Breakfast, on Vaillant's
    controls). Two unrelated sources, the same number.

    It is derivable rather than guessed: balance = indoor - internal_gains_W / heat_loss_W_per_K.
    A 150 W/K house with 600 W of gains balances at 21 - 4 = 17 C.

    THIS IS WHY THE TWO BUGS HID EACH OTHER. The spread used to be scaled by load, which made the
    curve too COOL in mild weather; omitting the gains made it too HOT in mild weather. The errors
    are largest in the same place and point in opposite directions, so together they matched NIBE's
    curve to a fifth of a degree, and fixing either one alone made the fit worse.
    """
    balance = indoor_setpoint if balance_point_temp is None else balance_point_temp

    load = balance - outdoor_temp
    if load <= 0:
        # Warmer than the balance point: the house is heating itself.
        return indoor_setpoint

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
