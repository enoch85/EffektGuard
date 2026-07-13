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

INTERNAL GAINS ARE REAL, AND A CURVE FIT CANNOT MEASURE THEM.

OEM's tool has no gains term; a real house does. Demand is linear in (balance - T_out), not in
(T_room - T_out). So this law takes a balance point - but the caller must DERIVE it from watts
(`indoor - gains_W / heat_loss_W_per_K`), never fit it, because the fit is degenerate:

    constant spread lifts the curve by   (spread / 2) * (1 - phi ** (1/n))
    a balance point drops it by          a term with the same shape and the opposite sign

Both vanish at the design point and grow in mild weather - the SAME basis function. They are not
separately identifiable, so any assumed spread manufactures a matching "gains" figure out of
nothing. Fit this law to Kuhne's Vaillant curve, which contains PROVABLY ZERO gains, and a
spurious balance point appears anyway, scaling with whatever spread you assumed: 0.3 K at spread
0, 2.6 K at spread 5, 4.9 K at spread 10.

This is worth stating plainly because an earlier version of this file DID fit the balance point
against NIBE's curve 9, reported "RMS 0.31 C vs 1.70 C for no gains", and presented that as
evidence. It was not evidence. It was the constant-spread term being read back out.
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
