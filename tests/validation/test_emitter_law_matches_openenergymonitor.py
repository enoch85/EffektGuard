"""Our flow-temperature curve is checked against OpenEnergyMonitor's, not against our own opinion.

The emitter law is the one number the whole weather-compensation layer rests on. It decides how hot
the water has to be, at every outdoor temperature, forever. If it is wrong, everything downstream is
wrong in a way no amount of tuning will reveal - it will just quietly hold the house at the wrong
temperature and call it optimisation.

So it is pinned to a published, independent implementation: OpenEnergyMonitor's weather-compensation
tool, whose source is public.

    // github.com/openenergymonitor/tools : www/tools/weathercomp/weathercomp.js
    let HTC         = heat_loss / (room_temperature - design_outsideT);
    let heat_demand = HTC * (room_temperature - outsideT);
    let DT          = Math.pow((heat_demand / rated_emitter_output_dt50), 1 / 1.3) * 50;
    let MWT         = room_temperature + DT;
    let flowT       = MWT + (systemDT * 0.5);

Two things in that source are worth stating plainly, because this project got one of them wrong and
worried unnecessarily about the other.

**The spread is constant.** `systemDT * 0.5`, not `systemDT * phi * 0.5`. A fixed-speed circulator
gives constant mass flow and a spread proportional to load - that is a wet boiler. A heat pump
modulates its circulator to hold the commissioned spread and varies the flow rate. This file used to
scale it, and the error pivoted exactly on the design point, so it was invisible there and grew in
both directions: 1.63 C too cool at +12 C outdoor, 0.98 C too hot at -12 C.

**WeatherComp has no internal-gains term, and it is the ODD ONE OUT.** An earlier draft of this file
took that omission as gospel and wrote "heat demand is linear in (room - outdoor), full stop". It is
not. OpenEnergyMonitor contradict themselves, and the rest of their work says so:

  * their SCOP tool carries the naive formula COMMENTED OUT, with the note
    "This approach would need to take into account gains, hence use of degree days approach",
    and uses `baseTemp: 15.5` against `roomT: 19.3` - a 3.8 K base difference;
  * their measured-heat-demand tool fits `base_DT` from real data, default 4 K;
  * across 383 monitored systems on heatpumpmonitor.org the median fitted `base_DT` is 2.5 K, and
    the median implied gains 583 W.

So this file uses WeatherComp to check the EMITTER LAW - the `^(1/1.3)` part, which is what it is
authoritative about - and holds the demand model identical on both sides to do it. The gains term is
checked against NIBE's own published curve instead, in the emitter module's own tests.

And the Vaillant heat curve that this project ran for a year is the same law in different clothes:

    TFlow = 2.55 * (HC * (Tset - Tout)) ** 0.78 + Tset          [Andre Kuhne]

1/0.78 = 1.28, which is the radiator exponent 1.3. He fitted it to Vaillant's published curves and
checked it against eBus readings from his own AroTherm to within 0.07 C. It is not a rival model. It
is this one, with the design point folded into a single dimensionless curve number.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import DEFAULT_BALANCE_POINT_OFFSET
from custom_components.effektguard.utils.emitter import en442_flow_temp

# OpenEnergyMonitor weathercomp.js defaults, verbatim from the source.
OEM_HEAT_LOSS_KW = 3.0
OEM_RATED_EMITTER_DT50_KW = 15.0
OEM_ROOM_TEMP = 20.0
OEM_DESIGN_OUTDOOR = -3.0
OEM_SYSTEM_DT = 5.0
OEM_EXPONENT = 1.3


def oem_weathercomp_flow_temp(outdoor: float) -> float:
    """weathercomp.js, transliterated line for line. This is the reference, not our code."""
    htc = OEM_HEAT_LOSS_KW / (OEM_ROOM_TEMP - OEM_DESIGN_OUTDOOR)
    heat_demand = htc * (OEM_ROOM_TEMP - outdoor)
    delta_t = (heat_demand / OEM_RATED_EMITTER_DT50_KW) ** (1 / OEM_EXPONENT) * 50
    mean_water_temp = OEM_ROOM_TEMP + delta_t
    return mean_water_temp + (OEM_SYSTEM_DT * 0.5)


OEM_DESIGN_FLOW = oem_weathercomp_flow_temp(OEM_DESIGN_OUTDOOR)


def ours(outdoor: float) -> float:
    """Our law with gains switched OFF, because WeatherComp has none - and OEM knows it.

    Their SCOP tool carries the naive formula COMMENTED OUT, with the note "This approach would need
    to take into account gains, hence use of degree days approach", and uses a base temperature of
    15.5 C against a 19.3 C room instead. Their measured-demand tool fits a base_DT from real data.
    WeatherComp is the outlier, not the authority - so it is used here to check the EMITTER LAW only,
    with the demand model held identical on both sides.
    """
    return en442_flow_temp(
        indoor_setpoint=OEM_ROOM_TEMP,
        outdoor_temp=outdoor,
        design_outdoor_temp=OEM_DESIGN_OUTDOOR,
        design_flow_temp=OEM_DESIGN_FLOW,
        design_spread=OEM_SYSTEM_DT,
        emitter_exponent=OEM_EXPONENT,
        balance_point_temp=OEM_ROOM_TEMP,  # no gains, matching weathercomp.js
    )


@pytest.mark.parametrize(
    "outdoor", [15.0, 12.0, 8.0, 5.0, 2.0, 0.0, -3.0, -6.0, -10.0, -15.0, -20.0]
)
def test_our_curve_is_openenergymonitors_curve(outdoor):
    """Across the whole Nordic range, to a hundredth of a degree."""
    reference = oem_weathercomp_flow_temp(outdoor)
    mine = ours(outdoor)

    assert mine == pytest.approx(reference, abs=0.01), (
        f"At {outdoor:+.1f} C outdoor, OpenEnergyMonitor's weather-compensation tool asks for "
        f"{reference:.2f} C of flow and we ask for {mine:.2f} C - a gap of {mine - reference:+.2f} C. "
        f"Their tool is public, published and independently used; ours drives a real heat pump. "
        f"Where they disagree, the burden is on us."
    )


def test_the_error_a_scaled_spread_produces_is_not_symmetric():
    """Why the old bug hid: it was zero exactly where anyone would have checked it.

    Scaling the spread with load pivots the whole curve about the design point. At the design point
    the error is exactly zero, which is where a sanity check naturally looks - and it grows in both
    directions from there, cooling the house in mild weather and cooking it in cold.
    """
    room, design_out, spread = OEM_ROOM_TEMP, OEM_DESIGN_OUTDOOR, OEM_SYSTEM_DT

    def with_scaled_spread(outdoor: float) -> float:
        phi = (room - outdoor) / (room - design_out)
        excess = (OEM_DESIGN_FLOW - spread / 2 - room) * phi ** (1 / OEM_EXPONENT)
        return room + excess + (spread * phi) / 2

    assert with_scaled_spread(design_out) == pytest.approx(
        ours(design_out), abs=0.01
    ), "precondition: at the design point the old bug is invisible"
    assert (
        with_scaled_spread(12.0) < ours(12.0) - 1.0
    ), "mild weather: the old model ran the house cool"
    assert with_scaled_spread(-12.0) > ours(-12.0) + 0.5, "cold weather: the old model ran it hot"


def test_the_vaillant_heat_curve_is_the_same_law():
    """Kuhne's formula and ours are one model. Neither is a rival to the other.

    HC is not a heat loss coefficient - it is Vaillant's dimensionless curve number, 0.1 to 4.0,
    defaulting to 0.6 for a heat pump. It is obtained by INVERTING the formula at the design point,
    which is the same information our design_flow_temp carries. Protons for Breakfast works the
    example: 45 C of flow needed at -5 C outdoor for a 20 C room gives heat curve 0.75.
    """
    room, design_out, design_flow = 20.0, -5.0, 45.0

    hc = ((design_flow - room) / 2.55) ** (1 / 0.78) / (room - design_out)
    assert hc == pytest.approx(0.75, abs=0.01), (
        f"Inverting Kuhne at the published worked example gives HC {hc:.3f}, not the 0.75 that "
        f"Protons for Breakfast reports. If this fails, our reading of the formula is wrong."
    )

    def kuhne(outdoor: float) -> float:
        return 2.55 * (hc * (room - outdoor)) ** 0.78 + room

    for outdoor in (10.0, 5.0, 0.0, -5.0, -10.0, -15.0):
        theirs = kuhne(outdoor)
        mine = en442_flow_temp(
            indoor_setpoint=room,
            outdoor_temp=outdoor,
            design_outdoor_temp=design_out,
            design_flow_temp=design_flow,
            design_spread=5.0,
            emitter_exponent=1.3,
        )
        assert mine == pytest.approx(theirs, abs=2.0), (
            f"At {outdoor:+.1f} C, Vaillant's curve (via Kuhne) wants {theirs:.1f} C and we want "
            f"{mine:.1f} C. These are supposed to be the same physics; a real divergence here means "
            f"one of us has the emitter law wrong."
        )


# NIBE's own published heating curve 9, digitised. Room 21 C, operating spread 5 K.
NIBE_CURVE_9 = {-15.0: 52.6, -10.0: 48.6, -5.0: 44.9, 0.0: 41.0, 5.0: 36.9, 10.0: 32.5}


def test_the_balance_point_is_what_makes_our_curve_match_nibes():
    """A house does not start needing heat the moment it is a degree cooler outside than in.

    Bodies, appliances and the sun cover its losses until about four degrees below the setpoint. A
    model linear in (indoor - outdoor) therefore asks for too much flow in mild weather - which is
    where most of a season's kWh are delivered, and where every excess degree of flow costs 2.5-3 %
    of COP on OEM's measured fleet.

    Fitted against NIBE's own curve 9, anchored at its -15 C end and asked to reproduce the rest:

        balance    -15C    -10C     -5C     +0C     +5C    +10C |  RMS
          21.0    +0.00   +0.84   +1.26   +1.72   +2.19   +2.69 |  1.70   <- no gains
          17.0    +0.00   +0.43   +0.41   +0.39   +0.28   +0.04 |  0.31   <- what we use

    17 C, from NIBE. 15.5 C against a 19.3 C room, from OEM's SCOP tool. A 2.5 K median base_DT
    across 383 monitored systems, from heatpumpmonitor.org. Three independent sources, one answer.
    """
    room, dut, spread = 21.0, -15.0, 5.0
    balance = room - DEFAULT_BALANCE_POINT_OFFSET

    def rms(balance_point: float) -> float:
        errors = [
            en442_flow_temp(
                indoor_setpoint=room,
                outdoor_temp=outdoor,
                design_outdoor_temp=dut,
                design_flow_temp=NIBE_CURVE_9[dut],
                design_spread=spread,
                emitter_exponent=1.3,
                balance_point_temp=balance_point,
            )
            - nibe
            for outdoor, nibe in NIBE_CURVE_9.items()
        ]
        return (sum(e * e for e in errors) / len(errors)) ** 0.5

    with_gains = rms(balance)
    without_gains = rms(room)

    assert with_gains < 0.5, (
        f"Our curve is {with_gains:.2f} C RMS away from NIBE's own published curve 9. The emitter "
        f"law is supposed to reproduce it - that is the whole basis for trusting it to say how hot "
        f"the water should be."
    )
    assert with_gains < without_gains / 2, (
        f"Modelling internal gains barely helps ({with_gains:.2f} C RMS with, {without_gains:.2f} C "
        f"without). Either the balance point is wrong or NIBE's curve is not the emitter law."
    )
