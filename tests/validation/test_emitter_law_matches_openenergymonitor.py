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
authoritative about - and holds the demand model identical on both sides to do it.

**The gains term is NOT checked against a curve, because it cannot be.** An earlier draft fitted it
to NIBE's published curve 9 and reported a triumphant RMS. Two separate tests below now show why
that was worthless: the constant-spread and balance-point terms are the same basis function with
opposite signs (so any assumed spread manufactures a matching "gains" figure, even from a curve
with provably zero gains), AND curve 9 is a straight line to within 0.19 C, which cannot resolve
curvature at all. Gains are WATTS over W/K. See const.py.

And the Vaillant heat curve that this project ran for a year is the same law in different clothes:

    TFlow = 2.55 * (HC * (Tset - Tout)) ** 0.78 + Tset          [Andre Kuhne]

1/0.78 = 1.28, which is the radiator exponent 1.3. He fitted it to Vaillant's published curves and
checked it against eBus readings from his own AroTherm to within 0.07 C. It is not a rival model. It
is this one, with the design point folded into a single dimensionless curve number.
"""

from __future__ import annotations

import pytest

from custom_components.effektguard.const import (
    DEFAULT_DESIGN_SPREAD,
    DEFAULT_HEAT_LOSS_COEFFICIENT,
    INTERNAL_GAINS_W,
)
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


def _rms_against_nibe(balance_point: float, spread: float) -> float:
    """RMS error of our curve against NIBE's curve 9, anchored at its -15 C end."""
    room, dut = 21.0, -15.0
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


def test_nibes_published_curve_is_a_straight_line_and_validates_nothing():
    """NIBE's curve cannot be used as evidence for our law, and this is why.

    An earlier version of this suite treated the six digitised points of NIBE's curve 9 as the
    ground truth that "validates" the emitter law - and fitted the internal-gains constant to them.
    Both were mistakes, and this test exists to make them impossible to repeat.

    Fit a straight line to those six points and the residual is 0.19 C. They ARE a straight line.
    Worse, their successive slopes wobble non-monotonically:

        -0.800, -0.740, -0.780, -0.820, -0.880  C per C

    A real emitter law steepens MONOTONICALLY toward cold. This steepens toward WARM in the middle
    of the range. That is digitisation noise, and it is larger than the curvature anyone was trying
    to detect.

    Collinear points confirm every model fitted to them. Curve 9 cannot tell the emitter law from a
    ruler, and it certainly cannot resolve a balance point - it will just fit one to its own noise.
    NIBE's controller interpolates its curves linearly; ours follows EN 442. The gap between them
    is not our error, it is THE TRIM - the whole reason this layer exists.
    """
    ts = sorted(NIBE_CURVE_9)
    n = len(ts)
    sx, sy = sum(ts), sum(NIBE_CURVE_9[t] for t in ts)
    sxx = sum(t * t for t in ts)
    sxy = sum(t * NIBE_CURVE_9[t] for t in ts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - sx * slope) / n
    linear_rms = (sum((slope * t + intercept - NIBE_CURVE_9[t]) ** 2 for t in ts) / n) ** 0.5

    assert linear_rms < 0.25, (
        f"NIBE's published curve 9 now departs from a straight line by {linear_rms:.2f} C RMS. If "
        f"it has become genuinely curved, it could finally discriminate between emitter models - "
        f"and this whole test, plus the reasoning in const.py about why gains cannot be fitted to "
        f"it, would want revisiting."
    )

    step_slopes = [(NIBE_CURVE_9[b] - NIBE_CURVE_9[a]) / (b - a) for a, b in zip(ts, ts[1:])]
    assert step_slopes != sorted(step_slopes, reverse=True), (
        "Curve 9's slopes have become monotonic in the direction a real emitter law predicts. That "
        "would make it evidence rather than noise; re-examine this test before trusting it."
    )


def test_our_curve_stays_within_sight_of_nibes():
    """A sanity BOUND, not a validation. We trim NIBE's curve; we must not fight it.

    The emitter law and NIBE's linear interpolation genuinely disagree - that disagreement is the
    correction this layer is for. But a trim that wandered degrees away from the pump's own curve
    would mean one of the two is broken, and `WEATHER_COMP_MAX_OFFSET` (3.0 C) would then be
    clipping every decision. This keeps us honest without pretending curve 9 proves anything.
    """
    balance = 21.0 - INTERNAL_GAINS_W / DEFAULT_HEAT_LOSS_COEFFICIENT
    rms = _rms_against_nibe(balance, DEFAULT_DESIGN_SPREAD)

    assert rms < 1.0, (
        f"Our flow-temperature curve now sits {rms:.2f} C RMS from NIBE's own published curve 9. "
        f"We are supposed to be trimming that curve, not replacing it. A gap this size means the "
        f"design point, the spread or the gains are misconfigured - and every offset we emit would "
        f"be a correction toward our own error."
    )


def test_a_curve_fit_cannot_measure_internal_gains():
    """The trap that produced the wrong constant, nailed down so nobody walks into it again.

    A previous version of this suite fitted the balance point against NIBE's curve 9, got 4.0 K,
    reported "RMS 0.31 C with gains vs 1.70 C without", and shipped that as evidence. It was not
    evidence. The fit is DEGENERATE:

        a constant spread LIFTS the curve by   (spread / 2) * (1 - phi ** (1/n))
        a balance point   DROPS  the curve by  a term of the same shape, opposite sign

    Both are zero at the design point and grow in mild weather. They are the same basis function.
    So whatever spread you assume, the fit hands you a "gains" figure that absorbs it - and it will
    do so even when the curve you are fitting contains no gains AT ALL.

    Proof, run here rather than asserted: fit our law to Kuhne's Vaillant curve, which is a pure
    power law with PROVABLY ZERO gains, and watch a balance point appear anyway, tracking the
    spread we assumed.

    The lesson is in const.py: gains are WATTS, divided by the house's W/K. Never degrees off a fit.
    """
    room, dut = 20.0, -15.0
    hc = 0.75  # Vaillant curve number, Protons for Breakfast's worked example

    def kuhne(outdoor: float) -> float:
        return 2.55 * (hc * (room - outdoor)) ** 0.78 + room

    def best_fit_offset(assumed_spread: float) -> float:
        """The balance-point offset a fitter would 'discover' in a curve that has none."""
        probes = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0]

        def rms(offset: float) -> float:
            errs = [
                en442_flow_temp(
                    indoor_setpoint=room,
                    outdoor_temp=t,
                    design_outdoor_temp=dut,
                    design_flow_temp=kuhne(dut),
                    design_spread=assumed_spread,
                    emitter_exponent=1.3,
                    balance_point_temp=room - offset,
                )
                - kuhne(t)
                for t in probes
            ]
            return (sum(e * e for e in errs) / len(errs)) ** 0.5

        return min((n / 10.0 for n in range(0, 90)), key=rms)

    near_zero = best_fit_offset(0.01)
    at_five = best_fit_offset(5.0)
    at_ten = best_fit_offset(10.0)

    assert near_zero < 1.0, (
        f"With no spread to absorb, fitting a zero-gains curve should recover ~zero gains; it "
        f"recovered {near_zero:.1f} K. If this fails the degeneracy argument itself is wrong."
    )
    assert at_five > near_zero + 1.5 and at_ten > at_five + 1.5, (
        f"The 'gains' a curve fit reports must track the spread it was given - that is what makes "
        f"the fit worthless as evidence. Got {near_zero:.1f} K / {at_five:.1f} K / {at_ten:.1f} K "
        f"for spreads of 0 / 5 / 10 K. If they no longer diverge, the two terms have stopped being "
        f"degenerate and the balance point could legitimately be fitted after all - which would be "
        f"news, and would want a very careful look before anyone acts on it."
    )
