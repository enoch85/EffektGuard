"""Month-long EffektGuard simulation harness (issue #18 deep test).

Drives the REAL DecisionEngine (custom_components.effektguard.optimization)
through January 2026 at 5-minute steps with REAL weather (open-meteo archive,
Stockholm) and REAL Nordpool SE3 day-ahead prices, against a physical
house + NIBE pump plant model. Two house configurations:

  wooden:   timber frame, radiators, thermal_mass 0.7, HLC 150 W/K, tau ~30 h,
            NIBE F750 profile (most common ASHP)
  concrete: concrete slab UFH, thermal_mass 1.8, HLC 180 W/K, tau ~80 h,
            NIBE F1155 profile (GSHP, added in this change)

Target indoor temperature: 22.0 C for both.

Plant model (deliberately simple but honest):
  - Heating curve: flow_target = 20 + slope*(20 - Tout)/30 + offset
    (NIBE curve 9 approximation: slope 1.5 like the adapter's estimate)
  - Compressor hysteresis on degree minutes: start at DM <= -60, stop DM >= 0
  - Flow ramps toward target when running (0.5 C/min), decays toward
    indoor temperature when off (0.1 C/min)
  - DM integrates (flow_actual - flow_target) minutes, clamped to [-3000, 100]
  - Heat output Q = K_EMIT * (flow - Tin); K_EMIT sized for design point
  - Electrical power = Q / COP(Tout) from the pump profile curve
  - Aux heat: DM below -1500 adds electric aux steps (like real NIBE)

The engine's wall-clock reads (dt_util.now/utcnow) are monkeypatched to the
simulation clock each step so price-quarter and forecast logic see sim time.

Outputs: summary JSON + violations + downsampled trace per config, written to
sim-results/. Run: .venv/bin/python sim_harness.py [--selftest]
"""

import asyncio
import functools
import json
import sys
import zoneinfo

import numpy as np
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter, PriceData
from custom_components.effektguard.const import (
    CONF_GESPOT_ENTITY,
    INTERNAL_GAINS_W,
    POWER_SOURCE_EXTERNAL_METER,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)
from custom_components.effektguard.utils.emitter import en442_flow_temp
from custom_components.effektguard.utils.offset import integer_offset_for
from custom_components.effektguard.utils.time_utils import QUARTERS_PER_HOUR
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)

from custom_components.effektguard.models.nibe import (
    NibeF730Profile,
    NibeF750Profile,
    NibeF1155Profile,
    NibeF2040Profile,
    NibeS1155Profile,
)
from custom_components.effektguard.optimization.billing_period import BillingPeriodAccumulator
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TZ = zoneinfo.ZoneInfo("Europe/Stockholm")
STEP_MIN = 5
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

# The archived Nordpool files quote SEK/MWh; GE-Spot publishes what the user
# configured, which for a Swedish user is conventionally öre/kWh.
# 1 SEK/MWh = 0.1 öre/kWh.
ORE_PER_KWH_FROM_SEK_PER_MWH = 0.1
GESPOT_UNIT_ORE = "öre/kWh"

# Plant constants
DM_START = -60.0
DM_STOP = 0.0
# THE F2040 HAS NO IMMERSION HEATER. It is an outdoor monobloc; its electric addition lives in the
# indoor module it is paired with (a VVM or SMO), which this package does not model. Every other
# machine's heater is on its profile, from its datasheet. This is the fallback for the F2040 alone,
# and it is an ASSUMPTION about that indoor module - not a NIBE figure - so it is named as one.
#
# It matters: the immersion burn is a headline number in the saturated-compressor finding, and the
# simulator used to apply this one invented value to all five machines, matching none of them.
ASSUMED_INDOOR_MODULE_HEATER_KW = 3.0
STANDBY_KW = 0.1  # controller, pumps, standby losses
J_PER_KWH = 3_600_000.0

# Float arithmetic only. A real leak is orders of magnitude bigger: the one this replaced a fake
# audit to catch was 183 kWh.
WATER_NODE_LEAK_BUDGET_KWH = 0.5

# How far the run's seasonal COP may exceed the datasheet's own figure for the weather it saw.
# The healthy range, measured across all five houses and all four scenarios, is 0.72 to 1.03; the
# margin is for the mild hours when the curve runs water below the W35 rating point and the pump
# legitimately beats its rating. Doubling the plant's COP lands at 1.5 to 2.1 and is caught on
# every house - which is the whole point, because the identity this replaced called that PASS.
COP_ENVELOPE_TOLERANCE = 1.15

# Heat capacity of the water loop and the emitter metal it fills. Roughly 70 L of water
# (0.081 kWh/K) plus the steel of the radiators. Without this the plant HANDS OUT the heat stored
# in the water for free every time the compressor stops, and charges nothing to put it back.
WATER_LOOP_J_PER_K = 350_000.0  # ~0.10 kWh/K
COMPRESSOR_RESPONSE_S = 900.0  # how briskly the compressor closes on its flow target

# Bounds on the degree-minute integrator. Reaching the floor is not a normal operating state: it
# means the deficit grew without limit despite the curve offset AND the auxiliary heater, so the
# recovery system failed. The harness treats it as such.
DM_INTEGRATOR_FLOOR = -3000.0
DM_INTEGRATOR_CEILING = 100.0

# Above this the house is not "warm", it is being cooked - and on a heat pump it is usually the
# immersion heater doing it, at COP 1.0.
INDOOR_CEILING = 26.0
TOMORROW_VISIBLE_HOUR = 13  # Nordpool day-ahead published ~12:45 CET
QUARTER_MINUTES = 15
SIM_DAYS = 31
# The --dst run: Sat 24 Oct through Mon 26 Oct 2026, spanning the fall-back night.
DST_SIM_DAYS = 3
# 2026-10-25: at 03:00 CEST the clock goes back to 02:00 CET, so the day is 25 hours long and
# the wall-clock hour 02 is metered twice. From the tz database, not from an assumption.
DST_FALL_BACK_DAY = "2026-10-25"
DST_FALL_BACK_HOURS = 25

# CAPACITY AND COP NOW COME FROM THE DATASHEET. See HouseConfig.capacity_kw_at / cop_at.
#
# What used to be here was ASHP_DERATE_PER_C = 0.025, "fraction of rated output lost per C below
# A7", justified by a comment claiming the EN 14511 rating points "trace a near-linear decline".
# They trace a near-linear RISE. The whole derating was invented, backwards, and cited to a
# standard that says the opposite. It is gone.
#
# COP is set by the LIFT, not by the weather. These place the source and the condenser.
KELVIN = 273.15
# The exergy penalty for hotter water, BEYOND what Carnot already accounts for. Measured on the
# machines whose datasheets identify it (F1155/S1155: -0.0055/K; F2040: -0.0028/K) and imported as
# a STATED ASSUMPTION by the two whose datasheets cannot (F750/F730 confound load with flow).
FLOW_EXERGY_PENALTY_PER_K = -0.0046

# Physical bounds on the exergy efficiency. A real machine achieves 30-70% of Carnot; these only
# stop a fit extrapolating off the end of its own data into nonsense, which the first version did.
# a + b*load + c*(flow-35). Three of them, so a fit needs at least four points to have any
# degrees of freedom at all - see HouseConfig.exergy_fit.
EXERGY_FIT_PARAMETERS = 3

MIN_EXERGY_EFFICIENCY = 0.15
MAX_EXERGY_EFFICIENCY = 0.80

COP_RATING_FLOW_C = 35.0  # EN 14511 rating point is W35: the profile's COP curve is measured here
CONDENSER_APPROACH_K = 5.0  # refrigerant condenses this far above the water it is heating
EVAPORATOR_APPROACH_K = 5.0  # and evaporates this far below the source it is drawing from
MIN_LIFT_K = 10.0  # a compressor cannot usefully run at zero lift; bound the division
EXHAUST_AIR_SOURCE_C = 20.0  # F750/F730 draw ~20 C indoor extract air, all year
BRINE_SOURCE_C = 0.0  # F1155/S1155 draw ~0 C brine, stable year-round


# Comfort accounting matches the engine's configured tolerance (not a looser
# ad-hoc band): minutes below target-tolerance count as under-heating.
TARGET_INDOOR = 22.0
COMFORT_TOLERANCE = 0.5
# THE DESIGN TEMPERATURE IS THE SIZING CONVENTION, AND IT IS LOAD-BEARING.
#
# Houses are sized from their pump's Pdesignh, so the design temperature decides how big each house
# is - and therefore whether the pump ever saturates at all. It moved the F750 between "saturates in
# a cold snap" and "does not". That is exactly the kind of arbitrary, unexamined choice this audit
# exists to find, and it used to be -15.0 with no justification whatsoever.
#
# NIBE declares Pdesignh at BOTH EN 14825 reference climates, and both are published:
#
#     cold    (-22 C)   the Nordic reference. A Swedish house is sized here.
#     average (-10 C)   the central-European reference. The F730's ErP block confirms it by
#                       declaring TOL = -10 C.
#
# This is a Swedish integration simulating a Swedish January, so the COLD reference is the honest
# default. The average-climate sizing is not discarded - it is a real case (a pump under-sized for
# its house, which is the commonest installation fault there is) and `--undersized` runs it. The
# saturation finding is reported across BOTH, because it must not depend on which one I picked.
EN14825_COLD_DESIGN_C = -22.0
EN14825_AVERAGE_DESIGN_C = -10.0
DESIGN_OUTDOOR = EN14825_COLD_DESIGN_C

# Sizing a house at the average-climate design point instead of the cold one makes it this much
# bigger for the same pump - i.e. it is the same as fitting a pump one size too small.
UNDERSIZED_PUMP_FACTOR = (22.0 - EN14825_COLD_DESIGN_C) / (22.0 - EN14825_AVERAGE_DESIGN_C)
DESIGN_SPREAD = 5.0
RADIATOR_EXPONENT = 1.3  # EN 442
UFH_EXPONENT = 1.1  # EN 1264
OVERSHOOT_TOLERANCE = 1.5  # overshoot band stays wider; heat is banked, not lost

# Illustrative Swedish effect tariff (SEK per kW of the mean of the top-3
# daily quarter-hour-mean peaks, per month). Rate is fictional-but-typical;
# the point is comparing runs, not billing accuracy.
# Ellevio's published rate, and it lives in const.py now - see SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH.
# The harness used to carry its own copy and call it "fictional-but-typical". It is neither: it is
# Ellevio's real 81,25 kr/kW/month, and production carried a DIFFERENT unsourced number (50.0).
EFFECT_TARIFF_SEK_PER_KW = SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH


@dataclass
class HouseConfig:
    name: str
    thermal_mass: float  # EffektGuard relative scale
    insulation_quality: float
    hlc_w_per_k: float  # heat loss coefficient
    tau_hours: float  # thermal time constant
    profile: object
    heating_type: str
    design_flow: float  # flow temp at design outdoor -15C

    @property
    def capacity_j_per_k(self) -> float:
        return self.hlc_w_per_k * self.tau_hours * 3600.0

    @property
    def emitter_exponent(self) -> float:
        """EN 442 / EN 1264 exponent for this house's emitters."""
        return UFH_EXPONENT if self.heating_type != "radiator" else RADIATOR_EXPONENT

    @property
    def design_excess(self) -> float:
        """Mean water temperature above the room at the design point."""
        return self.design_flow - DESIGN_SPREAD / 2.0 - TARGET_INDOOR

    @property
    def design_heat_w(self) -> float:
        """Emitter output at the design point - net of the free heat the house makes itself."""
        return self.hlc_w_per_k * (TARGET_INDOOR - DESIGN_OUTDOOR) - INTERNAL_GAINS_W

    def heat_output_w(self, flow: float, indoor: float) -> float:
        """Emitter output, by the EN 442 characteristic equation.

            Q / Q_design = (dT_mean / dT_mean_design) ** n

        A LINEAR emitter (n = 1) is not a radiator: it exaggerates output at low flow
        temperatures, which flatters a controller that under-supplies.

        THE SPREAD IS CONSTANT. This model used to widen it with load - `DESIGN_SPREAD *
        load_ratio` - and iterate to a fixed point. That is a FIXED-SPEED circulator on a wet
        boiler: constant mass flow, so the flow-return spread rises and falls with the heat being
        carried. A NIBE modulates its circulator (GP1) to HOLD the commissioned spread and varies
        the flow RATE instead, which is why the controller's own emitter law holds it constant too.

        With the spread fixed there is no fixed point left to solve: the mean water temperature is
        just `flow - spread/2`, and the output follows directly.
        """
        excess = flow - DESIGN_SPREAD / 2.0 - indoor
        if excess <= 0:
            return 0.0
        return self.design_heat_w * (excess / self.design_excess) ** self.emitter_exponent

    def curve_flow_temp(self, outdoor: float, tuned: bool = False) -> float:
        """The supply temperature the pump's own heating curve calls for, at offset 0.

        A correctly tuned NIBE curve follows the emitter law, not a straight line. NIBE's
        published curve 9 (offset 0) reads 41 C at 0 C outdoor; the emitter law gives 40.6 C,
        a straight line between the same anchors gives 38.7 C. Modelling the curve as linear
        makes it under-supply everywhere between its endpoints, and the house cannot hold target
        even with the controller switched off.
        """
        # A STOCK NIBE CURVE HAS NO INTERNAL-GAINS TERM, and that is not an oversight in this
        # model - it is what the hardware does. The installer picks a curve number and the pump
        # draws a line from the design point; nothing in it knows that the occupants and the
        # fridge are supplying several hundred watts. So a stock curve OVER-SUPPLIES in mild
        # weather, and the simulated baseline house duly sits at 22.5 C against a 22.0 C target.
        #
        # WHICH MEANS THE DEFAULT BASELINE IS A SOFT ONE, AND I WAS QUOTING SAVINGS AGAINST IT.
        # A diligent owner trims the curve down until the house actually holds target, and against
        # THAT baseline the optimiser's saving falls from 1.5-4.4 % to 0.5-1.4 %. Most of what I
        # reported was the controller correcting a mis-tuned curve rather than optimising anything.
        #
        # Both yardsticks are real and they answer different questions, so the harness offers both:
        # `--tuned-baseline` gives the pump a curve that knows about the gains, which is the honest
        # question "what is this worth to someone whose pump is already set up properly?"
        return en442_flow_temp(
            indoor_setpoint=TARGET_INDOOR,
            outdoor_temp=outdoor,
            design_outdoor_temp=DESIGN_OUTDOOR,
            design_flow_temp=self.design_flow,
            design_spread=DESIGN_SPREAD,
            emitter_exponent=self.emitter_exponent,
            balance_point_temp=(
                TARGET_INDOOR - INTERNAL_GAINS_W / self.hlc_w_per_k if tuned else None
            ),
        )

    @property
    def immersion_heater_kw(self) -> float:
        """This machine's immersion heater, from its datasheet. The F2040 has none.

        The plant used to give every house the same invented 3.0 kW, which is no machine's actual
        setting. NIBE ships the F750 and F730 with a 6.5 kW heater set to 3.5 kW at delivery, and
        the F1155-12/S1155-12 with a 7 kW heater in seven automatic steps.
        """
        published = float(getattr(self.profile, "immersion_heater_kw", 0.0))
        return published if published > 0.0 else ASSUMED_INDOOR_MODULE_HEATER_KW

    @property
    def dm_aux_limit(self) -> float:
        """Aux-heat threshold, taken from the pump profile rather than restated.

        The correct value for real NIBE hardware is contested (see the audit's
        F-112). Reading it from the profile means the plant model tracks whatever
        the integration believes, instead of silently diverging from it.
        """
        return float(self.profile.dm_threshold_aux_swedish)

    def source_temp_c(self, outdoor_temp: float) -> float:
        """The temperature of the heat SOURCE the compressor is lifting from.

        A heat pump's efficiency is set by the LIFT - how far it has to raise the heat - not by
        the weather as such. What the weather changes is the source, and only for some machines:

        - Outdoor air (F2040): the source IS the outdoor air.
        - Exhaust air (F750, F730): ~20 C indoor extract air, all year. The weather barely touches
          it, which is why these pumps hold their COP through a cold snap.
        - Ground source (F1155, S1155): ~0 C brine, stable year-round.
        """
        if getattr(self.profile, "supports_exhaust_airflow", False):
            return EXHAUST_AIR_SOURCE_C
        if "GSHP" in getattr(self.profile, "model_type", ""):
            return BRINE_SOURCE_C
        return outdoor_temp

    @functools.cached_property
    def exergy_fit(self) -> tuple[float, float, float]:
        """(a, b, c) in  eta = a + b*load + c*(flow - 35),  fitted to this machine's OWN datasheet.

        THE COP MODEL USED TO BE ANCHORED ON A CURVE THAT WAS INVENTED. Every profile carried an
        outdoor-keyed `cop_curve`, called "Real-world COP curve (tested and validated)" and sourced
        to "NIBE F750 datasheet, Swedish NIBE forum validation". The F750 and F730 shipped
        byte-identical curves despite being different machines, and the number 5.0 - labelled "Best
        COP" - appears in neither datasheet. The simulator computed a month of kWh and SEK from it
        and I published the savings.

        A heat pump's COP is the Carnot limit between its source and its sink, degraded by how good
        the machine is, how hard it is pushed, and how hot the water is. All of that is IN the
        datasheet:

            eta  = COP_published / Carnot(source, flow)    at each published rating point
            load = PH_published / PH_max                   at that same point

        AND MY FIRST VERSION OF THIS FIT WAS ITSELF A FICTION. Fitting all three of the F750's
        points gave b = +0.586 - efficiency RISING with load, which is backwards - and it
        extrapolated to COP 9.86 at full load and 35 C flow. The simulator visits that condition,
        and the Carnot guard (ceiling 12.5 there) would have waved it straight through.

        The cause was in the datasheet and I had not read it closely enough: the F750's two
        MINIMUM-frequency points differ by AIRFLOW (108 vs 252 m3/h), not by compressor load. More
        ventilation air, more source heat, higher output AND higher COP. They are not a load pair.
        Drop the off-rating airflow point and only TWO usable points remain - and between them load
        and flow move TOGETHER, so the F750's datasheet cannot separate the two effects at all. Any
        fit that claims to is fitting noise.

        So the flow penalty is MEASURED where the data identifies it, and IMPORTED where it does
        not, and the difference is stated rather than hidden:

            F1155 / S1155   c = -0.0055 /K   measured (0/35 vs 0/45, and 10/35 vs 10/45)
            F2040           c = -0.0028 /K   measured (7/35 vs 7/45, and 2/35 vs 2/45)
            F750 / F730     NOT IDENTIFIABLE - the mean of the above, as a stated ASSUMPTION

        That assumption is not a measurement of the F750 and nothing here pretends it is.
        """
        rated_airflow = max(
            (p.airflow_m3h for p in self.profile.datasheet_points if p.airflow_m3h), default=None
        )
        points = [
            p
            for p in self.profile.datasheet_points
            if rated_airflow is None or p.airflow_m3h == rated_airflow
        ]
        ph_max = self.profile.max_heat_output_kw

        def eta(point) -> float:
            return point.cop / self.carnot_at(point.source_temp_c, point.flow_temp_c)

        # THE FLOW PENALTY IS ONLY IDENTIFIABLE WITH MORE POINTS THAN PARAMETERS.
        #
        # My first identifiability test asked whether any two points shared a source temperature
        # and differed in flow. The F750's two rated-airflow points do - but they ALSO differ in
        # load, so the two effects are still confounded, and lstsq happily solved 3 unknowns from
        # 2 equations and returned a minimum-norm answer with b = +0.10: efficiency rising with
        # load. Backwards again, from a test I wrote to catch exactly that.
        #
        # Three parameters need at least four points. That is the whole condition.
        if len(points) > EXERGY_FIT_PARAMETERS:
            design = np.array(
                [
                    [1.0, p.heat_output_kw / ph_max, p.flow_temp_c - COP_RATING_FLOW_C]
                    for p in points
                ]
            )
            target = np.array([eta(p) for p in points])
            a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
            return float(a), float(b), float(c)

        c = FLOW_EXERGY_PENALTY_PER_K
        design = np.array([[1.0, p.heat_output_kw / ph_max] for p in points])
        target = np.array([eta(p) - c * (p.flow_temp_c - COP_RATING_FLOW_C) for p in points])
        a, b = np.linalg.lstsq(design, target, rcond=None)[0]
        return float(a), float(b), c

    def exergy_efficiency(self, load_fraction: float, flow_temp: float) -> float:
        """How much of Carnot this machine actually achieves, here. From its own datasheet."""
        a, b, c = self.exergy_fit
        eta = a + b * min(max(load_fraction, 0.0), 1.0) + c * (flow_temp - COP_RATING_FLOW_C)
        return min(max(eta, MIN_EXERGY_EFFICIENCY), MAX_EXERGY_EFFICIENCY)

    def cop_at(self, outdoor_temp: float, flow_temp: float, load_fraction: float = 1.0) -> float:
        """COP = exergy_efficiency(load, flow) x Carnot(source, flow). No invented curve.

        Note what is NOT here: the outdoor temperature. It enters only through `source_temp_c`, and
        for four of the five machines it does not enter at all - an exhaust-air pump breathes 20 C
        house air and a ground-source pump drinks 0 C brine, whatever the weather is doing. The
        model this replaces dropped an F1155's COP from 5.3 to 3.3 because the air outside got
        cold, while its heat source sat at 0 C and never moved.
        """
        source = self.source_temp_c(outdoor_temp)
        return max(
            1.0,
            self.exergy_efficiency(load_fraction, flow_temp) * self.carnot_at(source, flow_temp),
        )

    def carnot_at(self, source_temp: float, flow_temp: float) -> float:
        """The thermodynamic ceiling between a SOURCE and a SINK."""
        t_cond = flow_temp + CONDENSER_APPROACH_K + KELVIN
        t_evap = source_temp - EVAPORATOR_APPROACH_K + KELVIN
        return t_cond / max(t_cond - t_evap, MIN_LIFT_K)

    def carnot_cop(self, outdoor_temp: float, flow_temp: float) -> float:
        """The Carnot bound at this weather. The harness asserts the plant never beats it."""
        return self.carnot_at(self.source_temp_c(outdoor_temp), flow_temp)

    def capacity_kw_at(self, outdoor_temp: float) -> float:
        """The most heat this machine can make right now. FROM ITS DATASHEET.

        AND IT DOES NOT DERATE AS IT GETS COLDER. It rises.

        This method used to be:

            derate = 1.0 - ASHP_DERATE_PER_C * max(0.0, ASHP_RATING_POINT_C - outdoor_temp)
            return rated * max(ASHP_MIN_CAPACITY_FRACTION, derate)

        with a comment claiming "the EN 14511 rating points (A7/W35, A2/W35, A-7/W35, A-15/W35)
        trace a near-linear decline". They trace a near-linear RISE. The F2040-8's published
        capacity goes 3.86 -> 5.11 -> 6.60 kW from +7 to +2 to -7 C, because it is an INVERTER: at
        its +7 rating point it is throttled back to part load, and as the weather cools it ramps
        the compressor UP. What collapses in the cold is the COP (4.65 -> 3.76 -> 2.68), not the
        capacity. There is no derating table in the datasheet because there is no derating.

        I invented that citation and got the sign of the effect backwards, and the entire
        saturated-compressor finding (F-124) was built on the result.

        The capacity is now interpolated from the machine's own published points, against its own
        SOURCE temperature - which for four of the five machines is a constant, so their capacity
        is flat, which is correct and is what the datasheets show. Below the coldest published
        point the curve is HELD, because NIBE tabulates nothing there (only a graph), and holding
        is the honest thing to do with the end of the evidence.
        """
        # THE MODULATION ENVELOPE WINS WHERE THE DATASHEET PUBLISHES ONE.
        #
        # "Heating capacity (PH): 3 - 12 kW" is what an F1155-12 can actually deliver. Its 0/35
        # rating point of 5.06 kW is its output at NOMINAL (50 Hz) frequency, and reading THAT as
        # the machine's ceiling would halve a 12 kW heat pump. The exhaust-air pumps publish their
        # maximum directly - their third rating point is explicitly "max compressor frequency" - so
        # for them the envelope and the top rating point are the same number.
        if self.profile.heating_capacity_range_kw[1] > 0.0:
            return self.profile.heating_capacity_range_kw[1]

        # Only the F2040 has no envelope row, and it is the only machine whose source IS the
        # weather. Its capacity is the EN 14511 curve against SOURCE temperature, HELD below the
        # coldest published point - because NIBE tabulates nothing below -7 C, only a graph.
        #
        # THAT MEANS THIS UNDERSTATES THE F2040. Its true maximum below -7 C is not a number I
        # have. Any saturation the simulator shows for this machine is therefore an UPPER BOUND on
        # the real thing, and must never be reported as a measured failure. F-124 was.
        source = self.source_temp_c(outdoor_temp)
        by_source = sorted(
            {
                point.source_temp_c: point
                for point in self.profile.datasheet_points
                if point.flow_temp_c == COP_RATING_FLOW_C
            }.items()
        )
        if len(by_source) < 2:
            return self.profile.max_heat_output_kw

        temps = [t for t, _ in by_source]
        caps = [point.heat_output_kw for _, point in by_source]

        # BELOW THE COLDEST RATING POINT, NIBE'S OWN ErP DECLARATION CLOSES THE MODEL.
        #
        # The manual tabulates the F2040's maximum output down to -7 C and no further - below that
        # it gives a graph. But the ErP says the machine covers a Pdesignh design load with Psup of
        # supplementary heat, so the COMPRESSOR must deliver (Pdesignh - Psup) at the design
        # temperature. For the F2040-8 that is 8.2 - 1.1 = 7.1 kW at -10 C, against 6.60 kW at -7 C.
        #
        # So capacity keeps RISING below -7 C, and the rate is not invented - it is whatever gets
        # from the last measured point to the manufacturer's own declaration. Below the design
        # temperature the curve is HELD, because that is where every published statement stops.
        #
        # The old model derated 2.5 %/C in the opposite direction and blamed EN 14511 for it.
        pdesign = self.profile.design_heat_load_kw
        psup = self.profile.supplementary_heat_kw
        if source < temps[0] and pdesign > 0.0 and psup > 0.0:
            at_design = pdesign - psup
            if EN14825_COLD_DESIGN_C < temps[0]:
                span = temps[0] - EN14825_COLD_DESIGN_C
                frac = min(1.0, (temps[0] - source) / span)
                return caps[0] + (at_design - caps[0]) * frac
            return at_design

        return float(np.interp(source, temps, caps))


# EVERY HOUSE IS SIZED FROM ITS PUMP'S OWN Pdesignh. It used to be sized from nothing at all.
#
# NIBE declares, for every machine, the design heat load it is certified for. That is the
# manufacturer's own statement of how big a house the pump is for, and it is the only sourced way
# to size a simulated building:
#
#     hlc = (Pdesignh + internal_gains) / (target_indoor - design_outdoor)
#
# The houses used to carry invented heat-loss coefficients, and three of the five paired a pump
# with a house it was far too big for:
#
#     concrete_f1155   6.06 kW house, 12 kW pump   -> 2.0x oversized
#     villa_s1155      5.32 kW house, 12 kW pump   -> 2.3x oversized
#     apartment_f730   2.73 kW house,  5 kW pump   -> 1.8x oversized
#
# THAT DECIDED WHAT THE SIMULATION WAS ABLE TO FIND. A pump with twice the capacity its house needs
# cannot saturate, cannot fall behind, and cannot reach for its immersion heater - so it can never
# exercise the degree-minute recovery ladder at all. I reported that "the ground-source houses never
# engage the emergency ladder" as if it were a fact about the controller. It was a fact about my
# sizing. The only two correctly-sized systems in the set were the only two that failed.
#
# DESIGN_OUTDOOR is the Swedish DVUT (dimensionerande vinterutetemperatur) for mid-Sweden; Boverket
# puts Stockholm near -16 C. It is a stated convention of this harness, not a datasheet figure, and
# every house is sized against it consistently, so the PAIRING is what is being asserted here.
# WHERE EVERY NUMBER IN THIS PLANT MODEL CAME FROM.
#
# This table exists because the numbers that came from nowhere were the ones that decided what the
# simulation was able to find, and nobody could tell them apart from measurements. The COP curves
# were called "Real-world ... (tested and validated)" and sourced to "NIBE F750 datasheet, Swedish
# NIBE forum validation"; they were in neither. The capacity derating cited EN 14511 and ran in the
# opposite direction to it. The houses had heat-loss coefficients from nowhere at all.
#
# Each entry is exactly one of two things, and the difference is the point:
#
#   SOURCED  a document, quoted, that a reader can open.
#   ASSUMED  no published source exists. Then the sensitivity is MEASURED and stated here, because
#            an unsourced number that moves the answer is a finding about the modeller.
#
# tests/validation/test_every_simulator_constant_says_where_it_came_from.py enforces it: a new
# physical constant cannot be added to this file without declaring one or the other.
PROVENANCE: dict[str, str] = {
    # ---- the pump, from NIBE ----
    "DM_START": (
        "SOURCED: NIBE starts the compressor at -60 degree minutes. docs/research/01_degree_"
        "minutes.md, from the NIBE manual (menu 4.9.3)."
    ),
    "DM_STOP": "SOURCED: NIBE stops the compressor at 0 degree minutes. docs/research/01.",
    "COP_RATING_FLOW_C": (
        "SOURCED: EN 14511 rates heat pumps at W35. Every NIBE datasheet's rating points say so - "
        "'A20(12)W35', '0/35 nominal', 'A7/W35'."
    ),
    "DESIGN_SPREAD": (
        "SOURCED: EN 14511 dT5K - the 5 K water-side temperature difference the standard rates at. "
        "The F2040 installer manual's table header says it verbatim: 'Output data according to "
        "EN 14511 dT5K'. IHB EN 1848-8/231846 p.65."
    ),
    "RADIATOR_EXPONENT": "SOURCED: EN 442 panel radiators, n = 1.3. docs/research/02_emitter_law.md.",
    "UFH_EXPONENT": "SOURCED: EN 1264 underfloor heating, n = 1.1. docs/research/02_emitter_law.md.",
    "EXHAUST_AIR_SOURCE_C": (
        "SOURCED: the F750 and F730 are rated at A20(12) - 20 C dry-bulb extract air. That IS their "
        "heat source, and it does not change with the weather. NIBE F750 datasheet, part no. "
        "066 063."
    ),
    "BRINE_SOURCE_C": (
        "SOURCED: the F1155 and S1155 are rated at B0 - 0 C incoming brine. Their capacity chart's "
        "x-axis is labelled 'Incoming brine temp, C'. F1155 installer manual IHB EN 2008-5/331379."
    ),
    "DST_FALL_BACK_HOURS": (
        "SOURCED: the IANA time zone database (https://www.iana.org/time-zones), zone "
        "Europe/Stockholm. On 2026-10-25 the offset goes from +02:00 to +01:00 at 03:00 local, so "
        "the wall-clock hour 02 is metered twice and the day is 25 hours long. EU Directive "
        "2000/84/EC fixes the transition to the last Sunday of October across the union. Verified "
        "by stepping the absolute time line through the zone rather than by assuming it: the "
        "harness counts 25 distinct billing hours on that date and fails the run if it does not."
    ),
    "EN14825_COLD_DESIGN_C": (
        "SOURCED: EN 14825 cold-climate reference design temperature. NIBE declares a Pdesignh at "
        "this reference for every machine, and the houses are sized from it."
    ),
    "EN14825_AVERAGE_DESIGN_C": (
        "SOURCED: EN 14825 average-climate reference. The F730's own ErP block confirms it by "
        "declaring TOL = -10 C. Used by --undersized, which sizes a house here and fits it a pump "
        "certified for the cold reference - the commonest installation fault there is."
    ),
    "STANDBY_KW": (
        "SOURCED (as a range; the value is mid-band): the F750 datasheet, part no. 066 063, "
        "publishes its running auxiliaries - 'Drive output heating medium pump 2: 5-45 W' and "
        "'Driving power exhaust air fan: 25-140 W', so 30-185 W with the compressor running. "
        "0.1 kW sits inside that. Swept "
        "across the full published band the saturation findings do not move at all (2.1-2.2x vs "
        "2.2x baseline)."
    ),
    "FLOW_EXERGY_PENALTY_PER_K": (
        "SOURCED where the datasheets identify it, ASSUMED where they cannot, and the difference "
        "is stated in HouseConfig.exergy_fit. Measured from the EN 14511 rating points of the "
        "F1155/S1155 (IHB EN 2008-5/331379: -0.0055/K) and the F2040 (IHB EN 1848-8/231846: "
        "-0.0028/K), which publish W35 and W45 at the same source and load. The F750/F730 confound "
        "load with flow and cannot identify it, so they inherit the mean of the two."
    ),
    "ASSUMED_INDOOR_MODULE_HEATER_KW": (
        "ASSUMED. The F2040 has NO immersion heater - it is an outdoor monobloc and its electric "
        "addition lives in the paired indoor module (VVM/SMO), which this package does not model. "
        "Every other machine's heater is on its profile, from its datasheet. Sensitivity: the F750's "
        "cold-snap burn moved 38.1 -> 35.8 kWh when the heaters were sourced per machine, and the "
        "saturation finding did not move."
    ),
    # ---- the plant, where NIBE publishes nothing ----
    "CONDENSER_APPROACH_K": (
        "ASSUMED. No datasheet publishes the refrigerant's approach temperatures. The exergy fit "
        "ABSORBS them at the rating points - a different Carnot gives a different eta that "
        "reproduces the same published COP - so the datasheet is matched whatever this is. Away "
        "from the rating points it matters, by up to 42 % on an extrapolation to W55 full load. "
        "Sensitivity, swept 3-7 K through the whole simulation: seasonal cost moves +/-2 %, and the "
        "saturation finding does not move AT ALL, because saturation is a capacity constraint and "
        "not an efficiency one."
    ),
    "EVAPORATOR_APPROACH_K": "ASSUMED. See CONDENSER_APPROACH_K - same assumption, same sensitivity.",
    "WATER_LOOP_J_PER_K": (
        "ASSUMED, and only half of it could be sourced. The F750 publishes its own buffer: 'Volume "
        "boiler section (of which buffer vessel) litre 35 (25)' - 35 L of water is 146 kJ/K. The "
        "EMITTER side is a property of the HOUSE, and NIBE publishes no system volume for it (the "
        "manuals only say 'if the climate system volume is too small ... supplement with a buffer "
        "vessel'). 350 kJ/K is about 84 L of water-equivalent: the pump's 35 L plus a radiator "
        "circuit. Sensitivity, halved and doubled: the saturation finding holds throughout "
        "(2.0-3.0x over what physics forces, houses cooked to 27.8-30.3 C, against 2.0-2.5x and "
        "29.1-29.8 C at the committed value)."
    ),
    "COMPRESSOR_RESPONSE_S": (
        "ASSUMED. How briskly the compressor closes on its flow target. No datasheet publishes it. "
        "Sensitivity, swept 300-1800 s: the saturation finding holds throughout (1.9-2.4x, houses "
        "at 29.1-29.4 C)."
    ),
}

HOUSES = [
    HouseConfig(
        name="wooden_f750",  # exhaust air, radiators, light timber frame. ~130 m2.
        thermal_mass=0.7,
        insulation_quality=1.0,
        hlc_w_per_k=127.0,  # F750 Pdesignh 5.0 kW at the EN 14825 COLD design point (-22 C)
        tau_hours=30.0,
        profile=NibeF750Profile(),
        heating_type="radiator",
        design_flow=50.0,
    ),
    HouseConfig(
        name="concrete_f1155",  # ground source, underfloor, heavy slab. A LARGE villa, ~280 m2.
        thermal_mass=1.8,
        insulation_quality=1.2,
        hlc_w_per_k=286.0,  # F1155-12 Pdesignh 12 kW at -22 C. Was 180 - the pump was twice the house.
        tau_hours=80.0,
        profile=NibeF1155Profile(),
        heating_type="concrete_ufh",
        design_flow=38.0,
    ),
    HouseConfig(
        name="apartment_f730",  # DELIBERATELY OVERSIZED, and that is the point of this one.
        #
        # The F730 is the SMALLEST exhaust-air machine NIBE makes, and a small flat cannot buy a
        # smaller one. So a 2.7 kW flat gets a 5 kW pump, and that is not a modelling error - it is
        # what actually happens. It is kept, and named, so that the set contains one system where
        # the pump has headroom to spare. The difference between this house and the other four is
        # now a STATED scenario rather than an accident of numbers nobody checked.
        thermal_mass=0.9,
        insulation_quality=1.3,
        hlc_w_per_k=90.0,  # 2.7 kW load against a 5 kW pump: 1.8x oversized, on purpose
        tau_hours=45.0,
        profile=NibeF730Profile(),
        heating_type="radiator",
        design_flow=45.0,
    ),
    HouseConfig(
        name="villa_s1155",  # S-series ground source, timber underfloor. A large villa.
        thermal_mass=1.2,
        insulation_quality=1.1,
        hlc_w_per_k=286.0,  # S1155-12 Pdesignh 12 kW at -22 C. Was 160 - the pump was 2.3x the house.
        tau_hours=55.0,
        profile=NibeS1155Profile(),
        heating_type="timber_ufh",
        design_flow=40.0,
    ),
    HouseConfig(
        name="airsource_f2040",  # outdoor air. The only machine whose source IS the weather.
        #
        # EVERY NUMBER HERE COMES FROM THE SAME COLUMN OF THE DATASHEET, and it did not used to.
        # NIBE publishes the F2040-8's capacity and COP at 35 C flow, and its Pdesignh separately
        # for the 35 C and 55 C applications (9.0 and 10.0 kW in a cold climate). The house was
        # sized from the 35 C Pdesignh and then run at a 55 C design flow, where the machine is
        # weaker - three inputs from three different columns. It is a low-temperature (underfloor)
        # system now, so the capacity curve, the COP and the design load all describe one machine in
        # one application.
        #
        # WHAT REMAINS UNKNOWN, and it bounds every conclusion drawn from this house: NIBE tabulates
        # the F2040's maximum output only down to -7 C. Below that the manual gives a GRAPH and no
        # numbers, so the model holds capacity at the -7 C figure. A Swedish January goes lower. The
        # results for this house below -7 C therefore rest on an assumption, and are reported as a
        # bound rather than a measurement. The F750 carries no such caveat - see the F-124 test.
        thermal_mass=1.0,
        insulation_quality=0.9,
        hlc_w_per_k=218.0,  # F2040-8 Pdesignh 9.0 kW (COLD climate, 35 C application)
        tau_hours=40.0,
        profile=NibeF2040Profile(),
        heating_type="concrete_ufh",
        design_flow=35.0,  # the flow temperature its published capacity curve is measured at
    ),
]


def apply_coldsnap(times, temps):
    """Synthetic stress variant: shift Jan 12-18 by -12 C (tests DM/aux
    behavior in a deep cold spell; documented as synthetic)."""
    out = list(temps)
    for i, t in enumerate(times):
        if 12 <= t.day <= 18 and t.month == 1:
            out[i] = temps[i] - 12.0
    return out


def _to_gespot_shape(days: dict, ore_per_unit: float) -> dict[str, list[dict[str, Any]]]:
    """Normalise a raw price file into the GE-Spot attribute shape.

    Every price source ends up as {"time": iso8601, "value": <display unit>} so a
    single code path - the real adapter - parses all of them. Hourly sources are
    expanded to four identical quarters, which is what an hourly market genuinely
    means for a quarter-hour tariff.
    """
    utc = zoneinfo.ZoneInfo("UTC")
    out: dict[str, list[dict[str, Any]]] = {}
    for day, raw in days.items():
        entries: list[dict[str, Any]] = []
        expand = 1 if len(raw) >= 90 else QUARTERS_PER_HOUR
        for item in raw:
            start = datetime.fromisoformat(item["start"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=TZ)

            # THE QUARTERS ARE STEPPED ON THE ABSOLUTE LINE, AND THE FOLD IS WHY.
            #
            # This used to convert to Europe/Stockholm and then do
            # `(start + timedelta(minutes=15 * q)).isoformat()`. Adding a timedelta to an AWARE
            # datetime is wall-clock arithmetic, and - this is the part that is easy to miss -
            # `datetime.__add__` RESETS `fold` TO 0. Even at q = 0, where the timedelta is zero.
            #
            # So on the night the clocks go back, the second 02:00 (CET, fold=1) came back out of
            # here stamped +02:00: an exact duplicate of the first 02:00 (CEST), and the CET hour's
            # prices vanished. The harness then reported that the integration could not price the
            # repeated hour - a defect in the INSTRUMENT, presented as a defect in the code it was
            # measuring. The real adapter parses GE-Spot's own timestamps, which carry the right
            # offset, and compares them interzone (i.e. in UTC); it prices that hour correctly.
            #
            # Stepping UTC and converting back keeps each quarter the instant it actually is.
            base = start.astimezone(utc)
            for q in range(expand):
                moment = (base + timedelta(minutes=QUARTER_MINUTES * q)).astimezone(TZ)
                entries.append({"time": moment.isoformat(), "value": item["price"] * ore_per_unit})
        out[day] = entries
    return out


def load_live_se4() -> tuple[dict[str, list[dict[str, Any]]], str]:
    """The real SE4 day captured from a live GE-Spot integration.

    Already in GE-Spot's own attribute shape, so it is handed to the adapter
    untouched - byte-for-byte what the integration sees in production.
    """
    payload = json.loads((DATA_DIR / "gespot_live_se4.json").read_text(encoding="utf-8"))
    attrs = payload["attributes"]
    days: dict[str, list[dict[str, Any]]] = {}
    for key in ("today_interval_prices", "tomorrow_interval_prices"):
        for item in attrs.get(key) or []:
            day = datetime.fromisoformat(item["time"]).date().isoformat()
            days.setdefault(day, []).append({"time": item["time"], "value": item["value"]})
    return days, attrs["unit_of_measurement"]


def _synthetic_days(start: datetime, days: int):
    """Synthetic weather + quarter-hourly prices, generated on the ABSOLUTE time line.

    Everything here steps UTC and converts back, because the wall clock is not a uniform ruler. The
    day the clocks go back is 25 hours long and carries 100 quarter-hour prices, not 96 - and a
    generator that assumes 96 would quietly manufacture a day that no market ever published, which
    is the opposite of what a harness is for.
    """
    start_absolute = start.astimezone(zoneinfo.ZoneInfo("UTC"))
    end_absolute = (start + timedelta(days=days)).astimezone(zoneinfo.ZoneInfo("UTC"))

    hours = int((end_absolute - start_absolute).total_seconds() // 3600)
    times = [(start_absolute + timedelta(hours=h)).astimezone(TZ) for h in range(hours)]
    temps = [-5.0 + 4.0 * ((t.hour % 24) / 24.0) for t in times]

    raw: dict = {}
    quarters = int((end_absolute - start_absolute).total_seconds() // (60 * QUARTER_MINUTES))
    for q in range(quarters):
        moment = (start_absolute + timedelta(minutes=QUARTER_MINUTES * q)).astimezone(TZ)
        # The expensive blocks are wall-clock ones (morning and evening peaks), so they are keyed
        # off the LOCAL quarter-of-day - which is what a price area actually does.
        local_quarter = moment.hour * 4 + moment.minute // QUARTER_MINUTES
        price = 500.0 + 400.0 * (1 if 28 <= local_quarter <= 40 or 68 <= local_quarter <= 80 else 0)
        raw.setdefault(moment.date().isoformat(), []).append(
            {"start": moment.isoformat(), "price": price}
        )
    return times, temps, _to_gespot_shape(raw, ORE_PER_KWH_FROM_SEK_PER_MWH), GESPOT_UNIT_ORE


def load_data(selftest: bool, live_se4: bool = False, dst: bool = False):
    """Load real weather + prices, or synthetic data for --selftest / --dst."""
    if dst:
        # The last Sunday of October 2026: at 03:00 CEST the clock goes back to 02:00 CET, so the
        # wall-clock hour 02 happens TWICE and the day is 25 hours long. This is the day on which
        # the coordinator used to DELETE a billing hour - see
        # tests/unit/coordinator/test_the_billing_hour_survives_the_clocks_going_back.py - and the
        # harness could not see it, because its own clock advanced by wall time and its tariff
        # periods were keyed on (date, hour), which those two hours share.
        return _synthetic_days(datetime(2026, 10, 24, tzinfo=TZ), 3)
    if selftest:
        return _synthetic_days(datetime(2026, 1, 1, tzinfo=TZ), 2)

    weather = json.load(open(DATA_DIR / "weather_jan2026.json"))
    times = [
        datetime.fromisoformat(t).replace(tzinfo=TZ) if "T" in t else None
        for t in weather["hourly"]["time"]
    ]
    temps = weather["hourly"]["temperature_2m"]

    if live_se4:
        # Real captured SE4 prices, replayed against January weather. The market
        # day is a July one; the point is the price SHAPE (a 41x spread between
        # cheapest and dearest quarter), which is far harsher on the optimiser
        # than the January SE3 profile.
        se4_days, unit = load_live_se4()
        days: dict[str, list[dict[str, Any]]] = {}
        # Take the price SHAPE (the ordered quarters) and re-stamp it onto the simulated days.
        # The timestamps must be REBUILT in the simulation's timezone, not edited: the captured
        # day is a July one at UTC+02:00 and the simulated days are January at UTC+01:00, so
        # rewriting only the date leaves every interval an hour out of place - which is exactly
        # what the adapter's timestamp lookup then refuses to price, and rightly so.
        shape = [values for _, values in sorted(se4_days.items())]
        midnight = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
        for index in range(SIM_DAYS + 1):
            start = midnight + timedelta(days=index)
            entries = shape[index % len(shape)]
            days[start.date().isoformat()] = [
                {
                    "time": (start + timedelta(minutes=QUARTER_MINUTES * quarter)).isoformat(),
                    "value": entry["value"],
                }
                for quarter, entry in enumerate(entries)
            ]
        return times, temps, days, unit

    prices = json.load(open(DATA_DIR / "prices_jan2026.json"))["days"]
    return (
        times,
        temps,
        _to_gespot_shape(prices, ORE_PER_KWH_FROM_SEK_PER_MWH),
        GESPOT_UNIT_ORE,
    )


def outdoor_at(times, temps, when: datetime) -> float:
    """Linear interpolation of hourly outdoor temperature."""
    idx = int((when - times[0]).total_seconds() // 3600)
    idx = max(0, min(idx, len(temps) - 2))
    frac = ((when - times[idx]).total_seconds() / 3600.0) if idx < len(times) else 0.0
    frac = max(0.0, min(frac, 1.0))
    return temps[idx] * (1 - frac) + temps[idx + 1] * frac


class _StubState:
    """The two fields GESpotAdapter reads off a Home Assistant state object."""

    def __init__(self, state: str, attributes: dict[str, Any]):
        self.state = state
        self.attributes = attributes


class _StubStates:
    def __init__(self) -> None:
        self._states: dict[str, _StubState] = {}

    def set(self, entity_id: str, state: _StubState) -> None:
        self._states[entity_id] = state

    def get(self, entity_id: str) -> _StubState | None:
        return self._states.get(entity_id)


class _StubHass:
    """Just enough Home Assistant to run the real adapter against."""

    def __init__(self) -> None:
        self.states = _StubStates()


class PriceSource:
    """Feeds the simulation through the REAL GESpotAdapter.

    The harness used to construct QuarterPeriod objects by hand, which meant the
    adapter that actually runs in production - unit detection, timestamp parsing,
    the sort by absolute instant, the DST-aware interval lookup - was never
    exercised by any simulation. A price-parsing regression could not have been
    caught here. Now the day's intervals are published as a Home Assistant state
    shaped exactly like a live GE-Spot entity, and the adapter parses it.

    PriceData is cached per (day, tomorrow-visible), so the adapter runs ~62 times
    across a month rather than once per 5-minute step.
    """

    ENTITY_ID = "sensor.gespot_current_price_sim"

    def __init__(self, days: dict[str, list[dict[str, Any]]], unit: str):
        self._days = days
        self._unit = unit
        self._hass = _StubHass()
        self._adapter = GESpotAdapter(
            self._hass,  # type: ignore[arg-type]
            {CONF_GESPOT_ENTITY: self.ENTITY_ID},
        )
        self._cache: dict[tuple[str, bool], PriceData] = {}

    @property
    def unit(self) -> str:
        """The unit the adapter detected off the entity (not what we assumed)."""
        return self._adapter.price_unit or self._unit

    def get(self, now: datetime) -> PriceData:
        today_key = now.date().isoformat()
        tomorrow_key = (now + timedelta(days=1)).date().isoformat()
        tomorrow_visible = now.hour >= TOMORROW_VISIBLE_HOUR
        cache_key = (today_key, tomorrow_visible)
        if cache_key in self._cache:
            return self._cache[cache_key]

        today_raw = self._days.get(today_key, [])
        tomorrow_raw = self._days.get(tomorrow_key, []) if tomorrow_visible else []

        current = today_raw[0]["value"] if today_raw else 0.0
        self._hass.states.set(
            self.ENTITY_ID,
            _StubState(
                state=str(current),
                attributes={
                    "unit_of_measurement": self._unit,
                    "currency": "SEK",
                    "today_interval_prices": today_raw,
                    "tomorrow_interval_prices": tomorrow_raw,
                },
            ),
        )
        price_data = asyncio.run(self._adapter.get_prices())
        self._cache[cache_key] = price_data
        return price_data


# Reference thermal-battery controller. Not a proposal for production - a YARDSTICK. It knows
# nothing about degree minutes, weather, peaks or the pump; it only charges the house when power
# is cheap and coasts when it is dear, inside a hard comfort band. If EffektGuard cannot beat
# this, the sophistication is not paying for itself.
BATTERY_BAND = 1.0  # °C swing around target the house is allowed to use as storage
BATTERY_CHARGE_OFFSET = 4.0  # curve offset while charging on cheap power
BATTERY_COAST_OFFSET = -4.0  # curve offset while coasting on dear power
BATTERY_CHEAP_PERCENTILE = 30  # below this percentile of the day, charge
BATTERY_DEAR_PERCENTILE = 70  # above this percentile of the day, coast


def battery_reference_offset(price_data: PriceData, now: datetime, indoor: float) -> float:
    """Charge the fabric when power is cheap, coast when dear, never leave the comfort band."""
    if indoor > TARGET_INDOOR + BATTERY_BAND:
        return BATTERY_COAST_OFFSET  # full - stop charging
    if indoor < TARGET_INDOOR - BATTERY_BAND:
        return BATTERY_CHARGE_OFFSET  # flat - must heat regardless of price

    prices = [q.price for q in price_data.today]
    period = price_data.get_period(now)
    if not prices or period is None:
        return 0.0

    ordered = sorted(prices)
    cheap = ordered[int(len(ordered) * BATTERY_CHEAP_PERCENTILE / 100)]
    dear = ordered[int(len(ordered) * BATTERY_DEAR_PERCENTILE / 100)]

    if period.price <= cheap:
        return BATTERY_CHARGE_OFFSET
    if period.price >= dear:
        return BATTERY_COAST_OFFSET
    return 0.0


def build_engine(
    house: HouseConfig,
    mode: str = "balanced",
    enable_price: bool = True,
    enable_weather: bool = True,
    tuned_curve: bool = False,
):
    """Build the real DecisionEngine for this house.

    `enable_price` / `enable_weather` exist so the harness can ABLATE a layer and attribute the
    result. "The optimiser costs 2 % more than doing nothing" is not actionable; "the price layer
    costs 3 % and the weather compensation saves 1 %" is.
    """
    hass = MagicMock()
    effect = EffectManager(hass)
    # The harness has no Home Assistant storage; the peak history lives for the run only.
    effect._store = MagicMock()
    effect._store.async_save = AsyncMock()
    thermal = ThermalModel(house.thermal_mass, house.insulation_quality)
    config = {
        "target_indoor_temp": TARGET_INDOOR,
        "tolerance": COMFORT_TOLERANCE,
        "optimization_mode": mode,
        "enable_weather_compensation": enable_weather,
        "enable_peak_protection": True,
        "enable_price_optimization": enable_price,
        "latitude": 59.33,
        "heating_type": house.heating_type,
        "heat_loss_coefficient": house.hlc_w_per_k,
        "thermal_mass": house.thermal_mass,
        "insulation_quality": house.insulation_quality,
    }
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=effect,
        thermal_model=thermal,
        config=config,
        heat_pump_model=house.profile,
    )
    return engine, effect


def simulate(
    house: HouseConfig,
    times,
    temps,
    price_source: PriceSource,
    days: int,
    mode: str = "balanced",
    baseline: bool = False,
    fixed_offset: float | None = None,
    battery: bool = False,
    enable_price: bool = True,
    enable_weather: bool = True,
    tuned_curve: bool = False,
    forecast_available: bool = True,
):
    engine, effect = build_engine(house, mode, enable_price, enable_weather)

    start = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    steps = days * 24 * 60 // STEP_MIN

    indoor = 22.0
    indoor_start = indoor
    dm = -30.0
    offset_applied = 0  # integer offset "in the pump" (register 47011)
    compressor_on = True
    flow = 30.0

    violations = []
    trace = []
    stats = {
        "indoor_min": 99.0,
        "indoor_max": -99.0,
        "indoor_sum": 0.0,
        "dm_min": 0.0,
        "cost_sek": 0.0,
        "energy_kwh": 0.0,
        "aux_kwh": 0.0,
        "unavoidable_aux_kwh": 0.0,
        "writes": 0,
        "offset_min": 0,
        "offset_max": 0,
        "exceptions": 0,
        "comfort_minutes_below": 0,
        "comfort_minutes_above": 0,
        "compressor_starts": 0,
        "sign_flips": 0,
        "heat_kwh": 0.0,
        "loss_kwh": 0.0,
        "layer_votes": {},
        "water_node_leak_kwh": 0.0,
        "flow_target_max": -999.0,
        "compressor_heat_kwh": 0.0,
        "datasheet_cop_x_heat": 0.0,
    }
    best_published_cop = max(p.cop for p in house.profile.datasheet_points)
    last_offsets = []
    # The REAL one, from the integration. Not a copy of it.
    billing = BillingPeriodAccumulator()
    daily_peaks: dict = {}  # date -> max HOURLY-mean kW (the billed quantity)
    # date -> how many billing hours the PRODUCTION accumulator actually billed on it. A day is not
    # always 24 hours long,
    # and the tariff bills every hour the meter recorded: the fall-back day has 25 and the
    # spring-forward day 23. Counting them is how this harness proves it is actually TRAVERSING
    # the transition rather than merely surviving it - a flat night load is priced identically
    # whether the repeated hour is billed once or twice, so the tariff figure alone cannot tell.
    billing_hours: dict = {}
    # Highest completed quarter-hour MEAN so far: what the coordinator publishes as
    # peak_this_month, and therefore what the effect layer is defending. Starts at
    # zero, as it does on a fresh install.
    running_peak_kw = 0.0

    # THE CLOCK ADVANCES ON THE ABSOLUTE TIME LINE, NOT THE WALL CLOCK.
    #
    # This was `now = start + timedelta(minutes=STEP_MIN * step)`, and `start` is aware
    # (Europe/Stockholm). Adding a timedelta to an AWARE datetime is WALL-CLOCK arithmetic: the
    # digits advance uniformly and the UTC offset is recomputed from wherever they land. Real time
    # does not work that way. Across a spring-forward that clock walks through a wall time that never
    # happened; across a fall-back it passes the repeated hour once instead of twice.
    #
    # So the harness could not have experienced a DST transition honestly even if pointed straight
    # at one - and the coordinator bug that deleted a billing hour on the fall-back night (a peak of
    # 9 kW recorded as 1) would have been invisible to it. Step UTC; derive local from it.
    start_absolute = start.astimezone(zoneinfo.ZoneInfo("UTC"))

    for step in range(steps):
        now = (start_absolute + timedelta(minutes=STEP_MIN * step)).astimezone(TZ)
        # Freeze engine wall clock to sim time
        dt_util.now = lambda tz=None, _n=now: _n
        dt_util.utcnow = lambda _n=now: _n.astimezone(zoneinfo.ZoneInfo("UTC"))

        tout = outdoor_at(times, temps, now)

        # --- plant step ---
        # S1 IS CLAMPED TO THE PUMP'S MAXIMUM SUPPLY TEMPERATURE, as it is on the real hardware.
        #
        # This clamp was missing while `flow` (BT25) was clamped, twelve lines below. Degree
        # minutes are the integral of (BT25 - S1), so the plant was integrating against a setpoint
        # the pump was physically forbidden to reach: in the F2040 cold snap the curve asked for up
        # to 4.1 C above max_flow_temp for 513 samples, and DM therefore fell at up to 4.1 per
        # minute NO MATTER WHAT ANY CONTROLLER DID. Degree minutes ran to the integrator floor on
        # their own, and the harness reported it as a control failure. It was a plant artefact.
        #
        # A NIBE limits the calculated supply temperature to the configured maximum; it does not
        # chase a setpoint it cannot make. Removing this artefact is what makes the residual trap
        # underneath it (F-124) measurable at its true size rather than at an inflated one.
        max_flow = float(house.profile.max_flow_temp)
        flow_target = min(house.curve_flow_temp(tout, tuned_curve) + offset_applied, max_flow)

        # The compressor's capacity now bounds the water node directly (see below), so the flow
        # saturates below target of its own accord when the pump runs out - which is what lets
        # degree minutes actually run away, and is the real mechanism behind an undersized pump
        # falling back on its immersion heater in a cold snap.

        # THE WATER LOOP IS A THERMAL MASS, NOT A RAMP RATE.
        #
        # This used to move `flow` toward its target at a fixed C/min and then compute the room's
        # heat from wherever the flow happened to be - including while the compressor was OFF, so
        # the decaying water heated the room for free and nothing ever charged for putting the heat
        # in. The plant manufactured energy in proportion to how long the compressor spent idle,
        # which systematically flattered whichever controller ran the pump least.
        #
        # The physics is simply a first-order node: the compressor heats the water, the water heats
        # the room, and the flow temperature is what the balance between them leaves behind.
        #
        #     C_water * dT_flow/dt = Q_compressor - Q_emitters
        #
        # Now every joule the room receives was paid for, the loop is a buffer rather than a
        # source, and a controller that swings the flow pays the real cost of doing so.
        q_emit_w = house.heat_output_w(flow, indoor)

        capacity_w = house.capacity_kw_at(tout) * 1000.0
        if compressor_on:
            # The compressor modulates toward the flow its curve is asking for, bounded by what it
            # can actually deliver - which comes from the datasheet, not from an invented derating.
            demand_w = q_emit_w + WATER_LOOP_J_PER_K * (flow_target - flow) / COMPRESSOR_RESPONSE_S
            q_comp_w = max(0.0, min(demand_w, capacity_w))
        else:
            q_comp_w = 0.0

        # How hard the compressor is being pushed, which is what sets its efficiency. No
        # circularity: q_comp is fixed by demand and capacity, both computed above.
        load_fraction = q_comp_w / capacity_w if capacity_w > 0 else 0.0

        # THE IMMERSION HEATER IS THERMOSTATIC, because every real one is.
        #
        # It used to dump a flat 3 kW into the water node whenever degree minutes passed the aux
        # limit - including when the node was already at its ceiling. In a five-minute step that is
        # 900 kJ into a 350 kJ/K loop: 2.6 K of overshoot per step, which the clamp below then
        # deleted. The heater was metered, paid for, and its heat thrown away, 183 kWh of it in the
        # F2040 cold snap, while every energy "audit" in the harness reported 0.00 % error.
        #
        # A real immersion heater has a high-limit thermostat and cycles on the water temperature.
        # So it injects at most what fits under the ceiling: the heat the emitters are taking out,
        # less what the compressor is already putting in, plus whatever headroom the node has left.
        aux_headroom_w = (
            WATER_LOOP_J_PER_K * (max_flow - flow) / (STEP_MIN * 60.0) + q_emit_w - q_comp_w
        )
        aux_w = 0.0
        if dm <= house.dm_aux_limit:
            aux_w = min(house.immersion_heater_kw * 1000.0, max(0.0, aux_headroom_w))

        flow_unclamped = (
            flow + (q_comp_w + aux_w - q_emit_w) * (STEP_MIN * 60.0) / WATER_LOOP_J_PER_K
        )
        flow = max(indoor, min(flow_unclamped, max_flow))

        # THE ONLY ENERGY STATEMENT IN THIS PLANT THAT CAN ACTUALLY FAIL.
        #
        # Everything downstream of here - the room ODE, the "first law residual", the compressor
        # audit - is an algebraic rearrangement of the same two lines and CANNOT disagree with
        # itself. This clamp is different: it overwrites a state variable AFTER the ODE has
        # integrated it, so every joule it removes is energy the meter charged for and the room
        # never received. Nothing else in the harness can see that, and it measured 0.00 % error
        # while 183 kWh vanished in the F2040 cold snap.
        #
        # In a healthy plant the clamp never binds and this stays at zero. It is an assertion, not
        # a statistic.
        stats["water_node_leak_kwh"] += WATER_LOOP_J_PER_K * (flow - flow_unclamped) / J_PER_KWH

        # THE DATASHEET, AT THE WEATHER THIS RUN ACTUALLY SAW. Accumulated here, asserted in
        # check_invariants. The plant's COP is the manufacturer's rated figure scaled by the Carnot
        # ratio between the flow it is making and the W35 rating point, so whenever the water is
        # HOTTER than W35 the scale is below one and the realised COP cannot exceed the datasheet.
        # That is a bound the energy bookkeeping does not determine, which is exactly why it can
        # fail - and a doubled COP, the bug the deleted identity waved through, breaks it on every
        # house.
        heat_kwh_this_step = q_comp_w / 1000.0 * STEP_MIN / 60.0
        stats["compressor_heat_kwh"] += heat_kwh_this_step
        stats["datasheet_cop_x_heat"] += best_published_cop * heat_kwh_this_step

        q_w = q_emit_w

        aux_kw = aux_w / 1000.0

        # Indoor temperature ODE
        # INTERNAL GAINS. The simulated house used to have none: its only heat source was the
        # emitters. A real house is warmed by its occupants, its fridge, its lighting and the sun
        # to the tune of a few hundred watts, all year - which is why heat demand reaches zero at
        # the BALANCE POINT (~17 C outdoor) rather than at room temperature.
        #
        # Leaving them out did not just make the plant unrealistic, it made it BLIND: the
        # controller models 600 W of gains and asks for correspondingly less flow, so a house with
        # zero gains would be systematically under-supplied - and deleting the controller's gains
        # term (a real regression) would have been INVISIBLE here, because the two errors cancel.
        d_indoor = (
            q_w + INTERNAL_GAINS_W - house.hlc_w_per_k * (indoor - tout)
        ) / house.capacity_j_per_k
        indoor += d_indoor * STEP_MIN * 60.0

        # DM dynamics + compressor hysteresis
        dm += (flow - flow_target) * STEP_MIN
        dm = max(DM_INTEGRATOR_FLOOR, min(dm, DM_INTEGRATOR_CEILING))
        if not compressor_on and dm <= DM_START:
            compressor_on = True
            stats["compressor_starts"] += 1
        elif compressor_on and dm >= DM_STOP:
            compressor_on = False

        cop = house.cop_at(tout, flow, load_fraction)

        # THE SECOND LAW. No machine can beat Carnot between the temperatures it is working across.
        #
        # Unlike the energy "audits" this replaces, this one is not derived from the plant's own
        # bookkeeping - it is an external physical bound on the COP MODEL, so it can disagree with
        # it. It catches a wrong anchor, a flipped exponent or bad approach temperatures. It does
        # NOT catch a COP that is merely too generous but still sub-Carnot; the datasheet envelope
        # in check_invariants is what covers that, and between them they bracket the model from
        # both sides.
        if cop > house.carnot_cop(tout, flow):
            violations.append(
                {
                    "t": now.isoformat(),
                    "type": "cop_beats_carnot",
                    "detail": f"COP {cop:.2f} > Carnot {house.carnot_cop(tout, flow):.2f}",
                }
            )

        power_kw = (q_comp_w / 1000.0) / cop + aux_kw + STANDBY_KW
        hz = 40 + int(min(50, max(0, (flow_target - indoor)))) if compressor_on else 0

        # --- price/weather context (parsed by the REAL GE-Spot adapter) ---
        price_data = price_source.get(now)
        cur_q = (now.hour * 4) + now.minute // 15
        # Locate the interval by timestamp, exactly as the integration does, rather
        # than indexing by quarter number - the two disagree on DST days.
        cur_period = price_data.get_period(now)
        if cur_period is None:
            violations.append(
                {"t": now.isoformat(), "type": "no_price_for_instant", "detail": f"q{cur_q}"}
            )
            cur_price_ore = 100.0
        else:
            cur_price_ore = cur_period.price

        fc = [
            WeatherForecastHour(
                datetime=now + timedelta(hours=h),
                temperature=outdoor_at(times, temps, now + timedelta(hours=h)),
            )
            for h in range(1, 49)
        ]
        # A weather entity is vol.Optional in the config flow, and with none configured
        # WeatherAdapter.get_forecast() returns None outright ("Weather forecast disabled - no
        # entity configured in setup"). That is a SUPPORTED install, and until this flag existed the
        # harness had never simulated it: it fed a perfect 48 h forecast to every run.
        #
        # Note this is NOT --no-weather. That flag clears enable_weather_compensation, which kills
        # the Math WC layer - the core control law, voting 100% of the time - and which the config
        # flow never writes, so production cannot reach it. Withholding the FORECAST is the thing a
        # real user can do, and it is the weaker ablation: Math WC still runs off outdoor and flow
        # temperature. Only the forecast-fed layers go quiet.
        weather = (
            WeatherData(current_temp=tout, forecast_hours=fc, source_entity="sim")
            if forecast_available
            else None
        )

        nibe = NibeState(
            outdoor_temp=round(tout, 1),
            indoor_temp=round(indoor, 2),
            supply_temp=round(flow, 1),
            return_temp=round(flow - 5.0, 1),
            degree_minutes=round(dm, 0),
            current_offset=float(offset_applied),
            is_heating=compressor_on,
            is_hot_water=False,
            timestamp=now,
            compressor_hz=hz,
            power_kw=round(power_kw, 2),
        )

        # --- the real decision engine (or neutral baseline) ---
        if battery:
            calc_offset = battery_reference_offset(price_data, now, indoor)
        elif fixed_offset is not None:
            calc_offset = fixed_offset
        elif baseline:
            calc_offset = 0.0
        else:
            try:
                # The peak the effect layer defends is the one this simulation has
                # actually produced so far, not a constant. A hardcoded 6.0 kW meant
                # the layer was always defending a peak the plant never set, and the
                # "no peak recorded yet" path (where predictive protection must stay
                # silent) was never reached at all.
                decision = engine.calculate_decision(
                    nibe_state=nibe,
                    price_data=price_data,
                    weather_data=weather,
                    current_peak=running_peak_kw,
                    current_power=power_kw,
                )
                calc_offset = decision.offset
                # WHICH LAYERS ACTUALLY VOTED. "5/5 PASS" says nothing about a layer that never
                # fired - and this harness has already shipped a run where the Peak layer voted
                # weight 0.00 in all 8928 steps of every run ever made, while reporting PASS. A
                # green run over silent code is not evidence, and the only way to know which it is
                # is to count.
                for layer in decision.layers:
                    if layer.weight > 0.0:
                        stats["layer_votes"][layer.name] = (
                            stats["layer_votes"].get(layer.name, 0) + 1
                        )
            except Exception as err:  # noqa: BLE001 - we are hunting bugs
                stats["exceptions"] += 1
                violations.append(
                    {
                        "t": now.isoformat(),
                        "type": "exception",
                        "detail": f"{type(err).__name__}: {err}",
                    }
                )
                calc_offset = 0.0

        # The REAL quantisation the adapter uses, not a copy of it. This harness used to carry its
        # own transcription of that arithmetic - including the int() truncation - which is exactly
        # how a plant model and the code it is meant to be testing drift apart unnoticed.
        new_int = integer_offset_for(calc_offset, offset_applied)
        if new_int != offset_applied:
            offset_applied = new_int
            stats["writes"] += 1

        # --- invariants & stats ---
        # A degree-minute deficit that reaches the integrator floor means the recovery system -
        # the curve offset AND the auxiliary heater together - failed to arrest it. That is the
        # signal worth failing on.
        #
        # The previous invariant here ("DM below the aux limit while aux is off") was a FALSE
        # POSITIVE: aux is decided from the degree minutes at the START of the step and the check
        # ran against the value at the END, so a deficit that crossed the limit mid-step tripped
        # it even though aux engages on the very next step - which is simply what a controller
        # sampling at an interval does. Worse, it could never catch a real defect, because aux
        # engages exactly when DM crosses the limit. It was unfalsifiable in both directions.
        if dm <= DM_INTEGRATOR_FLOOR:
            violations.append(
                {"t": now.isoformat(), "type": "dm_runaway", "detail": f"DM floored at {dm:.0f}"}
            )

        # Nothing here used to fail on OVERHEATING. The harness counted comfort_minutes_above and
        # asserted nothing about it, so a run that cooked the house to 35 C reported "violations:
        # 0". Overheating is a comfort failure, an efficiency failure, and - when it is auxiliary
        # heat doing it - an expensive one.
        if indoor > INDOOR_CEILING:
            violations.append(
                {
                    "t": now.isoformat(),
                    "type": "indoor_above_ceiling",
                    "detail": f"indoor {indoor:.2f}",
                }
            )
        if indoor < 18.0:
            violations.append(
                {"t": now.isoformat(), "type": "indoor_below_18", "detail": f"indoor {indoor:.2f}"}
            )
        if not -10 <= calc_offset <= 10:
            violations.append(
                {
                    "t": now.isoformat(),
                    "type": "offset_out_of_range",
                    "detail": f"offset {calc_offset:.2f}",
                }
            )

        last_offsets.append(offset_applied)
        if len(last_offsets) > 9:
            last_offsets.pop(0)
            deltas = [b - a for a, b in zip(last_offsets, last_offsets[1:])]
            flips = sum(1 for a, b in zip(deltas, deltas[1:]) if a * b < 0)
            if flips >= 3:
                stats["sign_flips"] += 1

        stats["indoor_min"] = min(stats["indoor_min"], indoor)
        stats["indoor_max"] = max(stats["indoor_max"], indoor)
        stats["indoor_sum"] += indoor
        stats["dm_min"] = min(stats["dm_min"], dm)
        # What the plant actually ASKED the pump for. Degree minutes integrate (BT25 - S1), so if
        # S1 can exceed what the pump may make, DM falls forever regardless of the controller. The
        # number is published so a test can check the plant rather than recompute the clamp and
        # assert on its own arithmetic - which is what the first version of that test did.
        stats["flow_target_max"] = max(stats["flow_target_max"], flow_target)
        stats["offset_min"] = min(stats["offset_min"], offset_applied)
        stats["offset_max"] = max(stats["offset_max"], offset_applied)
        energy = power_kw * STEP_MIN / 60.0
        stats["energy_kwh"] += energy

        # First-law audit. Heat INTO the room, and heat OUT of it. Over a month these must balance
        # to within the change in the fabric's stored energy - otherwise the plant is inventing or
        # destroying energy and every cost number it produces is fiction.
        stats["heat_kwh"] += q_w * STEP_MIN / 60.0 / 1000.0
        stats["loss_kwh"] += (
            (house.hlc_w_per_k * (indoor - tout) - INTERNAL_GAINS_W) * STEP_MIN / 60.0 / 1000.0
        )
        stats["aux_kwh"] += aux_kw * STEP_MIN / 60.0

        # THE RESISTIVE HEAT PHYSICS FORCES, as opposed to the resistive heat the optimiser causes.
        #
        # A correctly-sized air-source system in Sweden is BIVALENT: NIBE declares Tbiv = -9 C for
        # the F2040-8, below which the machine cannot meet the design load and supplementary heat is
        # REQUIRED. The harness used to assert that a healthy pump burns no resistive heat at all,
        # which is an assertion about a machine that does not exist - and it duly failed the only
        # correctly-sized air-source house in the set, for doing exactly what it is designed to do.
        #
        # What CAN be asked, and is worth asking, is whether the optimiser burns more resistive heat
        # than the pump's own capacity deficit forces. That is computable here: the house's heat
        # demand at this instant, against what the compressor can physically deliver. Anything above
        # it is the controller's doing, not the weather's.
        demand_now_w = house.hlc_w_per_k * (indoor - tout) - INTERNAL_GAINS_W
        stats["unavoidable_aux_kwh"] += (
            max(0.0, demand_now_w - capacity_w) / 1000.0 * STEP_MIN / 60.0
        )
        stats["cost_sek"] += energy * cur_price_ore / 100.0

        # EFFECT TARIFF BASIS: THE HOURLY MEAN. Not the quarter-hour, which is what this used to
        # accumulate, and not the instantaneous sample, which is what it accumulated before that.
        #
        # Ellevio: "the measurement uses hourly averages". Energimarknadsinspektionen:
        # "elnatsforetagen mater din elanvandning per timme". A 15-minute hot-water cycle at 9 kW
        # inside an otherwise idle hour has an hourly mean of 3 kW, and the harness was pricing the
        # 9 - so every tariff figure it produced was up to fourfold too high.
        # THE BILLED QUANTITY IS COMPUTED BY THE PRODUCTION CODE, NOT BY A LOOKALIKE.
        #
        # This used to be the harness's OWN accumulator: `sum(period_samples) / len(period_samples)`,
        # keyed on its own idea of an hour. The coordinator has always used a TIME-WEIGHTED mean over
        # an absolute hour. Two implementations of the single most consequential number this
        # integration computes - and the harness was validating the one nobody runs.
        #
        # They agreed only because this loop steps a perfectly uniform five minutes, which Home
        # Assistant does not. And they were both wrong on the night the clocks go back, INDEPENDENTLY,
        # so neither could see the other's bug: the coordinator merged the repeated hour and deleted a
        # 9 kW billing peak. An instrument that re-implements the thing it measures cannot measure it.
        #
        # `BillingPeriodAccumulator` is now the only definition, and this is the real one. Break it
        # and --dst fails here as well as in the unit tests.
        completed = billing.add(now, power_kw)
        if completed is not None:
            # COUNT WHAT THE ACCUMULATOR ACTUALLY BILLED, not what this loop thinks an hour is.
            #
            # The first version of this counter re-derived the hour key here, from `now`, and so it
            # kept reporting 25 hours on the fall-back day even when the production accumulator was
            # merging the two 02:00s into one. It was measuring the harness, not the code under test
            # - the exact vacuity this whole commit exists to remove, reintroduced one line below the
            # comment complaining about it. Verified by mutation: reinstate the DST bug in
            # billing_period.py and this now reports 24 hours and fails the run.
            # COUNTED, not collected in a set: on the fall-back day both 02:00 hours carry the SAME
            # local `started_at`, and PEP 495 makes those two datetimes compare EQUAL (and hash
            # equal), so a set would silently merge them back into one and report 24 again - passing
            # the check by making the same mistake it exists to catch.
            billing_hours[completed.started_at.date()] = (
                billing_hours.get(completed.started_at.date(), 0) + 1
            )

            day = completed.started_at.date()
            daily_peaks[day] = max(daily_peaks.get(day, 0.0), completed.mean_power_kw)
            running_peak_kw = max(running_peak_kw, completed.mean_power_kw)

            # THE EFFECT LAYER WAS NEVER GIVEN A PEAK HISTORY. The harness computed
            # `running_peak_kw` and handed it to the engine, but never called
            # `record_quarter_measurement()` - so `EffectManager._monthly_peaks` stayed empty for
            # all 8928 steps, and `should_limit_power()` short-circuits on an empty history:
            #
            #     if not self._monthly_peaks:
            #         return PowerLimitDecision(should_limit=False, severity="OK", ...)
            #
            # The peak layer therefore voted weight 0.00 on every single step of every run. Every
            # claim this harness made about effect-tariff protection - the feature the integration
            # is named for - was vacuous. (The coordinator had the mirror-image bug for meter-less
            # houses; this is the same hole, in the instrument that was supposed to catch it.)
            asyncio.run(
                effect.record_period_measurement(
                    power_kw=completed.mean_power_kw,
                    period=completed.billing_hour,
                    timestamp=completed.started_at,
                    source=POWER_SOURCE_EXTERNAL_METER,
                )
            )

        if indoor < TARGET_INDOOR - COMFORT_TOLERANCE:
            stats["comfort_minutes_below"] += STEP_MIN
        elif indoor > TARGET_INDOOR + OVERSHOOT_TOLERANCE:
            stats["comfort_minutes_above"] += STEP_MIN

        if step % 6 == 0:  # 30-min trace resolution
            trace.append(
                {
                    "t": now.isoformat(),
                    "tout": round(tout, 1),
                    "tin": round(indoor, 2),
                    "flow": round(flow, 1),
                    "dm": round(dm),
                    "offset": offset_applied,
                    "calc": round(calc_offset, 2),
                    "kw": round(power_kw, 2),
                    "price": round(cur_price_ore, 1),
                    "comp": int(compressor_on),
                }
            )

    # The run ends on an hour boundary, and that final hour is complete in sim-time. Production
    # never flushes - Home Assistant keeps running, and an hour cut short by a shutdown was never
    # measured and is not a bill.
    final = billing.flush()
    if final is not None:
        day = final.started_at.date()
        daily_peaks[day] = max(daily_peaks.get(day, 0.0), final.mean_power_kw)
        billing_hours[day] = billing_hours.get(day, 0) + 1
    top3 = sorted(daily_peaks.values(), reverse=True)[:3]
    tariff_kw = sum(top3) / len(top3) if top3 else 0.0
    stats["peak_kw_hourly_mean"] = round(max(daily_peaks.values()), 2) if daily_peaks else 0.0
    stats["tariff_top3_kw"] = round(tariff_kw, 2)
    stats["billing_hours_by_day"] = {
        day.isoformat(): count for day, count in sorted(billing_hours.items())
    }
    stats["tariff_cost_sek"] = round(tariff_kw * EFFECT_TARIFF_SEK_PER_KW, 0)
    stats["total_cost_sek"] = round(stats["cost_sek"] + stats["tariff_cost_sek"], 0)

    stats["indoor_mean"] = round(stats["indoor_sum"] / steps, 2)
    del stats["indoor_sum"]

    # THE ROOM-SIDE BALANCE IS AN IDENTITY, AND I SPENT SEVERAL COMMITS QUOTING IT AS EVIDENCE.
    #
    #     residual = heat_in - loss - stored
    #     the ODE   d_indoor = (q_w + GAINS - HLC*(indoor - tout)) / C
    #
    # are the same terms rearranged, so the residual is zero by construction. It says the room ODE
    # integrates consistently and NOTHING ELSE. Proved by making the compressor pay for only HALF
    # the heat it produced: electricity fell from 912 to 487 kWh and the residual stayed at 0.00.
    #
    # And that is precisely where the original free-heat bug lived - the COMPRESSOR side. So the
    # room balance could never have caught it, and I found it by reasoning rather than by the check
    # I built to find it. It is kept because a non-zero value would still mean the ODE is broken,
    # but it is no longer the thing being claimed.
    stored_kwh = house.capacity_j_per_k * (indoor - indoor_start) / J_PER_KWH
    residual = stats["heat_kwh"] - stats["loss_kwh"] - stored_kwh
    stats["heat_kwh"] = round(stats["heat_kwh"], 1)
    stats["loss_kwh"] = round(stats["loss_kwh"], 1)
    stats["energy_balance_residual_kwh"] = round(residual, 2)

    # AND SO WAS THE COMPRESSOR-SIDE "AUDIT" I ADDED TO REPLACE IT. It is deleted here.
    #
    #     power_kw = q_comp/cop + aux + standby          (the plant)
    #     metered  = power_kw - aux - standby            (the "meter")
    #     owed     = q_comp/cop                          (the "independent" figure)
    #
    # Substitute the first into the second and you get the third, exactly: x - y + y = x. Two
    # symbols, one line, and I called them "two independent expressions of the same joules" in the
    # code and in a test docstring. Doubling the compressor's COP - which halves the bill, a
    # catastrophic plant bug - left it reporting 0.00 % error and PASS.
    #
    # There is no exact energy audit to be had inside a closed ODE plant: every residual you can
    # write is a rearrangement of the equations that produced it. What CAN fail is a statement
    # about something the bookkeeping does not determine, and there are exactly two of those:
    #
    #   * water_node_leak_kwh - the flow clamp overwrites a state variable AFTER the ODE has
    #     integrated it, so it can destroy metered joules. It measured 183 kWh in the F2040 cold
    #     snap while every "audit" above read 0.00 %.
    #   * the second law (per step, above) and the datasheet envelope (in check_invariants), which
    #     bracket the COP model from above and below using data the plant's energy accounting does
    #     not reference.
    stats["water_node_leak_kwh"] = round(stats["water_node_leak_kwh"], 1)
    stats["datasheet_cop"] = round(
        stats["datasheet_cop_x_heat"] / max(stats["compressor_heat_kwh"], 1e-9), 2
    )
    del stats["datasheet_cop_x_heat"]
    del stats["compressor_heat_kwh"]
    stats["mean_cop"] = round(
        stats["heat_kwh"] / max(stats["energy_kwh"] - STANDBY_KW * steps * STEP_MIN / 60.0, 1e-9), 2
    )
    stats["violations"] = len(violations)
    return stats, violations, trace


# Safety invariants. A run that trips one of these has demonstrated the optimiser
# doing something it must never do, and the harness exits non-zero so that a human
# - or CI - cannot mistake a bad run for a good one. Previously every run exited 0
# no matter what it found, so the simulation could not fail and therefore could not
# hold anything up.
FATAL_VIOLATIONS = frozenset(
    {
        "indoor_below_18",  # comfort floor breached: the pump was starved
        "indoor_above_ceiling",  # house cooked, usually by the immersion heater
        "offset_out_of_range",  # engine emitted an offset the register cannot hold
        "exception",  # engine raised while controlling a heat pump
        "dm_runaway",  # the deficit outran the curve offset AND the aux heater
        "no_price_for_instant",  # adapter could not price a moment that exists
        "cop_beats_carnot",  # the PLANT broke the second law: every cost it reports is fiction
    }
)

# The optimiser is allowed to move heat around, but not to make the house colder
# than a do-nothing controller would. Baseline mean indoor is the comparison.
MIN_MEAN_INDOOR_C = TARGET_INDOOR - COMFORT_TOLERANCE

# THE INVARIANTS BELOW USED TO BE UNFALSIFIABLE, AND SO DID THIS WHOLE HARNESS.
#
# Every mutation of a safety constant still printed "PASS: all safety invariants held":
#
#     MIN_TEMP_LIMIT        18.0 -> 5.0     PASS      <- the comfort floor, gutted
#     DM_THRESHOLD_AUX_LIMIT -1500 -> -400  PASS      <- immersion heater at shallow debt
#     WEATHER_GENTLE_OFFSET  0.83 -> 2.0    PASS      <- the overheat bug, hand-tuned against
#     INTERNAL_GAINS_W        600 -> 0      PASS
#     comfort-layer abstention removed      PASS
#
# The reason was not the invariants themselves - it was that a mild January never brings the
# house within reach of any of them, and three of the most telling numbers were COUNTED AND
# NEVER ASSERTED. `aux_kwh` was tracked and ignored, so driving the pump into the immersion
# heater was free. `comfort_minutes_below` and `comfort_minutes_above` were tracked and ignored,
# so the house could sit outside its comfort band for the entire month and still report zero
# violations. The file's own comment complains about exactly this pattern ("The harness counted
# comfort_minutes_above and asserted nothing about it") - and then did it again, twice.
#
# A test that cannot fail cannot detect. These now bite, and the gate runs the COLD SNAP as well
# as the mild month, so the house is actually taken near its limits.

# The immersion heater is a COP-1.0 resistive element. On a correctly sized pump in a Swedish
# January the optimiser must never reach for it: that is the whole point of the degree-minute
# ladder. A little is tolerated in a deep cold snap on an air-source pump whose capacity has
# genuinely collapsed - that is physics, not a control failure - so the budget is per-scenario.
# How much MORE resistive heat than physics forces the optimiser may burn. Not an absolute budget:
# a bivalent system is designed to use its immersion heater below Tbiv, and asserting otherwise is
# asserting something about a machine NIBE does not sell.
AUX_OVER_PHYSICS_TOLERANCE = 1.25
AUX_SLACK_KWH = 5.0  # so a house that needs essentially none is not failed by rounding

# Degree minutes must stay clear of the aux limit by a real margin. Skimming it means the ladder
# is only just holding, and the next colder night tips into resistive heat.
DM_AUX_MARGIN = 200.0

# Minutes outside the comfort band, per 31-day month. Not zero - the optimiser is ALLOWED to
# coast into the band's edge to dodge a price peak, that is its job - but a house that spends
# whole days out of band is not being optimised, it is being neglected.
MAX_COMFORT_MINUTES_BELOW = 240
MAX_COMFORT_MINUTES_ABOVE = 720


def check_invariants(tag: str, stats: dict, violations: list, house=None) -> list[str]:
    """Return the reasons this run must be treated as a failure."""
    failures = []

    fatal = [v for v in violations if v["type"] in FATAL_VIOLATIONS]
    if fatal:
        kinds = sorted({v["type"] for v in fatal})
        failures.append(f"{len(fatal)} safety violation(s): {', '.join(kinds)}")

    if stats["indoor_min"] < 18.0:
        failures.append(f"indoor fell to {stats['indoor_min']:.2f} C (floor is 18.0)")

    if stats["indoor_mean"] < MIN_MEAN_INDOOR_C:
        failures.append(
            f"mean indoor {stats['indoor_mean']:.2f} C is below the comfort band "
            f"({MIN_MEAN_INDOOR_C:.2f} C) - the optimiser under-heated the house"
        )

    if stats["exceptions"]:
        failures.append(f"{stats['exceptions']} engine exception(s)")

    # THE PLANT MAY NOT DESTROY ENERGY THE METER CHARGED FOR. The flow clamp overwrites the water
    # node's temperature after the ODE has integrated it, so it is the one place in the harness
    # where joules can go missing without any residual noticing - and 183 kWh did, in the F2040
    # cold snap, while the "audits" that preceded this reported 0.00 % error.
    if abs(stats["water_node_leak_kwh"]) > WATER_NODE_LEAK_BUDGET_KWH:
        failures.append(
            f"the flow clamp destroyed {abs(stats['water_node_leak_kwh']):.1f} kWh that the meter "
            f"charged for and the room never received - the plant is deleting energy, and every "
            f"cost number it produces is fiction by that much"
        )

    # THE COP MODEL, BOUNDED BY THE MANUFACTURER'S OWN BEST FIGURE. And the bound is ONE-WAY.
    #
    # This check used to compare the run's seasonal COP against the datasheet point nearest to the
    # flow temperature - which is a FULL-LOAD figure. A heat pump at part load is legitimately more
    # efficient than its full-load rating (the F750 publishes COP 4.72 at minimum frequency and 2.43
    # at maximum), so the check failed an honest plant the moment the models became real.
    #
    # And the F2040 legitimately runs BELOW its published range: NIBE's coldest rating point is
    # -7 C, and a Swedish January reaches -11.6 C. Going below the datasheet there is physics, not
    # a bug.
    #
    # So only one direction is a defect: a plant that buys heat MORE CHEAPLY than the machine can
    # possibly make it. That is what a doubled COP looks like, and that is what this catches.
    if house is not None and stats["datasheet_cop"] > 0:
        ratio = stats["mean_cop"] / stats["datasheet_cop"]
        if ratio > COP_ENVELOPE_TOLERANCE:
            failures.append(
                f"the run's seasonal COP was {stats['mean_cop']:.2f}, against a best published "
                f"figure of {stats['datasheet_cop']:.2f} for this machine at ANY of its rating "
                f"points ({ratio:.2f}x) - the plant is buying heat more cheaply than the machine "
                f"can make it, so every cost number in this run is too low"
            )

    # THE OPTIMISER MAY NOT BURN MORE RESISTIVE HEAT THAN THE PUMP'S CAPACITY DEFICIT FORCES.
    #
    # This used to be an absolute budget - 0 kWh in a mild month, 25 kWh in a cold snap - and it was
    # a statement about a machine that does not exist. A correctly-sized air-source system is
    # BIVALENT by design: NIBE declares Tbiv = -9 C for the F2040-8, with 1.1 kW of supplementary
    # heat, and below that temperature the immersion heater is SUPPOSED to run. The absolute budget
    # failed the only correctly-sized air-source house in the set for doing what it was built to do.
    #
    # The physics-grounded question is the one worth asking, and the plant can answer it: at every
    # step, how much heat did the house need that the compressor could not physically deliver? Sum
    # that, and it is the resistive heat the WEATHER forces. Everything above it is the CONTROLLER's.
    unavoidable = stats["unavoidable_aux_kwh"]
    allowed = unavoidable * AUX_OVER_PHYSICS_TOLERANCE + AUX_SLACK_KWH

    if stats["aux_kwh"] > allowed:
        failures.append(
            f"the immersion heater burned {stats['aux_kwh']:.1f} kWh, but the pump's capacity "
            f"deficit only forced {unavoidable:.1f} kWh of it "
            f"({stats['aux_kwh'] / max(unavoidable, 1e-9):.1f}x). The rest is the controller's "
            f"doing: resistive heat at COP 1.0, bought because the offset was pinned at maximum "
            f"against a compressor that had nothing left to give"
        )

    if stats["comfort_minutes_below"] > MAX_COMFORT_MINUTES_BELOW:
        failures.append(
            f"{stats['comfort_minutes_below']} minutes below the comfort band "
            f"(budget {MAX_COMFORT_MINUTES_BELOW}) - the optimiser starved the house"
        )

    if stats["comfort_minutes_above"] > MAX_COMFORT_MINUTES_ABOVE:
        failures.append(
            f"{stats['comfort_minutes_above']} minutes above the comfort band "
            f"(budget {MAX_COMFORT_MINUTES_ABOVE}) - the optimiser cooked the house"
        )

    return failures


def main() -> int:
    selftest = "--selftest" in sys.argv
    coldsnap = "--coldsnap" in sys.argv
    baseline = "--baseline" in sys.argv
    battery = "--battery" in sys.argv
    live_se4 = "--live-se4" in sys.argv
    no_price = "--no-price" in sys.argv
    no_weather = "--no-weather" in sys.argv
    tuned_curve = "--tuned-baseline" in sys.argv
    undersized = "--undersized" in sys.argv
    no_forecast = "--no-forecast" in sys.argv
    dst = "--dst" in sys.argv
    mode = "balanced"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    # --dst spans the fall-back weekend: 3 days, one of them 25 hours long.
    days = DST_SIM_DAYS if dst else (2 if selftest else SIM_DAYS)
    times, temps, price_days, unit = load_data(selftest, live_se4, dst)
    if coldsnap:
        temps = apply_coldsnap(times, temps)
    OUT_DIR.mkdir(exist_ok=True)

    price_source = PriceSource(price_days, unit)
    exit_code = 0

    houses = HOUSES
    if undersized:
        # THE COMMONEST INSTALLATION FAULT THERE IS: a pump one size too small for its house.
        #
        # Sizing a house at the EN 14825 AVERAGE-climate design point while fitting it with a pump
        # certified at the COLD one is exactly that, and both figures are published, so the gap is
        # the manufacturer's own. It is not a hypothetical - it is what happens when a European-spec
        # sizing meets a Swedish winter.
        houses = [
            replace(h, name=f"{h.name}", hlc_w_per_k=h.hlc_w_per_k * UNDERSIZED_PUMP_FACTOR)
            for h in HOUSES
        ]

    for house in houses:
        stats, violations, trace = simulate(
            house,
            times,
            temps,
            price_source,
            days,
            mode,
            baseline,
            battery=battery,
            enable_price=not no_price,
            enable_weather=not no_weather,
            tuned_curve=tuned_curve,
            forecast_available=not no_forecast,
        )
        stats["price_unit_seen_by_adapter"] = price_source.unit
        tag = f"{house.name}{'-selftest' if selftest else ''}"
        if mode != "balanced":
            tag += f"-{mode}"
        if coldsnap:
            tag += "-coldsnap"
        if undersized:
            tag += "-undersized"
        if live_se4:
            tag += "-live-se4"
        if battery:
            tag += "-battery"
        if baseline:
            tag += "-baseline"
        if no_price:
            tag += "-noprice"
        if no_weather:
            tag += "-noweather"
        if no_forecast:
            tag += "-noforecast"
        if dst:
            tag += "-dst"
        if tuned_curve:
            tag += "-tuned"

        # The baseline run is a do-nothing controller used as a yardstick. It is
        # expected to breach comfort - that is the point of it - so it reports but
        # does not gate.
        failures = [] if (baseline or battery) else check_invariants(tag, stats, violations, house)

        if dst:
            # THE DST RUN MUST BE ABLE TO FAIL, OR IT IS DECORATION.
            #
            # A green --dst run proves very little on its own: the October night load is flat and
            # low, so merging the two 02:00 hours into one two-hour period produces the SAME mean,
            # the same tariff figure, and the same PASS. I checked - reverting the harness's period
            # key to the ambiguous `(date, hour)` moved not one of the reported numbers.
            #
            # What the merge DOES change is how many billable hours the day contains. A fall-back
            # day has 25. Count them, and the run can fail for the reason it exists.
            hours_on_the_long_day = stats["billing_hours_by_day"].get(DST_FALL_BACK_DAY)
            if hours_on_the_long_day != DST_FALL_BACK_HOURS:
                failures.append(
                    f"{DST_FALL_BACK_DAY} was billed as {hours_on_the_long_day} hours. The clocks "
                    f"go back that night, so it is {DST_FALL_BACK_HOURS} hours long and every one "
                    f"of them is separately metered. Billing 24 means the two 02:00 hours - which "
                    f"print the same digits and are an hour apart - were merged into one."
                )

        json.dump(
            {
                "house": house.name,
                "days": days,
                "stats": stats,
                "failures": failures,
                "violations": violations[:200],
            },
            open(OUT_DIR / f"summary-{tag}.json", "w"),
            indent=1,
        )
        json.dump(trace, open(OUT_DIR / f"trace-{tag}.json", "w"))
        votes = stats.pop("layer_votes", {})
        print(f"[{tag}] {json.dumps(stats)}")
        if votes:
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            total = max(days * 24 * 60 // STEP_MIN, 1)
            share = ", ".join(f"{n} {100 * h / total:.0f}%" for n, h in ranked)
            print(f"[{tag}] layers that voted: {share}")
        if violations:
            print(f"[{tag}] first violations: {violations[:5]}")
        if failures:
            exit_code = 1
            for failure in failures:
                print(f"[{tag}] FAIL: {failure}")
        else:
            print(f"[{tag}] PASS: all safety invariants held")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
