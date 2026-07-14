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
import json
import sys
import zoneinfo
from dataclasses import dataclass
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
AUX_STEP_KW = 3.0  # one aux step
STANDBY_KW = 0.1  # controller, pumps, standby losses
# Float arithmetic only. Any real discrepancy is orders of magnitude bigger than this.
COMPRESSOR_ENERGY_TOLERANCE_PCT = 0.1

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

# OUTDOOR-air capacity derating. Applies ONLY to a pump whose source is outdoor air (F2040).
#
# It does NOT apply to the exhaust-air pumps (F750, F730), whose source is ~20 C indoor
# ventilation air, nor to ground-source pumps drawing stable brine. Applying it to an exhaust-air
# pump saturated it in January, drove degree minutes to the integrator floor and let the emergency
# layer cook the house to 35.4 C - a defect in this plant model, not in the integration.
#
# An ASHP's heat output falls as the source air gets colder: less enthalpy in the
# air, and the compressor works across a wider lift. The EN 14511 rating points
# (A7/W35, A2/W35, A-7/W35, A-15/W35) trace a near-linear decline, so the plant
# model derates the profile's rated output linearly below the A7 rating point and
# floors it at the manufacturer's stated minimum.
#
# This is what lets degree minutes actually run away: when the demanded flow
# exceeds what the pump can deliver, supply saturates BELOW target and DM
# integrates downward without limit - the real mechanism behind an undersized
# pump falling back on the immersion heater in a cold snap.
#
# A ground-source pump draws from ~0 C brine year-round, so its capacity is flat
# against outdoor temperature and it is not derated here.
# COP is set by the LIFT, not by the weather. These place the source and the condenser.
KELVIN = 273.15
COP_RATING_FLOW_C = 35.0  # EN 14511 rating point is W35: the profile's COP curve is measured here
CONDENSER_APPROACH_K = 5.0  # refrigerant condenses this far above the water it is heating
EVAPORATOR_APPROACH_K = 5.0  # and evaporates this far below the source it is drawing from
MIN_LIFT_K = 10.0  # a compressor cannot usefully run at zero lift; bound the division
EXHAUST_AIR_SOURCE_C = 20.0  # F750/F730 draw ~20 C indoor extract air, all year
BRINE_SOURCE_C = 0.0  # F1155/S1155 draw ~0 C brine, stable year-round

ASHP_RATING_POINT_C = 7.0  # EN 14511 A7/W35
ASHP_DERATE_PER_C = 0.025  # fraction of rated output lost per C below A7
ASHP_MIN_CAPACITY_FRACTION = 0.45  # floor; below this the pump is aux-assisted

# Comfort accounting matches the engine's configured tolerance (not a looser
# ad-hoc band): minutes below target-tolerance count as under-heating.
TARGET_INDOOR = 22.0
COMFORT_TOLERANCE = 0.5
DESIGN_OUTDOOR = -15.0
DESIGN_SPREAD = 5.0
RADIATOR_EXPONENT = 1.3  # EN 442
UFH_EXPONENT = 1.1  # EN 1264
OVERSHOOT_TOLERANCE = 1.5  # overshoot band stays wider; heat is banked, not lost

# Illustrative Swedish effect tariff (SEK per kW of the mean of the top-3
# daily quarter-hour-mean peaks, per month). Rate is fictional-but-typical;
# the point is comparing runs, not billing accuracy.
EFFECT_TARIFF_SEK_PER_KW = 81.25


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
    def dm_aux_limit(self) -> float:
        """Aux-heat threshold, taken from the pump profile rather than restated.

        The correct value for real NIBE hardware is contested (see the audit's
        F-112). Reading it from the profile means the plant model tracks whatever
        the integration believes, instead of silently diverging from it.
        """
        return float(self.profile.dm_threshold_aux_swedish)

    @property
    def derates_with_outdoor_temp(self) -> bool:
        """Whether the pump's capacity falls as it gets colder OUTSIDE.

        Only a pump whose SOURCE is outdoor air does. Read from the profile rather than asserted
        here, because getting this wrong is not a detail: derating an exhaust-air pump by outdoor
        temperature saturated it in a January simulation, which drove degree minutes to the
        integrator floor and let the emergency layer cook the house to 35.4 C. That was a defect in
        THIS model, not in the integration.

        - Exhaust air (F750, F730): the source is ~20 C indoor ventilation air, which does not get
          colder when the weather does. The F750 profile documents this itself.
        - Ground source (F1155, S1155): ~0 C brine, stable year-round.
        - Outdoor air (F2040): genuinely derates.
        """
        if getattr(self.profile, "supports_exhaust_airflow", False):
            return False
        return "GSHP" not in getattr(self.profile, "model_type", "")

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

    def cop_at(self, outdoor_temp: float, flow_temp: float) -> float:
        """COP as a function of the LIFT, anchored on the manufacturer's own COP curve.

        THE SIMULATED PUMP USED TO IGNORE FLOW TEMPERATURE ENTIRELY:

            cop = house.cop_at(tout, flow)     # outdoor only

        A heat pump's COP is governed by the lift between the water it makes and the source it
        draws from - Carnot, degraded by a real machine's exergy efficiency. OpenEnergyMonitor's
        measured fleet puts the penalty at 2.5-3 % of COP per degree of flow temperature.

        Ignoring that does not just make the plant unrealistic, it makes the harness INCAPABLE OF
        MEASURING ITS OWN PRODUCT: running cooler water is the entire mechanism by which weather
        compensation saves money, and with a flow-blind COP a lower curve buys no efficiency at
        all - only less heat, to be paid back later. The optimiser could therefore only ever look
        like a loss, and it duly did (+4 % against a do-nothing controller). That was an artefact
        of this model, not a finding about the integration.

        The profile's published COP curve is kept as the anchor - it is real manufacturer data,
        measured at the W35 rating point - and Carnot supplies the flow-temperature dependence
        around it. So at 35 C flow this returns exactly what it always returned.

        The sensitivity this produces is NOT one number, and it should not be: it runs from about
        1.9 %/C on a ground-source pump lifting from 0 C brine to about 3.7 %/C on an exhaust-air
        pump lifting from 20 C extract air. A small lift is proportionally more sensitive to a
        degree of flow than a large one. OEM's measured 2.5-3 %/C is a fleet average of mostly
        air-source machines and sits inside that range - which is the check, rather than a target
        to be hit by tuning an exponent.
        """
        source = self.source_temp_c(outdoor_temp)
        rated = float(self.profile.get_cop_at_temperature(outdoor_temp))

        def carnot(flow: float) -> float:
            t_cond = flow + CONDENSER_APPROACH_K + KELVIN
            t_evap = source - EVAPORATOR_APPROACH_K + KELVIN
            return t_cond / max(t_cond - t_evap, MIN_LIFT_K)

        return max(1.0, rated * carnot(flow_temp) / carnot(COP_RATING_FLOW_C))

    def capacity_kw_at(self, outdoor_temp: float) -> float:
        """Compressor heat output the pump can actually deliver right now."""
        rated = float(self.profile.rated_power_kw[1])
        if not self.derates_with_outdoor_temp:
            return rated
        derate = 1.0 - ASHP_DERATE_PER_C * max(0.0, ASHP_RATING_POINT_C - outdoor_temp)
        return rated * max(ASHP_MIN_CAPACITY_FRACTION, derate)


# Every pump the integration ships a profile for. Two houses could not exercise the paths that
# only exist for some hardware: an ASHP is the ONLY kind that derates as the weather gets colder,
# so it is the only one that can saturate, drive degree minutes away and reach for the immersion
# heater - which is precisely the failure the safety layers exist to prevent, and it was never
# once simulated.
HOUSES = [
    HouseConfig(
        name="wooden_f750",  # exhaust air, radiators, light timber frame
        thermal_mass=0.7,
        insulation_quality=1.0,
        hlc_w_per_k=150.0,
        tau_hours=30.0,
        profile=NibeF750Profile(),
        heating_type="radiator",
        design_flow=50.0,
    ),
    HouseConfig(
        name="concrete_f1155",  # ground source, underfloor, heavy slab
        thermal_mass=1.8,
        insulation_quality=1.2,
        hlc_w_per_k=180.0,
        tau_hours=80.0,
        profile=NibeF1155Profile(),
        heating_type="concrete_ufh",
        design_flow=38.0,
    ),
    HouseConfig(
        name="apartment_f730",  # small exhaust-air pump, tight modern flat
        thermal_mass=0.9,
        insulation_quality=1.3,
        hlc_w_per_k=90.0,
        tau_hours=45.0,
        profile=NibeF730Profile(),
        heating_type="radiator",
        design_flow=45.0,
    ),
    HouseConfig(
        name="villa_s1155",  # S-series ground source, timber underfloor
        thermal_mass=1.2,
        insulation_quality=1.1,
        hlc_w_per_k=160.0,
        tau_hours=55.0,
        profile=NibeS1155Profile(),
        heating_type="timber_ufh",
        design_flow=40.0,
    ),
    HouseConfig(
        name="airsource_f2040",  # THE HARD ONE: outdoor air, so capacity collapses in a cold snap
        thermal_mass=1.0,
        insulation_quality=0.9,
        hlc_w_per_k=220.0,
        tau_hours=40.0,
        profile=NibeF2040Profile(),
        heating_type="radiator",
        design_flow=55.0,
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
    out: dict[str, list[dict[str, Any]]] = {}
    for day, raw in days.items():
        entries: list[dict[str, Any]] = []
        expand = 1 if len(raw) >= 90 else QUARTERS_PER_HOUR
        for item in raw:
            start = datetime.fromisoformat(item["start"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=TZ)
            start = start.astimezone(TZ)
            for q in range(expand):
                entries.append(
                    {
                        "time": (start + timedelta(minutes=QUARTER_MINUTES * q)).isoformat(),
                        "value": item["price"] * ore_per_unit,
                    }
                )
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


def load_data(selftest: bool, live_se4: bool = False):
    """Load real weather + prices, or synthetic 2-day data for --selftest."""
    if selftest:
        start = datetime(2026, 1, 1, tzinfo=TZ)
        hours = 48
        temps = [-5.0 + 4.0 * ((h % 24) / 24.0) for h in range(hours)]
        times = [start + timedelta(hours=h) for h in range(hours)]
        raw = {}
        for d in range(2):
            day = (start + timedelta(days=d)).date().isoformat()
            raw[day] = [
                {
                    "start": (start + timedelta(days=d, minutes=15 * q)).isoformat(),
                    "price": 500.0 + 400.0 * (1 if 28 <= q <= 40 or 68 <= q <= 80 else 0),
                }
                for q in range(96)
            ]
        return times, temps, _to_gespot_shape(raw, ORE_PER_KWH_FROM_SEK_PER_MWH), GESPOT_UNIT_ORE

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
        "compressor_elec_metered_kwh": 0.0,
        "compressor_elec_owed_kwh": 0.0,
    }
    last_offsets = []
    quarter_samples: list[float] = []
    quarter_id = None
    daily_peaks: dict = {}  # date -> max quarter-mean kW
    # Highest completed quarter-hour MEAN so far: what the coordinator publishes as
    # peak_this_month, and therefore what the effect layer is defending. Starts at
    # zero, as it does on a fresh install.
    running_peak_kw = 0.0

    for step in range(steps):
        now = start + timedelta(minutes=STEP_MIN * step)
        # Freeze engine wall clock to sim time
        dt_util.now = lambda tz=None, _n=now: _n
        dt_util.utcnow = lambda _n=now: _n.astimezone(zoneinfo.ZoneInfo("UTC"))

        tout = outdoor_at(times, temps, now)

        # --- plant step ---
        flow_target = house.curve_flow_temp(tout, tuned_curve) + offset_applied

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

        if compressor_on:
            # The compressor modulates toward the flow its curve is asking for, bounded by what it
            # can actually deliver at this outdoor temperature.
            demand_w = q_emit_w + WATER_LOOP_J_PER_K * (flow_target - flow) / COMPRESSOR_RESPONSE_S
            q_comp_w = max(0.0, min(demand_w, house.capacity_kw_at(tout) * 1000.0))
        else:
            q_comp_w = 0.0

        aux_w = 0.0
        if dm <= house.dm_aux_limit:
            aux_w = AUX_STEP_KW * 1000.0

        flow += (q_comp_w + aux_w - q_emit_w) * (STEP_MIN * 60.0) / WATER_LOOP_J_PER_K
        flow = max(indoor, min(flow, float(house.profile.max_flow_temp)))

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

        cop = house.cop_at(tout, flow)
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
        weather = WeatherData(current_temp=tout, forecast_hours=fc, source_entity="sim")

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
        stats["offset_min"] = min(stats["offset_min"], offset_applied)
        stats["offset_max"] = max(stats["offset_max"], offset_applied)
        energy = power_kw * STEP_MIN / 60.0
        stats["energy_kwh"] += energy

        # THE COMPRESSOR-SIDE AUDIT. What the meter recorded for the compressor, and - computed
        # independently, from the heat it made and the COP it made it at - what that heat SHOULD
        # have cost. These are two different expressions of the same joules, so they can disagree,
        # which is the whole point: the room-side balance below CANNOT.
        stats["compressor_elec_metered_kwh"] += (
            max(0.0, power_kw - aux_kw - STANDBY_KW) * STEP_MIN / 60.0
        )
        stats["compressor_elec_owed_kwh"] += (q_comp_w / 1000.0) / cop * STEP_MIN / 60.0
        # First-law audit. Heat INTO the room, and heat OUT of it. Over a month these must balance
        # to within the change in the fabric's stored energy - otherwise the plant is inventing or
        # destroying energy and every cost number it produces is fiction.
        stats["heat_kwh"] += q_w * STEP_MIN / 60.0 / 1000.0
        stats["loss_kwh"] += (
            (house.hlc_w_per_k * (indoor - tout) - INTERNAL_GAINS_W) * STEP_MIN / 60.0 / 1000.0
        )
        stats["aux_kwh"] += aux_kw * STEP_MIN / 60.0
        stats["cost_sek"] += energy * cur_price_ore / 100.0

        # Effect tariff basis: quarter-hour MEAN power (Swedish effektavgift),
        # never the instantaneous sample.
        this_quarter = (now.date(), cur_q)
        if quarter_id is not None and this_quarter != quarter_id:
            q_mean = sum(quarter_samples) / len(quarter_samples)
            day = quarter_id[0]
            daily_peaks[day] = max(daily_peaks.get(day, 0.0), q_mean)
            running_peak_kw = max(running_peak_kw, q_mean)

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
                effect.record_quarter_measurement(
                    power_kw=q_mean,
                    quarter=quarter_id[1],
                    timestamp=now,
                    source=POWER_SOURCE_EXTERNAL_METER,
                )
            )
            quarter_samples = []
        quarter_id = this_quarter
        quarter_samples.append(power_kw)

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

    if quarter_samples and quarter_id is not None:
        q_mean = sum(quarter_samples) / len(quarter_samples)
        day = quarter_id[0]
        daily_peaks[day] = max(daily_peaks.get(day, 0.0), q_mean)
    top3 = sorted(daily_peaks.values(), reverse=True)[:3]
    tariff_kw = sum(top3) / len(top3) if top3 else 0.0
    stats["peak_kw_quarter_mean"] = round(max(daily_peaks.values()), 2) if daily_peaks else 0.0
    stats["tariff_top3_kw"] = round(tariff_kw, 2)
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
    stored_kwh = house.capacity_j_per_k * (indoor - indoor_start) / 3_600_000.0
    residual = stats["heat_kwh"] - stats["loss_kwh"] - stored_kwh
    stats["heat_kwh"] = round(stats["heat_kwh"], 1)
    stats["loss_kwh"] = round(stats["loss_kwh"], 1)
    stats["energy_balance_residual_kwh"] = round(residual, 2)

    # THE CHECK THAT CAN ACTUALLY FAIL. What the meter charged for the compressor, against what its
    # heat owed at the COP it was made at. Two independent expressions of the same joules.
    metered = stats["compressor_elec_metered_kwh"]
    owed = stats["compressor_elec_owed_kwh"]
    stats["compressor_elec_metered_kwh"] = round(metered, 1)
    stats["compressor_elec_owed_kwh"] = round(owed, 1)
    stats["compressor_energy_error_pct"] = round(100.0 * (metered - owed) / max(owed, 1e-9), 2)
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
AUX_BUDGET_KWH_MILD = 0.0
AUX_BUDGET_KWH_COLDSNAP = 25.0

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

    # The plant is not allowed to invent or destroy energy on the compressor side. This is the
    # check the room-side residual could never be: the meter's compressor electricity against what
    # that heat owed at the COP it was made at, computed independently.
    if abs(stats["compressor_energy_error_pct"]) > COMPRESSOR_ENERGY_TOLERANCE_PCT:
        failures.append(
            f"the compressor was metered {stats['compressor_elec_metered_kwh']:.1f} kWh but its "
            f"heat owed {stats['compressor_elec_owed_kwh']:.1f} kWh at the COP it was made at "
            f"({stats['compressor_energy_error_pct']:+.1f}%) - the plant is inventing or destroying "
            f"energy, and every cost number it produces is fiction"
        )

    # Tracked since the harness was written. Asserted for the first time here.
    aux_budget = AUX_BUDGET_KWH_COLDSNAP if "coldsnap" in tag else AUX_BUDGET_KWH_MILD
    if stats["aux_kwh"] > aux_budget:
        failures.append(
            f"the immersion heater burned {stats['aux_kwh']:.1f} kWh (budget {aux_budget:.0f}) - "
            f"the degree-minute ladder failed to recover the house before the aux limit"
        )

    if house is not None:
        aux_limit = house.dm_aux_limit
        if stats["dm_min"] <= aux_limit + DM_AUX_MARGIN:
            failures.append(
                f"degree minutes reached {stats['dm_min']:.0f}, within {DM_AUX_MARGIN:.0f} of the "
                f"{aux_limit:.0f} aux limit - the ladder is only just holding"
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
    mode = "balanced"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    days = 2 if selftest else SIM_DAYS
    times, temps, price_days, unit = load_data(selftest, live_se4)
    if coldsnap:
        temps = apply_coldsnap(times, temps)
    OUT_DIR.mkdir(exist_ok=True)

    price_source = PriceSource(price_days, unit)
    exit_code = 0

    for house in HOUSES:
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
        )
        stats["price_unit_seen_by_adapter"] = price_source.unit
        tag = f"{house.name}{'-selftest' if selftest else ''}"
        if mode != "balanced":
            tag += f"-{mode}"
        if coldsnap:
            tag += "-coldsnap"
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
        if tuned_curve:
            tag += "-tuned"

        # The baseline run is a do-nothing controller used as a yardstick. It is
        # expected to breach comfort - that is the point of it - so it reports but
        # does not gate.
        failures = [] if (baseline or battery) else check_invariants(tag, stats, violations, house)

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
