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

import json
import sys
import zoneinfo
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)

try:
    from custom_components.effektguard.models.nibe import NibeF750Profile, NibeF1155Profile
except ImportError:  # F1155 profile ships with the multi-source PR (#19)
    from custom_components.effektguard.models.nibe import NibeF750Profile
    from custom_components.effektguard.models.nibe import NibeS1155Profile as NibeF1155Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TZ = zoneinfo.ZoneInfo("Europe/Stockholm")
STEP_MIN = 5
DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

# Plant constants
FLOW_RAMP_ON = 0.5  # C/min toward target while compressor runs
FLOW_DECAY_OFF = 0.1  # C/min toward indoor when off
DM_START = -60.0
DM_STOP = 0.0
DM_AUX = -1500.0
AUX_STEP_KW = 3.0  # one aux step
TOMORROW_VISIBLE_HOUR = 13  # Nordpool day-ahead published ~12:45 CET

# Comfort accounting matches the engine's configured tolerance (not a looser
# ad-hoc band): minutes below target-tolerance count as under-heating.
TARGET_INDOOR = 22.0
COMFORT_TOLERANCE = 0.5
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
    max_heat_kw: float

    @property
    def capacity_j_per_k(self) -> float:
        return self.hlc_w_per_k * self.tau_hours * 3600.0

    @property
    def k_emit(self) -> float:
        # Sized so design heat demand is met at design flow with Tin=22
        design_q = self.hlc_w_per_k * (22.0 - (-15.0))
        return design_q / (self.design_flow - 22.0)

    @property
    def curve_slope(self) -> float:
        # Curve calibrated so the plant balances at 22 C indoor with offset 0
        # (a correctly tuned NIBE): flow_target(-15) == design_flow
        return (self.design_flow - 22.0) / 37.0


HOUSES = [
    HouseConfig(
        name="wooden_f750",
        thermal_mass=0.7,
        insulation_quality=1.0,
        hlc_w_per_k=150.0,
        tau_hours=30.0,
        profile=NibeF750Profile(),
        heating_type="radiator",
        design_flow=50.0,
        max_heat_kw=8.0,
    ),
    HouseConfig(
        name="concrete_f1155",
        thermal_mass=1.8,
        insulation_quality=1.2,
        hlc_w_per_k=180.0,
        tau_hours=80.0,
        profile=NibeF1155Profile(),
        heating_type="concrete_ufh",
        design_flow=38.0,
        max_heat_kw=12.0,
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


def load_data(selftest: bool):
    """Load real weather + prices, or synthetic 2-day data for --selftest."""
    if selftest:
        start = datetime(2026, 1, 1, tzinfo=TZ)
        hours = 48
        temps = [-5.0 + 4.0 * ((h % 24) / 24.0) for h in range(hours)]
        times = [start + timedelta(hours=h) for h in range(hours)]
        prices = {}
        for d in range(2):
            day = (start + timedelta(days=d)).date().isoformat()
            prices[day] = [
                {
                    "start": (start + timedelta(days=d, minutes=15 * q)).isoformat(),
                    "price": 500.0 + 400.0 * (1 if 28 <= q <= 40 or 68 <= q <= 80 else 0),
                }
                for q in range(96)
            ]
        return times, temps, prices

    weather = json.load(open(DATA_DIR / "weather_jan2026.json"))
    times = [
        datetime.fromisoformat(t).replace(tzinfo=TZ) if "T" in t else None
        for t in weather["hourly"]["time"]
    ]
    temps = weather["hourly"]["temperature_2m"]
    prices = json.load(open(DATA_DIR / "prices_jan2026.json"))["days"]
    return times, temps, prices


def outdoor_at(times, temps, when: datetime) -> float:
    """Linear interpolation of hourly outdoor temperature."""
    idx = int((when - times[0]).total_seconds() // 3600)
    idx = max(0, min(idx, len(temps) - 2))
    frac = ((when - times[idx]).total_seconds() / 3600.0) if idx < len(times) else 0.0
    frac = max(0.0, min(frac, 1.0))
    return temps[idx] * (1 - frac) + temps[idx + 1] * frac


def quarters_for_day(prices: dict, day: datetime) -> list[QuarterPeriod]:
    """Build QuarterPeriods (ore/kWh) for a day; expand hourly data if needed."""
    raw = prices.get(day.date().isoformat())
    if not raw:
        return []
    periods = []
    if len(raw) >= 90:  # 15-min data
        for entry in raw:
            st = datetime.fromisoformat(entry["start"])
            if st.tzinfo is None:
                st = st.replace(tzinfo=TZ)
            periods.append(
                QuarterPeriod(start_time=st.astimezone(TZ), price=entry["price"] / 10.0)
            )  # SEK/MWh -> ore/kWh
    else:  # hourly -> repeat 4x
        for entry in raw:
            st = datetime.fromisoformat(entry["start"])
            if st.tzinfo is None:
                st = st.replace(tzinfo=TZ)
            st = st.astimezone(TZ)
            for q in range(4):
                periods.append(
                    QuarterPeriod(
                        start_time=st + timedelta(minutes=15 * q), price=entry["price"] / 10.0
                    )
                )
    return periods


def build_engine(house: HouseConfig, mode: str = "balanced"):
    hass = MagicMock()
    effect = EffectManager(hass)
    thermal = ThermalModel(house.thermal_mass, house.insulation_quality)
    config = {
        "target_indoor_temp": TARGET_INDOOR,
        "tolerance": COMFORT_TOLERANCE,
        "optimization_mode": mode,
        "enable_weather_compensation": True,
        "enable_peak_protection": True,
        "enable_price_optimization": True,
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
    prices,
    days: int,
    mode: str = "balanced",
    baseline: bool = False,
):
    engine, effect = build_engine(house, mode)

    start = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    steps = days * 24 * 60 // STEP_MIN

    indoor = 22.0
    dm = -30.0
    offset_applied = 0  # integer offset "in the pump" (register 47011)
    accumulator_ref = 0  # mirrors adapter _last_nibe_offset behaviour
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
    }
    last_offsets = []
    quarter_samples: list[float] = []
    quarter_id = None
    daily_peaks: dict = {}  # date -> max quarter-mean kW

    for step in range(steps):
        now = start + timedelta(minutes=STEP_MIN * step)
        # Freeze engine wall clock to sim time
        dt_util.now = lambda tz=None, _n=now: _n
        dt_util.utcnow = lambda _n=now: _n.astimezone(zoneinfo.ZoneInfo("UTC"))

        tout = outdoor_at(times, temps, now)

        # --- plant step ---
        flow_target = 22.0 + house.curve_slope * (22.0 - tout) + offset_applied
        if compressor_on:
            flow = min(flow + FLOW_RAMP_ON * STEP_MIN, flow_target + 1.0)
        else:
            flow = max(flow - FLOW_DECAY_OFF * STEP_MIN, indoor)

        q_w = max(0.0, house.k_emit * (flow - indoor))
        q_w = min(q_w, house.max_heat_kw * 1000.0)

        aux_kw = 0.0
        if dm <= DM_AUX:
            aux_kw = AUX_STEP_KW
            q_w += aux_kw * 1000.0

        # Indoor temperature ODE
        d_indoor = (q_w - house.hlc_w_per_k * (indoor - tout)) / house.capacity_j_per_k
        indoor += d_indoor * STEP_MIN * 60.0

        # DM dynamics + compressor hysteresis
        dm += (flow - flow_target) * STEP_MIN
        dm = max(-3000.0, min(dm, 100.0))
        if not compressor_on and dm <= DM_START:
            compressor_on = True
            stats["compressor_starts"] += 1
        elif compressor_on and dm >= DM_STOP:
            compressor_on = False

        cop = house.profile.get_cop_at_temperature(tout)
        power_kw = (q_w / 1000.0 - aux_kw) / cop + aux_kw + 0.1 if compressor_on or aux_kw else 0.1
        hz = 40 + int(min(50, max(0, (flow_target - indoor)))) if compressor_on else 0

        # --- price/weather context ---
        today_q = quarters_for_day(prices, now)
        tomorrow_q = (
            quarters_for_day(prices, now + timedelta(days=1))
            if now.hour >= TOMORROW_VISIBLE_HOUR
            else []
        )
        price_data = PriceData(today=today_q, tomorrow=tomorrow_q, has_tomorrow=bool(tomorrow_q))
        cur_q = (now.hour * 4) + now.minute // 15
        cur_price_ore = today_q[cur_q].price if len(today_q) > cur_q else 100.0

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
        if baseline:
            calc_offset = 0.0
        else:
            try:
                decision = engine.calculate_decision(
                    nibe_state=nibe,
                    price_data=price_data,
                    weather_data=weather,
                    current_peak=6.0,
                    current_power=power_kw,
                )
                calc_offset = decision.offset
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

        # Adapter-faithful integer write (fractional accumulator, threshold 1.0)
        if abs(calc_offset - accumulator_ref) >= 1.0:
            new_int = accumulator_ref + int(calc_offset - accumulator_ref)
            new_int = int(max(-10, min(10, new_int)))
            if new_int != offset_applied:
                offset_applied = new_int
                accumulator_ref = new_int
                stats["writes"] += 1

        # --- invariants & stats ---
        if dm < -1500 and aux_kw == 0:
            violations.append(
                {"t": now.isoformat(), "type": "dm_below_aux_limit", "detail": f"DM {dm:.0f}"}
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
        stats["aux_kwh"] += aux_kw * STEP_MIN / 60.0
        stats["cost_sek"] += energy * cur_price_ore / 100.0

        # Effect tariff basis: quarter-hour MEAN power (Swedish effektavgift),
        # never the instantaneous sample.
        this_quarter = (now.date(), cur_q)
        if quarter_id is not None and this_quarter != quarter_id:
            q_mean = sum(quarter_samples) / len(quarter_samples)
            day = quarter_id[0]
            daily_peaks[day] = max(daily_peaks.get(day, 0.0), q_mean)
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
    stats["violations"] = len(violations)
    return stats, violations, trace


def main():
    selftest = "--selftest" in sys.argv
    coldsnap = "--coldsnap" in sys.argv
    baseline = "--baseline" in sys.argv
    mode = "balanced"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    days = 2 if selftest else 31
    times, temps, prices = load_data(selftest)
    if coldsnap:
        temps = apply_coldsnap(times, temps)
    OUT_DIR.mkdir(exist_ok=True)

    for house in HOUSES:
        stats, violations, trace = simulate(house, times, temps, prices, days, mode, baseline)
        tag = f"{house.name}{'-selftest' if selftest else ''}"
        if mode != "balanced":
            tag += f"-{mode}"
        if coldsnap:
            tag += "-coldsnap"
        if baseline:
            tag += "-baseline"
        json.dump(
            {"house": house.name, "days": days, "stats": stats, "violations": violations[:200]},
            open(OUT_DIR / f"summary-{tag}.json", "w"),
            indent=1,
        )
        json.dump(trace, open(OUT_DIR / f"trace-{tag}.json", "w"))
        print(f"[{tag}] {json.dumps(stats)}")
        if violations:
            print(f"[{tag}] first violations: {violations[:5]}")


if __name__ == "__main__":
    main()
