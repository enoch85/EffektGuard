"""Battery mode (experimental): charge in the day's cheapest hours, coast through its dearest.

Drives the real battery_adjustment and the real DecisionEngine. The day used: hours 0-5 at 10, 6-9 at
100, 10-17 at 50, 18-19 at 90, 20-23 at 40 (one price unit, any unit: the rules are relative).

The shipped default leaves coasting to the price layer (BATTERY_MODE_COAST 0); the coast path is tested
at COAST_TEST through the `coasting` fixture, so it stays correct for anyone who turns it on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.const import (
    BATTERY_MODE_AUX_START_FRACTION,
    BATTERY_MODE_CHARGE,
    BATTERY_MODE_COAST,
    CONF_ENABLE_BATTERY_MODE,
)
from custom_components.effektguard.optimization import battery as battery_module
from custom_components.effektguard.optimization.battery import battery_adjustment
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TZ = ZoneInfo("Europe/Stockholm")
DAY = datetime(2026, 1, 14, tzinfo=TZ)
HOURLY = [10.0] * 6 + [100.0] * 4 + [50.0] * 8 + [90.0] * 2 + [40.0] * 4
TARGET = 21.0
BAND = 0.48  # tolerance 1.2 x 0.4
COAST_TEST = 2.0
AUX_START = NibeF750Profile().aux_start_dm
HZ_MAX = 90  # a pump that publishes its maximum frequency (43123 on an F750)
FULL = (
    TARGET + BAND
)  # where the comfort layer's overshoot ramp starts (the band beats balanced's dead zone)


@pytest.fixture
def coasting(monkeypatch):
    monkeypatch.setattr(battery_module, "BATTERY_MODE_COAST", COAST_TEST)


@pytest.fixture(autouse=True)
def _stockholm():
    old = dt_util.get_default_time_zone()
    dt_util.set_default_time_zone(TZ)
    yield
    dt_util.set_default_time_zone(old)


def _prices(hourly=HOURLY, day=DAY) -> PriceData:
    quarters, t = [], day
    for price in hourly:
        for _ in range(4):
            quarters.append(QuarterPeriod(start_time=t, price=price))
            t += timedelta(minutes=15)
    return PriceData(today=quarters, tomorrow=[], has_tomorrow=False)


def _adjust(
    now,
    prices,
    room=TARGET + 0.2,
    tier="OK",
    zone="NONE",
    saturated=False,
    valid=True,
    dm=-100.0,
    aux_start=AUX_START,
    peak=False,
    hz_max=HZ_MAX,
):
    return battery_adjustment(
        now=now,
        price_data=prices,
        nibe_state=SimpleNamespace(
            indoor_temp=room, degree_minutes=dm, indoor_temp_valid=valid, compressor_hz_max=hz_max
        ),
        target_temp=TARGET,
        tolerance_range=BAND,
        full_at=FULL,
        emergency_tier=tier,
        proactive_zone=zone,
        compressor_saturated=saturated,
        aux_start_dm=aux_start,
        peak_protection_active=peak,
    )


def _move(hour, prices=None, **state):
    now = DAY + timedelta(hours=hour, minutes=5)
    return _adjust(now, _prices() if prices is None else prices, **state)[0]


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (2, BATTERY_MODE_CHARGE),
        (7, -COAST_TEST),
        (19, -COAST_TEST),
        (12, 0.0),
        (21, 0.0),
    ],
    ids=["cheap night", "dear morning", "dear evening", "middling day", "middling evening"],
)
def test_the_day_is_charged_cheap_and_coasted_dear(coasting, hour, expected):
    assert _move(hour) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("hour", "kwargs"),
    [
        (2, {"room": FULL}),  # full: where the comfort layer starts pulling the room down
        (2, {"hz_max": None}),  # no published maximum: headroom unknown (an F2040)
        (2, {"dm": AUX_START * BATTERY_MODE_AUX_START_FRACTION}),  # half way to the immersion start
        (2, {"aux_start": None}),  # the pump's immersion start unknown: missing input abstains
        (2, {"peak": True}),  # peak protection acting: a charge would outvote it
        (7, {"room": TARGET - BAND}),  # empty: at the band's lower edge
        (2, {"valid": False}),
        (7, {"valid": False}),
        (2, {"saturated": True}),  # no frequency left: a charge only buys immersion heat
        (7, {"saturated": True}),  # nothing to refill a coast with later
        (2, {"zone": "Z3"}),  # degree minutes past half their normal range
        (7, {"zone": "Z5"}),
        (2, {"tier": "T1"}),  # repaying debt
        (7, {"tier": "T2"}),
        (2, {"tier": "EMERGENCY"}),  # past the aux limit (F-124)
        (7, {"tier": "ANTI_WINDUP"}),
    ],
    ids=[
        "charged full",
        "no published maximum frequency",
        "half way to the immersion heater",
        "immersion start unknown",
        "peak protection acting",
        "coasted empty",
        "no room reading, cheap",
        "no room reading, dear",
        "saturated, cheap",
        "saturated, dear",
        "Z3",
        "Z5",
        "T1",
        "T2",
        "EMERGENCY",
        "anti-windup",
    ],
)
def test_a_guard_stops_the_battery(coasting, hour, kwargs):
    assert _move(hour, **kwargs) == 0.0


@pytest.mark.parametrize("zone", ["Z1", "Z2"])
def test_the_first_prevention_rungs_leave_the_battery_on(coasting, zone):
    """Z1 starts at 2 % of the normal degree-minute range: guarding on it, the battery never ran."""
    assert (_move(2, zone=zone), _move(7, zone=zone)) == pytest.approx(
        (BATTERY_MODE_CHARGE, -COAST_TEST)
    )


def test_the_shipped_default_leaves_coasting_to_the_price_layer():
    assert _adjust(DAY + timedelta(hours=7, minutes=5), _prices()) == (0.0, "")


@pytest.mark.parametrize("kwargs", [{"peak": True}, {"hz_max": None}], ids=["peak", "no maximum"])
def test_a_charge_guard_leaves_a_coast_alone(coasting, kwargs):
    """Only a charge adds power and asks for headroom: a coast passes these guards."""
    assert _move(7, **kwargs) == pytest.approx(-COAST_TEST)


def test_a_flat_day_is_left_alone():
    """A spread under 5 % of the price scale (the price layer's own flat-day rule) ranks noise."""
    assert _move(2, prices=_prices([50.0 + 0.01 * hour for hour in range(24)])) == 0.0


def test_a_step_day_charges_only_the_hours_below_its_median():
    """Three hours at -10, the rest at 120: the 'six cheapest' would reach into the dear plateau."""
    step = [-10.0] * 3 + [120.0] * 21
    assert _move(1, prices=_prices(step)) == pytest.approx(BATTERY_MODE_CHARGE)
    assert _move(4, prices=_prices(step)) == 0.0


@pytest.mark.parametrize("prices", [None, PriceData(today=[], tomorrow=[], has_tomorrow=False)])
def test_no_prices_no_battery(prices):
    assert _adjust(DAY + timedelta(hours=2), prices) == (0.0, "")


def test_yesterdays_prices_are_not_todays():
    assert _move(2, prices=_prices(day=DAY - timedelta(days=1))) == 0.0


@pytest.mark.parametrize(
    ("day", "hours"),
    [(datetime(2026, 3, 29, tzinfo=TZ), 23), (datetime(2026, 10, 25, tzinfo=TZ), 25)],
    ids=["clocks forward", "clocks back"],
)
def test_a_clock_change_day_still_ranks_by_local_hour(day, hours):
    """92 or 100 quarters; on the long day the repeated 02:00 hour is one hour of eight quarters."""
    start = day.astimezone(ZoneInfo("UTC"))
    quarters = [
        QuarterPeriod(
            start_time=(start + timedelta(minutes=15 * i)).astimezone(TZ),
            price=10.0 if i < 16 else 100.0,
        )
        for i in range(hours * 4)
    ]
    prices = PriceData(today=quarters, tomorrow=[], has_tomorrow=False)
    move = _adjust(day + timedelta(hours=1, minutes=5), prices)[0]
    assert move == pytest.approx(BATTERY_MODE_CHARGE)


def _engine(battery: bool | None, profile=NibeF750Profile()) -> DecisionEngine:
    config = {"target_indoor_temp": TARGET, "tolerance": 1.2, "latitude": 55.6}
    if battery is not None:
        config[CONF_ENABLE_BATTERY_MODE] = battery
    return DecisionEngine(
        price_analyzer=__import__(
            "custom_components.effektguard.optimization.price_layer", fromlist=["PriceAnalyzer"]
        ).PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(thermal_mass=2.0, insulation_quality=1.0),
        config=config,
        heat_pump_model=profile,
    )


def _house(hour: int, room: float = TARGET + 0.2) -> NibeState:
    return NibeState(
        outdoor_temp=2.0,
        indoor_temp=room,
        supply_temp=30.0,
        return_temp=26.0,
        degree_minutes=50.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=DAY + timedelta(hours=hour),
        compressor_hz=40,
        compressor_hz_max=HZ_MAX,
        indoor_temp_valid=True,
    )


@pytest.mark.parametrize(
    ("hour", "move"),
    [(2, BATTERY_MODE_CHARGE), (7, -BATTERY_MODE_COAST), (12, 0.0)],
    ids=["cheap", "dear", "middling"],
)
def test_the_engine_adds_the_battery_to_the_layers_decision(freezer, hour, move):
    freezer.move_to(DAY + timedelta(hours=hour, minutes=5))
    decide = lambda engine: engine.calculate_decision(  # noqa: E731
        nibe_state=_house(hour),
        price_data=_prices(),
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )
    off, on = decide(_engine(None)), decide(_engine(True))

    assert on.offset - off.offset == pytest.approx(move), (off.reasoning, on.reasoning)
    assert ("Battery" in on.reasoning) is (move != 0.0)


def test_battery_mode_is_off_unless_switched_on(freezer):
    freezer.move_to(DAY + timedelta(hours=2, minutes=5))
    decide = lambda engine: engine.calculate_decision(  # noqa: E731
        nibe_state=_house(2),
        price_data=_prices(),
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )
    assert decide(_engine(False)).offset == decide(_engine(None)).offset


def test_an_engine_without_a_pump_profile_does_not_charge(freezer):
    """The immersion start comes from the pump's profile; without one the charge abstains."""
    freezer.move_to(DAY + timedelta(hours=2, minutes=5))
    decide = lambda engine: engine.calculate_decision(  # noqa: E731
        nibe_state=_house(2),
        price_data=_prices(),
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )
    assert decide(_engine(True, profile=None)).offset == decide(_engine(None, profile=None)).offset


@pytest.mark.parametrize(
    ("tolerance", "mode", "edge"),
    [(1.2, "balanced", 0.48), (0.5, "savings", 0.3)],
    ids=["the band", "savings' wider dead zone"],
)
@pytest.mark.parametrize("below", [True, False], ids=["just below", "at the edge"])
def test_full_is_where_the_comfort_layer_starts_pulling_down(freezer, tolerance, mode, edge, below):
    """A charge past the comfort layer's overshoot edge is a fight it wins with a deep cut (concrete
    F1155, savings: 465 minutes below its band, budget 240). Measured from the room's own past-day
    mean instead, each day's charge raised the next day's ceiling."""
    freezer.move_to(DAY + timedelta(hours=2, minutes=5))
    room = TARGET + edge - (0.05 if below else 0.0)

    def decide(battery):
        config = {"target_indoor_temp": TARGET, "tolerance": tolerance, "latitude": 55.6}
        config["optimization_mode"] = mode
        if battery:
            config[CONF_ENABLE_BATTERY_MODE] = True
        engine = DecisionEngine(
            price_analyzer=__import__(
                "custom_components.effektguard.optimization.price_layer", fromlist=["PriceAnalyzer"]
            ).PriceAnalyzer(),
            effect_manager=EffectManager(MagicMock()),
            thermal_model=ThermalModel(thermal_mass=2.0, insulation_quality=1.0),
            config=config,
            heat_pump_model=NibeF750Profile(),
        )
        assert engine.comfort_layer.overshoot_edge() == pytest.approx(edge)
        return engine.calculate_decision(
            nibe_state=_house(2, room),
            price_data=_prices(),
            weather_data=None,
            current_peak=0.0,
            current_power=2.0,
        ).offset

    assert decide(True) - decide(False) == pytest.approx(BATTERY_MODE_CHARGE if below else 0.0)


@pytest.mark.parametrize(
    ("band", "dead_zone"), [(0.48, 0.2), (0.2, 0.3)], ids=["band wider", "dead zone wider"]
)
def test_the_comfort_layer_s_edge_is_the_wider_of_band_and_dead_zone(band, dead_zone):
    layer = ComfortLayer(tolerance_range=band)
    layer.mode_config = SimpleNamespace(dead_zone=dead_zone)
    assert layer.overshoot_edge() == pytest.approx(max(band, dead_zone))


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.9, BATTERY_MODE_CHARGE), (1.0, 0.0), (1.5, 0.0)],
    ids=["short of the line", "at it", "past it"],
)
def test_the_charge_stops_short_of_the_pump_s_immersion_start(share, expected):
    dm = AUX_START * BATTERY_MODE_AUX_START_FRACTION * share
    assert _move(2, dm=dm) == pytest.approx(expected)
