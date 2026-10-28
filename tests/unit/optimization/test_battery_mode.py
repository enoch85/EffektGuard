"""Battery mode (experimental): charge in the day's cheapest hours, coast through its dearest.

Drives the real battery_adjustment and the real DecisionEngine. The day used: hours 0-5 at 10, 6-9 at
100, 10-17 at 50, 18-19 at 90, 20-23 at 40 (one price unit, any unit: the rules are relative).
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
from custom_components.effektguard.const import (
    BATTERY_MODE_CHARGE,
    BATTERY_MODE_COAST,
    CONF_ENABLE_BATTERY_MODE,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
)
from custom_components.effektguard.optimization.battery import battery_adjustment
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

TZ = ZoneInfo("Europe/Stockholm")
DAY = datetime(2026, 1, 14, tzinfo=TZ)
HOURLY = [10.0] * 6 + [100.0] * 4 + [50.0] * 8 + [90.0] * 2 + [40.0] * 4
TARGET = 21.0
BAND = 0.48  # tolerance 1.2 x 0.4
ALLOWED = MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED].preheat_overshoot_allowed


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


def _move(
    hour, room=TARGET + 0.2, prices=None, tier="OK", zone="NONE", saturated=False, valid=True
):
    return battery_adjustment(
        now=DAY + timedelta(hours=hour, minutes=5),
        price_data=_prices() if prices is None else prices,
        nibe_state=SimpleNamespace(indoor_temp=room, indoor_temp_valid=valid),
        target_temp=TARGET,
        tolerance_range=BAND,
        overshoot_allowed=ALLOWED,
        emergency_tier=tier,
        proactive_zone=zone,
        compressor_saturated=saturated,
    )[0]


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (2, BATTERY_MODE_CHARGE),
        (7, -BATTERY_MODE_COAST),
        (19, -BATTERY_MODE_COAST),
        (12, 0.0),
        (21, 0.0),
    ],
    ids=["cheap night", "dear morning", "dear evening", "middling day", "middling evening"],
)
def test_the_day_is_charged_cheap_and_coasted_dear(hour, expected):
    assert _move(hour) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("hour", "kwargs"),
    [
        (2, {"room": TARGET + ALLOWED}),  # full: at the allowed pre-heat overshoot
        (7, {"room": TARGET - BAND}),  # empty: at the band's lower edge
        (2, {"valid": False}),
        (7, {"valid": False}),
        (2, {"saturated": True}),  # no frequency left: a charge only buys immersion heat
        (7, {"saturated": True}),  # nothing to refill a coast with later
        (2, {"zone": "Z1"}),  # degree minutes sliding toward debt
        (7, {"zone": "Z3"}),
        (2, {"tier": "T1"}),  # repaying debt
        (7, {"tier": "T2"}),
        (2, {"tier": "EMERGENCY"}),  # past the aux limit (F-124)
        (7, {"tier": "ANTI_WINDUP"}),
    ],
    ids=[
        "charged full",
        "coasted empty",
        "no room reading, cheap",
        "no room reading, dear",
        "saturated, cheap",
        "saturated, dear",
        "Z1",
        "Z3",
        "T1",
        "T2",
        "EMERGENCY",
        "anti-windup",
    ],
)
def test_a_guard_stops_the_battery(hour, kwargs):
    assert _move(hour, **kwargs) == 0.0


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
    out = battery_adjustment(
        now=DAY + timedelta(hours=2),
        price_data=prices,
        nibe_state=SimpleNamespace(indoor_temp=TARGET, indoor_temp_valid=True),
        target_temp=TARGET,
        tolerance_range=BAND,
        overshoot_allowed=ALLOWED,
        emergency_tier="OK",
        proactive_zone="NONE",
        compressor_saturated=False,
    )
    assert out == (0.0, "")


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
    move = battery_adjustment(
        now=day + timedelta(hours=1, minutes=5),
        price_data=prices,
        nibe_state=SimpleNamespace(indoor_temp=TARGET, indoor_temp_valid=True),
        target_temp=TARGET,
        tolerance_range=BAND,
        overshoot_allowed=ALLOWED,
        emergency_tier="OK",
        proactive_zone="NONE",
        compressor_saturated=False,
    )[0]
    assert move == pytest.approx(BATTERY_MODE_CHARGE)


def _engine(battery: bool | None) -> DecisionEngine:
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
    )


def _house(hour: int) -> NibeState:
    return NibeState(
        outdoor_temp=2.0,
        indoor_temp=TARGET + 0.2,
        supply_temp=30.0,
        return_temp=26.0,
        degree_minutes=50.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=DAY + timedelta(hours=hour),
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
