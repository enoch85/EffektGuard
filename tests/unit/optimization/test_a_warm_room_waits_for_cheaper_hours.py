"""In an expensive quarter a warm room waits for the cheaper hours close ahead, on a warming day.

Production on 5 and 6 Nov 2026, 07-08 local: EXPENSIVE quarters (284-390 ore), the room 0.2 and
0.6 C over its 21 C target after a cold night, and the layers asked -1 to -2, so the pump raced
(DM -538..-375) while the day warmed from -2.8 to +7.2 C (5 Nov) and from -0.3 to +5.7 C (6 Nov).
The room then reached 22.2 and 22.5 C in the afternoon.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import (
    WeatherData,
    WeatherForecastHour,
)
from custom_components.effektguard.const import (
    DM_TIER_EMERGENCY,
    OVERSHOOT_PROTECTION_DEEPEST,
    WAIT_CHEAPER_ROOM_OVER,
    WAIT_CHEAPER_WARMING,
    WAIT_CHEAPER_WARMING_HOURS,
    QuarterClassification,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.optimization.weather_layer import forecast_rise

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
NOW = MIDNIGHT + timedelta(hours=12, minutes=5)  # inside the dearest two hours
TARGET = 21.0
WARM = TARGET + 0.2
SHALLOW = -100.0  # a pump at ease: the wait does not ask for one repaying degree minutes
OUTDOOR = -2.0


def _engine(target: float = TARGET) -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": target,
            "tolerance": 1.5,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
        },
        heat_pump_model=NibeF750Profile(),
    )


def _prices(cheaper_after: bool = True) -> PriceData:
    """Noon's two hours at 400 ore; after them the ordinary 50-68 ore, or 400 to the end of the day."""
    prices = [50.0 + (q % 7) * 3.0 for q in range(96)]
    for q in range(46, 54 if cheaper_after else 96):
        prices[q] = 400.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _weather(temps: list[float], at: datetime = NOW) -> WeatherData:
    """A forecast from the hour at hand: temps[n] for n hours ahead."""
    hour = at.replace(minute=0, second=0, microsecond=0)
    return WeatherData(
        current_temp=OUTDOOR,
        forecast_hours=[
            WeatherForecastHour(datetime=hour + timedelta(hours=n), temperature=t)
            for n, t in enumerate(temps)
        ],
    )


def _warming(rise: float, by_hour: int = WAIT_CHEAPER_WARMING_HOURS) -> list[float]:
    """From OUTDOOR, `rise` warmer by `by_hour` hours ahead, then level for the rest of the day."""
    return [OUTDOOR + rise * min(n, by_hour) / by_hour for n in range(24)]


WARMING_DAY = _warming(3.9)  # 5 Nov 2026 at 06:45, the forecast as issued (Open-Meteo)
LEVEL_DAY = [OUTDOOR] * 24


def _state(indoor: float, at: datetime = NOW) -> NibeState:
    return NibeState(
        outdoor_temp=OUTDOOR,
        indoor_temp=indoor,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=SHALLOW,
        current_offset=-1.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=at,
        compressor_hz=58,
        power_kw=2.5,
    )


def _decide(engine, indoor, forecast=WARMING_DAY, prices=None, at: datetime = NOW):
    weather = None if forecast is None else _weather(forecast, at)
    return engine.calculate_decision(_state(indoor, at), prices or _prices(), weather, 0.0, 0.0)


@pytest.fixture(autouse=True)
def _noon(freezer):
    freezer.move_to(NOW)


def test_a_warm_room_waits_in_an_expensive_quarter_on_a_warming_day():
    decision = _decide(_engine(), WARM)

    assert decision.offset == OVERSHOOT_PROTECTION_DEEPEST, decision.reasoning
    assert "Warm room waits for cheaper hours lowered" in decision.reasoning
    assert "price -" in decision.reasoning, "the note names the cheaper hours ahead"


@pytest.mark.parametrize(
    ("indoor", "forecast", "cheaper_after"),
    [
        pytest.param(TARGET + WAIT_CHEAPER_ROOM_OVER / 2, WARMING_DAY, True, id="room-not-warm"),
        pytest.param(WARM, _warming(WAIT_CHEAPER_WARMING - 0.5), True, id="day-warms-too-little"),
        pytest.param(WARM, LEVEL_DAY, True, id="day-does-not-warm"),
        pytest.param(WARM, None, True, id="no-forecast"),
        pytest.param(WARM, WARMING_DAY, False, id="no-cheaper-hours-ahead"),
    ],
)
def test_without_each_condition_the_room_does_not_wait(indoor, forecast, cheaper_after):
    decision = _decide(_engine(), indoor, forecast, _prices(cheaper_after))

    assert decision.offset > OVERSHOOT_PROTECTION_DEEPEST, decision.reasoning
    assert "Warm room waits" not in decision.reasoning


def test_a_room_read_in_tenths_is_warm_at_the_first_tenth_over():
    """21.4 - 21.3 is 0.0999... in floating point: the reading is a tenth over all the same."""
    decision = _decide(_engine(target=21.3), 21.4)

    assert decision.offset == OVERSHOOT_PROTECTION_DEEPEST, decision.reasoning


def test_warmth_at_the_window_s_last_hour_counts():
    late = LEVEL_DAY[:WAIT_CHEAPER_WARMING_HOURS] + [OUTDOOR + WAIT_CHEAPER_WARMING] * 18

    assert forecast_rise(_weather(late), WAIT_CHEAPER_WARMING_HOURS) == WAIT_CHEAPER_WARMING
    assert _decide(_engine(), WARM, late).offset == OVERSHOOT_PROTECTION_DEEPEST


def test_warmth_after_the_window_does_not_count():
    later = LEVEL_DAY[: WAIT_CHEAPER_WARMING_HOURS + 1] + [OUTDOOR + 2 * WAIT_CHEAPER_WARMING] * 17

    assert forecast_rise(_weather(later), WAIT_CHEAPER_WARMING_HOURS) == 0.0
    assert "Warm room waits" not in _decide(_engine(), WARM, later).reasoning


def test_the_forecast_is_read_against_itself():
    """A forecast running 7 C warmer than the pump's outdoor sensor is not a warming day."""
    warm_forecast = [OUTDOOR + 7.0] * 24

    assert forecast_rise(_weather(warm_forecast), WAIT_CHEAPER_WARMING_HOURS) == 0.0
    assert "Warm room waits" not in _decide(_engine(), WARM, warm_forecast).reasoning


def test_an_ordinary_quarter_does_not_wait():
    """08:05 at the day's ordinary price, with a cheaper hour from 09:00: only the class stops it."""
    prices = [100.0] * 96
    for q in range(36, 40):
        prices[q] = 50.0
    for q in range(46, 54):
        prices[q] = 400.0
    day = PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )
    morning = MIDNIGHT + timedelta(hours=8, minutes=5)
    decision = _decide(_engine(), WARM, WARMING_DAY, day, at=morning)

    assert "Warm room waits" not in decision.reasoning


def test_once_waiting_the_room_keeps_waiting_as_the_warming_passes():
    """Later in the day the forecast no longer rises; the expensive run is what the room waits out."""
    engine = _engine()
    assert _decide(engine, WARM).offset == OVERSHOOT_PROTECTION_DEEPEST

    later = NOW + timedelta(minutes=5)
    assert _decide(engine, WARM, LEVEL_DAY, at=later).offset == OVERSHOOT_PROTECTION_DEEPEST


def test_the_wait_ends_with_the_warm_room_and_does_not_restart_on_a_level_day():
    engine = _engine()
    _decide(engine, WARM)
    cooled = _decide(engine, TARGET, LEVEL_DAY, at=NOW + timedelta(minutes=5))
    warm_again = _decide(engine, WARM, LEVEL_DAY, at=NOW + timedelta(minutes=10))

    assert "Warm room waits" not in cooled.reasoning
    assert "Warm room waits" not in warm_again.reasoning, "the wait starts only on a warming day"


def test_the_thermal_debt_ladder_is_never_cut():
    engine = _engine()
    engine.price.update_prices(_prices())
    emergency = SimpleNamespace(tier=DM_TIER_EMERGENCY)

    offset, note = engine._wait_for_cheaper_hours(
        0.0, _state(WARM), _prices(), _weather(WARMING_DAY), QuarterClassification.PEAK, emergency
    )

    assert (offset, note) == (0.0, "")
