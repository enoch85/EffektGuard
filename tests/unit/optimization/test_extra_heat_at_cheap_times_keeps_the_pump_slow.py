"""Heat bought at a cheap time for later keeps the pump slow.

A raise makes heat by pulling degree minutes down, and past WEATHER_COMP_DEFER_DM_CRITICAL the
compressor races, where it makes the least heat per kWh. So a cheap quarter's raise, and the
pre-heat for a dearer period, fade over PRICE_RAISE_FADE_DM above that point and are gone at it.
At a price of zero or below the energy costs nothing, and the raise stays whole. Only where the heat
source does not weaken with the weather: an outdoor-air pump needs what it banks for the cold.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pytest_homeassistant.common import MockConfigEntry

from custom_components.effektguard import _create_coordinator
from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    CONF_HEAT_PUMP_MODEL,
    CONF_OPTIMIZATION_MODE,
    CONF_TARGET_INDOOR_TEMP,
    CONF_TOLERANCE,
    DOMAIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    PRICE_FORECAST_DM_DEBT_OFFSET,
    WEATHER_COMP_DEFER_DM_CRITICAL,
    QuarterClassification,
)
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

MIDNIGHT = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)
CHEAP_FROM, CHEAP_TO = 48, 64  # 12:00-16:00 at 20 ore
PEAK_FROM, PEAK_TO = 72, 80  # 18:00-20:00 at 300 ore
IN_THE_CHEAP_RUN = 52  # 13:00, the peak five hours ahead
BEFORE_THE_CHEAP_RUN = 40  # 10:00, at 100 ore, the cheap run two hours ahead
AFTER_THE_CHEAP_RUN = 64  # 16:00, at 100 ore, the peak two hours ahead
BEFORE_THE_PEAK = 68  # 17:00, at 100 ore, the peak an hour ahead


def _prices(cheap: float = 20.0) -> PriceData:
    prices = [100.0] * 96
    for q in range(CHEAP_FROM, CHEAP_TO):
        prices[q] = cheap
    for q in range(PEAK_FROM, PEAK_TO):
        prices[q] = 300.0
    return PriceData(
        today=[
            QuarterPeriod(MIDNIGHT + timedelta(minutes=15 * q), p) for q, p in enumerate(prices)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )


def _state(now: datetime, degree_minutes: float) -> NibeState:
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=now,
    )


def _vote(freezer, quarter: int, degree_minutes: float, cheap: float = 20.0):
    now = MIDNIGHT + timedelta(minutes=15 * quarter + 5)
    freezer.move_to(now)
    analyzer = PriceAnalyzer()
    prices = _prices(cheap)
    analyzer.update_prices(prices)
    return analyzer.evaluate_layer(
        nibe_state=_state(now, degree_minutes),
        price_data=prices,
        thermal_mass=1.0,
        target_temp=21.0,
        tolerance=1.5,
        mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
        raises_fade=True,
    )


@pytest.mark.parametrize(
    ("degree_minutes", "share"),
    [
        pytest.param(-200.0, 1.0, id="slow"),  # the owner's F750 ran a median 20 Hz down to -300
        pytest.param(-300.0, 0.5, id="half-way"),
        pytest.param(-400.0, 0.0, id="racing"),  # and 76 Hz from -400 to -500
        pytest.param(-700.0, 0.0, id="deeper"),
    ],
)
def test_a_cheap_quarter_s_raise_fades_as_the_pump_speeds_up(freezer, degree_minutes, share):
    slow = _vote(freezer, IN_THE_CHEAP_RUN, -100.0)
    vote = _vote(freezer, IN_THE_CHEAP_RUN, degree_minutes)

    assert slow.classification == QuarterClassification.VERY_CHEAP, slow.reason
    assert "pre-heat" in slow.reason, "the cheap quarter also buys heat for the peak ahead"
    assert slow.offset > 0
    assert vote.offset == pytest.approx(slow.offset * share), vote.reason


def test_a_cut_is_never_faded(freezer):
    """An hour before the peak the pre-PEAK cut outweighs the pre-heat: the vote is a cut."""
    calm = _vote(freezer, BEFORE_THE_PEAK, -100.0)
    vote = _vote(freezer, BEFORE_THE_PEAK, -300.0)

    assert "pre-heat" in calm.reason and calm.offset < 0, calm.reason
    assert vote.offset == pytest.approx(calm.offset), vote.reason


def test_a_normal_quarter_s_pre_heat_for_a_dearer_period_fades_too(freezer):
    """A lighter house buys heat for the peak in a normal quarter; it is heat for later all the same."""
    slow = _vote(freezer, AFTER_THE_CHEAP_RUN, WEATHER_COMP_DEFER_DM_CRITICAL + 300.0)
    vote = _vote(freezer, AFTER_THE_CHEAP_RUN, WEATHER_COMP_DEFER_DM_CRITICAL)

    assert slow.classification == QuarterClassification.NORMAL, slow.reason
    assert "pre-heat" in slow.reason and slow.offset > 0, slow.reason
    assert vote.offset == 0.0, vote.reason


def test_the_note_says_the_pump_was_kept_slow(freezer):
    vote = _vote(freezer, IN_THE_CHEAP_RUN, WEATHER_COMP_DEFER_DM_CRITICAL + 50.0)

    assert "pump kept slow at DM -350" in vote.reason, vote.reason


@pytest.mark.parametrize("price", [pytest.param(0.0, id="zero"), pytest.param(-5.0, id="negative")])
def test_at_a_price_of_zero_or_below_the_raise_stays_whole(freezer, price):
    racing = WEATHER_COMP_DEFER_DM_CRITICAL - 100.0
    slow = _vote(freezer, IN_THE_CHEAP_RUN, WEATHER_COMP_DEFER_DM_CRITICAL + 300.0, cheap=price)
    vote = _vote(freezer, IN_THE_CHEAP_RUN, racing, cheap=price)

    assert slow.offset > 0, slow.reason
    assert vote.offset == pytest.approx(slow.offset), vote.reason


def test_the_debt_gate_s_small_raise_is_not_heat_for_later(freezer):
    """A normal quarter with the cheap run ahead would wait; in debt it votes the small raise."""
    vote = _vote(freezer, BEFORE_THE_CHEAP_RUN, WEATHER_COMP_DEFER_DM_CRITICAL - 100.0)

    assert vote.classification == QuarterClassification.NORMAL, vote.reason
    assert vote.offset == pytest.approx(PRICE_FORECAST_DM_DEBT_OFFSET), vote.reason


@pytest.mark.parametrize(
    ("model", "fades"),
    [
        pytest.param("nibe_f750", True, id="exhaust-air"),
        pytest.param("nibe_f2040", False, id="outdoor-air"),
    ],
)
async def test_only_a_heat_source_that_holds_in_the_cold_fades(hass, freezer, model, fades):
    """The F2040 declares a bivalent temperature: below it, it cannot heat the house alone.

    The engine as Home Assistant builds it. Setup attaches the model after the engine, and a flag
    stored at construction kept every install from fading while the simulator, which builds the
    engine with the model, faded.
    """
    now = MIDNIGHT + timedelta(minutes=15 * IN_THE_CHEAP_RUN + 5)
    freezer.move_to(now)
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_HEAT_PUMP_MODEL: model},
        options={
            CONF_TARGET_INDOOR_TEMP: 21.0,
            CONF_TOLERANCE: 1.5,
            CONF_OPTIMIZATION_MODE: OPTIMIZATION_MODE_BALANCED,
        },
    )
    entry.add_to_hass(hass)
    engine = (await _create_coordinator(hass, entry)).engine
    racing = WEATHER_COMP_DEFER_DM_CRITICAL - 100.0
    decision = engine.calculate_decision(_state(now, racing), _prices(), None, 0.0, 0.0)
    price = next(layer for layer in decision.layers if layer.name == "Spot Price")

    assert (price.offset == 0.0) == fades, price.reason
