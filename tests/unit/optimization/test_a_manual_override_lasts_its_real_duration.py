"""A manual override lasts the minutes the user asked for - on the nights the clocks change too.

The deadline was `dt_util.now() + timedelta(minutes=duration)`, wall-clock arithmetic on a local
datetime. A 60-minute override set at 02:30 CEST on the night the clocks go back expired at 03:30
CET, 120 real minutes later; one set at 01:30 CET on the night they go forward expired at a 02:30
that never happens, which 03:00 CEST is already past - 30 real minutes. The deadline is now kept
in UTC, where a minute is a minute.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import MAX_OFFSET
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

STOCKHOLM = ZoneInfo("Europe/Stockholm")
UTC = timezone.utc
STOCKHOLM_LATITUDE = 59.33
OVERRIDE_MINUTES = 60


def _clock_at(monkeypatch, instant_utc: datetime) -> None:
    """Both of dt_util's clocks from one instant: `now()` local with `fold`, as HA hands it out."""
    monkeypatch.setattr(dt_util, "now", lambda tz=None: instant_utc.astimezone(tz or STOCKHOLM))
    monkeypatch.setattr(dt_util, "utcnow", lambda: instant_utc)


def _engine() -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
        config={"target_indoor_temp": 21.0, "tolerance": 0.5, "latitude": STOCKHOLM_LATITUDE},
    )


def _override_holds(engine: DecisionEngine, instant_utc: datetime) -> bool:
    """Whether a cycle at `instant_utc` still applies the user's offset."""
    state = NibeState(
        outdoor_temp=5.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=instant_utc,
    )
    decision = engine.calculate_decision(
        nibe_state=state, price_data=None, weather_data=None, current_peak=0.0, current_power=0.0
    )
    return decision.is_manual_override


@pytest.mark.parametrize(
    "night, set_at_utc, checked_at_utc, still_held",
    [
        (
            "the clocks go back: 02:30 CEST plus 61 real minutes is 02:31 CET",
            datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
            datetime(2026, 10, 25, 1, 31, tzinfo=UTC),
            False,
        ),
        (
            "the clocks go forward: 01:30 CET plus 30 real minutes is 03:00 CEST",
            datetime(2027, 3, 28, 0, 30, tzinfo=UTC),
            datetime(2027, 3, 28, 1, 0, tzinfo=UTC),
            True,
        ),
    ],
)
def test_a_sixty_minute_override_holds_for_sixty_real_minutes(
    monkeypatch, night, set_at_utc, checked_at_utc, still_held
):
    engine = _engine()
    _clock_at(monkeypatch, set_at_utc)
    engine.set_manual_override(MAX_OFFSET, OVERRIDE_MINUTES)
    assert _override_holds(engine, set_at_utc), "precondition: the override applies when set"

    _clock_at(monkeypatch, checked_at_utc)

    assert _override_holds(engine, checked_at_utc) is still_held, (
        f"{night}. A {OVERRIDE_MINUTES}-minute override set at "
        f"{set_at_utc.astimezone(STOCKHOLM):%H:%M %Z} was "
        f"{'already over' if still_held else 'still applying'} at "
        f"{checked_at_utc.astimezone(STOCKHOLM):%H:%M %Z}, "
        f"{(checked_at_utc - set_at_utc).total_seconds() / 60:.0f} real minutes later - the "
        f"deadline was wall-clock arithmetic on the local time."
    )
