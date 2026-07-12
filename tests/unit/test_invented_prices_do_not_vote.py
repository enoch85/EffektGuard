"""With no price source, the integration invents 96 identical prices and lets them vote.

Found on the owner's live Home Assistant: the config entry had `gespot_entity = None` (it was
created before GE-Spot was installed). The adapter is honest about that - it raises
`ValueError("No GE-Spot entity configured")`. The coordinator catches it and fabricates:

    except (AttributeError, KeyError, ValueError, TypeError) as err:
        _LOGGER.warning("Price data unavailable, using fallback: %s", err)
        price_data = get_fallback_prices()          # 96 quarters, all price = 1.0

This is the F-013/F-014 pattern exactly - the NIBE adapter used to invent degree minutes, was made
to raise instead, and here the fabrication has simply moved one layer up.

The audit called it "price optimisation is silently inert". It is worse than inert. The invented
prices are a **weighted vote in the control decision**, and they drag the house colder:

    price_data = None          ->  offset +1.00 °C   (engine abstains; thermal layers decide)
    price_data = fabricated    ->  offset +0.27 °C   ("[Spot Price] Q89: NORMAL (night)")

A 73 % cut in the heat commanded, sourced from a number nobody measured. And the reasoning string
shown to the user reports "[Spot Price] ... NORMAL" as though a real price had been analysed, while
`sensor.effektguard_current_electricity_price` publishes the invented 1.0 as if it were the going
rate - all with `enable_price_optimization = True`, so the user believes it is working.

The honest answer to "what is the electricity price?" when there is no price source is **nothing**,
not one. The engine already handles `price_data=None` correctly: the price layer abstains and the
thermal, comfort and safety layers decide on their own. And the user has to be TOLD - through a
Home Assistant repair issue, not a log line nobody reads.
"""

from __future__ import annotations

import inspect
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

CONFIG = {
    "target_indoor_temp": 21.0,
    "tolerance": 0.5,
    "optimization_mode": "balanced",
    "latitude": 59.33,
    "heating_type": "radiator",
    "heat_loss_coefficient": 150.0,
    "thermal_mass": 0.7,
    "insulation_quality": 1.0,
}


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config=CONFIG,
        heat_pump_model=NibeF750Profile(),
    )


@pytest.fixture
def state() -> NibeState:
    """A house mildly in debt, on a cold-ish day. Nothing dramatic."""
    return NibeState(
        outdoor_temp=-5.0,
        indoor_temp=21.0,
        supply_temp=38.0,
        return_temp=33.0,
        degree_minutes=-150.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 12, 0),
        compressor_hz=50,
        power_kw=2.0,
    )


def test_the_coordinator_does_not_invent_prices():
    """The adapter raises honestly. The coordinator must not undo that."""
    source = inspect.getsource(EffektGuardCoordinator)

    assert "get_fallback_prices" not in source, (
        "The coordinator calls get_fallback_prices() when the price source is missing or fails. "
        "That returns 96 quarters all priced 1.0 - a number nobody measured - and the decision "
        "engine then WEIGHS it. The adapter was fixed to raise rather than fabricate (F-013/F-014); "
        "catching that and fabricating one layer up puts the defect straight back."
    )


def test_the_fabrication_is_gone_entirely():
    """No dead code, no second way back in."""
    from custom_components.effektguard.optimization import price_layer

    assert not hasattr(price_layer, "get_fallback_prices"), (
        "get_fallback_prices() still exists. Nothing may invent a price: if it is there, someone "
        "will call it."
    )


def test_a_missing_price_source_is_raised_as_a_repair_issue():
    """A warning in the log is not telling the user. A repair issue is."""
    source = inspect.getsource(EffektGuardCoordinator)

    assert "async_create_issue" in source, (
        "When there is no electricity price source, price optimisation does not run - and the user "
        "has `enable_price_optimization` switched on and believes it does. They are told by a "
        "_LOGGER.warning, which nobody reads. Home Assistant has a repair-issue registry for "
        "exactly this."
    )


def test_abstaining_heats_the_house_more_than_inventing_a_price(engine, state):
    """The reason this matters, in one number.

    The invented prices are not neutral. They classify as NORMAL, the price layer casts a real
    weighted vote, and the aggregate is pulled down - so the fabrication takes heat AWAY from the
    house on the strength of a price that does not exist.
    """
    honest = engine.calculate_decision(
        nibe_state=state,
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )

    # Reproduce what the fallback used to be: 96 identical quarters, for TODAY.
    #
    # The date matters, and getting it wrong HIDES the bug. PriceData.get_period_index(now) looks up
    # the CURRENT quarter, so a day stamped with some other date matches nothing, the price layer
    # abstains, and the fabricated case comes out identical to the honest one - the test passes and
    # proves nothing. get_fallback_prices() built its invented day around dt_util.now(), which is
    # precisely why it had a vote to cast. (I wrote it with a fixed date first, and it silently
    # agreed with me.)
    from homeassistant.util import dt as dt_util

    from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod

    base = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    invented = PriceData(
        today=[
            QuarterPeriod(
                start_time=base.replace(hour=q // 4, minute=(q % 4) * 15),
                price=1.0,
            )
            for q in range(96)
        ],
        tomorrow=[],
        has_tomorrow=False,
    )

    fabricated = engine.calculate_decision(
        nibe_state=state,
        price_data=invented,
        weather_data=None,
        current_peak=0.0,
        current_power=2.0,
    )

    assert honest.offset > fabricated.offset, (
        f"Abstaining commands {honest.offset:+.2f} °C; the invented prices command "
        f"{fabricated.offset:+.2f} °C. The fabrication is not neutral - it votes, and it votes the "
        f"house colder."
    )
    assert "Spot Price" not in honest.reasoning, (
        "With no price data the reasoning must not mention a spot price at all. It said: "
        f"{honest.reasoning!r}"
    )


def test_the_repair_issue_can_be_cleared_after_a_restart():
    """The flag lives on the coordinator. The issue lives in Home Assistant.

    `_price_issue_active` is an instance attribute, and a Home Assistant restart builds a NEW
    coordinator with it set False - while the repair issue, which HA persists in its registry,
    is still sitting there. Guarding the DELETE on that flag means:

        boot 1   no price source   -> issue raised, flag True
        (user configures GE-Spot, restarts HA)
        boot 2   prices fine       -> flag is False again, so the delete returns early
                                      and the issue stays raised. Forever.

    The user is then nagged about a problem they have already fixed, and nothing they do will
    clear it. Caught on a live Home Assistant, not here - which is the point of running it.

    async_delete_issue is a no-op when there is nothing to delete, so the clear path must simply
    not be conditional on in-memory state that does not survive the thing it is tracking.
    """
    source = inspect.getsource(EffektGuardCoordinator._clear_price_source_issue)

    assert "if not self._price_issue_active" not in source, (
        "_clear_price_source_issue() returns early when the in-memory flag is False. That flag is "
        "reset by every restart; the repair issue is not. So an issue raised before a restart can "
        "never be cleared after one, and the user is told to fix something they already fixed."
    )
    assert "async_delete_issue" in source, "the clear path must actually delete the issue"
