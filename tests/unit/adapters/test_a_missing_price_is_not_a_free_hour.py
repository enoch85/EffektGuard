"""A price entry with no price becomes the cheapest quarter of the day.

    price = float(item.get("value", 0.0))          # gespot_adapter

A GE-Spot entry that arrives without its `value` key silently becomes **0.0 öre**. Zero is the
cheapest possible price, so that quarter is ranked the best of the day and classified VERY_CHEAP -
and `PRICE_OFFSET_VERY_CHEAP` is **+4.0 °C**, commented in const.py as *"exceptional prices,
aggressive pre-heating!"*.

**So a quarter with no data commands the most aggressive pre-heating the price layer can ask for.**

Two lines above, the same function treats the TIMESTAMP with exactly the care the price is denied:

    start_time = dt_util.parse_datetime(time_str)
    if start_time is None:
        # parse_datetime signals invalid input with None, not an exception; letting it
        # through would break the whole day at sort time instead of one interval here
        _LOGGER.warning("Skipping price period with invalid time: %s", time_str)
        continue

An unparseable time is skipped, loudly, with a comment explaining why. An absent price is invented.

And there is already an exception handler that would do the right thing:

    except (ValueError, TypeError, KeyError) as err:
        _LOGGER.warning("Failed to parse price period: %s", err)
        continue

`item["value"]` would raise KeyError, be caught there, and the bad interval would be dropped. But
`.get("value", 0.0)` supplies a default, so the KeyError never fires. **The handler that would have
saved us can never run, because the default swallows the error before it reaches it.**

The last twist is what makes this unrecoverable. **Zero is a real Nordic price** - exactly-zero
quarters occur roughly a hundred hours a year per SE zone (audit F-040). So after the fact there is
nothing to distinguish "electricity was free" from "we were never told". The fabrication is
indistinguishable from the truth.

Missing data has one honest representation, and it is not a number. Drop the interval: the period
lookup is by timestamp, so a gap simply means that quarter has no price, and the price layer
abstains for it - which is the same thing that happens when there is no price source at all (F-123).
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.gespot_adapter import GESpotAdapter
from custom_components.effektguard.optimization.price_layer import (
    PriceAnalyzer,
    QuarterClassification,
)


def _adapter() -> GESpotAdapter:
    return GESpotAdapter(MagicMock(), {"gespot_entity": "sensor.gespot"})


def _raw_day(broken_at: int | None = None) -> list[dict[str, object]]:
    """A realistic SE4 day. One entry may arrive without its `value` key."""
    base = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    day: list[dict[str, object]] = []

    for i in range(96):
        item: dict[str, object] = {"time": (base + timedelta(minutes=15 * i)).isoformat()}
        if i != broken_at:
            item["value"] = 40.0 + 50.0 * (i % 24) / 24.0
        day.append(item)

    return day


def test_a_good_day_parses():
    """The precondition. If this fails, the parser is rejecting everything."""
    periods = _adapter()._parse_periods(_raw_day())

    assert len(periods) == 96
    assert min(p.price for p in periods) >= 40.0


def test_an_entry_with_no_price_is_dropped_not_invented():
    """The interval has no price. That is not the same as a price of zero."""
    periods = _adapter()._parse_periods(_raw_day(broken_at=50))

    assert len(periods) == 95, (
        "A GE-Spot entry with no `value` key was still turned into a price period. "
        f"`.get('value', 0.0)` invented 0.0 for it - and 0.0 is the cheapest possible price."
    )
    assert all(p.price != 0.0 for p in periods), (
        "A fabricated 0.0 öre survived into the parsed day. Zero is a REAL Nordic price (~100 h a "
        "year per SE zone), so nothing downstream can ever tell it apart from a genuinely free "
        "quarter."
    )


def test_a_quarter_with_no_data_is_not_the_best_quarter_of_the_day():
    """The consequence, end to end: no data ranks as the cheapest hour there is."""
    periods = _adapter()._parse_periods(_raw_day(broken_at=50))

    classes = PriceAnalyzer().classify_quarterly_periods(periods)

    assert QuarterClassification.VERY_CHEAP not in set(classes.values()) or all(
        periods[i].price > 0.0 for i, c in classes.items() if c is QuarterClassification.VERY_CHEAP
    ), (
        "A quarter the adapter had no price for was classified VERY_CHEAP - the best quarter of the "
        "day - because it was invented as 0.0. PRICE_OFFSET_VERY_CHEAP is +4.0 °C, 'aggressive "
        "pre-heating'. The heat pump would be driven hardest in the interval nobody sent us a price "
        "for."
    )


def test_a_day_where_every_price_is_missing_yields_no_day_at_all():
    """The schema-change case: GE-Spot renames the key and every entry breaks.

    All 96 intervals drop, `today` comes back empty, and the coordinator's no-price-source path
    takes over: the price layer abstains entirely and a repair issue tells the user (F-123). That is
    the correct outcome. The wrong one is 96 quarters of invented 0.0, every one of them VERY_CHEAP,
    with the pump pre-heating aggressively around the clock.
    """
    broken = [{"time": item["time"]} for item in _raw_day()]

    periods = _adapter()._parse_periods(broken)

    assert periods == [], (
        f"Every entry was missing its price and {len(periods)} periods came back anyway. If they "
        f"are all invented zeros, every quarter of the day classifies VERY_CHEAP and the pump "
        f"pre-heats aggressively, around the clock, on a day nobody sent us a single price for."
    )
