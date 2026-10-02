"""The effect tariff bills the OWNER'S 15-minute period mean - stated as configuration, not fact.

HISTORY, because this file has asserted the opposite twice and both versions cited sources. The
integration originally measured 15-minute peaks; the audit re-based it on the HOUR, citing Ellevio
("the measurement uses hourly averages") and Energimarknadsinspektionen ("per timme") - and those
citations are real, but they describe operators the owner is not billed by. Operator models vary
across thousands of DSOs, which is finding F-107 and precisely why the government ordered the
effect-charge framework repealed and rebuilt. THE OWNER'S tariff measures 15-minute intervals, so
that is what this integration bills: an owner-model configuration, not a claim about Sweden.

Invariants: BILLING_PERIOD_MINUTES is 15; the simulator's illustrative rate stays Ellevio's
published 81.25 kr/kW/month; the night window (22:00-06:00) counts half; a full period is billed at
its time-weighted mean; only the top three periods are kept, at most one per day.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
    BILLING_PERIODS_PER_DAY,
    NIGHT_TARIFF_WEIGHT,
    POWER_SOURCE_EXTERNAL_METER,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

JANUARY = datetime(2026, 1, 15, tzinfo=timezone.utc)
PERIOD_10_00 = 10 * 4  # the quarter starting 10:00 - daytime
PERIOD_02_00 = 2 * 4  # the quarter starting 02:00 - inside the night window


def _manager() -> EffectManager:
    manager = EffectManager(MagicMock())
    manager._store = MagicMock()
    manager._store.async_save = AsyncMock()
    manager._monthly_peaks = []
    return manager


def test_the_rate_is_a_real_published_figure():
    """81.25 kr/kW/month is Ellevio's published rate, kept as the simulator's example.

    It is ILLUSTRATIVE - effect charges are set per grid company - but every SEK figure shown is
    denominated in it, so it must at least be a number somebody has actually charged.
    """
    assert SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH == 81.25
    assert (
        NIGHT_TARIFF_WEIGHT == 0.5
    ), "between 22:00 and 06:00 half the peak counts - the owner's configured night discount"


def test_the_billing_period_is_the_owners_quarter():
    """15 minutes is the owner's tariff cadence. Changing it changes what every peak means."""
    assert BILLING_PERIOD_MINUTES == 15, (
        f"The billing period is {BILLING_PERIOD_MINUTES} minutes. The owner's grid company "
        f"measures 15-minute intervals (operator models vary - F-107). This is configuration; "
        f"change it deliberately or not at all."
    )
    assert BILLING_PERIODS_PER_DAY == 24 * 60 // BILLING_PERIOD_MINUTES


@pytest.mark.asyncio
async def test_a_period_is_billed_at_its_own_mean():
    """Under a 15-minute tariff a sustained hot-water cycle genuinely IS the billed peak.

    There is no quiet 45 minutes to average it away - that was the HOUR model. What the accumulator
    guarantees instead is that the recorded number is the period's time-weighted MEAN, not an
    instantaneous spike (see test_one_definition_of_the_billed_quantity.py).
    """
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=9.0,  # the quarter's mean while the hot water ran
        period=PERIOD_10_00,
        timestamp=JANUARY.replace(hour=10),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_the_night_discount_runs_from_22_to_06():
    """A 6 kW period at 02:00 is billed as 3 kW - half."""
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0,
        period=PERIOD_02_00,
        timestamp=JANUARY.replace(hour=2),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_a_daytime_period_is_billed_in_full():
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0,
        period=PERIOD_10_00,
        timestamp=JANUARY.replace(hour=10),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_only_the_top_three_periods_are_billed_and_one_per_day():
    """The monthly charge is the mean of the three highest periods, at most one per day."""
    manager = _manager()

    for day, kw in ((10, 5.0), (11, 6.0), (12, 5.5), (13, 2.0)):
        await manager.record_period_measurement(
            power_kw=kw,
            period=PERIOD_10_00,
            timestamp=JANUARY.replace(day=day, hour=10),
            source=POWER_SOURCE_EXTERNAL_METER,
        )

    peaks = sorted((p.effective_power for p in manager._monthly_peaks), reverse=True)

    assert len(peaks) == 3, (
        f"The tariff bills the mean of the THREE highest periods of the month, so only three are "
        f"kept. {len(peaks)} are: {peaks}"
    )
    assert peaks == pytest.approx(
        [6.0, 5.5, 5.0]
    ), "and they must be the three highest - the 2.0 kW period is not billed at all"
