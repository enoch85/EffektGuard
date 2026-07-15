"""The Swedish effect tariff bills the mean power of a billing HOUR, not a 15-minute quarter.

Ellevio (whose model this implements) bills the average of the three highest hourly peaks of the
month, one per day, with 22:00-06:00 counted at half. An hourly mean averages the quiet 45 minutes
around a spike, so a 15-minute hot-water cycle recorded as a quarter-hour peak reads at up to three
times its billed value - and the effect layer throttles the pump to defend a peak on no bill.

Invariants: BILLING_PERIOD_MINUTES is 60; the tariff rate and night weight match the published
figures (SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH 81.25, NIGHT_TARIFF_WEIGHT 0.5); a full hour is
billed at its mean, the night discount halves it, and only the top three hours are kept.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.effektguard.const import (
    BILLING_PERIOD_MINUTES,
    NIGHT_TARIFF_WEIGHT,
    POWER_SOURCE_EXTERNAL_METER,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

JANUARY = datetime(2026, 1, 15, tzinfo=timezone.utc)


def _manager() -> EffectManager:
    manager = EffectManager(MagicMock())
    manager._store = MagicMock()
    manager._store.async_save = AsyncMock()
    manager._monthly_peaks = []
    return manager


def test_the_rate_is_the_one_a_real_company_publishes():
    """The tariff rate is Ellevio's published 81,25 kr/kW/month, and the night weight is a half.

    Every SEK figure the owner is shown is denominated in this number, so it must be one somebody
    actually charges.
    """
    assert SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH == 81.25, (
        f"The effect tariff is {SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH} SEK/kW/month. Ellevio "
        f"publishes 81,25 kr per kilowatt per manad. Every SEK figure the owner is shown is "
        f"denominated in this number, so it had better be one somebody actually charges."
    )
    assert (
        NIGHT_TARIFF_WEIGHT == 0.5
    ), "Ellevio: between 22:00 and 06:00 'raknas bara halva effekttoppen' - half the peak counts."


def test_the_billing_period_is_an_hour():
    """The constant said 15 and called itself "Swedish Effektavgift measurement period"."""
    assert BILLING_PERIOD_MINUTES == 60, (
        f"The billing period is {BILLING_PERIOD_MINUTES} minutes. Ellevio: 'the measurement uses "
        f"hourly averages'. Energimarknadsinspektionen: 'elnatsforetagen mater din elanvandning per "
        f"timme'. A quarter-hour mean is not a quantity anyone is billed on."
    )


@pytest.mark.asyncio
async def test_a_hot_water_cycle_is_not_a_billing_peak():
    """THE BUG. One 15-minute cycle inside an otherwise quiet hour, recorded at three times its
    billed value - and the effect layer throttles the pump to defend it.
    """
    manager = _manager()

    # The hour, as the meter sees it: a hot-water cycle, then the house idling.
    await manager.record_period_measurement(
        power_kw=(9.0 + 1.0 + 1.0 + 1.0) / 4,  # the HOUR's mean, which is what the tariff bills
        period=10,
        timestamp=JANUARY.replace(hour=10),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    recorded = manager.get_monthly_peak_summary()["highest"]

    assert recorded == pytest.approx(3.0), (
        f"EffektGuard recorded a billing peak of {recorded:.2f} kW for an hour whose mean power was "
        f"3.00 kW. The 9 kW quarter is a hot-water cycle, and the tariff averages it with the "
        f"quiet 45 minutes around it. At 81.25 SEK/kW the difference is a phantom "
        f"{(recorded - 3.0) * 81.25:.0f} SEK a month - and the effect layer throttles the heat pump "
        f"to protect it."
    )


@pytest.mark.asyncio
async def test_the_night_discount_runs_from_22_to_06():
    """Ellevio: between 22:00 and 06:00 "raknas bara halva effekttoppen". Hours, not quarters."""
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0,
        period=2,
        timestamp=JANUARY.replace(hour=2),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(3.0), (
        "A 6 kW hour at 02:00 is billed as 3 kW - half - and that is the whole reason the "
        "distinction between actual and effective power exists."
    )


@pytest.mark.asyncio
async def test_a_daytime_hour_is_billed_in_full():
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0,
        period=10,
        timestamp=JANUARY.replace(hour=10),
        source=POWER_SOURCE_EXTERNAL_METER,
    )

    assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_only_the_top_three_hours_are_billed_and_one_per_day():
    """Ellevio: "the average of the three highest peaks", one per day, on three different days."""
    manager = _manager()

    for day, kw in ((10, 5.0), (11, 6.0), (12, 5.5), (13, 2.0)):
        await manager.record_period_measurement(
            power_kw=kw,
            period=10,
            timestamp=JANUARY.replace(day=day, hour=10),
            source=POWER_SOURCE_EXTERNAL_METER,
        )

    peaks = sorted((p.effective_power for p in manager._monthly_peaks), reverse=True)

    assert len(peaks) == 3, (
        f"The tariff bills the mean of the THREE highest hours of the month, so only three are "
        f"kept. {len(peaks)} are: {peaks}"
    )
    assert peaks == pytest.approx(
        [6.0, 5.5, 5.0]
    ), "and they must be the three highest - the 2.0 kW hour is not billed at all"
