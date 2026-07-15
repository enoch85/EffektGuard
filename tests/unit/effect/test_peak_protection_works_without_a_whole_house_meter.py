"""The whole-house meter is optional; peak protection is not.

`should_limit_power` returns "OK, no peaks recorded yet" on an empty history, and the history is
filled only by the peak recorder. Gating that recorder on BILLABILITY - as a first billing fix did -
leaves a house with no whole-house meter recording nothing, so peak protection never fires.

BILLABLE and USABLE-AS-A-CONTROL-THRESHOLD are different questions. NIBE phase currents are a valid
control threshold (the pump is the dominant controllable load, compared against its own recorded
history) but are not whole-house grid import - so the PeakEvent carries provenance and is never
reported as a bill. Estimates drive neither.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.const import (
    BILLABLE_POWER_SOURCES,
    PEAK_CONTROL_POWER_SOURCES,
    POWER_SOURCE_ESTIMATE,
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NIBE_CURRENTS,
    POWER_SOURCE_NONE,
)
from custom_components.effektguard.optimization.effect_layer import EffectManager

JANUARY = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
MIDDAY_HOUR = 10  # inside DAYTIME, so no night weighting confuses the arithmetic


def _manager() -> EffectManager:
    manager = EffectManager(MagicMock())
    manager._monthly_peaks = []
    return manager


def test_a_guess_is_not_a_control_threshold():
    """Estimates drive nothing. This is the line the billing fix was right about."""
    assert POWER_SOURCE_ESTIMATE not in PEAK_CONTROL_POWER_SOURCES
    assert POWER_SOURCE_ESTIMATE not in BILLABLE_POWER_SOURCES
    assert POWER_SOURCE_NONE not in PEAK_CONTROL_POWER_SOURCES


def test_phase_currents_control_but_do_not_bill():
    """The distinction the whole fix turns on, stated once."""
    assert POWER_SOURCE_NIBE_CURRENTS in PEAK_CONTROL_POWER_SOURCES, (
        "NIBE phase currents were excluded from peak RECORDING because they are not billable. But "
        "an empty peak history makes should_limit_power return OK forever, so every user without a "
        "whole-house meter - and the meter is optional - lost peak protection entirely."
    )
    assert POWER_SOURCE_NIBE_CURRENTS not in BILLABLE_POWER_SOURCES
    assert BILLABLE_POWER_SOURCES < PEAK_CONTROL_POWER_SOURCES, (
        "Everything billable must also be usable for control. If these sets ever cross, a reading "
        "could bill the owner without being allowed to protect them from the bill."
    )


@pytest.mark.asyncio
async def test_peak_protection_actually_fires_for_a_house_with_no_meter():
    """The regression, end to end: record from phase currents, then demand a limit."""
    manager = _manager()

    # A cold January stretch. One counted hour per day - the tariff's own rule - fills the top 3.
    for day_offset, kw in enumerate((6.0, 5.5, 5.0)):
        await manager.record_period_measurement(
            power_kw=kw,
            period=MIDDAY_HOUR,
            timestamp=JANUARY + timedelta(days=day_offset),
            source=POWER_SOURCE_NIBE_CURRENTS,
        )

    assert len(manager._monthly_peaks) == 3, (
        "Nothing was recorded. A house whose only power measurement is the pump's own phase "
        "currents has no monthly peak history at all, and should_limit_power short-circuits to "
        "'OK - no peaks recorded yet' on an empty history."
    )

    # Now the pump goes past the lowest of the top three. Protection must engage.
    decision = manager.should_limit_power(current_power=7.0, current_period=MIDDAY_HOUR)

    assert decision.should_limit, (
        f"The house is drawing 7.0 kW against a recorded monthly peak of 5.0 kW and peak "
        f"protection said {decision.severity!r}: {decision.reason!r}. This is the integration's "
        f"headline feature, and for every user without a whole-house meter it never fired."
    )
    assert decision.severity == "CRITICAL"
    assert decision.recommended_offset < 0.0, "protection must REDUCE heat, not add it"


@pytest.mark.asyncio
async def test_the_resulting_peak_is_flagged_as_not_a_bill():
    """It controls the pump. It must never be shown to the owner as money."""
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0,
        period=MIDDAY_HOUR,
        timestamp=JANUARY,
        source=POWER_SOURCE_NIBE_CURRENTS,
    )
    summary = manager.get_monthly_peak_summary()

    assert summary["highest"] == pytest.approx(6.0)
    assert summary["billable"] is False, (
        "A monthly peak built from the pump's own phase currents was reported as billable. BE1/BE2/"
        "BE3 measure the heat pump - not the oven, not the EV charger - and the Swedish effect "
        "tariff bills whole-house grid import."
    )
    assert summary["peaks"][0]["source"] == POWER_SOURCE_NIBE_CURRENTS


@pytest.mark.asyncio
async def test_one_unmetered_quarter_taints_the_whole_billing_figure():
    """The tariff charges the top THREE quarters together, so the set is billable or it is not."""
    manager = _manager()

    await manager.record_period_measurement(
        power_kw=6.0, period=MIDDAY_HOUR, timestamp=JANUARY, source=POWER_SOURCE_EXTERNAL_METER
    )
    await manager.record_period_measurement(
        power_kw=5.0,
        period=MIDDAY_HOUR,
        timestamp=JANUARY + timedelta(days=1),
        source=POWER_SOURCE_NIBE_CURRENTS,
    )

    summary = manager.get_monthly_peak_summary()

    assert summary["count"] == 2
    assert summary["billable"] is False, (
        "Two of the month's top quarters, one measured at the meter and one at the pump, were "
        "reported together as a billing figure. The tariff is charged on the three together; one "
        "pump-only quarter in the set means the total is not what the grid delivered."
    )


@pytest.mark.asyncio
async def test_a_metered_house_is_unaffected():
    """The regression guard on the guard: none of this may change a properly metered install."""
    manager = _manager()

    for kw in (6.0, 5.5, 5.0):
        await manager.record_period_measurement(
            power_kw=kw,
            period=MIDDAY_HOUR,
            timestamp=JANUARY,
            source=POWER_SOURCE_EXTERNAL_METER,
        )

    summary = manager.get_monthly_peak_summary()
    assert summary["billable"] is True
    assert summary["highest"] == pytest.approx(6.0)
    assert manager.should_limit_power(7.0, MIDDAY_HOUR).should_limit
