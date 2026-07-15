"""`mW` and `MW` differ only in case, and one is 10^9 times the other, so the unit must NOT be folded.

HA ships both `UnitOfPower.MILLIWATT` ("mW") and `UnitOfPower.MEGA_WATT` ("MW"); case-folding
collapses them, and the table mapped that key to MEGAWATTS - so a 5000 mW (5 W) sensor read as
5 000 000 kW, persisted as the month's tariff peak. That does not throttle the house: every real
quarter then looks safe against the astronomical threshold, so peak protection is silently disabled
until the month rolls over. `power_kw_from_state` keys the table case-SENSITIVELY, and a second line
of defence (TestTheSecondLineOfDefence) refuses any peak above what a domestic supply can deliver.

The ambiguity is derived from `UnitOfPower` itself, so a future case-colliding pair fails here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.const import UnitOfPower

from custom_components.effektguard.const import POWER_SOURCE_EXTERNAL_METER
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.utils.power import (
    POWER_UNIT_FACTORS_KW,
    power_kw_from_state,
)


def _sensor(value: str, unit: str | None) -> MagicMock:
    state = MagicMock()
    state.entity_id = "sensor.house_power"
    state.state = value
    state.attributes = {"unit_of_measurement": unit} if unit is not None else {}
    return state


def test_home_assistant_really_does_ship_two_units_that_differ_only_in_case():
    """The precondition. If this ever stops being true, the guard below is guarding nothing."""
    folded = [unit.value.lower() for unit in UnitOfPower]
    collisions = {f for f in folded if folded.count(f) > 1}

    assert collisions == {"mw"}, (
        f"Home Assistant's UnitOfPower now case-collides on {collisions or 'nothing'}, not just "
        f"{{'mw'}}. Every colliding pair is a silent unit-conversion bug in any code that folds "
        f"case before looking a unit up. Check power.py handles each one."
    )
    assert UnitOfPower.MILLIWATT.value == "mW"
    assert UnitOfPower.MEGA_WATT.value == "MW"


def test_a_milliwatt_sensor_is_not_read_as_megawatts():
    """The bug: 5000 mW read as 5 000 000 kW. A factor of 10^9, straight into the billing peak."""
    reading = power_kw_from_state(_sensor("5000", UnitOfPower.MILLIWATT))

    assert reading == pytest.approx(0.005), (
        f"5000 mW is 5 watts, i.e. 0.005 kW. It was read as {reading} kW. Case-folding the unit "
        f"collapses 'mW' onto 'MW' and applies the MEGAWATT factor - a factor of 10^9 - and the "
        f"result is classified billable and persisted as the month's tariff peak."
    )


def test_a_megawatt_sensor_is_still_read_as_megawatts():
    """The other half of the pair must not be broken by fixing the first."""
    assert power_kw_from_state(_sensor("2", UnitOfPower.MEGA_WATT)) == pytest.approx(2000.0)


@pytest.mark.parametrize(
    ("value", "unit", "expected_kw"),
    [
        ("1500", UnitOfPower.WATT, 1.5),
        ("1.5", UnitOfPower.KILO_WATT, 1.5),
        ("1500000", UnitOfPower.MILLIWATT, 1.5),
        ("0.0015", UnitOfPower.MEGA_WATT, 1.5),
    ],
)
def test_every_power_unit_converts_to_the_same_kilowatts(value, unit, expected_kw):
    """The same 1.5 kW, spelled four ways. All four must agree."""
    assert power_kw_from_state(_sensor(value, unit)) == pytest.approx(expected_kw)


class TestTheSecondLineOfDefence:
    """A number persisted for a month gets a plausibility CEILING, the symmetric partner of the
    PEAK_RECORDING_MINIMUM floor. Peaks above what a domestic supply can deliver are refused, so a
    mis-scaled unit is contained even before the table is fixed.
    """

    @pytest.mark.asyncio
    async def test_an_impossible_reading_never_becomes_a_tariff_peak(self):
        """5 000 000 kW is not a peak. It is a broken sensor, and it costs a month of protection."""
        manager = EffectManager(MagicMock())
        manager._store = MagicMock()
        manager._store.async_save = AsyncMock()
        manager._monthly_peaks = []

        what_the_old_code_produced = 5_000_000.0  # 5000 mW, read as megawatts

        event = await manager.record_period_measurement(
            power_kw=what_the_old_code_produced,
            period=10,
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
            source=POWER_SOURCE_EXTERNAL_METER,
        )

        assert event is None and not manager._monthly_peaks, (
            f"{what_the_old_code_produced:,.0f} kW was recorded as this month's tariff peak. No "
            f"domestic main fuse can pass it. Once it is in the record, every real quarter looks "
            f"safe against it - the effect layer reports 'Safe margin: 4999994 kW below peak' on a "
            f"6 kW January cold snap - so peak protection goes quiet until the month rolls over "
            f"and the owner blows the real peak the feature exists to prevent."
        )

    @pytest.mark.asyncio
    async def test_peak_protection_still_works_after_the_refusal(self):
        """The point of refusing it: the month is not written off."""
        manager = EffectManager(MagicMock())
        manager._store = MagicMock()
        manager._store.async_save = AsyncMock()
        manager._monthly_peaks = []

        await manager.record_period_measurement(
            power_kw=5_000_000.0,
            period=10,
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
            source=POWER_SOURCE_EXTERNAL_METER,
        )
        # A real quarter, after the bad one.
        await manager.record_period_measurement(
            power_kw=6.0,
            period=10,
            timestamp=datetime(2026, 1, 15, 10, 15, tzinfo=timezone.utc),
            source=POWER_SOURCE_EXTERNAL_METER,
        )

        assert manager.get_monthly_peak_summary()["highest"] == pytest.approx(6.0), (
            "the real 6 kW quarter must be the month's peak - the impossible one was refused, so "
            "it cannot be sitting above it making everything else look safe"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("power_kw", [6.0, 17.0, 24.0, 99.0])
    async def test_every_power_a_real_house_can_draw_is_still_recorded(self, power_kw):
        """The ceiling must never refuse a real house. 25 A three-phase is 17 kW; 35 A is 24 kW."""
        manager = EffectManager(MagicMock())
        manager._store = MagicMock()
        manager._store.async_save = AsyncMock()
        manager._monthly_peaks = []

        event = await manager.record_period_measurement(
            power_kw=power_kw,
            period=10,
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
            source=POWER_SOURCE_EXTERNAL_METER,
        )

        assert event is not None, (
            f"{power_kw} kW was refused as implausible. A large Swedish villa on a 35 A service "
            f"with an EV charging draws 24 kW, and the ceiling exists to catch unit errors, not "
            f"customers."
        )


def test_the_canonical_units_are_keyed_case_sensitively():
    """A regression guard on the TABLE, not just its outputs.

    If someone re-lowercases these keys the conversions above still pass for exact-cased units - the
    bug only bites the ambiguous pair. So the table's own shape is pinned.
    """
    assert "mW" in POWER_UNIT_FACTORS_KW and "MW" in POWER_UNIT_FACTORS_KW
    assert POWER_UNIT_FACTORS_KW["mW"] < POWER_UNIT_FACTORS_KW["MW"]
    assert POWER_UNIT_FACTORS_KW["MW"] / POWER_UNIT_FACTORS_KW["mW"] == pytest.approx(1e9)


class TestForgivingWhereItIsSafeToBe:
    """A hand-written template sensor may not match HA's capitalisation. That much is fine."""

    @pytest.mark.parametrize("unit", ["w", "W", "kw", "kW", "KW"])
    def test_unambiguous_case_variants_are_accepted(self, unit):
        """Refusing "kw" would break working installations and buy no safety."""
        assert power_kw_from_state(_sensor("1000", unit)) is not None

    @pytest.mark.parametrize("unit", ["mw", "Mw", "MW ", " mW"])
    def test_an_ambiguous_spelling_is_refused_rather_than_guessed(self, unit):
        """`mw` is BOTH milliwatts and megawatts. There is no safe guess, so there is no guess.

        Note ' mW' and 'MW ' are stripped first and then match exactly - those are fine. The ones
        that must be refused are the ones whose case does not identify the unit.
        """
        result = power_kw_from_state(_sensor("1000", unit))

        if unit.strip() in POWER_UNIT_FACTORS_KW:
            assert result is not None, "an exactly-spelled unit must still work after stripping"
        else:
            assert result is None, (
                f"A sensor reporting {unit!r} was converted to {result} kW. That spelling is both "
                f"milliwatts and megawatts - a factor of 10^9 - and this reading decides whether "
                f"the house is about to set a monthly billing peak. Refuse, do not guess."
            )


class TestRefusalIsStillRefusal:
    """The original contract must survive the fix."""

    @pytest.mark.parametrize("unit", [None, "", "kWh", "Wh", "%", "°C", "A"])
    def test_a_non_power_unit_is_refused(self, unit):
        assert power_kw_from_state(_sensor("1234", unit)) is None

    def test_a_non_numeric_reading_is_refused(self):
        assert power_kw_from_state(_sensor("unavailable", UnitOfPower.WATT)) is None
        assert power_kw_from_state(_sensor("banana", UnitOfPower.WATT)) is None
