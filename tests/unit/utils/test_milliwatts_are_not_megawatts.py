"""`mW` and `MW` differ only in case, and one of them is a billion times the other.

`power_kw_from_state` case-folded the unit before looking it up. Home Assistant ships BOTH
`UnitOfPower.MILLIWATT` ("mW") and `UnitOfPower.MEGA_WATT` ("MW"), so `.lower()` collapsed them onto
the same key - and the table mapped that key to MEGAWATTS.

A sensor reporting 5000 mW (five watts) was therefore read as 5 000 000 kW. That number is
classified billable, recorded as a quarter-hour mean, and persisted as the month's tariff peak. The
effect layer then believes the house has already blown its billing peak and pins itself to CRITICAL
for the rest of the month, throttling heat in January to protect a peak that never happened.

The module's own docstring says "There is no defensible default... An unrecognised unit is refused."
It then silently guessed on the single genuinely ambiguous case in Home Assistant's whole unit enum.

These tests derive the ambiguity from `UnitOfPower` itself rather than hardcoding "mW"/"MW", so if
Home Assistant ever adds another case-colliding pair they fail here instead of in someone's January
electricity bill.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.const import UnitOfPower

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
