"""A projection is not a meter reading, and a price is not a sum of money.

Two sensors carry `device_class=MONETARY`, and Home Assistant is strict about what that means:

    DEVICE_CLASS_STATE_CLASSES[SensorDeviceClass.MONETARY] == {SensorStateClass.TOTAL}

TOTAL tells the recorder to keep a **sum** - it is the state class of a meter that accumulates.

`savings_estimate` is `device_class=MONETARY`, `state_class=TOTAL`, unit hardcoded `"SEK"`. Its
value is `savings.monthly_estimate`: a **forward-looking projection** that goes up and down as the
forecast changes. So Home Assistant's long-term statistics **accumulate a projection as though it
were a running total**, and the number that lands in the Energy dashboard is meaningless. The only
state class MONETARY permits is the one that is semantically wrong for this quantity.

The unit, though, is right, and that is worth recording because it is a trap. It looks like a
Swedish-centric oversight, and the obvious "fix" - derive the currency from the user's spot-price
entity - is a 100x error: that entity reports **öre/kWh**, while `monthly_estimate` is **kronor**
(its tariff component is SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH, and the spot component is DROPPED
when the price unit is not SEK-compatible rather than converted at a guessed rate). Showing a
Norwegian a SEK figure computed from a Swedish grid tariff is a real problem - with the tariff
MODEL, not the label. That is F-107, and it is open with the owner.

`current_price` is `device_class=MONETARY` with **no state class** and a *dynamic* unit read off the
spot-price entity - typically `"öre/kWh"`, which is not a currency at all. A price per kilowatt-hour
is a **rate**, not an amount of money. The inline comment says "monetary device_class doesn't
support state_class", which is simply untrue (it supports TOTAL), and the consequence of believing
it is that the price sensor produces **no long-term statistics at all** - the one sensor a user most
wants to plot.

Neither sensor should be MONETARY. A projection is a number; a price is a measurement.
"""

from __future__ import annotations

from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass

from custom_components.effektguard.sensor import SENSORS


def _by_key(key: str):
    match = [d for d in SENSORS if d.key == key]
    assert match, f"no sensor description with key {key!r}"
    return match[0]


def test_a_projection_is_not_accumulated_into_the_energy_dashboard():
    """savings_estimate is a forecast. TOTAL makes the recorder sum it."""
    savings = _by_key("savings_estimate")

    assert savings.state_class != SensorStateClass.TOTAL, (
        "savings_estimate is state_class=TOTAL, so Home Assistant's recorder keeps a SUM of it - "
        "but the value is a forward-looking monthly PROJECTION that rises and falls with the "
        "forecast. The Energy and Statistics graphs accumulate it as if it were a meter."
    )


def test_the_savings_label_matches_the_unit_the_value_is_computed_in():
    """SEK is the RIGHT label here, and the reasoning matters more than the assertion.

    It is tempting to call a hardcoded "SEK" a Swedish-centric oversight and derive the unit from
    the user's spot-price entity instead. That entity reports **öre/kWh**. `monthly_estimate` is
    **kronor** - its effect-tariff component is SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH, and
    SavingsCalculator DROPS the spot component entirely when the price unit is not SEK-compatible
    rather than guessing an exchange rate. Deriving the label from the price feed therefore prints
    "öre" on a value denominated in SEK: a 100x error, dressed up as internationalisation.

    (This was written, and caught on a live Home Assistant, which recorded unit='öre' against a
    kronor value. The number a sensor shows and the unit it claims must be the same number.)

    Showing a Norwegian a SEK figure derived from a Swedish grid tariff is a real problem. It is a
    problem with the tariff MODEL, not with the label - audit F-107, open with the owner.
    """
    savings = _by_key("savings_estimate")

    assert savings.native_unit_of_measurement == "SEK", (
        "savings_estimate must be labelled SEK, because that is the unit its value is computed in: "
        "a Swedish effect tariff, plus a spot component that is dropped unless it is already "
        "SEK-compatible. Any other label misstates the magnitude."
    )


def test_a_price_per_kwh_is_not_a_sum_of_money():
    """MONETARY means an amount of currency. 'öre/kWh' is a rate."""
    price = _by_key("current_price")

    assert price.device_class != SensorDeviceClass.MONETARY, (
        "current_price is device_class=MONETARY, but its unit is read off the spot-price entity "
        "and is typically 'öre/kWh' - not a currency. A price per kilowatt-hour is a rate, not an "
        "amount of money."
    )


def test_the_price_sensor_produces_statistics():
    """The sensor a user most wants to plot recorded nothing at all.

    MONETARY permits only TOTAL, and TOTAL is wrong for a price, so the sensor was left with no
    state class - and a sensor with no state class gets no long-term statistics. MEASUREMENT is
    what a price is: the recorder keeps min, max and mean.
    """
    price = _by_key("current_price")

    assert price.state_class == SensorStateClass.MEASUREMENT, (
        "current_price has no state class, so Home Assistant records no long-term statistics for "
        "it. A price is a MEASUREMENT - min/max/mean over time is exactly what you want from it."
    )


def test_no_sensor_claims_monetary_without_earning_it():
    """Whatever else changes, MONETARY must come with the only state class HA allows for it."""
    for description in SENSORS:
        if description.device_class != SensorDeviceClass.MONETARY:
            continue

        assert description.state_class == SensorStateClass.TOTAL, (
            f"{description.key} declares device_class=MONETARY. Home Assistant permits exactly one "
            f"state class with it - TOTAL - and TOTAL means the recorder keeps a running sum. If "
            f"that is not what this sensor is, it is not MONETARY."
        )
