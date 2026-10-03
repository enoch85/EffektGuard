"""A projection is not a meter reading, and a price is not a sum of money.

MONETARY permits exactly one state class - TOTAL - which makes the recorder keep a running SUM.

`savings_estimate` must NOT be MONETARY: its value is a forward-looking monthly projection, and
summing it in the Energy dashboard is meaningless. Its unit stays hardcoded "SEK" (the effect-tariff
component is a Swedish tariff and the spot component is dropped unless already SEK-compatible, so the
value really is kronor - deriving the label from the öre/kWh price feed would be a 100x error).

`current_price` must NOT be MONETARY either: its unit is typically "öre/kWh", a rate, not currency.
It is MEASUREMENT, which is what gives a price long-term statistics (min/max/mean) at all.
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
    """SEK is the RIGHT label: `monthly_estimate` is kronor (a Swedish effect tariff plus a spot
    component dropped unless already SEK-compatible). Deriving the unit from the öre/kWh price feed
    would print "öre" on a SEK value - a 100x error. The Norwegian-user problem is the tariff MODEL,
    not the label (F-107, open with the owner).
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
