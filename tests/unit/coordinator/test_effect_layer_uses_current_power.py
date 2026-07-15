"""The coordinator must feed the decision engine live power, never the daily peak.

`peak_today` is a daily high-water mark that only ratchets up until the midnight reset.
Feeding it to the engine as "current power" let one unrelated household spike (an oven, a
kettle, an EV charger) pin the effect layer to CRITICAL (weight 1.0, offset -3.0 C) for the
rest of the day, regardless of what the heat pump was drawing. The engine must instead
receive the live reading PROJECTED over the billing hour, because the monthly record it is
compared against is an hourly mean.
"""

import inspect


class TestCoordinatorPowerContract:
    """The coordinator must feed the engine live power, not the daily maximum."""

    def test_decision_path_does_not_consume_peak_today(self):
        """`peak_today` (a daily maximum) and `current_power_kw` (the live reading the effect
        layer consumes) are different quantities and must not be aliased.
        """
        from custom_components.effektguard.coordinator import EffektGuardCoordinator

        update_src = inspect.getsource(EffektGuardCoordinator._read_and_decide)

        assert "current_power_for_decision = self.peak_today" not in update_src, (
            "The decision engine is being fed peak_today (a daily MAXIMUM) as current power. "
            "One morning spike would pin the effect layer to CRITICAL until midnight."
        )
        assert "projected_hour_mean" in update_src and "self.current_power_kw" in update_src, (
            "The decision engine must be fed the live reading PROJECTED over the billing hour "
            "- the monthly record it is compared against is an hourly mean, so an instantaneous "
            "spike is not the same quantity. See "
            "tests/unit/optimization/test_peak_protection_compares_like_with_like.py."
        )
