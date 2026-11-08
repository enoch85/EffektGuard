"""A cheap-hour bill comparison is possible without any optimizer or energy saving."""

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.optimization.savings_calculator import SavingsCalculator
from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor


def test_unmatched_off_period_peaks_cannot_inflate_the_price_timing_comparison():
    calc = SavingsCalculator()
    # Identical price timing, different weather/occupancy in the observed OFF period.
    estimates = [
        calc.estimate_monthly_savings(5.0, peak, average_spot_savings_per_day=2.0)
        for peak in (5.0, 10.0)
    ]
    assert estimates[0].monthly_estimate == estimates[1].monthly_estimate
    assert estimates[0].monthly_estimate == estimates[0].spot_savings


def test_even_a_positive_comparison_never_claims_a_measured_saving():
    calc = SavingsCalculator()
    calc.price_unit = "SEK/kWh"
    # This is metered consumption at cheap prices; there is no optimizer in this example.
    comparison = calc.calculate_spot_savings_per_cycle(2.0, 0.5, 1.0, 60.0)
    assert comparison == pytest.approx(1.0)
    estimate = calc.estimate_monthly_savings(5.0, 10.0, comparison)
    coord = MagicMock()
    coord.data = {"savings": estimate}
    entry = MagicMock()
    entry.entry_id = "comparison"
    sensor = EffektGuardSensor(
        coord, entry, next(s for s in SENSORS if s.key == "savings_estimate")
    )
    assert sensor.native_value > 0
    attrs = sensor.extra_state_attributes
    assert attrs["savings_measured"] is False
    assert attrs["reference"] == "Same metered energy at each day's average spot price"
    assert "effect_savings" not in attrs
    assert "optimized_cost" not in attrs


@pytest.mark.asyncio
@pytest.mark.parametrize("unit,expected_days", [("SEK/kWh", 2), (None, 1), ("EUR/kWh", 1)])
async def test_observed_zero_days_count_but_missing_units_do_not(unit, expected_days):
    from datetime import timedelta
    from unittest.mock import AsyncMock
    from freezegun import freeze_time
    from custom_components.effektguard.adapters.gespot_adapter import PriceData, QuarterPeriod
    from tests.unit.coordinator.test_a_read_does_not_drive_the_pump import START, _coordinator

    with freeze_time(START):
        coord, _, clock = _coordinator([0.0])
        coord.entry.data["enable_optimization"] = False
        coord.savings_calculator.record_spot_savings(6.0)
        coord.gespot.price_unit = unit
        midnight = START.replace(hour=0, minute=0, second=0)
        coord.gespot.get_prices = AsyncMock(
            return_value=PriceData(
                today=[QuarterPeriod(midnight + timedelta(minutes=15 * q), 1.0) for q in range(96)],
                tomorrow=[],
                has_tomorrow=False,
            )
        )
        unmetered = coord.nibe.get_current_state.side_effect

        async def metered():
            state = await unmetered()
            state.power_kw, state.power_is_estimated = 2.0, False
            return state

        coord.nibe.get_current_state = AsyncMock(side_effect=metered)
        await coord._do_aligned_refresh()
        assert coord._daily_spot_savings == 0.0

    clock["now"] = START + timedelta(days=1)
    with freeze_time(clock["now"]):
        await coord._async_update_data()
    assert coord.savings_calculator._optimization_days == expected_days
    assert coord.savings_calculator.average_daily_spot_savings == 6.0 / expected_days
