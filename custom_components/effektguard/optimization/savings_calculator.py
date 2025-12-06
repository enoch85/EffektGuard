"""Savings calculator for EffektGuard.

Estimates monthly cost savings from optimization:
1. Effect tariff savings (reducing monthly peak)
2. Spot price savings (heating during cheaper periods)

Calculations based on typical Swedish electricity costs and user's system.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..const import (
    BASELINE_EMA_WEIGHT_NEW,
    BASELINE_EMA_WEIGHT_OLD,
    BASELINE_PEAK_MULTIPLIER,
    DAYS_PER_MONTH,
    ORE_TO_SEK_CONVERSION,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class SavingsEstimate:
    """Estimated savings breakdown."""

    monthly_estimate: float  # Total estimated monthly savings (SEK)
    effect_savings: float  # Savings from peak reduction (SEK)
    spot_savings: float  # Savings from spot price optimization (SEK)
    baseline_cost: float  # Estimated cost without optimization (SEK)
    optimized_cost: float  # Estimated cost with optimization (SEK)


class SavingsCalculator:
    """Calculate estimated savings from EffektGuard optimization.

    Conservative estimates based on:
    - Swedish electricity market (15-minute effect tariff)
    - Typical spot price variations
    - Observed peak reductions and price avoidance
    """

    def __init__(self):
        """Initialize savings calculator."""
        # Swedish effect tariff typical costs (SEK/kW/month)
        # Based on common Swedish grid operators:
        # - Ellevio: ~55 SEK/kW/month
        # - Vattenfall Eldistribution: ~50 SEK/kW/month
        # - E.ON: ~50 SEK/kW/month
        # Using conservative average from const.py
        self.effect_tariff_sek_per_kw_month = SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH

        # Track baseline data for comparison
        self._baseline_monthly_peak: Optional[float] = None
        self._total_spot_savings: float = 0.0
        self._optimization_days: int = 0

    def estimate_monthly_savings(
        self,
        current_peak_kw: float,
        baseline_peak_kw: Optional[float] = None,
        average_spot_savings_per_day: float = 0.0,
    ) -> SavingsEstimate:
        """Estimate total monthly savings from optimization.

        Args:
            current_peak_kw: Current monthly peak with optimization (kW)
            baseline_peak_kw: Estimated baseline peak without optimization (kW)
                             If None, assumes +15% higher baseline
            average_spot_savings_per_day: Average daily savings from spot optimization (SEK)

        Returns:
            SavingsEstimate with breakdown
        """
        # Estimate baseline peak if not provided
        if baseline_peak_kw is None:
            # Conservative estimate: optimization typically reduces peak by 10-15%
            # Using 15% reduction assumption from const.py
            baseline_peak_kw = current_peak_kw * BASELINE_PEAK_MULTIPLIER

        # Calculate effect tariff savings
        # Reduction in peak × monthly cost per kW
        peak_reduction_kw = baseline_peak_kw - current_peak_kw
        effect_savings = max(0, peak_reduction_kw * self.effect_tariff_sek_per_kw_month)

        # Calculate spot price savings (30 days)
        spot_savings = average_spot_savings_per_day * DAYS_PER_MONTH

        # Total savings
        monthly_estimate = effect_savings + spot_savings

        # Calculate baseline and optimized costs
        baseline_cost = baseline_peak_kw * self.effect_tariff_sek_per_kw_month
        optimized_cost = current_peak_kw * self.effect_tariff_sek_per_kw_month

        _LOGGER.debug(
            "Savings estimate: Peak %.2f→%.2f kW (%.2f kW reduction), "
            "Effect: %.0f SEK, Spot: %.0f SEK/month, Total: %.0f SEK/month",
            baseline_peak_kw,
            current_peak_kw,
            peak_reduction_kw,
            effect_savings,
            spot_savings,
            monthly_estimate,
        )

        return SavingsEstimate(
            monthly_estimate=round(monthly_estimate, 0),
            effect_savings=round(effect_savings, 0),
            spot_savings=round(spot_savings, 0),
            baseline_cost=round(baseline_cost, 0),
            optimized_cost=round(optimized_cost, 0),
        )

    def calculate_spot_savings_per_cycle(
        self,
        actual_power_kw: float,
        current_price: float,
        average_price_today: float,
        cycle_minutes: float = 5.0,
    ) -> float:
        """Calculate actual spot savings for this cycle using real power data.

        This uses the ACTUAL power consumption from NIBE to calculate real savings.
        Savings = what it would cost at average price - what it actually cost.

        If current_price < average_price: positive savings (we used power during cheap time)
        If current_price > average_price: negative savings (we used power during expensive time)

        Args:
            actual_power_kw: Actual NIBE power consumption this cycle (kW)
            current_price: Current spot price (öre/kWh)
            average_price_today: Average spot price today (öre/kWh)
            cycle_minutes: Duration of this cycle in minutes (default 5)

        Returns:
            Savings for this cycle (SEK), positive = saved money, negative = cost more

        Example:
            Power: 2.5 kW for 5 minutes = 0.208 kWh
            Current price: 50 öre, Average: 100 öre
            Actual cost: 0.208 × 50 / 100 = 0.104 SEK
            Baseline cost: 0.208 × 100 / 100 = 0.208 SEK
            Savings: 0.208 - 0.104 = 0.104 SEK (we saved by using power when cheap!)
        """
        if actual_power_kw is None or actual_power_kw <= 0:
            return 0.0

        # Energy used this cycle
        cycle_hours = cycle_minutes / 60.0
        energy_kwh = actual_power_kw * cycle_hours

        # Actual cost at current price
        actual_cost = (energy_kwh * current_price) / ORE_TO_SEK_CONVERSION

        # What it would have cost at average price (baseline)
        baseline_cost = (energy_kwh * average_price_today) / ORE_TO_SEK_CONVERSION

        # Savings: positive if we paid less than average
        cycle_savings = baseline_cost - actual_cost

        _LOGGER.debug(
            "Spot savings: %.2f kW × %.2f min = %.3f kWh, "
            "price %.1f öre (avg %.1f), savings: %.4f SEK",
            actual_power_kw,
            cycle_minutes,
            energy_kwh,
            current_price,
            average_price_today,
            cycle_savings,
        )

        return cycle_savings

    def update_baseline_peak(self, observed_peak_kw: float) -> None:
        """Update baseline peak estimate from observation.

        Call this when you observe what the peak would have been
        without optimization (e.g., from manual mode periods).

        Args:
            observed_peak_kw: Observed peak without optimization (kW)
        """
        if self._baseline_monthly_peak is None:
            self._baseline_monthly_peak = observed_peak_kw
            _LOGGER.info("Baseline monthly peak set: %.2f kW", observed_peak_kw)
        else:
            # Update with exponential moving average (weights from const.py)
            self._baseline_monthly_peak = (
                BASELINE_EMA_WEIGHT_OLD * self._baseline_monthly_peak
                + BASELINE_EMA_WEIGHT_NEW * observed_peak_kw
            )
            _LOGGER.debug("Baseline monthly peak updated: %.2f kW", self._baseline_monthly_peak)

    def record_spot_savings(self, daily_savings: float) -> None:
        """Record actual spot price savings observed.

        Args:
            daily_savings: Savings observed today (SEK)
        """
        self._total_spot_savings += daily_savings
        self._optimization_days += 1
        _LOGGER.debug(
            "Spot savings recorded: %.2f SEK today, %.2f SEK total over %d days",
            daily_savings,
            self._total_spot_savings,
            self._optimization_days,
        )

    @property
    def average_daily_spot_savings(self) -> float:
        """Get average daily spot savings from recorded history.

        Returns:
            Average daily savings (SEK), or 0.0 if no data
        """
        if self._optimization_days == 0:
            return 0.0
        return self._total_spot_savings / self._optimization_days

    @property
    def baseline_monthly_peak(self) -> Optional[float]:
        """Get current baseline peak estimate.

        Returns:
            Baseline peak (kW), or None if not set
        """
        return self._baseline_monthly_peak
