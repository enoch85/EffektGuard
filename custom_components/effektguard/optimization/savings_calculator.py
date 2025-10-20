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
    CHEAP_PERIOD_BONUS_MULTIPLIER,
    DAYS_PER_MONTH,
    DEFAULT_HEAT_PUMP_POWER_KW,
    EMERGENCY_HEATING_COST_FACTOR,
    HEATING_FACTOR_PER_DEGREE,
    MULTIPLIER_BOOST_30_PERCENT,
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
    - Typical GE-Spot price variations
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

    def estimate_spot_savings_per_cycle(
        self,
        offset_applied: float,
        price_classification: str,
        average_price_today: float,
        current_price: float,
        heating_hours: float = 1.0,
        heat_pump_power_kw: float = DEFAULT_HEAT_PUMP_POWER_KW,
    ) -> float:
        """Estimate savings from a single optimization cycle.

        This estimates how much was saved (or cost) by applying offset
        during current price period compared to heating at average price.

        Args:
            offset_applied: Heating curve offset applied (°C)
            price_classification: Current price classification (cheap/normal/expensive/peak)
            average_price_today: Average electricity price today (öre/kWh)
            current_price: Current electricity price (öre/kWh)
            heating_hours: Hours of heating in this cycle (default 1.0)
            heat_pump_power_kw: Average heat pump power consumption (kW), default from const.py

        Returns:
            Estimated savings for this cycle (SEK), negative if cost increase
        """
        # Determine if offset increased or decreased heating
        # Positive offset = more heating
        # Negative offset = less heating
        heating_factor = 1.0 + (offset_applied * HEATING_FACTOR_PER_DEGREE)

        # Energy used this cycle
        energy_kwh = heat_pump_power_kw * heating_hours * heating_factor

        # Cost at current price vs average price
        cost_current = (energy_kwh * current_price) / ORE_TO_SEK_CONVERSION
        cost_average = (energy_kwh * average_price_today) / ORE_TO_SEK_CONVERSION

        # Savings: positive if we saved, negative if cost more
        cycle_savings = cost_average - cost_current

        # Additional consideration: did we shift heating away from expensive period?
        if price_classification in ["cheap", "CHEAP"]:
            # Heating during cheap: likely good decision
            # Bonus for preheating during cheap periods
            if offset_applied > 0:
                cycle_savings *= CHEAP_PERIOD_BONUS_MULTIPLIER
        elif price_classification in ["expensive", "EXPENSIVE", "peak", "PEAK"]:
            # Heating during expensive: likely cost increase unless necessary
            if offset_applied < 0:
                # Reduced heating during expensive period: good!
                cycle_savings *= MULTIPLIER_BOOST_30_PERCENT
            elif offset_applied > 0:
                # Increased heating during expensive: probably thermal debt emergency
                cycle_savings *= EMERGENCY_HEATING_COST_FACTOR

        _LOGGER.debug(
            "Cycle savings: offset %.2f°C, price %s (%.2f öre), "
            "energy %.2f kWh, savings %.2f SEK",
            offset_applied,
            price_classification,
            current_price,
            energy_kwh,
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
