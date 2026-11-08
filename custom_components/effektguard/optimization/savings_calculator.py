"""Metered cost comparisons, without attributing savings to the optimizer.

Same-energy spending at the daily average price is a price-timing reference, not a no-EffektGuard
counterfactual. OFF-period peaks also differ in weather, occupancy, other loads and stored heat.
Neither establishes what this house would have spent under comparable conditions without control.
Legacy Python names remain for stored-data and coordinator compatibility; the UI labels comparisons.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..const import (
    BASELINE_EMA_WEIGHT_NEW,
    BASELINE_EMA_WEIGHT_OLD,
    DAYS_PER_MONTH,
    ORE_TO_SEK_CONVERSION,
    PRICE_MAINUNIT_PREFIXES,
    PRICE_SUBUNIT_PREFIXES,
    SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class SavingsEstimate:
    """Legacy fields carrying separate descriptive comparisons, never measured savings.

    monthly_estimate and spot_savings: daily price-timing comparison projected over DAYS_PER_MONTH.
    effect_savings: signed OFF-period peak difference valued at the configured illustrative tariff.
    baseline_cost/optimized_cost: those separate peak references, not complete electricity bills.
    effect_baseline_measured: an OFF-period peak was observed, not a matched counterfactual.
    """

    monthly_estimate: float
    effect_savings: float
    spot_savings: float
    baseline_cost: float
    optimized_cost: float
    effect_baseline_measured: bool = False


class SavingsCalculator:
    """Compute descriptive price and peak comparisons using the existing persistence interface."""

    def price_to_main_unit_factor(self) -> float | None:
        """Factor converting the configured price unit to the main currency, or None.

        Sub-units (öre/cent) divide by 100; main units (SEK/EUR/NOK/DKK) pass through.

        An unrecognised or absent unit returns None - we do NOT guess. The old fallback
        assumed öre/kWh, but EVERY price integration publishes `<currency>/kWh` by default:

            Nord Pool (HA core)         -> SEK/kWh   (no cents option exists at all)
            custom-components/nordpool  -> SEK/kWh   (öre only if price_in_cents: true)
            GE-Spot                     -> SEK/kWh   (öre only if display format = subunit)

        So the öre assumption was 100x WRONG against all three. It fired whenever
        `price_unit` was None - which it is until the first successful price read. Reporting
        a savings figure that is 100x too large is worse than reporting none: skip the
        accumulation instead.
        """
        unit = (self.price_unit or "").lower().replace(" ", "")
        if unit.startswith(PRICE_SUBUNIT_PREFIXES):
            return 1.0 / ORE_TO_SEK_CONVERSION
        if unit.startswith(PRICE_MAINUNIT_PREFIXES):
            return 1.0

        if not self._unknown_unit_logged:
            self._unknown_unit_logged = True
            _LOGGER.warning(
                "Price unit %r not recognised - skipping monetary comparison for this cycle "
                "rather than guessing. Price-based OPTIMIZATION is unaffected (it ranks "
                "prices and does not need the unit).",
                self.price_unit,
            )
        return None

    @property
    def is_sek_price_unit(self) -> bool:
        """Return whether spot savings are denominated in SEK.

        The legacy money gauge is denominated in SEK. Another currency may guide ranking,
        but its numerical result cannot be labelled SEK without an explicit FX conversion.
        """
        unit = (self.price_unit or "").lower().replace(" ", "")
        return not unit or unit.startswith(("sek", "öre", "ore"))

    def __init__(self):
        """Initialize savings calculator."""
        self.price_unit: str | None = None
        self._unknown_unit_logged: bool = False
        # Illustrative configured tariff for the separate historical peak reference.
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
        """Return independent price-timing and historical peak comparisons.

        An observed OFF-period peak is not weather/comfort matched and is not the tariff's
        top-three daily-peak counterfactual. It must never inflate the price-timing projection.
        Keep its sign so a worse peak is visible; neither sign attributes cause to EffektGuard.
        """
        effect_baseline_measured = baseline_peak_kw is not None

        if not effect_baseline_measured:
            baseline_peak_kw = current_peak_kw

        peak_reduction_kw = baseline_peak_kw - current_peak_kw
        effect_savings = peak_reduction_kw * self.effect_tariff_sek_per_kw_month

        # Calculate spot price savings (30 days)
        spot_savings = average_spot_savings_per_day * DAYS_PER_MONTH

        # The gauge is the price-timing projection alone. The peak comparison stays separate.
        monthly_estimate = spot_savings

        # Calculate baseline and optimized costs
        baseline_cost = baseline_peak_kw * self.effect_tariff_sek_per_kw_month
        optimized_cost = current_peak_kw * self.effect_tariff_sek_per_kw_month

        _LOGGER.debug(
            "Cost comparisons (no savings attribution): Peak %.2f→%.2f kW (%.2f kW difference), "
            "Peak reference: %.0f SEK, Price timing: %.0f SEK/month, Projection: %.0f SEK/month",
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
            effect_baseline_measured=effect_baseline_measured,
        )

    def calculate_spot_savings_per_cycle(
        self,
        actual_power_kw: float,
        current_price: float,
        average_price_today: float,
        cycle_minutes: float = 5.0,
    ) -> float:
        """Compare the same metered energy at current price and the day's average price.

        Positive means below the price reference, negative above it. The same result can occur
        with optimization switched off. It measures neither energy saved nor incremental benefit:
        weather, comfort, efficiency, rebound and initial/final stored heat are not controlled.
        The configured price unit converts the result to its main currency.
        """
        if actual_power_kw is None or actual_power_kw <= 0:
            return 0.0

        # Energy used this cycle
        cycle_hours = cycle_minutes / 60.0
        energy_kwh = actual_power_kw * cycle_hours

        # Actual cost at current price. An unrecognised unit yields None - report no
        # savings rather than a figure that could be 100x out.
        to_main = self.price_to_main_unit_factor()
        if to_main is None:
            return 0.0

        actual_cost = energy_kwh * current_price * to_main

        # What it would have cost at average price (baseline)
        baseline_cost = energy_kwh * average_price_today * to_main

        # Savings: positive if we paid less than average
        cycle_savings = baseline_cost - actual_cost

        _LOGGER.debug(
            "Price-timing comparison: %.2f kW × %.2f min = %.3f kWh, "
            "price %.1f (avg %.1f), difference: %.4f main currency",
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

        An OFF-period peak is only a historical reference. Weather, occupancy and other
        loads are not matched; this observation does not establish avoided tariff charges.

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
        """Record a completed day's price-timing comparison.

        Args:
            daily_savings: Price-reference difference today (SEK)
        """
        self._total_spot_savings += daily_savings
        self._optimization_days += 1
        _LOGGER.debug(
            "Price-timing comparison recorded: %.2f SEK today, %.2f SEK total over %d days",
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
