"""Comfort Layer - Reactive adjustment to maintain comfort.

Phase 8 of layer refactoring: Comfort layer extraction.
Provides gentle steering toward target temperature even within tolerance zone.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from ..const import (
    COMFORT_CORRECTION_MILD,
    COMFORT_CORRECTION_MULT,
    HEAT_LOSS_DIVISOR,
    COMFORT_HEAT_LOSS_FLOOR,
    COMFORT_TOO_COLD_CORRECTION_MULT,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_HIGH,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OptimizationModeConfig,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
    OVERSHOOT_PROTECTION_OFFSET_MIN,
    OVERSHOOT_PROTECTION_START,
    PRICE_FORECAST_BASE_HORIZON,
    PRICE_FORECAST_EXPENSIVE_THRESHOLD,
)
from ..utils.time_utils import get_current_quarter

_LOGGER = logging.getLogger(__name__)


class ThermalModelProtocol(Protocol):
    """Protocol for thermal model interface.

    Avoids circular import with thermal_layer.py.
    """

    thermal_mass: float
    insulation_quality: float

    def get_prediction_horizon(self) -> float:
        """Get prediction horizon in hours."""
        ...


@dataclass
class ComfortLayerDecision:
    """Decision from the comfort layer.

    Encapsulates comfort-based temperature adjustments.
    """

    name: str
    offset: float
    weight: float
    reason: str
    # Additional diagnostic fields
    temp_deviation: float = 0.0
    buffer_hours: float = 0.0
    is_thermal_aware: bool = False


class ComfortLayer:
    """Comfort layer for reactive temperature adjustments.

    Provides gentle steering toward target temperature, with thermal-aware
    overshoot protection that considers weather forecast and price data.

    Mode affects behavior:
    - Comfort mode: Tighter dead zone (0.1°C), stronger weight multiplier (1.3x)
    - Balanced mode: Standard dead zone (0.2°C), normal weights
    - Savings mode: Wider dead zone (0.3°C), weaker weight multiplier (0.7x)
    """

    def __init__(
        self,
        get_thermal_trend: Optional[Callable[[], dict]] = None,
        thermal_model: Optional[ThermalModelProtocol] = None,
        mode_config: Optional[OptimizationModeConfig] = None,
        tolerance_range: float = 0.5,
        target_temp: float = 21.0,
    ):
        """Initialize comfort layer.

        Args:
            get_thermal_trend: Callable returning thermal trend dict
            thermal_model: ThermalModel for prediction horizon and characteristics
            mode_config: Mode configuration (dead_zone, comfort_weight_multiplier)
            tolerance_range: Temperature tolerance range (°C)
            target_temp: Target indoor temperature (°C)
        """
        self._get_thermal_trend = get_thermal_trend or (
            lambda: {"rate_per_hour": 0.0, "confidence": 0.0}
        )
        self.thermal = thermal_model
        self.mode_config = mode_config or MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED]
        self.tolerance_range = tolerance_range
        self.target_temp = target_temp

    def evaluate_layer(
        self,
        nibe_state,
        weather_data=None,
        price_data=None,
    ) -> ComfortLayerDecision:
        """Evaluate comfort layer decision.

        Thermal-aware overshoot protection (Dec 4, 2025):
        - Calculates REAL thermal buffer duration based on:
          1. Current indoor temp trend (observed cooling rate)
          2. Weather forecast (upcoming cold = higher heat loss)
          3. Building thermal characteristics (insulation, thermal mass)
        - Compares buffer hours to upcoming expensive period duration
        - If buffer is INSUFFICIENT, reduces weight to allow price layer pre-heating
        - If buffer is SUFFICIENT, allows gentle coasting with surplus-adjusted weight

        Args:
            nibe_state: Current NIBE state
            weather_data: Weather forecast (for cold snap → higher heat loss)
            price_data: Spot price data (for expensive period timing)

        Returns:
            ComfortLayerDecision with comfort correction
        """
        temp_deviation = nibe_state.indoor_temp - self.target_temp
        tolerance = self.tolerance_range
        dead_zone = self.mode_config.dead_zone
        weight_mult = self.mode_config.comfort_weight_multiplier

        if abs(temp_deviation) < dead_zone:
            return ComfortLayerDecision(
                name="Comfort",
                offset=0.0,
                weight=0.0,
                reason="At target",
                temp_deviation=temp_deviation,
            )

        elif abs(temp_deviation) < tolerance:
            # Within comfort zone but drifting from target
            correction = -temp_deviation * COMFORT_CORRECTION_MULT
            base_weight = LAYER_WEIGHT_COMFORT_MIN

            if temp_deviation > 0:
                reason = f"Slightly warm (+{temp_deviation:.1f}°C), gentle reduce"
            else:
                reason = f"Slightly cool ({temp_deviation:.1f}°C), gentle boost"

            return ComfortLayerDecision(
                name="Comfort",
                offset=correction,
                weight=base_weight * weight_mult,
                reason=reason,
                temp_deviation=temp_deviation,
            )

        elif temp_deviation > tolerance:
            # Overshoot - above target + tolerance
            overshoot = temp_deviation

            if overshoot >= OVERSHOOT_PROTECTION_START:
                # Thermal-aware overshoot protection
                result = self._evaluate_thermal_aware_overshoot(
                    nibe_state=nibe_state,
                    weather_data=weather_data,
                    price_data=price_data,
                    overshoot=overshoot,
                    temp_deviation=temp_deviation,
                )
                if result:
                    return result

                # Standard overshoot protection - no cold snap or sufficient buffer
                return self._standard_overshoot_protection(overshoot, temp_deviation, nibe_state)

            else:
                # Mild overshoot (below start threshold)
                correction = -temp_deviation * COMFORT_CORRECTION_MILD
                return ComfortLayerDecision(
                    name="Comfort",
                    offset=correction,
                    weight=LAYER_WEIGHT_COMFORT_HIGH,
                    reason=f"Warm: {temp_deviation:.1f}°C above target, gentle reduce",
                    temp_deviation=temp_deviation,
                )

        else:
            # Too cold, increase heating strongly
            correction = -(temp_deviation + tolerance) * COMFORT_TOO_COLD_CORRECTION_MULT
            return ComfortLayerDecision(
                name="Comfort",
                offset=correction,
                weight=LAYER_WEIGHT_COMFORT_MAX,
                reason=f"Too cold: {-temp_deviation:.1f}°C under",
                temp_deviation=temp_deviation,
            )

    def _evaluate_thermal_aware_overshoot(
        self,
        nibe_state,
        weather_data,
        price_data,
        overshoot: float,
        temp_deviation: float,
    ) -> Optional[ComfortLayerDecision]:
        """Evaluate thermal-aware overshoot protection.

        Problem: Small overshoot (0.8°C) triggers aggressive coasting (-7.7°C offset)
        which fights the price layer's smart pre-heating for upcoming expensive period.

        Solution: Calculate REAL thermal buffer duration:
        1. Current heat loss rate (from thermal trend + weather forecast)
        2. How many hours will our buffer last?
        3. When does expensive period start/end?
        4. Is buffer sufficient to coast through expensive hours?

        If buffer is INSUFFICIENT, we need to pre-heat NOW during cheap hours!

        Returns:
            ComfortLayerDecision if thermal-aware logic applies, None otherwise
        """
        # Get current thermal trend
        thermal_trend = self._get_thermal_trend()
        indoor_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Get thermal model parameters
        thermal_mass = self.thermal.thermal_mass if self.thermal else 1.0
        insulation = self.thermal.insulation_quality if self.thermal else 1.0

        # Calculate heat loss rate based on physics
        outdoor_temp = nibe_state.outdoor_temp
        indoor_temp = nibe_state.indoor_temp
        temp_diff = indoor_temp - outdoor_temp
        base_heat_loss = temp_diff / (insulation * HEAT_LOSS_DIVISOR)

        # Check weather forecast for upcoming cold
        forecast_heat_loss = base_heat_loss
        if weather_data and weather_data.forecast_hours:
            horizon = int(self.thermal.get_prediction_horizon()) if self.thermal else 12
            forecast_temps = [h.temperature for h in weather_data.forecast_hours[:horizon]]
            if forecast_temps:
                forecast_min_outdoor = min(forecast_temps)
                future_temp_diff = indoor_temp - forecast_min_outdoor
                forecast_heat_loss = future_temp_diff / (insulation * HEAT_LOSS_DIVISOR)

        # Use the WORSE of current trend or forecast-based loss
        effective_heat_loss = max(abs(indoor_rate), forecast_heat_loss)

        # Safety: minimum loss rate
        if effective_heat_loss <= 0.01:
            effective_heat_loss = COMFORT_HEAT_LOSS_FLOOR

        # Calculate buffer duration
        buffer_hours = overshoot / effective_heat_loss

        # Analyze prices for expensive periods
        expensive_result = self._analyze_expensive_periods(price_data, thermal_mass)

        if expensive_result:
            expensive_start_hours, expensive_duration_hours, price_increase_pct = expensive_result
            hours_needed = expensive_start_hours + expensive_duration_hours
            buffer_sufficient = buffer_hours >= hours_needed
            buffer_deficit = hours_needed - buffer_hours

            if not buffer_sufficient:
                # Buffer is NOT enough - need to pre-heat NOW
                correction = -temp_deviation * COMFORT_CORRECTION_MILD

                return ComfortLayerDecision(
                    name="Comfort",
                    offset=correction,
                    weight=LAYER_WEIGHT_COMFORT_MIN,
                    reason=(
                        f"Buffer {overshoot:.1f}°C = {buffer_hours:.1f}h @ {effective_heat_loss:.2f}°C/h loss | "
                        f"Need {hours_needed:.1f}h for {price_increase_pct:.0f}% spike in {expensive_start_hours:.1f}h | "
                        f"DEFICIT {buffer_deficit:.1f}h - pre-heat required!"
                    ),
                    temp_deviation=temp_deviation,
                    buffer_hours=buffer_hours,
                    is_thermal_aware=True,
                )

            else:
                # Buffer IS sufficient - gentle reduce
                surplus_ratio = buffer_hours / hours_needed
                adjusted_weight = min(
                    LAYER_WEIGHT_COMFORT_MIN * surplus_ratio, LAYER_WEIGHT_COMFORT_HIGH
                )
                coast_offset = -temp_deviation * COMFORT_CORRECTION_MILD * 2

                return ComfortLayerDecision(
                    name="Comfort",
                    offset=coast_offset,
                    weight=adjusted_weight,
                    reason=(
                        f"Buffer {overshoot:.1f}°C = {buffer_hours:.1f}h @ {effective_heat_loss:.2f}°C/h | "
                        f"Spike in {expensive_start_hours:.1f}h for {expensive_duration_hours:.1f}h | "
                        f"OK: {surplus_ratio:.1f}x surplus - reducing heat"
                    ),
                    temp_deviation=temp_deviation,
                    buffer_hours=buffer_hours,
                    is_thermal_aware=True,
                )

        return None

    def _analyze_expensive_periods(
        self, price_data, thermal_mass: float
    ) -> Optional[tuple[float, float, float]]:
        """Analyze price data for expensive periods.

        Returns:
            Tuple of (start_hours, duration_hours, increase_pct) or None
        """
        if not price_data or not price_data.today:
            return None

        current_quarter = get_current_quarter()

        if current_quarter >= len(price_data.today):
            return None

        current_price = price_data.today[current_quarter].price
        if current_price <= 0:
            return None

        # Scan forward to find expensive periods
        forecast_hours = PRICE_FORECAST_BASE_HORIZON * thermal_mass
        forecast_quarters = int(forecast_hours * 4)
        lookahead_end = min(current_quarter + 1 + forecast_quarters, 96)

        upcoming = list(price_data.today[current_quarter + 1 : lookahead_end])
        if price_data.has_tomorrow and lookahead_end >= 96:
            remaining = forecast_quarters - (96 - current_quarter - 1)
            upcoming.extend(price_data.tomorrow[:remaining])

        if not upcoming:
            return None

        expensive_start_hours = None
        expensive_duration_hours = 0.0
        price_increase_pct = 0.0

        for i, period in enumerate(upcoming):
            ratio = period.price / current_price
            if ratio >= PRICE_FORECAST_EXPENSIVE_THRESHOLD:
                if expensive_start_hours is None:
                    expensive_start_hours = (i + 1) * 0.25
                    price_increase_pct = (ratio - 1.0) * 100
                expensive_duration_hours += 0.25
            elif expensive_start_hours is not None:
                break

        if expensive_start_hours is not None:
            return (expensive_start_hours, expensive_duration_hours, price_increase_pct)

        return None

    def _standard_overshoot_protection(
        self, overshoot: float, temp_deviation: float, nibe_state
    ) -> ComfortLayerDecision:
        """Standard overshoot protection without thermal awareness.

        Graduated response based on how far above target:
        - 0.6°C above: -7°C offset, weight 0.7 (start coasting)
        - 1.5°C above: -10°C offset, weight 1.0 (full coast, overrides all)

        Args:
            overshoot: How far above target (positive value)
            temp_deviation: Total temperature deviation
            nibe_state: Current NIBE state

        Returns:
            ComfortLayerDecision for standard overshoot protection
        """
        overshoot_range = OVERSHOOT_PROTECTION_FULL - OVERSHOOT_PROTECTION_START
        fraction = min((overshoot - OVERSHOOT_PROTECTION_START) / overshoot_range, 1.0)

        # Scale offset from -7°C at 0.6°C to -10°C at 1.5°C overshoot
        coast_offset = OVERSHOOT_PROTECTION_OFFSET_MIN + fraction * (
            OVERSHOOT_PROTECTION_OFFSET_MAX - OVERSHOOT_PROTECTION_OFFSET_MIN
        )

        # Scale weight from 0.7 at 0.6°C to 1.0 at 1.5°C
        coast_weight = LAYER_WEIGHT_COMFORT_HIGH + fraction * (
            LAYER_WEIGHT_COMFORT_CRITICAL - LAYER_WEIGHT_COMFORT_HIGH
        )

        return ComfortLayerDecision(
            name="Comfort",
            offset=coast_offset,
            weight=coast_weight,
            reason=(
                f"Overshoot: {overshoot:.1f}°C above target, reducing heat "
                f"({coast_offset:.1f}°C @ {coast_weight:.2f})"
            ),
            temp_deviation=temp_deviation,
        )
