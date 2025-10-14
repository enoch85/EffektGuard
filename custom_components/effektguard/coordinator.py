"""Data update coordinator for EffektGuard."""

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import DOMAIN, UPDATE_INTERVAL_MINUTES

_LOGGER = logging.getLogger(__name__)


class EffektGuardCoordinator(DataUpdateCoordinator):
    """Coordinate data updates for EffektGuard.

    This coordinator orchestrates:
    - Data collection from NIBE, GE-Spot, and weather
    - Optimization decision calculation
    - State management and persistence
    - Updates to all entities

    Follows Home Assistant's DataUpdateCoordinator pattern for efficient
    data sharing across multiple entities.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        nibe_adapter,
        gespot_adapter,
        weather_adapter,
        decision_engine,
        effect_manager,
        entry: ConfigEntry,
    ):
        """Initialize coordinator with dependency injection."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
        )
        self.nibe = nibe_adapter
        self.gespot = gespot_adapter
        self.weather = weather_adapter
        self.engine = decision_engine
        self.effect = effect_manager
        self.entry = entry

        # State tracking
        self.current_offset: float = 0.0
        self.peak_today: float = 0.0
        self.peak_this_month: float = 0.0
        self.last_decision_time = None

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data and calculate optimal offset.

        This method:
        1. Gathers data from all sources (with graceful degradation)
        2. Runs optimization algorithm
        3. Returns updated state for all entities

        Returns:
            Dictionary containing:
            - nibe: Current NIBE state
            - price: GE-Spot price data (native 15-min intervals)
            - weather: Weather forecast
            - decision: Optimization decision with offset and reasoning
            - offset: Current heating curve offset
            - peak_today: Today's power peak
            - peak_this_month: This month's highest peak
        """
        _LOGGER.debug("Starting EffektGuard data update")

        # Gather core data (NIBE - must succeed)
        try:
            nibe_data = await self.nibe.get_current_state()
            _LOGGER.debug("NIBE data retrieved successfully")
        except Exception as err:
            _LOGGER.error("Failed to read NIBE data: %s", err)
            raise UpdateFailed(f"Cannot read NIBE data: {err}") from err

        # Gather optional data with graceful degradation
        # GE-Spot price data (native 15-minute intervals)
        try:
            price_data = await self.gespot.get_prices()
            _LOGGER.debug("GE-Spot data retrieved successfully")
        except Exception as err:
            _LOGGER.warning("Price data unavailable, using fallback: %s", err)
            price_data = self._get_fallback_prices()

        # Weather forecast
        try:
            weather_data = await self.weather.get_forecast()
            _LOGGER.debug("Weather data retrieved successfully")
        except Exception as err:
            _LOGGER.info("Weather forecast unavailable: %s", err)
            weather_data = None

        # Run optimization decision engine
        try:
            decision = await self.hass.async_add_executor_job(
                self.engine.calculate_decision,
                nibe_data,
                price_data,
                weather_data,
                self.peak_today,
            )
            _LOGGER.info(
                "Decision: offset %.2f°C, reasoning: %s",
                decision.offset,
                decision.reasoning,
            )
        except Exception as err:
            _LOGGER.error("Optimization failed: %s", err)
            # Fall back to safe operation (no offset)
            decision = self._get_safe_default_decision()

        # Update current state
        self.current_offset = decision.offset
        self.last_decision_time = dt_util.utcnow()

        # Update peak tracking
        await self._update_peak_tracking(nibe_data)

        # Save state periodically
        await self.effect.async_save()

        return {
            "nibe": nibe_data,
            "price": price_data,
            "weather": weather_data,
            "decision": decision,
            "offset": decision.offset,
            "peak_today": self.peak_today,
            "peak_this_month": self.peak_this_month,
        }

    def _get_fallback_prices(self):
        """Get fallback price data when GE-Spot unavailable.

        Returns neutral price classification to maintain safe operation
        without optimization.
        """
        from .optimization.price_analyzer import PriceData, QuarterPeriod

        _LOGGER.debug("Using fallback price data (no optimization)")

        # Create neutral periods - all classified as "normal"
        fallback_periods = []
        for quarter in range(96):  # 96 quarters per day (15-min intervals)
            hour = quarter // 4
            minute = (quarter % 4) * 15
            fallback_periods.append(
                QuarterPeriod(
                    quarter_of_day=quarter,
                    hour=hour,
                    minute=minute,
                    price=1.0,  # Neutral price
                    is_daytime=(6 <= hour < 22),
                )
            )

        return PriceData(
            today=fallback_periods,
            tomorrow=[],
            has_tomorrow=False,
        )

    def _get_safe_default_decision(self):
        """Get safe default decision when optimization fails.

        Returns zero offset to maintain current operation without changes.
        """
        from .optimization.decision_engine import LayerDecision, OptimizationDecision

        _LOGGER.debug("Using safe default decision (no changes)")

        return OptimizationDecision(
            offset=0.0,
            layers=[
                LayerDecision(
                    offset=0.0,
                    weight=1.0,
                    reason="Safe mode: optimization unavailable",
                )
            ],
            reasoning="Safe mode active - maintaining current operation",
        )

    async def _update_peak_tracking(self, nibe_data) -> None:
        """Update peak power tracking for effect tariff optimization.

        Tracks 15-minute power consumption windows for Swedish Effektavgift.
        """
        try:
            # Estimate current power consumption
            current_power = self._estimate_power_consumption(nibe_data)

            # Update daily peak
            if current_power > self.peak_today:
                self.peak_today = current_power
                _LOGGER.debug("New daily peak: %.2f kW", current_power)

            # Update monthly peak through effect manager
            now = dt_util.now()
            quarter_of_day = (now.hour * 4) + (now.minute // 15)  # 0-95

            peak_event = await self.effect.record_quarter_measurement(
                power_kw=current_power,
                quarter=quarter_of_day,
                timestamp=now,
            )

            if peak_event:
                self.peak_this_month = peak_event.effective_power
                _LOGGER.info("New monthly peak: %.2f kW", self.peak_this_month)

        except Exception as err:
            _LOGGER.warning("Failed to update peak tracking: %s", err)

    def _estimate_power_consumption(self, nibe_data) -> float:
        """Estimate heat pump power consumption from state.

        Estimates based on:
        - Compressor status
        - Supply/return temperature difference
        - Typical heat pump power ratings

        Returns:
            Estimated power consumption in kW
        """
        # Placeholder implementation - will be enhanced in Phase 2
        # For now, return a basic estimate based on compressor status

        if not nibe_data:
            return 0.0

        # Basic estimation:
        # - Compressor on: ~3-5 kW (depending on outdoor temp)
        # - Compressor off: ~0.1 kW (standby)
        is_heating = getattr(nibe_data, "is_heating", False)

        if is_heating:
            # Rough estimation: colder outdoor = higher power
            outdoor_temp = getattr(nibe_data, "outdoor_temp", 0.0)
            base_power = 4.0  # kW baseline

            # Adjust for outdoor temperature
            # Colder = more power needed
            if outdoor_temp < -10:
                return base_power * 1.3
            elif outdoor_temp < 0:
                return base_power * 1.1
            else:
                return base_power
        else:
            return 0.1  # Standby power

    async def async_set_offset(self, offset: float) -> None:
        """Apply heating curve offset to NIBE system.

        Args:
            offset: Offset value in °C (-10 to +10)
        """
        try:
            await self.nibe.set_curve_offset(offset)
            self.current_offset = offset
            _LOGGER.info("Applied offset: %.2f°C", offset)
        except Exception as err:
            _LOGGER.error("Failed to apply offset: %s", err)
            raise
