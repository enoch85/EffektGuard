"""Data update coordinator for EffektGuard."""

import logging
from datetime import timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CLIMATE_CENTRAL_SWEDEN,
    CLIMATE_MID_NORTHERN_SWEDEN,
    CLIMATE_NORTHERN_LAPLAND,
    CLIMATE_NORTHERN_SWEDEN,
    CLIMATE_SOUTHERN_SWEDEN,
    CONF_HEAT_PUMP_MODEL,
    DEFAULT_HEAT_PUMP_MODEL,
    DOMAIN,
    ESTIMATED_POWER_BASELINE,
    STORAGE_KEY_LEARNING,
    STORAGE_VERSION,
    UPDATE_INTERVAL_MINUTES,
)
from .models.nibe import (
    NibeF2040Profile,
    NibeF730Profile,
    NibeF750Profile,
    NibeS1155Profile,
)
from .optimization.adaptive_learning import AdaptiveThermalModel
from .optimization.thermal_predictor import ThermalStatePredictor
from .optimization.weather_learning import WeatherPatternLearner

_LOGGER = logging.getLogger(__name__)


# Model registry for quick lookup
HEAT_PUMP_MODELS = {
    "nibe_f730": NibeF730Profile,
    "nibe_f750": NibeF750Profile,
    "nibe_f2040": NibeF2040Profile,
    "nibe_s1155": NibeS1155Profile,
}


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

        # Load heat pump model profile
        model_key = entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL)
        model_class = HEAT_PUMP_MODELS.get(model_key, NibeF750Profile)
        self.heat_pump_model = model_class()

        _LOGGER.info(
            "Loaded heat pump model: %s (%s)",
            self.heat_pump_model.model_name,
            self.heat_pump_model.model_type,
        )

        # Learning modules (Phase 6)
        self.adaptive_learning = AdaptiveThermalModel()
        self.thermal_predictor = ThermalStatePredictor()
        self.weather_learner = WeatherPatternLearner()
        self.climate_region = self._detect_climate_region(hass)

        # Learning storage
        self.learning_store = Store(hass, STORAGE_VERSION, STORAGE_KEY_LEARNING)

        # State tracking
        self.current_offset: float = 0.0
        self.peak_today: float = 0.0
        self.peak_this_month: float = 0.0
        self.last_decision_time = None
        self._learned_data_changed = False  # Track if learning data needs saving

    def _detect_climate_region(self, hass: HomeAssistant) -> str:
        """Detect Swedish climate region based on Home Assistant location.

        Uses latitude to determine climate region for adaptive learning thresholds.

        Swedish climate regions (based on SMHI climate data):
        - Southern Sweden (55-58°N): Malmö, Gothenburg - Jan avg 0.1°C
        - Central Sweden (58-62°N): Stockholm, Gävle - Jan avg -3.7°C
        - Mid-Northern Sweden (62-65°N): Östersund, Umeå - Jan avg -7.9°C
        - Northern Sweden (65-67°N): Luleå, Boden - Jan avg -11.0°C
        - Northern Lapland (67-70°N): Kiruna, Gällivare - Jan avg -12.5°C

        Args:
            hass: HomeAssistant instance

        Returns:
            Climate region constant (southern_sweden, central_sweden, etc.)
        """
        try:
            # Get Home Assistant latitude
            latitude = hass.config.latitude

            if latitude is None:
                _LOGGER.warning("Latitude not configured, defaulting to central Sweden")
                return CLIMATE_CENTRAL_SWEDEN

            # Detect region based on latitude bands
            if latitude < 58.0:
                region = CLIMATE_SOUTHERN_SWEDEN
                region_name = "Southern Sweden (Malmö/Gothenburg)"
            elif latitude < 62.0:
                region = CLIMATE_CENTRAL_SWEDEN
                region_name = "Central Sweden (Stockholm/Gävle)"
            elif latitude < 65.0:
                region = CLIMATE_MID_NORTHERN_SWEDEN
                region_name = "Mid-Northern Sweden (Östersund/Umeå)"
            elif latitude < 67.0:
                region = CLIMATE_NORTHERN_SWEDEN
                region_name = "Northern Sweden (Luleå/Boden)"
            else:
                region = CLIMATE_NORTHERN_LAPLAND
                region_name = "Northern Lapland (Kiruna)"

            _LOGGER.info(
                "Detected climate region: %s (latitude: %.2f°N)",
                region_name,
                latitude,
            )

            return region

        except Exception as err:
            _LOGGER.warning("Failed to detect climate region: %s, defaulting to central", err)
            return CLIMATE_CENTRAL_SWEDEN

    async def async_initialize_learning(self) -> None:
        """Initialize learning modules by loading persisted data.

        Called once during coordinator setup to restore learned parameters
        from previous sessions.
        """
        _LOGGER.debug("Initializing learning modules...")

        try:
            learned_data = await self._load_learned_data()

            if learned_data:
                # Restore adaptive thermal model
                if "thermal_model" in learned_data:
                    thermal_data = learned_data["thermal_model"]
                    self.adaptive_learning.thermal_mass = thermal_data.get("thermal_mass", 1.0)
                    self.adaptive_learning.ufh_type = thermal_data.get("ufh_type", "unknown")
                    _LOGGER.info(
                        "Restored thermal model: mass=%.2f, UFH=%s",
                        self.adaptive_learning.thermal_mass,
                        self.adaptive_learning.ufh_type,
                    )

                # Restore weather patterns
                if "weather_patterns" in learned_data:
                    self.weather_learner.from_dict(learned_data["weather_patterns"])
                    summary = self.weather_learner.get_pattern_database_summary()
                    _LOGGER.info(
                        "Restored weather patterns: %d weeks of data",
                        summary.get("total_weeks", 0),
                    )

                _LOGGER.info("Learning modules initialized successfully")
            else:
                _LOGGER.info("No learned data found - starting fresh learning")

        except Exception as err:
            _LOGGER.warning("Failed to initialize learning modules: %s", err)
            # Continue with fresh learning

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
            _LOGGER.debug(
                "NIBE data retrieved: indoor %.1f°C, outdoor %.1f°C, flow %.1f°C, DM %.0f",
                nibe_data.indoor_temp,
                nibe_data.outdoor_temp,
                nibe_data.flow_temp,
                nibe_data.degree_minutes,
            )
        except Exception as err:
            _LOGGER.error("Failed to read NIBE data: %s", err)
            raise UpdateFailed(f"Cannot read NIBE data: {err}") from err

        # Gather optional data with graceful degradation
        # GE-Spot price data (native 15-minute intervals)
        try:
            price_data = await self.gespot.get_prices()
            if price_data and price_data.today:
                current_q = (dt_util.now().hour * 4) + (dt_util.now().minute // 15)
                current_price = (
                    price_data.today[current_q].price_ore
                    if current_q < len(price_data.today)
                    else 0
                )
                _LOGGER.debug(
                    "GE-Spot data retrieved: %d quarters today, current Q%d = %.1f öre/kWh",
                    len(price_data.today),
                    current_q,
                    current_price,
                )
            else:
                _LOGGER.debug("GE-Spot data empty, using fallback prices")
                price_data = self._get_fallback_prices()
        except Exception as err:
            _LOGGER.warning("Price data unavailable, using fallback: %s", err)
            price_data = self._get_fallback_prices()

        # Weather forecast
        try:
            weather_data = await self.weather.get_forecast()
            if weather_data:
                _LOGGER.debug(
                    "Weather data retrieved: current %.1f°C, %d hours forecast",
                    weather_data.current_temp,
                    len(weather_data.forecast_hours),
                )
            else:
                _LOGGER.debug("Weather forecast not available (optional feature disabled)")
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

        # Apply offset to NIBE heat pump via MyUplink integration
        # This sends the calculated offset to the MyUplink number entity (parameter 47011)
        # Rate limiting (5 min) handled in nibe_adapter to prevent excessive API calls
        try:
            await self.nibe.set_curve_offset(decision.offset)
            _LOGGER.info("Applied offset %.2f°C to NIBE via MyUplink", decision.offset)
        except Exception as err:
            _LOGGER.error("Failed to apply offset to NIBE: %s", err)
            # Continue anyway - next cycle will retry

        # Update peak tracking
        await self._update_peak_tracking(nibe_data)

        # Record observations for learning (Phase 6)
        await self._record_learning_observations(nibe_data, weather_data, decision.offset)

        # Save state periodically
        await self.effect.async_save()

        # Save learned data if changed (every hour to avoid excessive writes)
        if self._learned_data_changed:
            now = dt_util.now()
            if (
                not hasattr(self, "_last_learning_save")
                or (now - self._last_learning_save).total_seconds() > 3600
            ):
                await self._save_learned_data(
                    self.adaptive_learning,
                    self.thermal_predictor,
                    self.weather_learner,
                )
                self._last_learning_save = now
                self._learned_data_changed = False

        # Get current quarter classification from price analyzer
        from datetime import datetime

        now_time = datetime.now()
        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        current_classification = self.engine.price.get_current_classification(current_quarter)

        return {
            "nibe": nibe_data,
            "price": price_data,
            "weather": weather_data,
            "decision": decision,
            "offset": decision.offset,
            "peak_today": self.peak_today,
            "peak_this_month": self.peak_this_month,
            "current_quarter": current_quarter,
            "current_classification": current_classification,
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

        Note: Only records peaks when actual power sensor is available.
        Estimated power values are not stored to avoid recording standby/startup
        noise as legitimate peaks.
        """
        try:
            # Check if we have actual power sensor (not just estimation)
            has_power_sensor = hasattr(self.nibe, "_power_sensor_entity") and bool(
                self.nibe._power_sensor_entity
            )

            # Estimate current power consumption for decision-making
            current_power = self._estimate_power_consumption(nibe_data)

            # Update daily peak (always track for display)
            if current_power > self.peak_today:
                self.peak_today = current_power
                _LOGGER.debug("New daily peak: %.2f kW", current_power)

            # Only record monthly peaks if we have actual power sensor
            # Prevents storing estimated standby power as legitimate peaks
            if not has_power_sensor:
                _LOGGER.debug(
                    "Skipping monthly peak recording: using estimated power (%.2f kW)",
                    current_power,
                )
                return

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
            base_power = ESTIMATED_POWER_BASELINE  # kW baseline from const.py

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

    async def set_optimization_enabled(self, enabled: bool) -> None:
        """Enable or disable optimization.

        Args:
            enabled: True to enable optimization, False to disable
        """
        if enabled:
            _LOGGER.info("Optimization enabled")
            # Resume normal optimization
            await self.async_request_refresh()
        else:
            _LOGGER.info("Optimization disabled - resetting offset to neutral")
            # Reset offset to neutral (0.0)
            try:
                await self.async_set_offset(0.0)
            except Exception as err:
                _LOGGER.error("Failed to reset offset: %s", err)

    @property
    def current_peak(self) -> float:
        """Return current monthly peak power consumption.

        Returns:
            Monthly peak power in kW
        """
        return self.peak_this_month

    @property
    def model_profile(self):
        """Get heat pump model profile.

        Returns:
            Heat pump model profile instance
        """
        return self.heat_pump_model

    async def _record_learning_observations(
        self,
        nibe_data,
        weather_data,
        current_offset: float,
    ) -> None:
        """Record observations for all learning modules.

        Args:
            nibe_data: Current NIBE state
            weather_data: Weather forecast data
            current_offset: Current heating curve offset
        """
        try:
            now = dt_util.utcnow()

            # Record adaptive thermal observation
            self.adaptive_learning.record_observation(
                timestamp=now,
                indoor_temp=nibe_data.indoor_temp,
                outdoor_temp=nibe_data.outdoor_temp,
                heating_offset=current_offset,
            )

            # Record thermal state for predictor
            self.thermal_predictor.record_state(
                timestamp=now,
                indoor_temp=nibe_data.indoor_temp,
                outdoor_temp=nibe_data.outdoor_temp,
                heating_offset=current_offset,
                flow_temp=nibe_data.flow_temp,
                degree_minutes=nibe_data.degree_minutes,
            )

            # Record weather pattern once per day (at midnight or first update of day)
            if weather_data and weather_data.forecast_hours:
                current_date = now.date()

                if not hasattr(self, "_last_weather_record_date") or (
                    self._last_weather_record_date != current_date
                ):
                    # Extract daily temperature pattern from forecast
                    daily_temps = [hour.temperature for hour in weather_data.forecast_hours[:24]]

                    if daily_temps:
                        self.weather_learner.record_weather_pattern(
                            date=now,
                            daily_temps=daily_temps,
                        )
                        self._last_weather_record_date = current_date
                        _LOGGER.debug("Recorded weather pattern for %s", current_date)

            # Mark that learning data has changed
            self._learned_data_changed = True

            _LOGGER.debug("Recorded learning observations successfully")

        except Exception as err:
            _LOGGER.warning("Failed to record learning observations: %s", err)
            # Don't raise - learning is optional, don't break optimization

    async def _save_learned_data(
        self,
        adaptive_learning=None,
        thermal_predictor=None,
        weather_learner=None,
    ) -> None:
        """Persist learned parameters to storage.

        Args:
            adaptive_learning: AdaptiveThermalModel instance
            thermal_predictor: ThermalStatePredictor instance
            weather_learner: WeatherPatternLearner instance
        """
        try:
            learned_data = {
                "version": STORAGE_VERSION,
                "last_updated": dt_util.utcnow().isoformat(),
            }

            # Save thermal model parameters
            if adaptive_learning:
                learned_params = adaptive_learning.learned_parameters
                learned_data["thermal_model"] = {
                    "thermal_mass": adaptive_learning.thermal_mass,
                    "ufh_type": adaptive_learning.ufh_type,
                    "learned_parameters": learned_params,
                    "observation_count": len(adaptive_learning.observations),
                }

            # Save thermal predictor state
            if thermal_predictor:
                learned_data["thermal_predictor"] = {
                    "responsiveness": thermal_predictor._thermal_responsiveness,
                    "state_count": len(thermal_predictor.state_history),
                }

            # Save weather patterns
            if weather_learner:
                learned_data["weather_patterns"] = weather_learner.to_dict()
                summary = weather_learner.get_pattern_database_summary()
                learned_data["weather_summary"] = summary

            await self.learning_store.async_save(learned_data)
            _LOGGER.debug("Saved learned data to storage")

        except Exception as err:
            _LOGGER.error("Failed to save learned data: %s", err, exc_info=True)

    async def _load_learned_data(self) -> dict[str, Any] | None:
        """Load persisted learning data from storage.

        Returns:
            Dictionary with learned data, or None if not available
        """
        try:
            data = await self.learning_store.async_load()

            if data:
                _LOGGER.info(
                    "Loaded learned data from storage (version %s, updated %s)",
                    data.get("version"),
                    data.get("last_updated"),
                )
                return data
            else:
                _LOGGER.debug("No learned data in storage yet")
                return None

        except Exception as err:
            _LOGGER.warning("Failed to load learned data: %s", err)
            return None
