"""Data update coordinator for EffektGuard."""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

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
    DHW_CONTROL_MIN_INTERVAL_MINUTES,
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

        # DHW optimizer - pass climate detector for climate-aware thresholds
        from .optimization.dhw_optimizer import DHWDemandPeriod, IntelligentDHWScheduler

        # Configure DHW demand periods from options
        demand_periods = []

        # Morning demand period (e.g., shower time)
        if entry.options.get("dhw_morning_enabled", True):
            morning_hour = int(entry.options.get("dhw_morning_hour", 7))
            demand_periods.append(
                DHWDemandPeriod(
                    start_hour=morning_hour,
                    target_temp=55.0,  # Extra hot for morning shower
                    duration_hours=2,  # 2-hour window
                )
            )

        # Evening demand period (e.g., dishes, evening shower)
        if entry.options.get("dhw_evening_enabled", True):
            evening_hour = int(entry.options.get("dhw_evening_hour", 18))
            demand_periods.append(
                DHWDemandPeriod(
                    start_hour=evening_hour,
                    target_temp=55.0,  # Extra hot for dishes
                    duration_hours=3,  # 3-hour window
                )
            )

        # Pass climate detector from decision engine to DHW optimizer for dynamic thresholds
        self.dhw_optimizer = IntelligentDHWScheduler(
            demand_periods=demand_periods,
            climate_detector=decision_engine.climate_detector,
        )

        if demand_periods:
            try:
                # Format DHW periods for logging (handle both real values and test mocks)
                formatted_periods = []
                for p in demand_periods:
                    try:
                        hour = int(p.start_hour)
                        temp = float(p.target_temp)
                        formatted_periods.append(f"{hour:02d}:00 ({temp:.1f}°C)")
                    except (TypeError, ValueError):
                        # Fallback for mock objects in tests
                        formatted_periods.append(f"{p.start_hour}:00 ({p.target_temp}°C)")

                _LOGGER.info("DHW demand periods configured: %s", formatted_periods)

                # Debug logging for type validation
                _LOGGER.debug(
                    "DHW periods configured: %s (types: %s)",
                    [f"{p.start_hour}:00" for p in demand_periods],
                    [f"{type(p.start_hour).__name__}" for p in demand_periods],
                )
            except Exception as err:
                _LOGGER.debug("Could not format DHW periods: %s", err)

        # Learning storage
        self.learning_store = Store(hass, STORAGE_VERSION, STORAGE_KEY_LEARNING)

        # State tracking
        self.current_offset: float = 0.0
        self.peak_today: float = 0.0
        self.peak_this_month: float = 0.0
        self.last_decision_time = None
        self._learned_data_changed = False  # Track if learning data needs saving

        # DHW tracking
        self.last_dhw_heated = None  # Last time DHW was in heating mode
        self.last_dhw_temp = None  # Last BT7 temperature for trend analysis
        self.dhw_heating_start = None  # When current/last DHW cycle started
        self.dhw_heating_end = None  # When last DHW cycle ended
        self.dhw_was_heating = False  # Track state changes

        # Startup tracking - gracefully handle missing entities during HA startup
        # MyUplink integration can take 45-50 seconds to initialize entities
        self._first_successful_update = False

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

    async def async_shutdown(self) -> None:
        """Clean shutdown of coordinator.

        Saves all persistent state before unload:
        - Learning module data (thermal model, weather patterns)
        - Effect tracking state (monthly peaks)

        Called during integration unload or reload.
        """
        _LOGGER.debug("Shutting down EffektGuard coordinator")

        try:
            # Save learning state
            if self.adaptive_learning or self.thermal_predictor or self.weather_learner:
                await self._save_learned_data(
                    self.adaptive_learning, self.thermal_predictor, self.weather_learner
                )
                _LOGGER.debug("Saved learning data")

            # Save effect tracking state
            if self.effect:
                await self.effect.async_save()
                _LOGGER.debug("Saved effect tracking data")

            _LOGGER.info("Coordinator shutdown complete")

        except Exception as err:
            _LOGGER.error("Error during coordinator shutdown: %s", err, exc_info=True)
            # Don't raise - allow shutdown to complete

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

        # Gather core data (NIBE - required, but allow startup grace period)
        try:
            nibe_data = await self.nibe.get_current_state()
            _LOGGER.debug(
                "NIBE data retrieved: indoor %.1f°C, outdoor %.1f°C, flow %.1f°C, DM %.0f",
                nibe_data.indoor_temp,
                nibe_data.outdoor_temp,
                nibe_data.flow_temp,
                nibe_data.degree_minutes,
            )

            # Mark first successful update
            if not self._first_successful_update:
                self._first_successful_update = True
                _LOGGER.info("EffektGuard fully initialized - NIBE entities available")

        except Exception as err:
            # During startup, MyUplink entities may not be ready yet (takes ~45-50 seconds)
            # Gracefully handle this by returning minimal data until entities are available
            if not self._first_successful_update:
                _LOGGER.info(
                    "Waiting for NIBE MyUplink entities to become available: %s "
                    "(this is normal during HA startup, will retry in %d minutes)",
                    err,
                    UPDATE_INTERVAL_MINUTES,
                )
                # Return minimal coordinator data to allow entities to be created
                # They will show "unavailable" until NIBE data becomes ready
                return {
                    "nibe": None,
                    "price": None,
                    "weather": None,
                    "decision": None,
                    "offset": 0.0,
                    "peak_today": 0.0,
                    "peak_this_month": 0.0,
                    "startup_pending": True,  # Flag for entities to show proper status
                }
            else:
                # After first success, NIBE failures are real errors
                _LOGGER.error("Failed to read NIBE data: %s", err)
                raise UpdateFailed(f"Cannot read NIBE data: {err}") from err

        # Gather optional data with graceful degradation
        # GE-Spot price data (native 15-minute intervals)
        try:
            price_data = await self.gespot.get_prices()
            if price_data and price_data.today:
                current_q = (dt_util.now().hour * 4) + (dt_util.now().minute // 15)
                current_price = (
                    price_data.today[current_q].price if current_q < len(price_data.today) else 0
                )

                # Get unit from GE-Spot entity for accurate logging
                gespot_entity = self.hass.states.get(self.entry.data.get("gespot_entity"))
                unit = (
                    gespot_entity.attributes.get("unit_of_measurement", "units")
                    if gespot_entity
                    else "units"
                )

                _LOGGER.debug(
                    "GE-Spot data retrieved: %d quarters today, current Q%d = %.2f %s",
                    len(price_data.today),
                    current_q,
                    current_price,
                    unit,
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

        # Calculate DHW status and tracking
        dhw_status = "not_configured"
        dhw_next_boost = None
        dhw_last_heated = self.last_dhw_heated
        dhw_recommendation = "DHW sensor not configured"

        # Debug logging for DHW sensor detection
        if nibe_data:
            _LOGGER.debug(
                "DHW sensor check: has_dhw_top_temp=%s, dhw_top_temp=%s",
                hasattr(nibe_data, "dhw_top_temp"),
                getattr(nibe_data, "dhw_top_temp", "N/A"),
            )

        if nibe_data and hasattr(nibe_data, "dhw_top_temp") and nibe_data.dhw_top_temp is not None:
            current_dhw_temp = nibe_data.dhw_top_temp

            # Check if DHW is actively heating (from NIBE MyUplink sensor)
            # This is more accurate than temperature thresholds alone
            is_actively_heating = (
                nibe_data.is_hot_water if hasattr(nibe_data, "is_hot_water") else False
            )

            # Track DHW heating cycle transitions (start/stop)
            if is_actively_heating and not self.dhw_was_heating:
                # DHW heating just started
                self.dhw_heating_start = now_time
                _LOGGER.info("DHW heating started at %s", now_time.strftime("%H:%M:%S"))
            elif not is_actively_heating and self.dhw_was_heating:
                # DHW heating just stopped
                self.dhw_heating_end = now_time
                if self.dhw_heating_start:
                    duration = (now_time - self.dhw_heating_start).total_seconds() / 60
                    _LOGGER.info(
                        "DHW heating stopped at %s (duration: %.1f minutes)",
                        now_time.strftime("%H:%M:%S"),
                        duration,
                    )

            self.dhw_was_heating = is_actively_heating

            # Determine status using actual heating status + temperature
            if is_actively_heating:
                dhw_status = "heating"  # Compressor actively heating DHW
                # Track when we're in heating mode
                self.last_dhw_heated = now_time
                dhw_last_heated = self.last_dhw_heated
            elif current_dhw_temp < 35.0:
                dhw_status = "low"  # Below safety minimum, waiting to heat
            elif current_dhw_temp < 45.0:
                dhw_status = "pending"  # Below target, will heat soon
            elif current_dhw_temp < 52.0:
                dhw_status = "ready"  # At normal target
            else:
                dhw_status = "hot"  # Above normal (high demand met or Legionella cycle)

            # Predict next boost time based on temperature drop
            # Simple prediction: Assume 0.5°C/hour cooling rate (conservative)
            # NIBE typically starts DHW at ~45°C setpoint
            if current_dhw_temp >= 45.0:
                dhw_setpoint = 45.0  # Typical NIBE DHW start threshold
                cooling_rate = 0.5  # °C per hour (conservative estimate)
                temp_margin = current_dhw_temp - dhw_setpoint
                hours_until_boost = temp_margin / cooling_rate

                if hours_until_boost > 0:
                    dhw_next_boost = now_time + timedelta(hours=hours_until_boost)

            # Track temperature for trend analysis
            self.last_dhw_temp = current_dhw_temp

            # Update DHW optimizer with temperature history
            self.dhw_optimizer.update_bt7_temperature(current_dhw_temp, now_time)

            # Get DHW recommendation from optimizer with detailed planning
            try:
                dhw_result = self._calculate_dhw_recommendation(
                    nibe_data, price_data, weather_data, current_dhw_temp, now_time
                )
                dhw_recommendation = dhw_result["recommendation"]
                dhw_planning_summary = dhw_result["summary"]
                dhw_planning_details = dhw_result["details"]
            except Exception as e:
                _LOGGER.error(
                    "DHW recommendation calculation failed: %s. "
                    "Optimization continues without DHW recommendations.",
                    e,
                    exc_info=True,
                )
                dhw_recommendation = f"DHW calculation error: {str(e)[:50]}"
                dhw_planning_summary = "Error calculating DHW planning"
                dhw_planning_details = {}
                # Don't fail entire coordinator update for DHW subsystem issue
                # Core heating optimization continues to work

            # Apply DHW control based on optimizer decision (if hot water optimization enabled)
            if self.entry.data.get("enable_hot_water_optimization", False):
                await self._apply_dhw_control(
                    nibe_data, price_data, weather_data, current_dhw_temp, now_time
                )
        else:
            # DHW sensor not available - provide basic recommendation
            if nibe_data:
                _LOGGER.warning(
                    "DHW sensor (BT7) not found - check MyUplink integration has exposed BT7/40013 sensor"
                )
                dhw_recommendation = "DHW sensor not found - check MyUplink integration"
            else:
                _LOGGER.warning("NIBE data not available")
                dhw_recommendation = "NIBE data unavailable"

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
            "dhw_status": dhw_status,
            "dhw_next_boost": dhw_next_boost,
            "dhw_last_heated": dhw_last_heated,
            "dhw_heating_start": self.dhw_heating_start,
            "dhw_heating_end": self.dhw_heating_end,
            "dhw_recommendation": dhw_recommendation,
            "dhw_planning_summary": dhw_planning_summary,  # Human-readable summary
            "dhw_planning": dhw_planning_details,  # Detailed machine-readable data
        }

    def _calculate_dhw_recommendation(
        self, nibe_data, price_data, weather_data, current_dhw_temp: float, now_time: datetime
    ) -> dict[str, Any]:
        """Calculate DHW heating recommendation with detailed planning.

        Args:
            nibe_data: Current NIBE state
            price_data: GE-Spot price data
            weather_data: Weather forecast
            current_dhw_temp: Current DHW temperature (°C)
            now_time: Current datetime

        Returns:
            Dictionary with recommendation text and detailed planning info
        """
        # Get current price classification
        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        price_classification = (
            self.engine.price.get_current_classification(current_quarter)
            if price_data
            else "normal"
        )

        # Calculate space heating demand (simplified - from indoor temp deficit)
        indoor_temp = nibe_data.indoor_temp if nibe_data else 20.0
        target_indoor = 21.0  # Default target
        indoor_deficit = target_indoor - indoor_temp
        space_heating_demand = max(0, indoor_deficit * 2.0)  # Rough kW estimate

        # Get thermal debt
        thermal_debt = nibe_data.degree_minutes if nibe_data else -60.0

        # Get outdoor temp
        outdoor_temp = nibe_data.outdoor_temp if nibe_data else 0.0

        # Get climate zone thresholds
        from .optimization.climate_zones import get_dm_thresholds_for_temp

        climate_zone = (
            self.engine.climate_detector.zone_info if self.engine.climate_detector else None
        )
        dm_thresholds = get_dm_thresholds_for_temp(outdoor_temp, climate_zone)

        # Get recommendation from optimizer
        decision = self.dhw_optimizer.should_start_dhw(
            current_dhw_temp=current_dhw_temp,
            space_heating_demand_kw=space_heating_demand,
            thermal_debt_dm=thermal_debt,
            indoor_temp=indoor_temp,
            target_indoor_temp=target_indoor,
            outdoor_temp=outdoor_temp,
            price_classification=price_classification,
            current_time=now_time,
        )

        # Calculate optimal heating windows for today (next 24 hours)
        optimal_windows = self._find_optimal_dhw_windows(
            price_data, now_time, thermal_debt, dm_thresholds
        )

        # Build detailed planning attributes
        planning_details = {
            "should_heat": decision.should_heat,
            "priority_reason": decision.priority_reason,
            "current_temperature": current_dhw_temp,
            "target_temperature": decision.target_temp,
            "thermal_debt": thermal_debt,
            "thermal_debt_threshold_block": dm_thresholds["block"],
            "thermal_debt_threshold_abort": dm_thresholds["abort"],
            "thermal_debt_status": self._get_thermal_debt_status(
                thermal_debt, dm_thresholds
            ),
            "space_heating_demand_kw": round(space_heating_demand, 2),
            "current_price_classification": price_classification,
            "outdoor_temperature": outdoor_temp,
            "indoor_temperature": indoor_temp,
            "climate_zone": climate_zone.name if climate_zone else "Unknown",
            "optimal_heating_windows": optimal_windows,
            "next_optimal_window": optimal_windows[0] if optimal_windows else None,
        }

        # Check for weather opportunity
        if weather_data and hasattr(weather_data, "current_temp"):
            # Unusually warm weather is good for DHW
            zone_avg = climate_zone.winter_avg_low if climate_zone else 0.0
            temp_deviation = outdoor_temp - zone_avg
            if temp_deviation > 5.0:
                planning_details["weather_opportunity"] = (
                    f"Unusually warm (+{temp_deviation:.1f}°C), good for DHW heating"
                )

        # Convert decision to human-readable recommendation
        if not decision.should_heat:
            if decision.priority_reason == "CRITICAL_THERMAL_DEBT":
                recommendation = f"Block DHW - Thermal debt warning (DM: {thermal_debt:.0f}, zone: {planning_details['climate_zone']})"
            elif decision.priority_reason == "SPACE_HEATING_EMERGENCY":
                recommendation = f"Block DHW - House too cold ({indoor_temp:.1f}°C)"
            elif decision.priority_reason == "HIGH_SPACE_HEATING_DEMAND":
                recommendation = f"Delay DHW - High heating demand ({space_heating_demand:.1f} kW)"
            elif decision.priority_reason == "DHW_ADEQUATE":
                recommendation = f"DHW OK - Temperature adequate ({current_dhw_temp:.1f}°C)"
            else:
                recommendation = "Wait - Conditions not optimal"

            # Add next optimal window if available
            if optimal_windows:
                next_window = optimal_windows[0]
                recommendation += f" | Next window: {next_window['time_range']}"
        else:
            # Should heat - give specific recommendation
            if decision.priority_reason == "DHW_SAFETY_MINIMUM":
                recommendation = f"Heat now - Safety minimum ({current_dhw_temp:.1f}°C < 35°C)"
            elif decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY":
                recommendation = (
                    f"Heat now - Cheap electricity ({price_classification})"
                )
            elif decision.priority_reason.startswith("URGENT_DEMAND"):
                recommendation = "Heat now - Demand period approaching"
            elif decision.priority_reason.startswith("OPTIMAL_PREHEAT"):
                recommendation = (
                    f"Heat now - Pre-heating for demand ({price_classification})"
                )
            elif decision.priority_reason == "NORMAL_DHW_HEATING":
                recommendation = f"Heat now - Temperature low ({current_dhw_temp:.1f}°C)"
            else:
                recommendation = f"Heat recommended - Target: {decision.target_temp:.0f}°C"

        # Build human-readable planning summary
        planning_summary = self._format_dhw_planning_summary(
            recommendation=recommendation,
            current_temp=current_dhw_temp,
            target_temp=decision.target_temp,
            thermal_debt=thermal_debt,
            dm_thresholds=dm_thresholds,
            space_heating_demand=space_heating_demand,
            price_classification=price_classification,
            optimal_windows=optimal_windows,
            weather_opportunity=planning_details.get("weather_opportunity"),
        )

        # Return combined result with both machine-readable and human-readable data
        return {
            "recommendation": recommendation,
            "summary": planning_summary,
            "details": planning_details,
        }

    def _format_dhw_planning_summary(
        self,
        recommendation: str,
        current_temp: float,
        target_temp: float,
        thermal_debt: float,
        dm_thresholds: dict,
        space_heating_demand: float,
        price_classification: str,
        optimal_windows: list,
        weather_opportunity: Optional[str],
    ) -> str:
        """Format human-readable DHW planning summary.
        
        Args:
            recommendation: Base recommendation text
            current_temp: Current DHW temperature
            target_temp: Target DHW temperature
            thermal_debt: Current thermal debt (DM)
            dm_thresholds: Thermal debt thresholds
            space_heating_demand: Current heating demand in kW
            price_classification: Current price classification
            optimal_windows: List of optimal heating windows
            weather_opportunity: Weather opportunity text if any
            
        Returns:
            Multi-line human-readable summary
        """
        lines = []
        lines.append("DHW Planning Summary")
        lines.append("=" * 40)
        lines.append(f"Current: {current_temp:.1f}°C -> Target: {target_temp:.0f}°C")
        lines.append(f"Price: {price_classification}")
        
        # Thermal debt status
        if thermal_debt < dm_thresholds["abort"]:
            status_text = f"CRITICAL (DM {thermal_debt:.0f})"
        elif thermal_debt < dm_thresholds["block"]:
            status_text = f"WARNING (DM {thermal_debt:.0f})"
        else:
            status_text = f"OK (DM {thermal_debt:.0f})"
        lines.append(f"Thermal Debt: {status_text}")
        
        # Space heating status
        if space_heating_demand > 2.0:
            lines.append(f"Heating Demand: HIGH ({space_heating_demand:.1f} kW)")
        elif space_heating_demand > 0.5:
            lines.append(f"Heating Demand: MODERATE ({space_heating_demand:.1f} kW)")
        else:
            lines.append(f"Heating Demand: LOW ({space_heating_demand:.1f} kW)")
        
        # Weather opportunity
        if weather_opportunity:
            lines.append(f"Weather: {weather_opportunity}")
        
        # Next optimal windows
        if optimal_windows:
            lines.append("")
            lines.append("Optimal Heating Windows:")
            for i, window in enumerate(optimal_windows[:3], 1):
                duration = window['duration_hours']
                price = window['price_classification']
                time_range = window['time_range']
                lines.append(f"  {i}. {time_range} ({duration:.1f}h, {price})")
        else:
            lines.append("")
            lines.append("No optimal windows found")
        
        lines.append("")
        lines.append(f"Recommendation: {recommendation}")
        
        return "\n".join(lines)

    async def _apply_dhw_control(
        self, price_data, now_time: datetime, thermal_debt: float, dm_thresholds: dict
    ) -> list[dict]:
        """Find optimal DHW heating windows in the next 24 hours.

        Args:
            price_data: GE-Spot price data
            now_time: Current datetime
            thermal_debt: Current thermal debt (DM)
            dm_thresholds: Thermal debt thresholds for climate zone

        Returns:
            List of optimal heating windows with time ranges and reasoning
        """
        if not price_data or not hasattr(price_data, "quarters_today"):
            return []

        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        windows = []

        # Analyze price periods for next 24 hours (96 quarters)
        for i in range(current_quarter, min(current_quarter + 96, 96)):
            classification = self.engine.price.get_current_classification(i)

            # Consider CHEAP periods as opportunities
            if classification in ["CHEAP", "NORMAL"]:
                # Check if thermal debt allows DHW heating
                if thermal_debt > dm_thresholds["block"]:
                    hour = i // 4
                    minute = (i % 4) * 15
                    quarter_time = now_time.replace(
                        hour=hour, minute=minute, second=0, microsecond=0
                    )

                    # Group consecutive cheap quarters into windows
                    if windows and windows[-1]["end_quarter"] == i - 1:
                        # Extend existing window
                        windows[-1]["end_quarter"] = i
                        windows[-1]["end_time"] = (quarter_time + timedelta(minutes=15)).strftime("%H:%M")
                        windows[-1]["duration_hours"] = (i - windows[-1]["start_quarter"] + 1) * 0.25
                    else:
                        # Start new window
                        window = {
                            "start_quarter": i,
                            "end_quarter": i,
                            "start_time": quarter_time.strftime("%H:%M"),
                            "end_time": (quarter_time + timedelta(minutes=15)).strftime("%H:%M"),
                            "time_range": f"{quarter_time.strftime('%H:%M')}-{(quarter_time + timedelta(minutes=15)).strftime('%H:%M')}",
                            "price_classification": classification,
                            "duration_hours": 0.25,
                            "thermal_debt_ok": True,
                        }
                        windows.append(window)

        # Update time ranges for windows
        for window in windows:
            window["time_range"] = f"{window['start_time']}-{window['end_time']}"

        # Limit to next 3 optimal windows
        return windows[:3]

    def _get_thermal_debt_status(self, thermal_debt: float, dm_thresholds: dict) -> str:
        """Get human-readable thermal debt status.

        Args:
            thermal_debt: Current thermal debt (DM)
            dm_thresholds: Thresholds for climate zone

        Returns:
            Status string
        """
        if thermal_debt < dm_thresholds["abort"]:
            margin = abs(thermal_debt - dm_thresholds["abort"])
            return f"CRITICAL - {margin:.0f} DM from abort threshold"
        elif thermal_debt < dm_thresholds["block"]:
            margin = abs(thermal_debt - dm_thresholds["block"])
            return f"WARNING - {margin:.0f} DM from block threshold"
        else:
            margin = thermal_debt - dm_thresholds["block"]
            return f"OK - {margin:.0f} DM safety margin"

    async def _apply_dhw_control(
        self, nibe_data, price_data, weather_data, current_dhw_temp: float, now_time: datetime
    ) -> None:
        """Apply automatic DHW control based on optimizer decision.

        Controls NIBE temporary lux switch to heat or block DHW based on
        thermal debt, electricity prices, and demand periods.

        Args:
            nibe_data: Current NIBE state
            price_data: GE-Spot price data
            weather_data: Weather forecast
            current_dhw_temp: Current DHW temperature
            now_time: Current datetime
        """
        from .const import CONF_NIBE_TEMP_LUX_ENTITY

        # Get temporary lux entity from config
        temp_lux_entity = self.entry.data.get(CONF_NIBE_TEMP_LUX_ENTITY)
        if not temp_lux_entity:
            _LOGGER.debug(
                "DHW control disabled: No temporary lux entity configured (switch.temporary_lux_50004)"
            )
            return

        # Get current state of temporary lux switch
        temp_lux_state = self.hass.states.get(temp_lux_entity)
        if not temp_lux_state:
            _LOGGER.warning("Temporary lux entity %s not found", temp_lux_entity)
            return

        is_lux_on = temp_lux_state.state == "on"

        # Get decision from optimizer
        indoor_temp = nibe_data.indoor_temp if nibe_data else 21.0
        target_indoor = self.entry.data.get("target_indoor_temp", 21.0)
        indoor_deficit = target_indoor - indoor_temp
        space_heating_demand = max(0, indoor_deficit * 2.0)
        thermal_debt = nibe_data.degree_minutes if nibe_data else -60.0
        outdoor_temp = nibe_data.outdoor_temp if nibe_data else 0.0

        # Get current price classification
        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        price_classification = self.engine.price.get_current_classification(current_quarter)

        decision = self.dhw_optimizer.should_start_dhw(
            current_dhw_temp=current_dhw_temp,
            space_heating_demand_kw=space_heating_demand,
            thermal_debt_dm=thermal_debt,
            indoor_temp=indoor_temp,
            target_indoor_temp=target_indoor,
            outdoor_temp=outdoor_temp,
            price_classification=price_classification,
            current_time=now_time,
        )

        # Rate limiting: Don't change lux state too frequently (minimum 1 hour)
        if hasattr(self, "_last_dhw_control_time"):
            time_since_last = (now_time - self._last_dhw_control_time).total_seconds() / 60
            if time_since_last < DHW_CONTROL_MIN_INTERVAL_MINUTES:
                _LOGGER.debug(
                    "DHW control rate limited: %.1f min since last change (min %d min)",
                    time_since_last,
                    DHW_CONTROL_MIN_INTERVAL_MINUTES,
                )
                return

        # Apply control decision
        if decision.should_heat and not is_lux_on:
            # Turn ON temporary lux to boost DHW
            _LOGGER.info(
                "DHW control: Activating temporary lux - %s (DHW: %.1f°C, DM: %.0f)",
                decision.priority_reason,
                current_dhw_temp,
                thermal_debt,
            )
            try:
                await self.hass.services.async_call(
                    "switch",
                    "turn_on",
                    {"entity_id": temp_lux_entity},
                    blocking=False,
                )
                self._last_dhw_control_time = now_time
            except Exception as err:
                _LOGGER.error("Failed to turn on temporary lux: %s", err)

        elif not decision.should_heat and is_lux_on:
            # Turn OFF temporary lux to block/stop DHW
            _LOGGER.info(
                "DHW control: Deactivating temporary lux - %s (DHW: %.1f°C, DM: %.0f)",
                decision.priority_reason,
                current_dhw_temp,
                thermal_debt,
            )
            try:
                await self.hass.services.async_call(
                    "switch",
                    "turn_off",
                    {"entity_id": temp_lux_entity},
                    blocking=False,
                )
                self._last_dhw_control_time = now_time
            except Exception as err:
                _LOGGER.error("Failed to turn off temporary lux: %s", err)
        else:
            # No change needed
            _LOGGER.debug(
                "DHW control: No change needed (should_heat=%s, lux_on=%s, reason=%s)",
                decision.should_heat,
                is_lux_on,
                decision.priority_reason,
            )

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

    async def async_update_config(self, new_options: dict[str, Any]) -> None:
        """Update configuration without full reload.

        Allows hot-reload of runtime options like target temperature,
        thermal mass, and DHW settings without restarting the integration.

        Args:
            new_options: Dictionary of updated option values
        """
        _LOGGER.debug("Updating configuration: %s", new_options)

        # Update thermal model parameters
        if "thermal_mass" in new_options:
            self.engine.thermal.thermal_mass = new_options["thermal_mass"]
            _LOGGER.debug("Updated thermal mass: %.2f", new_options["thermal_mass"])

        if "insulation_quality" in new_options:
            self.engine.thermal.insulation_quality = new_options["insulation_quality"]
            _LOGGER.debug("Updated insulation quality: %.2f", new_options["insulation_quality"])

        # Update DHW settings
        dhw_config_changed = False
        if "dhw_morning_hour" in new_options or "dhw_morning_enabled" in new_options:
            dhw_config_changed = True
        if "dhw_evening_hour" in new_options or "dhw_evening_enabled" in new_options:
            dhw_config_changed = True

        if dhw_config_changed:
            # Rebuild DHW demand periods
            from .optimization.dhw_optimizer import DHWDemandPeriod

            demand_periods = []
            if new_options.get("dhw_morning_enabled", True):
                morning_hour = new_options.get("dhw_morning_hour", 7)
                demand_periods.append(
                    DHWDemandPeriod(
                        start_hour=morning_hour,
                        target_temp=55.0,
                        duration_hours=2,
                    )
                )
            if new_options.get("dhw_evening_enabled", True):
                evening_hour = new_options.get("dhw_evening_hour", 18)
                demand_periods.append(
                    DHWDemandPeriod(
                        start_hour=evening_hour,
                        target_temp=55.0,
                        duration_hours=3,
                    )
                )
            self.dhw_optimizer.demand_periods = demand_periods
            _LOGGER.debug("Updated DHW demand periods: %d periods", len(demand_periods))

        # Trigger immediate refresh with new settings
        await self.async_request_refresh()

        _LOGGER.info("Configuration updated without reload")

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
