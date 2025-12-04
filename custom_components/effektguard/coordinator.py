"""Data update coordinator for EffektGuard."""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_point_in_time
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
    DEFAULT_HEAT_PUMP_POWER_KW,
    DEFAULT_INDOOR_TEMP,
    DHW_CONTROL_MIN_INTERVAL_MINUTES,
    DHW_COOLING_RATE,
    DHW_READY_THRESHOLD,
    DHW_SAFETY_MIN,
    DM_DHW_ABORT_FALLBACK,
    DM_DHW_BLOCK_FALLBACK,
    DM_THRESHOLD_AUX_LIMIT,
    DM_THRESHOLD_START,
    DOMAIN,
    MIN_DHW_TARGET_TEMP,
    NIBE_DHW_START_THRESHOLD,
    SPACE_HEATING_DEMAND_HIGH_THRESHOLD,
    SPACE_HEATING_DEMAND_LOW_THRESHOLD,
    SPACE_HEATING_DEMAND_MODERATE_THRESHOLD,
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
from .optimization.decision_engine import LayerDecision, OptimizationDecision
from .optimization.thermal_predictor import ThermalStatePredictor
from .optimization.weather_learning import WeatherPatternLearner
from .utils.compressor_monitor import CompressorHealthMonitor

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
        # Pass climate zone info for seasonal-aware defaults
        climate_zone_info = decision_engine.climate_detector.zone_info
        self.weather_learner = WeatherPatternLearner(climate_zone_info=climate_zone_info)
        self.climate_region = self._detect_climate_region(hass)

        # Compressor health monitoring (Oct 19, 2025)
        self.compressor_monitor = CompressorHealthMonitor(max_history_hours=24)
        self.compressor_stats = None  # Latest CompressorStats from monitor

        # DHW optimizer - pass climate detector for climate-aware thresholds
        from .optimization.dhw_optimizer import DHWDemandPeriod, IntelligentDHWScheduler
        from .optimization.savings_calculator import SavingsCalculator

        # Get user-configured DHW target temperature (default 50°C)
        from .const import DEFAULT_DHW_TARGET_TEMP

        dhw_target_temp = float(entry.options.get("dhw_target_temp", DEFAULT_DHW_TARGET_TEMP))

        # Configure DHW demand periods from options
        demand_periods = []

        # Morning demand period (e.g., shower time)
        if entry.options.get("dhw_morning_enabled", True):
            morning_hour = int(entry.options.get("dhw_morning_hour", 7))
            demand_periods.append(
                DHWDemandPeriod(
                    start_hour=morning_hour,
                    target_temp=dhw_target_temp,  # User-configurable target
                    duration_hours=2,  # 2-hour window
                )
            )

        # Evening demand period (e.g., dishes, evening shower)
        if entry.options.get("dhw_evening_enabled", True):
            evening_hour = int(entry.options.get("dhw_evening_hour", 18))
            demand_periods.append(
                DHWDemandPeriod(
                    start_hour=evening_hour,
                    target_temp=dhw_target_temp,  # User-configurable target
                    duration_hours=3,  # 3-hour window
                )
            )

        # Pass climate detector from decision engine to DHW optimizer for dynamic thresholds
        self.dhw_optimizer = IntelligentDHWScheduler(
            demand_periods=demand_periods,
            climate_detector=decision_engine.climate_detector,
            user_target_temp=dhw_target_temp,
        )

        # Savings calculator
        self.savings_calculator = SavingsCalculator()

        # Airflow optimizer for exhaust air heat pumps (F750/F730)
        # Only active for exhaust air models - improves COP by enhanced ventilation
        from .optimization.airflow_optimizer import AirflowOptimizer
        from .const import (
            AIRFLOW_DEFAULT_ENHANCED,
            AIRFLOW_DEFAULT_STANDARD,
            CONF_AIRFLOW_ENHANCED_RATE,
            CONF_AIRFLOW_STANDARD_RATE,
            CONF_ENABLE_AIRFLOW_OPTIMIZATION,
        )

        airflow_enabled = entry.options.get(
            CONF_ENABLE_AIRFLOW_OPTIMIZATION,
            entry.data.get(CONF_ENABLE_AIRFLOW_OPTIMIZATION, False),
        )

        if airflow_enabled:
            flow_standard = float(
                entry.options.get(
                    CONF_AIRFLOW_STANDARD_RATE,
                    entry.data.get(CONF_AIRFLOW_STANDARD_RATE, AIRFLOW_DEFAULT_STANDARD),
                )
            )
            flow_enhanced = float(
                entry.options.get(
                    CONF_AIRFLOW_ENHANCED_RATE,
                    entry.data.get(CONF_AIRFLOW_ENHANCED_RATE, AIRFLOW_DEFAULT_ENHANCED),
                )
            )
            self.airflow_optimizer = AirflowOptimizer(
                flow_standard=flow_standard,
                flow_enhanced=flow_enhanced,
            )
            _LOGGER.info(
                "Airflow optimizer enabled: standard %.0f m³/h, enhanced %.0f m³/h",
                flow_standard,
                flow_enhanced,
            )
        else:
            self.airflow_optimizer = None
            _LOGGER.debug("Airflow optimizer disabled")

        # Track airflow enhancement state for minimum duration enforcement
        self._airflow_enhance_start: datetime | None = None

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
            except (AttributeError, TypeError, ValueError) as err:
                _LOGGER.debug("Could not format DHW periods: %s", err)

        # Learning storage
        self.learning_store = Store(hass, STORAGE_VERSION, STORAGE_KEY_LEARNING)

        # State tracking
        self.current_offset: float = 0.0
        self.last_applied_offset: float | None = None  # Last offset written to NIBE
        self.last_offset_timestamp: datetime | None = None  # When offset was last applied
        self.peak_today: float = 0.0
        self.peak_this_month: float = 0.0
        self.last_decision_time = None
        self._learned_data_changed = False  # Track if learning data needs saving
        self._last_predictor_save: datetime | None = None  # Track last thermal predictor save
        self._predictor_save_interval = timedelta(
            minutes=UPDATE_INTERVAL_MINUTES
        )  # Throttle to coordinator update interval

        # Peak tracking metadata (for sensor attributes)
        self.peak_today_time: datetime | None = None  # When today's peak occurred
        self.peak_today_source: str = "unknown"  # external_meter, nibe_currents, estimate
        self.peak_today_quarter: int | None = None  # 15-min quarter (0-95) for effect tariff
        self.yesterday_peak: float = 0.0  # Yesterday's peak for comparison

        # DHW tracking
        self.last_dhw_heated = None  # Last time DHW was in heating mode
        self.last_dhw_temp = None  # Last BT7 temperature for trend analysis
        self.dhw_heating_start = None  # When current/last DHW cycle started
        self.dhw_heating_end = None  # When last DHW cycle ended
        self.dhw_was_heating = False  # Track state changes

        # Spot price savings tracking (per-cycle accumulation)
        self._daily_spot_savings: float = 0.0  # Accumulates during day, recorded at midnight

        # Startup tracking - gracefully handle missing entities during HA startup
        # MyUplink integration can take 45-50 seconds to initialize entities
        self._first_successful_update = False
        self._startup_grace_period = True  # Skip first action to allow sensors to stabilize
        self._clock_aligned = False  # Wait for whole hour to align updates to :00, :05, :10...

        # Power sensor availability tracking (event-driven)
        # Event listener detects when external power sensor becomes available during startup
        # Listener unsubscribes after detection to avoid overhead
        self._power_sensor_available = False
        self._power_sensor_listener = None

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

        except (AttributeError, KeyError, ValueError) as err:
            _LOGGER.warning("Failed to detect climate region: %s, defaulting to central", err)
            return CLIMATE_CENTRAL_SWEDEN

    def _calculate_next_aligned_time(self) -> datetime:
        """Calculate next 5-minute boundary + 10 seconds.

        Returns:
            Next aligned datetime (e.g., 17:50:10, 17:55:10, 18:00:10)
        """
        now = dt_util.now()
        minutes_past = now.minute % UPDATE_INTERVAL_MINUTES
        seconds_past = now.second

        if minutes_past == 0 and seconds_past < 10:
            # Within current boundary, before :10
            seconds_to_next = 10 - seconds_past
        else:
            # Schedule for next boundary + 10 seconds
            minutes_to_next = UPDATE_INTERVAL_MINUTES - minutes_past
            seconds_to_next = (minutes_to_next * 60) - seconds_past + 10

        return now + timedelta(seconds=seconds_to_next)

    def _schedule_aligned_refresh(self) -> None:
        """Schedule next update at aligned time (bypasses base class drift).

        Updates are aligned to :XX:10 (10 seconds past each 5-minute mark).
        This gives sensors time to update before we read them, and aligns
        with 15-minute spot price intervals.
        """
        # Cancel any existing schedule
        if self._unsub_refresh:
            self._unsub_refresh()
            self._unsub_refresh = None

        next_time = self._calculate_next_aligned_time()

        @callback
        def _on_refresh(_now: datetime) -> None:
            self.hass.async_create_task(self._do_aligned_refresh())

        self._unsub_refresh = async_track_point_in_time(self.hass, _on_refresh, next_time)

        if not self._clock_aligned:
            self._clock_aligned = True
            _LOGGER.info(
                "Clock aligned to %s (updates every %d min at :XX:10)",
                next_time.strftime("%H:%M:%S"),
                UPDATE_INTERVAL_MINUTES,
            )
        else:
            _LOGGER.debug("Next update at %s", next_time.strftime("%H:%M:%S"))

    async def _do_aligned_refresh(self) -> None:
        """Perform refresh and schedule next aligned update."""
        try:
            self.data = await self._async_update_data()
            self.last_update_success = True
            self.async_set_updated_data(self.data)
        except Exception as err:  # noqa: BLE001
            self.last_update_success = False
            _LOGGER.error("Update failed: %s", err)

        # Schedule next aligned refresh (maintains alignment regardless of update duration)
        self._schedule_aligned_refresh()

    async def async_initialize_learning(self) -> None:
        """Initialize learning modules by loading persisted data.

        Called once during coordinator setup to restore learned parameters
        from previous sessions.
        """
        _LOGGER.debug("Initializing learning modules...")

        try:
            learned_data = await self._load_learned_data()

            if learned_data:
                # Restore last applied offset to avoid redundant API calls
                if "last_offset" in learned_data:
                    offset_data = learned_data["last_offset"]
                    self.last_applied_offset = offset_data.get("value")
                    timestamp_str = offset_data.get("timestamp")
                    if timestamp_str:
                        self.last_offset_timestamp = datetime.fromisoformat(timestamp_str)
                    _LOGGER.info(
                        "Restored last offset: %.1f°C from %s",
                        self.last_applied_offset,
                        timestamp_str or "unknown",
                    )

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

                # Restore thermal predictor (temperature trends)
                if "thermal_predictor" in learned_data:
                    self.thermal_predictor = ThermalStatePredictor.from_dict(
                        learned_data["thermal_predictor"]
                    )
                    _LOGGER.info(
                        "Restored thermal predictor with %d historical snapshots",
                        len(self.thermal_predictor.state_history),
                    )

                # Restore weather patterns
                if "weather_patterns" in learned_data:
                    self.weather_learner.from_dict(learned_data["weather_patterns"])
                    summary = self.weather_learner.get_pattern_database_summary()
                    _LOGGER.info(
                        "Restored weather patterns: %d weeks of data",
                        summary.get("total_weeks", 0),
                    )

                # Restore DHW optimizer state (critical for Legionella safety tracking)
                if "dhw_state" in learned_data and self.dhw_optimizer:
                    dhw_state = learned_data["dhw_state"]
                    if "last_legionella_boost" in dhw_state:
                        self.dhw_optimizer.last_legionella_boost = datetime.fromisoformat(
                            dhw_state["last_legionella_boost"]
                        )
                        _LOGGER.info(
                            "Restored last Legionella boost: %s",
                            self.dhw_optimizer.last_legionella_boost,
                        )

                # Initialize DHW history from Home Assistant recorder (resilience to restarts)
                # This checks past 14 days of BT7 data to detect recent Legionella cycles
                # even if the system was restarted after a high-temp cycle
                if self.dhw_optimizer and self.nibe:
                    # Ensure NIBE entities are discovered first
                    if not self.nibe._entity_cache:
                        await self.nibe._discover_nibe_entities()

                    bt7_entity = self.nibe._entity_cache.get("dhw_top_temp")
                    if bt7_entity:
                        await self.dhw_optimizer.initialize_from_history(self.hass, bt7_entity)
                    else:
                        _LOGGER.debug("BT7 sensor not available - skipping history initialization")

                _LOGGER.info("Learning modules initialized successfully")
            else:
                _LOGGER.info("No learned data found - starting fresh learning")

        except (OSError, ValueError, KeyError, AttributeError) as err:
            _LOGGER.warning("Failed to initialize learning modules: %s", err)
            # Continue with fresh learning

    async def async_restore_peaks(self) -> None:
        """Restore peak tracking values from effect manager's loaded data.

        Called after effect manager has loaded its persistent state to sync
        coordinator's peak values with stored data.
        """
        try:
            peak_summary = self.effect.get_monthly_peak_summary()
            if peak_summary and peak_summary.get("count", 0) > 0:
                self.peak_this_month = peak_summary["highest"]
                _LOGGER.info(
                    "Restored monthly peak: %.2f kW (%d peaks loaded)",
                    self.peak_this_month,
                    peak_summary["count"],
                )
            else:
                _LOGGER.info("No monthly peaks to restore - starting fresh")
        except (AttributeError, KeyError, ValueError) as err:
            _LOGGER.warning("Failed to restore peaks: %s", err)
            self.peak_this_month = 0.0

    def setup_power_sensor_listener(self) -> None:
        """Set up event listener to detect when external power sensor becomes available.

        Uses Home Assistant's event bus to react immediately when the power sensor
        state changes from unknown/unavailable to a valid value. This provides
        instant detection during startup without polling overhead.

        The listener automatically unsubscribes after detecting availability,
        so there's no ongoing event processing overhead.
        """
        # Check if we have an external power sensor configured
        if not hasattr(self.nibe, "_power_sensor_entity") or not self.nibe._power_sensor_entity:
            _LOGGER.debug("No external power sensor configured - skipping availability listener")
            return

        power_entity_id = self.nibe._power_sensor_entity

        # Check if sensor is already available (immediate check)
        power_state = self.hass.states.get(power_entity_id)
        if power_state and power_state.state not in ["unknown", "unavailable"]:
            self._power_sensor_available = True
            _LOGGER.info("External power sensor %s already available at startup", power_entity_id)
            return

        @callback
        def power_sensor_state_changed(event):
            """Handle power sensor state change event."""
            # Filter for this specific entity ID
            if event.data.get("entity_id") != power_entity_id:
                return

            new_state = event.data.get("new_state")
            # Filter out events without new_state (deletions)
            if new_state is None:
                return

            if new_state.state not in ["unknown", "unavailable"]:
                if not self._power_sensor_available:
                    _LOGGER.info(
                        "External power sensor %s is now available (state: %s)",
                        power_entity_id,
                        new_state.state,
                    )
                    self._power_sensor_available = True

                    # Unsubscribe - we don't need this listener anymore
                    if self._power_sensor_listener:
                        self._power_sensor_listener()
                        self._power_sensor_listener = None
                        _LOGGER.debug("Power sensor availability listener unsubscribed")

                    # Trigger immediate coordinator refresh to start using the sensor
                    self.hass.async_create_task(self.async_request_refresh())

        # Subscribe to state_changed events (no event_filter parameter)
        self._power_sensor_listener = self.hass.bus.async_listen(
            "state_changed",
            power_sensor_state_changed,
        )

        _LOGGER.debug(
            "Listening for external power sensor %s availability (current state: %s)",
            power_entity_id,
            power_state.state if power_state else "None",
        )

    async def async_shutdown(self) -> None:
        """Clean shutdown of coordinator.

        Saves all persistent state before unload:
        - Learning module data (thermal model, weather patterns)
        - Effect tracking state (monthly peaks)

        Called during integration unload or reload.
        """
        _LOGGER.debug("Shutting down EffektGuard coordinator")

        try:
            # Unsubscribe power sensor listener if still active
            if self._power_sensor_listener:
                self._power_sensor_listener()
                self._power_sensor_listener = None
                _LOGGER.debug("Power sensor availability listener unsubscribed")

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

        except (OSError, RuntimeError, ValueError) as err:
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

            # Update compressor health monitoring (Oct 19, 2025)
            # Context-aware monitoring (Oct 23, 2025): DHW vs space heating
            if nibe_data.compressor_hz is not None:
                # Determine heating mode for context-aware compressor monitoring
                # DHW heating runs at higher Hz (50°C target) than space heating (25-35°C)
                heating_mode = (
                    "dhw"
                    if hasattr(nibe_data, "is_hot_water") and nibe_data.is_hot_water
                    else "space"
                )

                self.compressor_stats = self.compressor_monitor.update(
                    nibe_data.compressor_hz, nibe_data.timestamp, heating_mode
                )
                # Log compressor diagnostics at debug level
                if self.compressor_stats:
                    risk_level, risk_reason = self.compressor_monitor.assess_risk(
                        self.compressor_stats
                    )
                    _LOGGER.debug(
                        "Compressor: %d Hz (1h avg: %.0f, 6h avg: %.0f, mode: %s) - %s: %s",
                        self.compressor_stats.current_hz,
                        self.compressor_stats.avg_1h,
                        self.compressor_stats.avg_6h,
                        heating_mode,
                        risk_level,
                        risk_reason,
                    )
            else:
                _LOGGER.debug("Compressor Hz not available from NIBE adapter")

        except (UpdateFailed, AttributeError, KeyError, ValueError, TypeError) as err:
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
        except (AttributeError, KeyError, ValueError, TypeError) as err:
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
        except (AttributeError, KeyError, ValueError, TypeError) as err:
            _LOGGER.info("Weather forecast unavailable: %s", err)
            weather_data = None

        # Run optimization decision engine
        is_grace_period = False

        # Check if optimization is enabled (master switch)
        if not self.entry.data.get("enable_optimization", True):
            _LOGGER.info("Optimization disabled by user - maintaining neutral offset")
            decision = OptimizationDecision(
                offset=0.0,
                reasoning="Optimization disabled by user",
                layers=[],
            )
        else:
            try:
                # Validate peak tracking data
                if self.peak_today is None or self.peak_today < 0:
                    _LOGGER.error(
                        "Peak tracking error: peak_today is %s (should have actual power measurement). "
                        "This indicates power sensor is not configured or unavailable. "
                        "Peak protection will be disabled until sensors are available.",
                        self.peak_today,
                    )
                    current_power_for_decision = 0.0  # Disable peak protection
                else:
                    current_power_for_decision = self.peak_today

                decision = await self.hass.async_add_executor_job(
                    self.engine.calculate_decision,
                    nibe_data,
                    price_data,
                    weather_data,
                    self.peak_this_month,  # Monthly peak threshold to protect
                    current_power_for_decision,  # Current whole-house power consumption
                )

                # Startup grace period: Skip first action to allow sensors/trends to stabilize
                if self._startup_grace_period and self._first_successful_update:
                    is_grace_period = True
                    _LOGGER.info(
                        "Startup grace period: Observing only. "
                        "Real decision would have been: %.2f°C (%s)",
                        decision.offset,
                        decision.reasoning,
                    )
                    # Don't force offset to 0.0 - let the calculated value stand for reporting
                    decision.reasoning = f"[Startup Grace Period] {decision.reasoning}"
                    self._startup_grace_period = False

                _LOGGER.info(
                    "Decision: offset %.2f°C, reasoning: %s",
                    decision.offset,
                    decision.reasoning,
                )
            except (AttributeError, KeyError, ValueError, TypeError, ZeroDivisionError) as err:
                _LOGGER.error("Optimization failed: %s", err)
                # Fall back to safe operation (no offset)
                decision = self._get_safe_default_decision()

        # Update current state
        self.current_offset = decision.offset
        self.last_decision_time = dt_util.utcnow()

        # Apply offset to NIBE heat pump via MyUplink integration
        # This sends the calculated offset to the MyUplink number entity (parameter 47011)
        # Rate limiting (5 min) handled in nibe_adapter to prevent excessive API calls
        #
        # Accumulation logic: We track fractional offsets but only write to NIBE when
        # the integer part changes. This prevents oscillation when calculated offsets
        # hover around boundaries (e.g., 0.48 ↔ 0.52 both stay at 0°C in NIBE).
        if is_grace_period:
            _LOGGER.info("Skipping offset application during startup grace period")
        elif self.last_applied_offset is not None and int(decision.offset) == int(
            self.last_applied_offset
        ):
            _LOGGER.debug(
                "Offset %.2f°C → int(%d°C) matches last applied int(%d°C), skipping adapter call",
                decision.offset,
                int(decision.offset),
                int(self.last_applied_offset),
            )
        else:
            try:
                was_applied = await self.nibe.set_curve_offset(decision.offset)
                if was_applied:
                    _LOGGER.info("Applied offset %.2f°C to NIBE via MyUplink", decision.offset)
                    # Track successfully applied offset to avoid redundant API calls on restart
                    self.last_applied_offset = decision.offset
                    self.last_offset_timestamp = dt_util.utcnow()
                    self._learned_data_changed = True  # Trigger save on shutdown
                else:
                    _LOGGER.debug(
                        "Offset %.2f°C unchanged (NIBE already at %d°C)",
                        decision.offset,
                        int(decision.offset),
                    )
            except (AttributeError, OSError, ValueError) as err:
                _LOGGER.error("Failed to apply offset to NIBE: %s", err)
                # Continue anyway - next cycle will retry

        # Calculate actual spot savings for this cycle using real NIBE power
        now_time = dt_util.now()
        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        if (
            price_data
            and hasattr(price_data, "today")
            and price_data.today
            and nibe_data
            and nibe_data.power_kw is not None
        ):
            prices_today = [q.price for q in price_data.today]
            if prices_today and current_quarter < len(price_data.today):
                average_price = sum(prices_today) / len(prices_today)
                current_price = price_data.today[current_quarter].price

                # Calculate savings using ACTUAL power consumption
                cycle_savings = self.savings_calculator.calculate_spot_savings_per_cycle(
                    actual_power_kw=nibe_data.power_kw,
                    current_price=current_price,
                    average_price_today=average_price,
                    cycle_minutes=UPDATE_INTERVAL_MINUTES,
                )
                self._daily_spot_savings += cycle_savings

        # Check for day change and save yesterday's peak
        now = dt_util.now()
        if not hasattr(self, "_last_update_date"):
            self._last_update_date = now.date()

        if now.date() != self._last_update_date:
            # New day detected - save yesterday's peak and reset
            self.yesterday_peak = self.peak_today
            _LOGGER.info(
                "Day change detected: Yesterday's peak was %.2f kW, resetting daily peak",
                self.yesterday_peak,
            )

            # Record yesterday's spot savings to the calculator
            if self._daily_spot_savings != 0.0:
                self.savings_calculator.record_spot_savings(self._daily_spot_savings)
                _LOGGER.info(
                    "Recorded daily spot savings: %.2f SEK",
                    self._daily_spot_savings,
                )
            self._daily_spot_savings = 0.0  # Reset for new day

            # Reset daily peak for new day
            self.peak_today = 0.0
            self.peak_today_time = None
            self.peak_today_source = "unknown"
            self.peak_today_quarter = None
            self._last_update_date = now.date()

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
        # Use Home Assistant timezone-aware helper to avoid naive datetimes
        now_time = dt_util.now()
        current_quarter = (now_time.hour * 4) + (now_time.minute // 15)
        current_classification = self.engine.price.get_current_classification(current_quarter)

        # Calculate estimated savings
        # Get average price today for comparison
        average_price_today = 0.0
        current_price = 0.0
        if price_data and hasattr(price_data, "today") and price_data.today:
            prices_today = [q.price for q in price_data.today]
            if prices_today:
                average_price_today = sum(prices_today) / len(prices_today)
            # Get current price
            if current_quarter < len(price_data.today):
                current_price = price_data.today[current_quarter].price

        # Calculate savings estimate
        savings_estimate = self.savings_calculator.estimate_monthly_savings(
            current_peak_kw=self.peak_this_month,
            baseline_peak_kw=self.savings_calculator.baseline_monthly_peak,
            average_spot_savings_per_day=self.savings_calculator.average_daily_spot_savings,
        )

        _LOGGER.debug(
            "Estimated savings: %.0f SEK/month (effect: %.0f SEK, spot: %.0f SEK)",
            savings_estimate.monthly_estimate,
            savings_estimate.effect_savings,
            savings_estimate.spot_savings,
        )

        # Calculate DHW status and tracking
        dhw_status = "not_configured"
        dhw_next_boost = None
        dhw_last_heated = self.last_dhw_heated
        dhw_recommendation = "DHW sensor not configured"
        dhw_planning_summary = "DHW sensor not configured"
        dhw_planning_details = {}

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
            elif current_dhw_temp < DHW_SAFETY_MIN:
                dhw_status = "low"  # Below safety minimum, waiting to heat
            elif current_dhw_temp < MIN_DHW_TARGET_TEMP:
                dhw_status = "pending"  # Below target, will heat soon
            elif current_dhw_temp < DHW_READY_THRESHOLD:
                dhw_status = "ready"  # At normal target
            else:
                dhw_status = "hot"  # Above normal (high demand met or Legionella cycle)

            # Track temperature for trend analysis
            self.last_dhw_temp = current_dhw_temp

            # Track previous Legionella boost time before update
            previous_legionella_boost = self.dhw_optimizer.last_legionella_boost

            # Update DHW optimizer with temperature history
            self.dhw_optimizer.update_bt7_temperature(current_dhw_temp, now_time)

            # If Legionella boost was newly detected, save state immediately
            if (
                self.dhw_optimizer.last_legionella_boost != previous_legionella_boost
                and self.dhw_optimizer.last_legionella_boost is not None
            ):
                _LOGGER.info(
                    "Legionella boost detected - saving DHW state to persist across reboots"
                )
                await self._save_dhw_state_immediate()

            # Get DHW recommendation from optimizer with detailed planning
            # Initialize defaults first to prevent UnboundLocalError if exception occurs
            dhw_result = {
                "recommendation": "DHW calculation pending",
                "summary": "Calculating DHW planning",
                "details": {},
                "decision": None,
            }

            try:
                dhw_result = await self._calculate_dhw_recommendation(
                    nibe_data, price_data, weather_data, current_dhw_temp, now_time
                )
                dhw_recommendation = dhw_result["recommendation"]
                dhw_planning_summary = dhw_result["summary"]
                dhw_planning_details = dhw_result["details"]

                # Use the optimizer's recommended start time (timezone-aware from GE-Spot)
                dhw_next_boost = dhw_planning_details.get("recommended_start_time")
            except (AttributeError, KeyError, ValueError, TypeError, ZeroDivisionError) as e:
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
            if self.entry.data.get("enable_hot_water_optimization", False) and dhw_result.get(
                "decision"
            ):
                await self._apply_dhw_control(
                    dhw_result["decision"],  # Reuse decision from recommendation
                    current_dhw_temp,
                    now_time,
                )
        else:
            # DHW sensor not available - provide basic recommendation
            if nibe_data:
                _LOGGER.warning(
                    "DHW sensor (BT7) not found - check MyUplink integration has exposed BT7/40013 sensor"
                )
                dhw_recommendation = "DHW sensor not found - check MyUplink integration"
                dhw_planning_summary = "DHW sensor not found"
                dhw_planning_details = {}
            else:
                _LOGGER.warning("NIBE data not available")
                dhw_recommendation = "NIBE data unavailable"
                dhw_planning_summary = "NIBE data unavailable"
                dhw_planning_details = {}

        # Get temperature trend from thermal predictor
        temperature_trend_data = self.thermal_predictor.get_current_trend()
        outdoor_trend_data = self.thermal_predictor.get_outdoor_trend()

        # Evaluate and control airflow for exhaust air heat pumps (F750/F730)
        airflow_decision = None
        if self.airflow_optimizer and nibe_data:
            try:
                # Get target temperature from config
                target_temp = float(
                    self.entry.options.get(
                        "target_indoor_temp",
                        self.entry.data.get("target_indoor_temp", DEFAULT_INDOOR_TEMP),
                    )
                )

                airflow_decision = self.airflow_optimizer.evaluate_from_nibe(
                    nibe_data=nibe_data,
                    target_temp=target_temp,
                    thermal_trend=temperature_trend_data,
                )

                # Automatic ventilation control based on decision
                await self._apply_airflow_decision(airflow_decision)

            except (AttributeError, ValueError, TypeError) as err:
                _LOGGER.warning("Airflow evaluation failed: %s", err)
                # Continue without airflow optimization

        # Start aligned scheduling on first update (replaces base class scheduling)
        if not self._clock_aligned:
            self._schedule_aligned_refresh()

        return {
            "nibe": nibe_data,
            "price": price_data,
            "weather": weather_data,
            "thermal": self.engine.thermal,  # Thermal model for predictions
            "thermal_trend": temperature_trend_data,  # Temperature trend from predictor
            "outdoor_trend": outdoor_trend_data,  # Outdoor temperature trend
            "decision": decision,
            "offset": decision.offset,
            "peak_today": self.peak_today,
            "peak_this_month": self.peak_this_month,
            "current_quarter": current_quarter,
            "current_classification": current_classification,
            "savings": savings_estimate,  # Estimated monthly savings
            "compressor_stats": self.compressor_stats,  # Compressor Hz monitoring (Oct 19, 2025)
            "dhw_status": dhw_status,
            "dhw_next_boost": dhw_next_boost,
            "dhw_last_heated": dhw_last_heated,
            "dhw_heating_start": self.dhw_heating_start,
            "dhw_heating_end": self.dhw_heating_end,
            "dhw_recommendation": dhw_recommendation,
            "dhw_planning_summary": dhw_planning_summary,  # Human-readable summary
            "dhw_planning": dhw_planning_details,  # Detailed machine-readable data
            "airflow_decision": airflow_decision,  # Airflow enhancement decision (F750/F730)
        }

    async def _calculate_dhw_recommendation(
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

        # Get thermal debt
        thermal_debt = nibe_data.degree_minutes if nibe_data else DM_THRESHOLD_START

        # Calculate space heating demand - use actual power sensor reading
        # This is the REAL current heating demand, not an estimate
        space_heating_demand = (
            nibe_data.power_kw if nibe_data and nibe_data.power_kw is not None else 0.0
        )

        # Get outdoor temp
        outdoor_temp = nibe_data.outdoor_temp if nibe_data else 0.0

        # Get indoor temperature and target
        indoor_temp = nibe_data.indoor_temp if nibe_data else DEFAULT_INDOOR_TEMP
        target_indoor = self.entry.data.get("indoor_temp", DEFAULT_INDOOR_TEMP)
        indoor_deficit = max(0.0, target_indoor - indoor_temp)

        # Get indoor temperature trend for predictive DHW blocking
        thermal_trend = self.thermal_predictor.get_current_trend() if self.thermal_predictor else {}
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # NEW: Block DHW if indoor cooling rapidly AND below target
        # Prevents DHW from causing comfort issues when house already struggling
        if indoor_deficit > 0.3 and trend_rate < -0.3:
            # House is below target AND cooling rapidly
            # DHW would make this worse - block it
            planning_summary = (
                f"⚠️ DHW Blocked - Space Heating Priority\n"
                f"Indoor: {indoor_temp:.1f}°C (target {target_indoor:.1f}°C)\n"
                f"Trend: Cooling {abs(trend_rate):.2f}°C/hour\n"
                f"Reason: Prevent further indoor temperature drop"
            )

            recommendation = (
                f"Block DHW - Indoor temp falling rapidly ({trend_rate:.2f}°C/h), "
                f"{indoor_deficit:.1f}°C below target. Prioritize space heating."
            )

            return {
                "recommendation": recommendation,
                "summary": planning_summary,
                "details": {
                    "should_heat": False,
                    "priority_reason": "INDOOR_COOLING_RAPIDLY",
                    "current_temperature": current_dhw_temp,
                    "target_temperature": 50.0,
                    "indoor_temp": indoor_temp,
                    "indoor_trend": trend_rate,
                    "indoor_deficit": indoor_deficit,
                },
            }

        # Get climate zone DM thresholds from climate detector
        # Climate detector is REQUIRED - don't use arbitrary fallback values
        if self.engine.climate_detector:
            dm_range = self.engine.climate_detector.get_expected_dm_range(outdoor_temp)
            dm_thresholds = {
                "block": dm_range["warning"],  # Block DHW at warning threshold
                "abort": dm_range["critical"],  # Abort DHW at critical threshold (always -1500)
            }
            climate_zone = self.engine.climate_detector.zone_info
        else:
            # Climate detector not initialized - use balanced fallback thresholds
            # -340/-500 provides reasonable DHW operation while maintaining safety margin
            _LOGGER.warning(
                "Climate detector not initialized - using balanced fallback DM thresholds "
                "(block: %d, abort: %d). Configure latitude in EffektGuard settings for "
                "accurate climate-aware thresholds.",
                DM_DHW_BLOCK_FALLBACK,
                DM_DHW_ABORT_FALLBACK,
            )
            dm_thresholds = {
                "block": DM_DHW_BLOCK_FALLBACK,  # -340: Never start DHW below this
                "abort": DM_DHW_ABORT_FALLBACK,  # -500: Abort DHW if reached during run
            }
            climate_zone = None

            # Show persistent notification in UI
            self.hass.components.persistent_notification.async_create(
                "EffektGuard could not detect your climate zone. "
                "Using balanced thermal debt thresholds. "
                "Configure latitude in integration settings for optimal climate-aware operation.",
                title="EffektGuard Configuration Recommended",
                notification_id="effektguard_climate_detection_missing",
            )  # Get hours since last DHW heating for max wait check
        hours_since_last = await self._calculate_hours_since_last_dhw()

        # Get price periods for window-based scheduling
        price_periods = []
        if price_data:
            price_periods = price_data.today + price_data.tomorrow

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
            price_periods=price_periods,
            hours_since_last_dhw=hours_since_last,
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
            "thermal_debt_status": self._get_thermal_debt_status(thermal_debt, dm_thresholds),
            "space_heating_demand_kw": round(space_heating_demand, 2),
            "current_price_classification": price_classification,
            "outdoor_temperature": outdoor_temp,
            "indoor_temperature": indoor_temp,
            "climate_zone": climate_zone.name if climate_zone else "Unknown",
            "recommended_start_time": decision.recommended_start_time,  # From optimizer
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

        else:
            # Should heat - give specific recommendation
            if decision.priority_reason == "DHW_SAFETY_MINIMUM":
                recommendation = (
                    f"Heat now - Safety minimum ({current_dhw_temp:.1f}°C < {DHW_SAFETY_MIN}°C)"
                )
            elif decision.priority_reason == "CHEAP_ELECTRICITY_OPPORTUNITY":
                recommendation = f"Heat now - Cheap electricity ({price_classification})"
            elif decision.priority_reason.startswith("URGENT_DEMAND"):
                recommendation = "Heat now - Demand period approaching"
            elif decision.priority_reason.startswith("OPTIMAL_PREHEAT"):
                recommendation = f"Heat now - Pre-heating for demand ({price_classification})"
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
            weather_opportunity=planning_details.get("weather_opportunity"),
        )

        # Return combined result with both machine-readable and human-readable data
        return {
            "recommendation": recommendation,
            "summary": planning_summary,
            "details": planning_details,
            "decision": decision,  # Include raw decision for control logic
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
        if space_heating_demand > SPACE_HEATING_DEMAND_MODERATE_THRESHOLD:
            lines.append(f"Heating Demand: HIGH ({space_heating_demand:.1f} kW)")
        elif space_heating_demand > SPACE_HEATING_DEMAND_LOW_THRESHOLD:
            lines.append(f"Heating Demand: MODERATE ({space_heating_demand:.1f} kW)")
        else:
            lines.append(f"Heating Demand: LOW ({space_heating_demand:.1f} kW)")

        # Weather opportunity
        if weather_opportunity:
            lines.append(f"Weather: {weather_opportunity}")

        lines.append("")
        lines.append(f"Recommendation: {recommendation}")

        return "\n".join(lines)

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

    async def _get_last_dhw_heating_time(self) -> datetime | None:
        """Get timestamp when temporary lux was last activated.

        Reads from MyUplink entity's last_changed attribute (source of truth).
        Falls back to Home Assistant history API if entity is currently OFF.

        Returns:
            datetime: Last time DHW heating was activated (UTC)
            None: If no history available
        """
        from homeassistant.const import STATE_ON

        from .const import CONF_NIBE_TEMP_LUX_ENTITY

        temp_lux_entity = self.entry.data.get(CONF_NIBE_TEMP_LUX_ENTITY)
        if not temp_lux_entity:
            _LOGGER.debug("No temporary lux entity configured")
            return None

        # Try to read from current entity state
        state = self.hass.states.get(temp_lux_entity)
        if state:
            if state.state == STATE_ON:
                # Currently ON - last_changed is when it turned ON
                _LOGGER.debug("DHW temp lux currently ON, last_changed: %s", state.last_changed)
                return state.last_changed

            # Entity is OFF - check history for last ON state
            try:
                from homeassistant.components import recorder

                # Look back 48 hours
                start_time = dt_util.utcnow() - timedelta(hours=48)
                end_time = dt_util.utcnow()

                # Get recorder instance and use its executor
                rec = recorder.get_instance(self.hass)
                if rec is None:
                    _LOGGER.warning("Recorder not available, cannot check DHW history")
                    return None

                # Get state history using recorder instance executor
                states = await rec.async_add_executor_job(
                    recorder.history.state_changes_during_period,
                    self.hass,
                    start_time,
                    end_time,
                    temp_lux_entity,
                )

                if temp_lux_entity in states:
                    entity_states = states[temp_lux_entity]

                    # Find most recent ON state
                    for state_obj in reversed(entity_states):
                        if state_obj.state == STATE_ON:
                            _LOGGER.debug(
                                "Last DHW heating from history: %s", state_obj.last_changed
                            )
                            return state_obj.last_changed

                _LOGGER.debug("No ON state in history for %s", temp_lux_entity)

            except (AttributeError, KeyError, ValueError, OSError) as err:
                _LOGGER.error("Failed to read DHW heating history: %s", err)

        else:
            _LOGGER.warning("Temporary lux entity %s not found", temp_lux_entity)

        # Fall back to stored value (if any)
        if hasattr(self, "_last_dhw_control_time") and self._last_dhw_control_time:
            _LOGGER.debug("Using fallback stored DHW time: %s", self._last_dhw_control_time)
            return self._last_dhw_control_time

        return None

    def _check_dhw_abort_conditions(
        self,
        abort_conditions: list[str],
        thermal_debt: float,
        indoor_temp: float,
        target_indoor: float,
    ) -> tuple[bool, str | None]:
        """Check if any DHW abort conditions are triggered.

        Abort conditions are returned by DHW optimizer to monitor during active heating.
        If triggered, we should stop DHW heating early to prioritize space heating.

        Args:
            abort_conditions: List of condition strings from DHW decision
                Examples: ["thermal_debt < -500", "indoor_temp < 21.5"]
            thermal_debt: Current degree minutes (DM) value
            indoor_temp: Current indoor temperature
            target_indoor: Target indoor temperature

        Returns:
            Tuple of (should_abort, reason_str)
            - should_abort: True if any condition triggered
            - reason_str: Human-readable abort reason or None
        """
        if not abort_conditions:
            return False, None

        for condition in abort_conditions:
            # Parse and evaluate "thermal_debt < THRESHOLD" condition
            if "thermal_debt <" in condition:
                try:
                    threshold = float(condition.split("<")[1].strip())
                    if thermal_debt < threshold:
                        return True, f"Thermal debt {thermal_debt:.0f} < {threshold:.0f}"
                except (ValueError, IndexError) as err:
                    _LOGGER.warning("Failed to parse abort condition '%s': %s", condition, err)
                    continue

            # Parse and evaluate "indoor_temp < THRESHOLD" condition
            elif "indoor_temp <" in condition:
                try:
                    threshold = float(condition.split("<")[1].strip())
                    if indoor_temp < threshold:
                        return True, f"Indoor {indoor_temp:.1f}°C < {threshold:.1f}°C"
                except (ValueError, IndexError) as err:
                    _LOGGER.warning("Failed to parse abort condition '%s': %s", condition, err)
                    continue

        return False, None

    async def _calculate_hours_since_last_dhw(self) -> float:
        """Calculate hours since last DHW heating.

        Returns:
            float: Hours since last heating (0.0+), or 24.0 if unknown
        """
        last_time = await self._get_last_dhw_heating_time()

        if last_time is None:
            _LOGGER.debug("Unknown DHW heating history - assuming 24 hours for safety")
            return 24.0  # Safe assumption

        now = dt_util.utcnow()
        hours = (now - last_time).total_seconds() / 3600

        return max(0.0, hours)  # Never negative

    async def _apply_airflow_decision(self, decision) -> None:
        """Apply automatic ventilation control based on airflow optimizer decision.

        Controls NIBE enhanced ventilation switch for exhaust air heat pumps (F750/F730).
        When enhanced airflow is beneficial, turns on increased ventilation to extract
        more heat from exhaust air. When not beneficial, returns to normal.

        Based on thermodynamic calculations:
        - Enhanced airflow extracts more heat from exhaust air
        - COP improves ~20% with warmer evaporator
        - Trade-off is ventilation penalty (more cold air to heat)

        Automatic stop conditions:
        - Indoor temp reaches target
        - Indoor trend turns positive (> +0.1°C/h)
        - Compressor drops below threshold
        - Maximum duration reached
        - Outdoor temp drops below -15°C (penalty exceeds gains)

        Args:
            decision: FlowDecision from airflow optimizer
        """
        from .const import NIBE_VENTILATION_MIN_ENHANCED_DURATION

        # Check if enhanced ventilation is currently active
        is_enhanced = await self.nibe.is_enhanced_ventilation_active()

        if is_enhanced is None:
            _LOGGER.debug("Ventilation control unavailable - no ventilation switch found")
            return

        # Determine target state based on decision
        if decision.should_enhance:
            # Only turn on if not already enhanced
            if not is_enhanced:
                success = await self.nibe.set_enhanced_ventilation(True)
                if success:
                    # Track when we started enhanced mode
                    self._airflow_enhance_start = dt_util.utcnow()
                    _LOGGER.info(
                        "🌀 Ventilation ENHANCED: ON for %d min (+%.2f kW gain) - %s",
                        decision.duration_minutes,
                        decision.expected_gain_kw,
                        decision.reason,
                    )
            else:
                _LOGGER.debug(
                    "Ventilation already enhanced - %s",
                    decision.reason,
                )
        else:
            # Only reduce if currently enhanced and minimum duration passed
            if is_enhanced:
                # Check minimum enhanced duration to prevent rapid cycling
                if hasattr(self, "_airflow_enhance_start") and self._airflow_enhance_start:
                    elapsed_minutes = (
                        dt_util.utcnow() - self._airflow_enhance_start
                    ).total_seconds() / 60

                    if elapsed_minutes < NIBE_VENTILATION_MIN_ENHANCED_DURATION:
                        _LOGGER.debug(
                            "Ventilation: keeping enhanced for %d more min (min duration)",
                            int(NIBE_VENTILATION_MIN_ENHANCED_DURATION - elapsed_minutes),
                        )
                        return

                success = await self.nibe.set_enhanced_ventilation(False)
                if success:
                    self._airflow_enhance_start = None
                    _LOGGER.info(
                        "🌀 Ventilation NORMAL: OFF - %s",
                        decision.reason,
                    )
            else:
                _LOGGER.debug(
                    "Ventilation at normal - %s",
                    decision.reason,
                )

    async def _apply_dhw_control(
        self, decision, current_dhw_temp: float, now_time: datetime
    ) -> None:
        """Apply automatic DHW control based on optimizer decision.

        Controls NIBE temporary lux switch to heat or block DHW based on
        pre-calculated decision from optimizer.

        Args:
            decision: DHWScheduleDecision from optimizer (reused from recommendation)
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

        # Use pre-calculated decision from _calculate_dhw_recommendation()
        # This avoids duplicate optimizer calls and log spam
        # Get thermal_debt and indoor_temp from coordinator data for abort conditions
        thermal_debt = (
            self.data.get("dhw_planning_details", {}).get("thermal_debt", 0.0)
            if self.last_update_success and hasattr(self, "data") and self.data is not None
            else 0.0
        )
        indoor_temp = (
            self.data.get("dhw_planning_details", {}).get("indoor_temperature", DEFAULT_INDOOR_TEMP)
            if self.last_update_success and hasattr(self, "data") and self.data is not None
            else DEFAULT_INDOOR_TEMP
        )
        target_indoor = self.entry.data.get("target_indoor_temp", DEFAULT_INDOOR_TEMP)

        # ABORT MONITORING: If DHW is currently heating, check abort conditions
        # This allows us to stop DHW early if conditions deteriorate (thermal debt, indoor temp)
        # thermal_debt and indoor_temp from coordinator data (same as used in decision calculation)
        if is_lux_on and decision.abort_conditions:
            should_abort, abort_reason = self._check_dhw_abort_conditions(
                decision.abort_conditions,
                thermal_debt,
                indoor_temp,
                target_indoor,
            )

            if should_abort:
                _LOGGER.warning(
                    "DHW heating aborted early: %s. Stopping DHW to prioritize space heating.",
                    abort_reason,
                )
                try:
                    await self.hass.services.async_call(
                        "switch",
                        "turn_off",
                        {"entity_id": temp_lux_entity},
                        blocking=False,
                    )
                    self._last_dhw_control_time = now_time
                except (AttributeError, OSError, ValueError) as err:
                    _LOGGER.error("Failed to abort DHW heating: %s", err)
                return  # Exit early - abort handled

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
            except (AttributeError, OSError, ValueError) as err:
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
            except (AttributeError, OSError, ValueError) as err:
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
        base_date = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for quarter in range(96):  # 96 quarters per day (15-min intervals)
            hour = quarter // 4
            minute = (quarter % 4) * 15
            start_time = base_date.replace(hour=hour, minute=minute)
            fallback_periods.append(QuarterPeriod(start_time=start_time, price=1.0))

        return PriceData(
            today=fallback_periods,
            tomorrow=[],
            has_tomorrow=False,
        )

    def _get_safe_default_decision(self):
        """Get safe default decision when optimization fails.

        Returns zero offset to maintain current operation without changes.
        """
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

        Handles grid import meters with solar/battery offset by using smart fallback:
        - If meter shows low reading BUT heat pump is running significantly,
          use estimated power instead (solar export may offset grid import)
        - Only record peaks above minimum threshold to avoid standby/noise
        """
        try:
            # Check if we have external power sensor (whole house)
            has_external_power_sensor = hasattr(self.nibe, "_power_sensor_entity") and bool(
                self.nibe._power_sensor_entity
            )

            # PRIORITY 1: External power meter (whole house including NIBE)
            # This is MOST IMPORTANT for peak billing - measures total house consumption
            # Used for: Monthly peak tracking (effect tariff billing)
            current_power = None
            if has_external_power_sensor:
                power_entity_id = self.nibe._power_sensor_entity
                power_state = self.hass.states.get(power_entity_id)

                # Only attempt to use external sensor if it's available
                # Availability is tracked via event listener for fast startup detection
                if power_state and power_state.state not in ["unknown", "unavailable"]:
                    try:
                        # Power sensor typically in watts, convert to kW
                        current_power = float(power_state.state) / 1000
                        _LOGGER.debug(
                            "📊 External power meter (whole house): %.3f kW from %s",
                            current_power,
                            power_entity_id,
                        )
                        # Mark sensor as available (in case event listener hasn't fired yet)
                        if not self._power_sensor_available:
                            self._power_sensor_available = True
                            _LOGGER.debug("External power sensor marked as available")
                    except (ValueError, TypeError) as e:
                        _LOGGER.warning(
                            "Failed to read power sensor %s (state: %s): %s",
                            power_entity_id,
                            power_state.state,
                            e,
                        )
                elif not self._power_sensor_available:
                    # Sensor still not available - skip peak tracking this cycle
                    # Event listener will trigger refresh when sensor becomes available
                    _LOGGER.debug(
                        "External power sensor %s not yet available (state: %s) - "
                        "skipping peak tracking (listener active: %s)",
                        power_entity_id,
                        power_state.state if power_state else "None",
                        self._power_sensor_listener is not None,
                    )
                    return  # Exit early, event listener will trigger refresh when ready

            # PRIORITY 2: NIBE phase currents (NIBE heat pump only - for reference/debugging)
            # Calculates real NIBE power from BE1/BE2/BE3 current sensors
            # Used when: No external meter available
            # Note: Only measures NIBE, not whole house (less useful for peak billing)
            if current_power is None and nibe_data.phase1_current is not None:
                current_power = self.nibe.calculate_power_from_currents(
                    nibe_data.phase1_current,
                    nibe_data.phase2_current,
                    nibe_data.phase3_current,
                )
                if current_power is not None:
                    _LOGGER.debug(
                        "⚡ NIBE power from phase currents: %.3f kW "
                        "(L1=%.1fA, L2=%.1fA, L3=%.1fA)",
                        current_power,
                        nibe_data.phase1_current,
                        nibe_data.phase2_current or 0.0,
                        nibe_data.phase3_current or 0.0,
                    )

            # PRIORITY 3: Estimate from compressor Hz (NOT FOR PEAK TRACKING!)
            # Only used for display/debugging when no real measurements available
            # WARNING: Never record estimated peaks - billing must use real measurements only
            if current_power is None and nibe_data.compressor_hz:
                current_power = self._estimate_power_from_compressor(
                    nibe_data.compressor_hz, nibe_data.outdoor_temp
                )
                _LOGGER.debug(
                    "⚙️  Power estimated from compressor: %.2f kW (%d Hz, %.1f°C outdoor) "
                    "[ESTIMATE ONLY - not used for peak billing]",
                    current_power,
                    nibe_data.compressor_hz,
                    nibe_data.outdoor_temp,
                )

            # PRIORITY 4: Last resort estimation (NOT FOR PEAK TRACKING!)
            # Only for display when nothing else available
            # WARNING: Never record estimated peaks
            if current_power is None:
                current_power = self._estimate_power_consumption(nibe_data)
                _LOGGER.warning(
                    "⚠️  Power estimation fallback: %.2f kW (no real data available) "
                    "[ESTIMATE ONLY - not used for peak billing]",
                    current_power,
                )

            # SMART FALLBACK for grid import meters with solar/battery
            # If meter shows unexpectedly low reading but compressor running significantly,
            # the meter likely shows NET import (actual consumption minus solar export)
            # In this case, use calculated/estimated heat pump power for peak tracking
            if has_external_power_sensor and current_power < 0.5:
                # Check if heat pump is actually working hard
                compressor_hz = getattr(nibe_data, "compressor_frequency", 0)
                is_heating = getattr(nibe_data, "is_heating", False)

                if is_heating and compressor_hz > 20:
                    # Compressor running significantly but meter shows low reading
                    # This indicates solar/battery offsetting grid import
                    estimated_power = self._estimate_power_from_compressor(nibe_data)

                    if estimated_power > 1.0:  # Estimated power seems reasonable
                        _LOGGER.info(
                            "Smart fallback: Using estimated power %.2f kW "
                            "(meter shows %.2f kW - likely solar/battery offset, compressor: %d Hz)",
                            estimated_power,
                            current_power,
                            compressor_hz,
                        )
                        current_power = estimated_power

            # Determine measurement source for metadata
            measurement_source = "unknown"
            if has_external_power_sensor and current_power is not None:
                # Check if this was from external meter or smart fallback
                if has_external_power_sensor and current_power >= 0.5:
                    measurement_source = "external_meter"
                elif nibe_data.phase1_current is not None:
                    measurement_source = "nibe_currents"
                else:
                    measurement_source = "estimate"
            elif nibe_data.phase1_current is not None:
                measurement_source = "nibe_currents"
            else:
                measurement_source = "estimate"

            # Get current timestamp for peak tracking
            now = dt_util.now()
            quarter_of_day = (now.hour * 4) + (now.minute // 15)  # 0-95

            # Update daily peak (always track for display, even if estimated)
            if current_power > self.peak_today:
                self.peak_today = current_power
                self.peak_today_time = now
                self.peak_today_source = measurement_source
                self.peak_today_quarter = quarter_of_day

                _LOGGER.info(
                    "New daily peak: %.2f kW at %s (quarter %d, source: %s)",
                    current_power,
                    now.strftime("%H:%M:%S"),
                    quarter_of_day,
                    measurement_source,
                )

            # CRITICAL: Only record monthly peaks with REAL measurements
            # Monthly peak billing requires accurate whole-house power measurement
            # We MUST have either:
            #   1. External whole-house power meter (best for billing) OR
            #   2. NIBE phase currents (accurate NIBE-only, but missing other house loads)
            # Estimates are NEVER used for monthly peak tracking - billing must be accurate
            has_real_measurement = has_external_power_sensor or nibe_data.phase1_current is not None
            if not has_real_measurement:
                _LOGGER.debug(
                    "Skipping monthly peak recording: No real power measurement available. "
                    "Current reading %.2f kW is estimated (not suitable for billing). "
                    "Configure external power meter for accurate peak tracking.",
                    current_power,
                )
                return

            # Update monthly peak through effect manager
            peak_event = await self.effect.record_quarter_measurement(
                power_kw=current_power,
                quarter=quarter_of_day,
                timestamp=now,
            )

            if peak_event:
                self.peak_this_month = peak_event.effective_power
                _LOGGER.info("New monthly peak: %.2f kW", self.peak_this_month)

        except (AttributeError, KeyError, ValueError, TypeError) as err:
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
            base_power = DEFAULT_HEAT_PUMP_POWER_KW  # kW baseline from const.py

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

    def _estimate_power_from_compressor(self, nibe_data) -> float:
        """Estimate heat pump power from compressor frequency and outdoor temperature.

        More accurate estimation when we know compressor is running.
        Used for smart fallback when grid meter shows solar/battery offset.

        Args:
            nibe_data: Current NIBE state with compressor frequency

        Returns:
            Estimated power consumption in kW
        """
        if not nibe_data:
            return 0.0

        compressor_hz = getattr(nibe_data, "compressor_frequency", 0)
        outdoor_temp = getattr(nibe_data, "outdoor_temp", 0.0)

        if compressor_hz == 0:
            return 0.1  # Standby

        # F750 power estimation based on compressor frequency and outdoor temp
        # Based on typical NIBE F750 performance curves:
        # - 20 Hz (minimum): ~1.5-2.0 kW
        # - 50 Hz (mid): ~3.5-4.5 kW
        # - 80 Hz (maximum): ~6.0-7.0 kW

        # Base power from frequency (linear approximation)
        # 20-80 Hz range maps to ~1.5-6.5 kW
        base_from_hz = 1.5 + (compressor_hz - 20) * (5.0 / 60)  # Linear interpolation
        base_from_hz = max(1.5, min(base_from_hz, 6.5))  # Clamp to reasonable range

        # Temperature adjustment (colder = more power needed for same output)
        if outdoor_temp < -15:
            temp_factor = 1.3
        elif outdoor_temp < -5:
            temp_factor = 1.2
        elif outdoor_temp < 0:
            temp_factor = 1.1
        else:
            temp_factor = 1.0

        estimated = base_from_hz * temp_factor

        _LOGGER.debug(
            "Power estimation: %d Hz at %.1f°C → %.2f kW (base: %.2f, temp_factor: %.2f)",
            compressor_hz,
            outdoor_temp,
            estimated,
            base_from_hz,
            temp_factor,
        )

        return estimated

    async def async_set_offset(self, offset: float) -> None:
        """Apply heating curve offset to NIBE system.

        Args:
            offset: Offset value in °C (-10 to +10)
        """
        try:
            await self.nibe.set_curve_offset(offset)
            self.current_offset = offset
            self.last_applied_offset = offset
            self.last_offset_timestamp = dt_util.utcnow()
            self._learned_data_changed = True  # Trigger save on shutdown
            _LOGGER.info("Applied offset: %.2f°C", offset)
        except (AttributeError, OSError, ValueError) as err:
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
            except (AttributeError, OSError, ValueError) as err:
                _LOGGER.error("Failed to reset offset: %s", err)

    async def async_update_config(self, new_options: dict[str, Any]) -> None:
        """Update configuration without full reload.

        Allows hot-reload of runtime options like target temperature,
        thermal mass, and DHW settings without restarting the integration.

        Args:
            new_options: Dictionary of updated option values
        """
        _LOGGER.debug("Updating configuration: %s", new_options)

        # Update decision engine cached configuration values
        # CRITICAL: Decision engine caches these at init, must update them here
        if "target_indoor_temp" in new_options:
            self.engine.target_temp = float(new_options["target_indoor_temp"])
            _LOGGER.debug("Updated target temperature: %.1f°C", self.engine.target_temp)

        if "tolerance" in new_options:
            self.engine.tolerance = float(new_options["tolerance"])
            self.engine.tolerance_range = self.engine.tolerance * 0.4  # Recalculate range
            _LOGGER.debug(
                "Updated tolerance: %.1f (range: %.1f°C)",
                self.engine.tolerance,
                self.engine.tolerance_range,
            )

        # Update thermal model parameters
        if "thermal_mass" in new_options:
            self.engine.thermal.thermal_mass = new_options["thermal_mass"]
            _LOGGER.debug("Updated thermal mass: %.2f", new_options["thermal_mass"])

        if "insulation_quality" in new_options:
            self.engine.thermal.insulation_quality = new_options["insulation_quality"]
            _LOGGER.debug("Updated insulation quality: %.2f", new_options["insulation_quality"])

        # Update optimization mode and recalculate mode config
        if "optimization_mode" in new_options:
            self.engine.config["optimization_mode"] = new_options["optimization_mode"]
            self.engine._update_mode_config()  # Recalculate mode-specific settings
            _LOGGER.debug("Updated optimization mode: %s", new_options["optimization_mode"])

        # Update control priority (note: stored in config dict, not cached)
        if "control_priority" in new_options:
            self.engine.config["control_priority"] = new_options["control_priority"]
            _LOGGER.debug("Updated control priority: %s", new_options["control_priority"])

        # Update switch states (stored in config dict, checked by layers)
        switch_keys = {
            "enable_optimization",
            "enable_peak_protection",
            "enable_price_optimization",
            "enable_weather_prediction",
            "enable_hot_water_optimization",
        }
        for key in switch_keys:
            if key in new_options:
                self.engine.config[key] = new_options[key]
                _LOGGER.debug("Updated switch %s: %s", key, new_options[key])

        # Update peak protection margin (note: stored in config dict, not cached)
        if "peak_protection_margin" in new_options:
            self.engine.config["peak_protection_margin"] = new_options["peak_protection_margin"]
            _LOGGER.debug(
                "Updated peak protection margin: %.2f", new_options["peak_protection_margin"]
            )

        # Update DHW settings
        dhw_config_changed = False
        dhw_keys = {
            "dhw_morning_hour",
            "dhw_morning_enabled",
            "dhw_evening_hour",
            "dhw_evening_enabled",
            "dhw_target_temp",
        }

        if any(key in new_options for key in dhw_keys):
            dhw_config_changed = True

        if dhw_config_changed:
            # Rebuild DHW demand periods
            from .optimization.dhw_optimizer import DHWDemandPeriod
            from .const import DEFAULT_DHW_TARGET_TEMP

            dhw_target = float(new_options.get("dhw_target_temp", DEFAULT_DHW_TARGET_TEMP))
            demand_periods = []

            if new_options.get("dhw_morning_enabled", True):
                morning_hour = int(new_options.get("dhw_morning_hour", 7))
                demand_periods.append(
                    DHWDemandPeriod(
                        start_hour=morning_hour,
                        target_temp=dhw_target,
                        duration_hours=2,
                    )
                )

            if new_options.get("dhw_evening_enabled", True):
                evening_hour = int(new_options.get("dhw_evening_hour", 18))
                demand_periods.append(
                    DHWDemandPeriod(
                        start_hour=evening_hour,
                        target_temp=dhw_target,
                        duration_hours=3,
                    )
                )

            self.dhw_optimizer.demand_periods = demand_periods
            # Update user_target_temp to match the new configuration
            self.dhw_optimizer.user_target_temp = dhw_target
            _LOGGER.debug(
                "Updated DHW demand periods: %d periods (target: %.1f°C)",
                len(demand_periods),
                dhw_target,
            )

        # Configuration is now updated in the engine's internal state
        # Next coordinator update cycle (≤5 min) will use these new values
        # No need for immediate refresh - that causes UI flicker and sensor unavailability

        _LOGGER.info("Configuration updated without reload - changes apply on next cycle")

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

            # Save thermal predictor state immediately (throttled to UPDATE_INTERVAL_MINUTES)
            # This ensures temperature trends persist across reboots
            await self._save_thermal_predictor_immediate()

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

        except (AttributeError, KeyError, ValueError, TypeError) as err:
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

            # Save last applied offset to avoid redundant API calls on restart
            if self.last_applied_offset is not None:
                learned_data["last_offset"] = {
                    "value": self.last_applied_offset,
                    "timestamp": (
                        self.last_offset_timestamp.isoformat()
                        if self.last_offset_timestamp
                        else None
                    ),
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

            # Save thermal predictor state (full state_history for trend analysis)
            if thermal_predictor:
                learned_data["thermal_predictor"] = thermal_predictor.to_dict()

            # Save weather patterns
            if weather_learner:
                learned_data["weather_patterns"] = weather_learner.to_dict()
                summary = weather_learner.get_pattern_database_summary()
                learned_data["weather_summary"] = summary

            # Save DHW optimizer state (critical for Legionella safety and max wait tracking)
            if self.dhw_optimizer:
                dhw_state = {}
                if self.dhw_optimizer.last_legionella_boost:
                    dhw_state["last_legionella_boost"] = (
                        self.dhw_optimizer.last_legionella_boost.isoformat()
                    )
                # Note: last_dhw_heating_time tracked by thermal debt tracker, not optimizer
                if dhw_state:
                    learned_data["dhw_state"] = dhw_state

            await self.learning_store.async_save(learned_data)
            _LOGGER.debug("Saved learned data to storage")

        except (OSError, ValueError, KeyError, AttributeError) as err:
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

        except (OSError, ValueError, KeyError) as err:
            _LOGGER.warning("Failed to load learned data: %s", err)
            return None

    async def _save_thermal_predictor_immediate(self) -> None:
        """Save thermal predictor state immediately (throttled to avoid excessive disk writes).

        Called after each temperature trend update to persist state_history across reboots.
        Uses throttling to limit disk writes to UPDATE_INTERVAL_MINUTES (same as coordinator updates).
        """
        now = dt_util.utcnow()

        # Throttle saves to UPDATE_INTERVAL_MINUTES to avoid excessive disk I/O
        if self._last_predictor_save is not None:
            time_since_last_save = (now - self._last_predictor_save).total_seconds()
            if time_since_last_save < self._predictor_save_interval.total_seconds():
                _LOGGER.debug(
                    "Skipping thermal predictor save - throttled (%.0fs since last save)",
                    time_since_last_save,
                )
                return

        try:
            # Load existing learning data to preserve other modules
            existing_data = await self.learning_store.async_load() or {}

            # Update only thermal predictor data
            if self.thermal_predictor:
                existing_data["thermal_predictor"] = self.thermal_predictor.to_dict()
                existing_data["last_updated"] = now.isoformat()
                existing_data["version"] = existing_data.get("version", STORAGE_VERSION)

                await self.learning_store.async_save(existing_data)
                self._last_predictor_save = now

                _LOGGER.debug(
                    "Saved thermal predictor state: %d snapshots, %.1f°C/h trend",
                    len(self.thermal_predictor.state_history),
                    self.thermal_predictor.get_current_trend().get("rate_per_hour", 0.0),
                )
            else:
                _LOGGER.debug("No thermal predictor to save")

        except (OSError, ValueError, KeyError, AttributeError) as err:
            _LOGGER.warning("Failed to save thermal predictor state: %s", err)
            # Don't raise - continue operation even if save fails

    async def _save_dhw_state_immediate(self) -> None:
        """Save DHW optimizer state immediately (Legionella boost tracking).

        Called when Legionella boost is detected to persist across reboots.
        Critical for safety - ensures we don't lose track of last boost time.
        """
        try:
            # Load existing learning data to preserve other modules
            existing_data = await self.learning_store.async_load() or {}

            # Update DHW state
            if self.dhw_optimizer and self.dhw_optimizer.last_legionella_boost:
                existing_data["dhw_state"] = {
                    "last_legionella_boost": self.dhw_optimizer.last_legionella_boost.isoformat()
                }
                existing_data["last_updated"] = dt_util.utcnow().isoformat()
                existing_data["version"] = existing_data.get("version", STORAGE_VERSION)

                await self.learning_store.async_save(existing_data)

                _LOGGER.debug(
                    "Saved DHW state: last Legionella boost at %s",
                    self.dhw_optimizer.last_legionella_boost,
                )
            else:
                _LOGGER.debug("No DHW state to save")

        except (OSError, ValueError, KeyError, AttributeError) as err:
            _LOGGER.warning("Failed to save DHW state: %s", err)
            # Don't raise - continue operation even if save fails
