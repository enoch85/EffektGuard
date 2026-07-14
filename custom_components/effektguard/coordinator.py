"""Data update coordinator for EffektGuard."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from homeassistant.components.persistent_notification import async_create
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_ON
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.event import async_track_point_in_time
from homeassistant.helpers.storage import Store
from homeassistant.helpers.issue_registry import (
    IssueSeverity,
    async_create_issue,
    async_delete_issue,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    DHW_CONTROL_ISSUE_ID,
    PRICE_SOURCE_ISSUE_ID,
    AIRFLOW_DEFAULT_ENHANCED,
    AIRFLOW_DEFAULT_STANDARD,
    CONF_AIRFLOW_ENHANCED_RATE,
    CONF_AIRFLOW_STANDARD_RATE,
    CONF_DHW_MIN_AMOUNT,
    CONF_ENABLE_AIRFLOW_OPTIMIZATION,
    CONF_ENABLE_OPTIMIZATION,
    CONF_HEAT_PUMP_MODEL,
    CONF_NIBE_TEMP_LUX_ENTITY,
    DEFAULT_DHW_EVENING_HOUR,
    DEFAULT_DHW_MORNING_HOUR,
    DEFAULT_DHW_TARGET_TEMP,
    DEFAULT_HEAT_PUMP_MODEL,
    DEFAULT_INDOOR_TEMP,
    DHW_CONTROL_MIN_INTERVAL_MINUTES,
    DHW_MIN_AMOUNT_DEFAULT,
    DHW_READY_THRESHOLD,
    DHW_SAFETY_MIN,
    DHW_WEATHER_COOLDOWN_MINUTES,
    DM_THRESHOLD_START,
    DOMAIN,
    MAX_BILLING_OBSERVATION_GAP_MINUTES,
    PEAK_CONTROL_POWER_SOURCES,
    LEARNING_OBSERVATION_INTERVAL_MINUTES,
    MIN_DHW_TARGET_TEMP,
    NIBE_VENTILATION_MIN_ENHANCED_DURATION,
    NIBE_VENTILATION_MIN_REST_DURATION,
    POWER_SOURCE_ESTIMATE,
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NIBE_CURRENTS,
    POWER_SOURCE_NONE,
    STORAGE_KEY_LEARNING,
    LEARNING_STORAGE_VERSION,
    TOLERANCE_RANGE_MULTIPLIER,
    STARTUP_GRACE_MIN_INTERVAL,
    STARTUP_MAX_GRACE_ATTEMPTS,
    STARTUP_GRACE_UPDATES,
    UPDATE_INTERVAL_MINUTES,
)
from .models.nibe import NibeF750Profile
from .models.registry import HeatPumpModelRegistry
from .optimization.adaptive_learning import AdaptiveThermalModel
from .optimization.airflow_optimizer import AirflowOptimizer
from .optimization.billing_period import BillingPeriodAccumulator
from .optimization.decision_engine import (
    OptimizationDecision,
    get_safe_default_decision,
)
from .optimization.dhw_optimizer import (
    DHWDemandPeriod,
    DHWRecommendation,
    IntelligentDHWScheduler,
)
from .optimization.prediction_layer import ThermalStatePredictor
from .optimization.savings_calculator import SavingsCalculator
from .optimization.weather_learning import WeatherPatternLearner
from .utils.compressor_monitor import CompressorHealthMonitor
from .utils.power import power_kw_from_state
from .utils.time_utils import get_current_billing_period
from .utils.volatile_helpers import OffsetVolatilityTracker

if TYPE_CHECKING:
    from .models.types import EffektGuardConfigDict

_LOGGER = logging.getLogger(__name__)


class EffektGuardCoordinator(DataUpdateCoordinator):
    """Coordinate data updates for EffektGuard.

    This coordinator orchestrates:
    - Data collection from NIBE, spot price, and weather
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
            # Hand HA the entry. Without it HA falls back to a deprecated ContextVar
            # (breaks_in_ha_version 2026.8), which is only set inside async_setup_entry - so
            # `self.config_entry` is None for a coordinator built anywhere else (audit F-073).
            config_entry=entry,
            name=DOMAIN,
            # Disable base class automatic scheduling - we use clock-aligned scheduling instead
            # This prevents drift from startup time and ensures updates at :00:10, :05:10, etc.
            update_interval=None,
        )
        self.nibe = nibe_adapter
        self.gespot = gespot_adapter
        self.weather = weather_adapter
        self.engine = decision_engine
        self.effect = effect_manager
        self.entry = entry

        # Load heat pump model profile from the single registry
        # (models self-register via decorator in models/nibe/)
        model_key = entry.data.get(CONF_HEAT_PUMP_MODEL, DEFAULT_HEAT_PUMP_MODEL)
        try:
            self.heat_pump_model = HeatPumpModelRegistry.get_model(model_key)
        except ValueError:
            _LOGGER.warning(
                "Unknown heat pump model '%s', falling back to %s - "
                "check the heat pump model setting",
                model_key,
                DEFAULT_HEAT_PUMP_MODEL,
            )
            self.heat_pump_model = NibeF750Profile()  # type: ignore[call-arg]

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

        # Compressor health monitoring (Oct 19, 2025)
        self.compressor_monitor = CompressorHealthMonitor(max_history_hours=24)
        # The monitor's verdict, fed to the decision engine. Its own risk ladder was computed and
        # written to a debug log; nothing consumed it, and the engine stayed free to demand +10
        # from a compressor already at maximum.
        self.compressor_risk: str | None = None
        self.compressor_stats = None  # Latest CompressorStats from monitor

        # DHW temporary lux entity (stored once, reused everywhere)
        self.temp_lux_entity = entry.data.get(CONF_NIBE_TEMP_LUX_ENTITY)

        # DHW optimizer - pass climate detector for climate-aware thresholds
        # Get user-configured DHW target temperature (default 50°C)
        dhw_target_temp = float(entry.options.get("dhw_target_temp", DEFAULT_DHW_TARGET_TEMP))

        # Get user-configured minimum hot water amount (default 5 minutes)
        dhw_min_amount = int(entry.options.get(CONF_DHW_MIN_AMOUNT, DHW_MIN_AMOUNT_DEFAULT))

        # Configure DHW demand periods from options
        demand_periods = []

        # Morning demand period (e.g., shower time)
        if entry.options.get("dhw_morning_enabled", True):
            morning_hour = int(entry.options.get("dhw_morning_hour", DEFAULT_DHW_MORNING_HOUR))
            demand_periods.append(
                DHWDemandPeriod(
                    availability_hour=morning_hour,
                    target_temp=dhw_target_temp,  # User-configurable target
                    duration_hours=2,  # 2-hour window
                    min_amount_minutes=dhw_min_amount,  # User-configurable min amount
                )
            )

        # Evening demand period (e.g., dishes, evening shower)
        if entry.options.get("dhw_evening_enabled", True):
            evening_hour = int(entry.options.get("dhw_evening_hour", DEFAULT_DHW_EVENING_HOUR))
            demand_periods.append(
                DHWDemandPeriod(
                    availability_hour=evening_hour,
                    target_temp=dhw_target_temp,  # User-configurable target
                    duration_hours=3,  # 3-hour window
                    min_amount_minutes=dhw_min_amount,  # User-configurable min amount
                )
            )

        # Pass climate detector, emergency layer, and price analyzer from decision engine
        # Phase 10: EmergencyLayer provides consistent thermal debt blocking logic
        # Phase 11: PriceAnalyzer provides shared price forecast and window search
        self.dhw_optimizer = IntelligentDHWScheduler(
            demand_periods=demand_periods,
            climate_detector=decision_engine.climate_detector,
            user_target_temp=dhw_target_temp,
            emergency_layer=decision_engine.emergency_layer,
            price_analyzer=decision_engine.price,
        )

        # Savings calculator
        self.savings_calculator = SavingsCalculator()

        # Airflow optimizer for exhaust air heat pumps (F750/F730)
        # Only created if model supports exhaust airflow optimization
        self.airflow_optimizer = None
        if self.heat_pump_model.supports_exhaust_airflow:
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
            _LOGGER.debug(
                "Airflow optimizer initialized: standard %.0f m³/h, enhanced %.0f m³/h",
                flow_standard,
                flow_enhanced,
            )
        else:
            _LOGGER.debug(
                "Airflow optimizer not available - model %s does not support exhaust airflow",
                self.heat_pump_model.model_name,
            )

        # Track airflow enhancement state for minimum duration enforcement
        self._airflow_enhance_start: datetime | None = None
        # How long the airflow optimizer asked this enhancement to run. It computes this
        # (15-60 min, by deficit) and it used to be logged and thrown away, so nothing bounded the
        # fan's cycling in either direction.
        self._airflow_enhance_minutes: int = NIBE_VENTILATION_MIN_ENHANCED_DURATION
        # When the fan last returned to normal, so it cannot be re-enhanced on the very next tick.
        self._airflow_normal_since: datetime | None = None

        if demand_periods:
            try:
                # Format DHW periods for logging (handle both real values and test mocks)
                formatted_periods = []
                for p in demand_periods:
                    try:
                        hour = int(p.availability_hour)
                        temp = float(p.target_temp)
                        formatted_periods.append(f"{hour:02d}:00 ({temp:.1f}°C)")
                    except (TypeError, ValueError):
                        # Fallback for mock objects in tests
                        formatted_periods.append(f"{p.availability_hour}:00 ({p.target_temp}°C)")

                _LOGGER.info("DHW demand periods configured: %s", formatted_periods)

                # Debug logging for type validation
                _LOGGER.debug(
                    "DHW periods configured: %s (types: %s)",
                    [f"{p.availability_hour}:00" for p in demand_periods],
                    [f"{type(p.availability_hour).__name__}" for p in demand_periods],
                )
            except (AttributeError, TypeError, ValueError) as err:
                _LOGGER.debug("Could not format DHW periods: %s", err)

        # Learning storage
        self.learning_store = Store(hass, LEARNING_STORAGE_VERSION, STORAGE_KEY_LEARNING)

        # State tracking
        self.current_offset: float = 0.0
        self.last_applied_offset: float | None = None  # Last offset written to NIBE
        self.last_offset_timestamp: datetime | None = None  # When offset was last applied
        # Daily high-water mark (display + diagnostics). Monotonically non-decreasing
        # until the midnight reset. NEVER pass this to the decision engine as
        # "current power" - see current_power_kw.
        self.peak_today: float = 0.0
        self.peak_this_month: float = 0.0
        # Instantaneous whole-house power (kW), refreshed every cycle by
        # _update_peak_tracking. None until the first successful measurement, in which
        # case peak protection stays disabled rather than acting on a guess.
        self.current_power_kw: float | None = None
        # Swedish quarter-hour tariffs bill the 15-minute MEAN power, not an
        # instantaneous sample: accumulate real measurements within the
        # What the effect tariff actually bills: the time-weighted mean power over a billing HOUR
        # (not the quarter-hour - a quarter-hour mean overstates the billed peak by up to fourfold).
        # The arithmetic lives in billing_period.py, once, and the simulator runs the same object -
        # it used to keep a second, different implementation, and validated that one instead.
        self._billing_period = BillingPeriodAccumulator()
        self.last_decision_time = None
        self._learned_data_changed = False  # Track if learning data needs saving
        self._last_learning_save: datetime | None = None  # Track last learned data save time
        self._last_predictor_save: datetime | None = None  # Track last thermal predictor save
        self._grace_period_ended_logged = False  # Track if grace period end was logged
        self._last_update_date = dt_util.now().date()  # Track date for daily resets
        self._last_dhw_control_time: datetime | None = None  # Track last DHW control action
        self._last_weather_record_date = dt_util.now().date()  # Track weather recording date
        self._predictor_save_interval = timedelta(
            minutes=UPDATE_INTERVAL_MINUTES
        )  # Throttle to coordinator update interval

        # Offset volatility tracking (prevents rapid back-and-forth offset changes)
        # Uses same min duration as price volatility (45 min) for consistency
        self._offset_volatility_tracker = OffsetVolatilityTracker()

        # Peak tracking metadata (for sensor attributes)
        self.peak_today_time: datetime | None = None  # When today's peak occurred
        self.peak_today_source: str = "unknown"  # external_meter, nibe_currents, estimate
        self.peak_today_period: int | None = None  # the billing HOUR (0-23) for the effect tariff
        self.yesterday_peak: float = 0.0  # Yesterday's peak for comparison

        # DHW tracking (unified: is_hot_water OR temp_lux active)
        self.last_dhw_heated = None  # Last time DHW was in heating mode
        self.last_dhw_temp = None  # Last BT7 temperature for trend analysis
        self.dhw_heating_start = None  # When current/last DHW cycle started
        self.dhw_heating_end = None  # When last DHW cycle ended
        self.dhw_was_active = False  # Track DHW state (is_hot_water OR temp_lux)
        # True while a temporary-lux boost that EFFEKTGUARD started is still running. The owner may
        # also start one from the heat pump's own panel or their own automation, and that one is
        # none of our business - so shutdown only cancels a boost we are responsible for.
        self._lux_boost_is_ours = False
        # While set, a boost_dhw SERVICE call owns the lux switch: ordinary price optimization
        # may not cancel it before this instant. Safety still may - see _apply_dhw_control.
        self._service_boost_until: datetime | None = None

        # Spot price savings tracking (per-cycle accumulation)
        self._daily_spot_savings: float = 0.0  # Accumulates during day, recorded at midnight

        # Startup tracking - gracefully handle missing entities during HA startup
        # MyUplink integration can take 45-50 seconds to initialize entities
        self._first_successful_update = False
        # Consecutive cycles spent waiting for the heat pump to appear. Bounded: see
        # STARTUP_MAX_GRACE_ATTEMPTS. Distinct from _startup_update_count below, which counts
        # observation cycles AFTER the pump is already answering.
        self._startup_grace_attempts = 0
        self._startup_update_count = 0  # Count updates before ending grace period
        self._startup_grace_updates = (
            STARTUP_GRACE_UPDATES  # Require N updates before active control
        )
        self._clock_aligned = False  # Wait for 5-min alignment (:00:10, :05:10, :10:10, etc.)
        self._unsub_aligned_refresh = None  # Clock-aligned refresh subscription
        # Startup grace: timeout after which observation cycles begin
        self._startup_grace_timeout = dt_util.now() + timedelta(seconds=STARTUP_GRACE_MIN_INTERVAL)

        # Whether the "no price source" repair issue is currently raised.
        self._price_issue_active = False
        self._dhw_issue_active = False

        # One writer at a time. See _drive_the_pump: the aligned control loop and a service that
        # commands the pump are both long coroutines, and asyncio interleaves them freely.
        self._control_lock = asyncio.Lock()

        # Power sensor availability tracking (event-driven)
        # Event listener detects when external power sensor becomes available during startup
        # Listener unsubscribes after detection to avoid overhead
        self._power_sensor_available = False
        # Learning observes hourly, not per control cycle - see _record_learning_observations
        self._last_learning_observation: datetime | None = None
        self._power_sensor_listener = None

    def _calculate_next_aligned_time(self) -> datetime:
        """Calculate next 5-minute boundary + 10 seconds.

        Aligns to 5-minute boundaries (:00:10, :05:10, :10:10, etc.)
        to ensure consistent timing relative to 15-minute spot price quarters.
        The +10 seconds gives sensors time to update before we read them.

        Returns:
            Next aligned datetime
        """
        now = dt_util.now()
        # Align to 5-minute boundaries (UPDATE_INTERVAL_MINUTES)
        minutes_past = now.minute % UPDATE_INTERVAL_MINUTES
        seconds_past = now.second

        if minutes_past == 0 and seconds_past < 10:
            # Within current 5-minute boundary, before :10
            seconds_to_next = 10 - seconds_past
        else:
            # Schedule for next 5-minute boundary + 10 seconds
            minutes_to_next = UPDATE_INTERVAL_MINUTES - minutes_past
            seconds_to_next = (minutes_to_next * 60) - seconds_past + 10

        return now + timedelta(seconds=seconds_to_next)

    def _schedule_aligned_refresh(self) -> None:
        """Schedule next update at aligned time (bypasses base class drift).

        Updates are aligned to :XX:10 (10 seconds past each 5-minute mark).
        This gives sensors time to update before we read them, and aligns
        with 15-minute spot price intervals.
        """
        # Never re-arm a coordinator that has been shut down.
        #
        # _do_aligned_refresh calls this from a `finally`, so an update already in flight
        # when the entry unloads would otherwise schedule a fresh timer on a dead object -
        # and the reload's new coordinator would arm its own. Two coordinators, one heat
        # pump, conflicting curve offsets, forever.
        if self._shutdown_requested:
            _LOGGER.debug("Coordinator shut down - not re-arming the aligned refresh")
            return

        # Cancel any existing schedule
        if self._unsub_aligned_refresh:
            self._unsub_aligned_refresh()
            self._unsub_aligned_refresh = None

        next_time = self._calculate_next_aligned_time()

        @callback
        def _on_refresh(_now: datetime) -> None:
            self.hass.async_create_task(self._do_aligned_refresh())

        self._unsub_aligned_refresh = async_track_point_in_time(self.hass, _on_refresh, next_time)

        if not self._clock_aligned:
            self._clock_aligned = True
            _LOGGER.info(
                "Clock aligned to %s (updates every 5 min at :00:10, :05:10, :10:10, etc.)",
                next_time.strftime("%H:%M:%S"),
            )
        else:
            _LOGGER.debug("Next update at %s", next_time.strftime("%H:%M:%S"))

    async def _do_aligned_refresh(self) -> None:
        """Perform one refresh and ALWAYS re-arm the next aligned update.

        This is the outermost frame of the coordinator's own scheduling loop, and it is the
        sole owner of the retry timer: the base class's scheduler is disabled
        (update_interval=None), so nothing else will ever re-arm it.

        That makes the broad `except Exception` correct here rather than sloppy. The
        previous except tuple was narrower than what the update path can actually raise -
        HomeAssistantError from a weather service call, IndexError from a price lookup on a
        DST 92/100-quarter day, ZeroDivisionError from the savings maths, numpy errors from
        the learning modules. Any one of those escaped, the task died, and
        _schedule_aligned_refresh() was never called again.

        The failure was silent and permanent: `last_update_success` stayed True, so every
        entity kept serving its last value and looked healthy, while the heat pump sat on
        the last offset written - until Home Assistant was restarted.

        The `finally` guarantees the loop survives any single bad cycle. Marking the update
        unsuccessful lets HA show the entities as unavailable, which is the honest signal.
        """
        try:
            # The one place the pump is driven on a schedule. `_drive_the_pump` holds the control
            # lock, so a service commanding the pump at the same moment waits its turn.
            self.data = await self._drive_the_pump()
            self.last_update_success = True
            self.async_set_updated_data(self.data)
        except UpdateFailed as err:
            # Expected degradation (e.g. required NIBE sensors unreadable).
            self.last_update_success = False
            _LOGGER.error("Update failed: %s", err)
        except Exception:  # noqa: BLE001 - supervisory loop; see docstring
            self.last_update_success = False
            _LOGGER.exception(
                "Unexpected error during EffektGuard update. The update loop will continue; "
                "entities are marked unavailable for this cycle and no offset was written."
            )
        finally:
            # ALWAYS re-arm. Without this the coordinator dies permanently on any
            # unhandled exception, because update_interval is None.
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
                # Restore last_applied_offset - try NIBE entity first (source of truth),
                # fall back to stored value if entity not available during startup
                offset_synced = False
                offset_entity = self.nibe.entity_cache.get("offset")
                if offset_entity:
                    state = self.hass.states.get(offset_entity)
                    if state and state.state not in ["unavailable", "unknown"]:
                        try:
                            self.last_applied_offset = float(int(float(state.state)))
                            _LOGGER.info(
                                "Synced with NIBE offset: %d°C", int(self.last_applied_offset)
                            )
                            offset_synced = True
                        except (ValueError, TypeError):
                            pass

                # Fall back to stored value if NIBE entity wasn't available
                if not offset_synced and "last_offset" in learned_data:
                    stored = learned_data["last_offset"].get("value")
                    if stored is not None:
                        self.last_applied_offset = float(int(stored))
                        _LOGGER.info(
                            "Restored offset from storage: %d°C (NIBE entity not yet available)",
                            int(self.last_applied_offset),
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
                    # Use the new restore method for all DHW state
                    self.dhw_optimizer.restore_from_persistence(dhw_state)

                # Initialize DHW history from Home Assistant recorder (resilience to restarts)
                # This checks past 14 days of BT7 data to detect recent Legionella cycles
                # even if the system was restarted after a high-temp cycle
                if self.dhw_optimizer and self.nibe:
                    # Ensure NIBE entities are discovered first
                    if not self.nibe.entity_cache:
                        await self.nibe.discover_entities()

                    bt7_entity = self.nibe.entity_cache.get("dhw_top_temp")
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
        if not self.nibe.power_sensor_entity:
            _LOGGER.debug("No external power sensor configured - skipping availability listener")
            return

        power_entity_id = self.nibe.power_sensor_entity

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

    async def _set_temporary_lux(self, on: bool) -> bool:
        """The ONE way this integration commands the hot-water boost, and the only place that records
        WHO STARTED IT - which is what lets `_cancel_our_dhw_boost` tell ours from the household's.

        Three call sites used to reach the switch directly and only one set `_lux_boost_is_ours`, so a
        boost our own service started was disowned on unload and left running to NIBE's lux timeout on
        the immersion heater.

        Starting from a shut-down coordinator is refused, as for the curve offset and the fan.
        STOPPING is not - that IS the cleanup, and it runs during shutdown.

        tests/unit/test_a_hot_water_boost_we_started_is_a_hot_water_boost_we_stop.py
        """
        if not self.temp_lux_entity:
            return False

        if on and self._shutdown_requested:
            _LOGGER.debug(
                "Coordinator is shut down - refusing to start a hot-water boost. The entry is "
                "unloaded; nothing would be left to stop it."
            )
            return False

        try:
            await self.hass.services.async_call(
                "switch",
                "turn_on" if on else "turn_off",
                {"entity_id": self.temp_lux_entity},
                blocking=True,
            )
        except (HomeAssistantError, AttributeError, OSError, ValueError) as err:
            _LOGGER.error("Failed to set temporary lux to %s: %s", on, err)
            return False

        # Ours if we switched it on; not ours once it is off, whoever asked for that.
        self._lux_boost_is_ours = on
        return True

    async def _cancel_our_dhw_boost(self) -> None:
        """Turn off a temporary-lux boost that EffektGuard started, if one is still running.

        Called on unload. A boost the OWNER started is left alone.
        """
        # Whatever happens below, the entry is going away - no service window survives it.
        self._service_boost_until = None
        if not (self._lux_boost_is_ours and self.temp_lux_entity):
            return

        state = self.hass.states.get(self.temp_lux_entity)
        if state is None or state.state != STATE_ON:
            self._lux_boost_is_ours = False
            return

        _LOGGER.info(
            "Cancelling the EffektGuard hot-water boost on %s before unload - it would otherwise "
            "run to NIBE's own timeout with nothing left to stop it",
            self.temp_lux_entity,
        )
        await self._set_temporary_lux(False)

    async def async_shutdown(self) -> None:
        """Clean shutdown of coordinator.

        Saves all persistent state before unload:
        - Learning module data (thermal model, weather patterns)
        - Effect tracking state (monthly peaks)

        Called during integration unload or reload.

        IDEMPOTENT BY DESIGN. This runs TWICE per unload: the base DataUpdateCoordinator
        registers `config_entry.async_on_unload(self.async_shutdown)` in its __init__, and
        async_unload_entry also calls it explicitly. Without the guard below, every unload
        saved the learning data and the effect peaks twice.
        """
        if self._shutdown_requested:
            _LOGGER.debug("Coordinator already shut down - ignoring repeat call")
            return

        _LOGGER.debug("Shutting down EffektGuard coordinator")

        # Base shutdown FIRST, and it is not optional. It sets `_shutdown_requested`,
        # cancels the base refresh handle, and shuts down the request debouncer.
        #
        # `_shutdown_requested` is what stops an in-flight refresh from RESURRECTING this
        # coordinator. `_do_aligned_refresh` runs on a task created with
        # hass.async_create_task (NOT entry.async_create_task), so HA cannot cancel it on
        # unload. Its `finally` block calls _schedule_aligned_refresh() - which, without
        # this flag, would re-arm a timer on a DEAD coordinator while the entry reload
        # creates a second, live one. BOTH would then write curve offsets to the same heat
        # pump, each with its own rate limiter and its own last_applied_offset, fighting
        # each other. Every reload would add another writer, permanently.
        #
        # The debouncer matters for the same reason: a trailing 10 s debounced refresh
        # queued by a service call can otherwise fire after unload and write an offset.
        await super().async_shutdown()

        try:
            # Unsubscribe aligned refresh timer (if active)
            unsub = getattr(self, "_unsub_aligned_refresh", None)
            if unsub is not None:
                unsub()
                self._unsub_aligned_refresh = None

            # Unsubscribe power sensor listener if still active
            if self._power_sensor_listener:
                self._power_sensor_listener()
                self._power_sensor_listener = None
                _LOGGER.debug("Power sensor availability listener unsubscribed")

            # CANCEL OUR OWN DHW BOOST. EffektGuard turns the temporary-lux switch ON to run a
            # high-temperature hot-water cycle, and it turned it OFF again on the next tick that
            # decided the cycle was done. But nothing turned it off on UNLOAD - so a reload, an
            # options change, or an HA restart in the middle of an EffektGuard-initiated boost left
            # the pump running that boost until NIBE's own timeout expired. A full high-temperature
            # DHW cycle, at the top of the tank where the immersion heater does the work, that
            # nobody asked for and nobody was left to stop.
            #
            # Only OUR boost. The owner may also start one from the heat pump's panel or their own
            # automation, and that one is none of our business - the DHW control path already says
            # so: "Stopping the lux boost cannot harm the pump - it only stops an
            # EffektGuard-initiated boost."
            await self._cancel_our_dhw_boost()

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

    async def _async_update_data(self) -> dict[str, object]:
        """Home Assistant's READ hook. Reads the world and decides. It NEVER writes.

        This is public, debounced, and called by anything that wants the coordinator refreshed: a
        Home Assistant reload, an options change, and services that have no business touching
        hardware. The heat-pump writes used to live in here, so `reset_peak_tracking` - a service
        whose entire job is to clear a stored counter - drove the pump.

        Writes belong to the control loop, and the control loop is `_do_aligned_refresh`: one
        owner, on the clock. Services that genuinely mean to command the pump call
        `async_refresh_and_apply`, and still take effect at once.
        """
        return await self._read_and_decide(apply=False)

    async def async_refresh_and_apply(self, *, explicit_command: bool = False) -> None:
        """Read, decide, and DRIVE THE PUMP. For services that genuinely command it.

        force_offset and boost_heating mean what they say and must land immediately, not at the
        next aligned tick. Bookkeeping services must NOT use this - they call
        `async_request_refresh()`, which reads and decides but writes nothing.
        """
        self.data = await self._drive_the_pump(explicit_command=explicit_command)
        self.async_set_updated_data(self.data)

    async def _drive_the_pump(self, *, explicit_command: bool = False) -> dict[str, object]:
        """The write path. Its sole owner, and the only place `apply=True` is passed.

        Two callers reach the pump - the aligned control loop every five minutes, and a service
        that explicitly commands it - and both are long coroutines that await at every step, so
        asyncio interleaves them freely. Without this lock:

            12:05:10  the aligned refresh reads the world and starts deciding
            12:05:11  force_offset(+3) sets the override, decides, and writes +3
            12:05:12  the aligned refresh - which snapshotted the engine BEFORE the override
                      existed - finishes and writes +0.5

        The forced offset is gone, overwritten by a decision that predates it. The same
        interleaving corrupts _apply_offset's rate limiting, which reads last_offset_timestamp
        and then writes it.

        Reads are deliberately NOT serialised: they touch no hardware, and blocking Home
        Assistant's refresh hook behind a write in progress would stall the entities for nothing.
        """
        async with self._control_lock:
            return await self._read_and_decide(
                apply=True,
                explicit_command=explicit_command,
            )

    def _report_no_price_source(self, reason: str) -> None:
        """Tell the user, in the UI, that price optimisation is not running.

        A _LOGGER.warning is not telling anyone. The user has `enable_price_optimization` switched
        on and believes the integration is trading on price; without a price source it simply is
        not, and nothing on screen says so (audit F-123).
        """
        if self._price_issue_active:
            return

        _LOGGER.warning(
            "No electricity price data (%s) - price optimization is NOT running", reason
        )
        async_create_issue(
            self.hass,
            DOMAIN,
            PRICE_SOURCE_ISSUE_ID,
            is_fixable=False,
            severity=IssueSeverity.WARNING,
            translation_key=PRICE_SOURCE_ISSUE_ID,
        )
        self._price_issue_active = True

    def _raise_dhw_control_issue(self) -> None:
        """Tell the user, in the UI, that hot-water optimisation is not running.

        They have `enable_hot_water_optimization` switched on, the DHW sensors are populated, and
        one of them is showing the time of a boost that will never happen.
        """
        if self._dhw_issue_active:
            return

        _LOGGER.warning(
            "No temporary-lux switch found (register 50004) - hot-water optimization is NOT "
            "running. Home Assistant's NIBE integration exposes this switch for F-series pumps "
            "only; an S-series pump has no equivalent entity."
        )
        async_create_issue(
            self.hass,
            DOMAIN,
            DHW_CONTROL_ISSUE_ID,
            is_fixable=False,
            severity=IssueSeverity.WARNING,
            translation_key=DHW_CONTROL_ISSUE_ID,
        )
        self._dhw_issue_active = True

    def _clear_dhw_control_issue(self) -> None:
        """A lux switch has appeared. Deliberately NOT guarded on the flag - see below."""
        async_delete_issue(self.hass, DOMAIN, DHW_CONTROL_ISSUE_ID)
        self._dhw_issue_active = False

    def _clear_price_source_issue(self) -> None:
        """Prices are flowing again.

        Deliberately NOT guarded on `_price_issue_active`. That flag lives on the coordinator and a
        restart builds a new one with it False - while the repair issue, which Home Assistant
        persists in its own registry, is still raised. Guarding the delete on it meant:

            boot 1   no price source  -> issue raised, flag True
            (the user configures GE-Spot and restarts)
            boot 2   prices fine      -> flag is False again, the delete returns early, and the
                                         issue stays raised. Forever, with nothing the user can do.

        `async_delete_issue` is a no-op when there is nothing to delete, so there is no cost to
        calling it. (Found on a live Home Assistant, not in the tests - the flag made the unit test
        of the raise path pass perfectly well.)
        """
        async_delete_issue(self.hass, DOMAIN, PRICE_SOURCE_ISSUE_ID)
        self._price_issue_active = False

    async def _read_and_decide(
        self,
        apply: bool,
        explicit_command: bool = False,
    ) -> dict[str, object]:
        """Fetch data and calculate optimal offset.

        Args:
            apply: Whether to drive the heat pump with the resulting decision. Only the control
                loop and the services that explicitly command the pump may pass True.
            explicit_command: A user command that bypasses startup observation and ordinary write
                cooldown. The decision engine's absolute safety floor still applies.

        This method:
        1. Gathers data from all sources (with graceful degradation)
        2. Runs optimization algorithm
        3. Returns updated state for all entities

        Returns:
            Dictionary containing:
            - nibe: Current NIBE state
            - price: Spot price data (native 15-min intervals)
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
            self.current_offset = float(nibe_data.current_offset)
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
                self._startup_grace_attempts = 0
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
                if self.compressor_stats:
                    # This is a CONTROL INPUT, not a log line. A compressor that has been at
                    # maximum frequency for a quarter of an hour has nothing left to give, and
                    # asking it for more heat buys only wear and a deeper DM deficit.
                    risk_level, risk_reason = self.compressor_monitor.assess_risk(
                        self.compressor_stats
                    )
                    self.compressor_risk = risk_level
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
                self._startup_grace_attempts += 1

                if self._startup_grace_attempts > STARTUP_MAX_GRACE_ATTEMPTS:
                    # The grace period is over. A heat pump that has not appeared by now is not
                    # slow, it is missing - a wrong entity, or none configured at all - and saying
                    # "still starting up" forever leaves the entry green while nothing is read and
                    # nothing is controlled.
                    _LOGGER.error(
                        "NIBE entities never became available after %d attempts (~%d minutes). "
                        "Check that the configured entities exist and their integration is "
                        "loaded. Last error: %s",
                        self._startup_grace_attempts - 1,
                        (self._startup_grace_attempts - 1) * UPDATE_INTERVAL_MINUTES,
                        err,
                    )
                    raise UpdateFailed(
                        f"NIBE entities unavailable after "
                        f"{self._startup_grace_attempts - 1} attempts: {err}"
                    ) from err

                _LOGGER.info(
                    "Waiting for NIBE entities to become available: %s "
                    "(this is normal during HA startup, attempt %d of %d, "
                    "will retry in %d minutes)",
                    err,
                    self._startup_grace_attempts,
                    STARTUP_MAX_GRACE_ATTEMPTS,
                    UPDATE_INTERVAL_MINUTES,
                )
                # Important: even though we return early, we must keep clock-aligned scheduling
                # running. With update_interval=None, nothing else will trigger retries.
                self._schedule_aligned_refresh()
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
            # After first success, NIBE failures are real errors
            _LOGGER.error("Failed to read NIBE data: %s", err)
            raise UpdateFailed(f"Cannot read NIBE data: {err}") from err

        # Gather optional data with graceful degradation
        # Spot price data (native 15-minute intervals)
        try:
            price_data = await self.gespot.get_prices()
            if price_data and price_data.today:
                current_index = price_data.get_period_index(dt_util.now())
                current_price = (
                    price_data.today[current_index].price if current_index is not None else 0
                )

                # Get unit from spot price entity for accurate logging
                gespot_id: str = self.entry.data.get("gespot_entity", "")
                gespot_entity = self.hass.states.get(gespot_id) if gespot_id else None
                unit = (
                    gespot_entity.attributes.get("unit_of_measurement", "units")
                    if gespot_entity
                    else "units"
                )

                _LOGGER.debug(
                    "Spot price data retrieved: %d quarters today, current Q%d = %.2f %s",
                    len(price_data.today),
                    current_index if current_index is not None else -1,
                    current_price,
                    unit,
                )
                self._clear_price_source_issue()
            else:
                self._report_no_price_source("the price entity returned no quarters")
                price_data = None
        except (AttributeError, KeyError, ValueError, TypeError) as err:
            # Do NOT fabricate. The old fallback returned 96 quarters all priced 1.0, and the
            # decision engine WEIGHED them: they classify NORMAL, the price layer casts a real
            # vote, and the aggregate is dragged down - +1.00 °C becomes +0.27 °C on a number
            # nobody measured. The reasoning string then told the user "[Spot Price] ... NORMAL"
            # as though a price had been analysed. (Audit F-123; same class as F-013/F-014, where
            # the NIBE adapter invented degree minutes.)
            #
            # None is the honest answer, and the engine handles it: the price layer abstains and
            # the thermal, comfort and safety layers decide on their own.
            self._report_no_price_source(str(err))
            price_data = None

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
        except (HomeAssistantError, AttributeError, KeyError, ValueError, TypeError) as err:
            # Weather is OPTIONAL - never let it take the whole update down.
            # HomeAssistantError was missing here: weather.get_forecasts raises it for any
            # entity without an hourly forecast, and it is not a subclass of the others.
            _LOGGER.info("Weather forecast unavailable: %s", err)
            weather_data = None

        # Run optimization decision engine
        is_grace_period = False

        # Check if optimization is enabled (master switch)
        optimization_enabled = self.entry.data.get("enable_optimization", True)
        if not optimization_enabled:
            _LOGGER.info("Optimization disabled by user - maintaining neutral offset")
            decision = OptimizationDecision(
                offset=0.0,
                reasoning="Optimization disabled by user",
                layers=[],
            )
        else:
            try:
                # The effect layer needs INSTANTANEOUS power to judge how close the current
                # quarter is to the monthly peak. Passing peak_today here (a daily maximum
                # that only ratchets upward until midnight) made a single morning spike pin
                # the effect layer to CRITICAL - weight 1.0, offset -3.0 - for the rest of
                # the day, even with the compressor idle.
                if self.current_power_kw is None or self.current_power_kw < 0:
                    _LOGGER.warning(
                        "No valid power measurement (%s) - disabling peak protection this "
                        "cycle rather than acting on a guess. Configure a power sensor or "
                        "NIBE phase currents for effect-tariff protection.",
                        self.current_power_kw,
                    )
                    current_power_for_decision = 0.0  # Disable peak protection
                else:
                    # LIKE FOR LIKE: the monthly record is an HOURLY MEAN, so the layer is
                    # compared against the hour this cycle projects to, not the instant. A
                    # five-minute oven spike early in the hour projects to almost nothing;
                    # the same spike at :55 has already committed most of the hour.
                    current_power_for_decision = self._billing_period.projected_hour_mean(
                        dt_util.now(), self.current_power_kw
                    )

                # Check if DHW is active (EITHER is_hot_water sensor OR temp_lux switch)
                # When NIBE heats DHW, flow temp reads charging temp (45-60°C), not space heating
                is_hot_water = (
                    nibe_data.is_hot_water if hasattr(nibe_data, "is_hot_water") else False
                )
                temp_lux_active = False
                if self.temp_lux_entity:
                    lux_state = self.hass.states.get(self.temp_lux_entity)
                    temp_lux_active = lux_state is not None and lux_state.state == STATE_ON

                # Unified DHW active state: either source means DHW is heating
                dhw_is_active = is_hot_water or temp_lux_active

                # Track DHW transition: active → inactive triggers weather layer cooldown
                # When DHW stops, flow temp remains elevated - needs cooldown period
                # Uses DHW_WEATHER_COOLDOWN_MINUTES (30 min) before weather comp re-enables
                if self.dhw_was_active and not dhw_is_active:
                    self.dhw_heating_end = dt_util.now()
                    _LOGGER.info(
                        "DHW heating stopped - triggering weather layer cooldown (%d min)",
                        DHW_WEATHER_COOLDOWN_MINUTES,
                    )
                elif not self.dhw_was_active and dhw_is_active:
                    self.dhw_heating_start = dt_util.now()
                    _LOGGER.info(
                        "DHW heating started (is_hot_water=%s, temp_lux=%s)",
                        is_hot_water,
                        temp_lux_active,
                    )
                self.dhw_was_active = dhw_is_active

                decision = await self.hass.async_add_executor_job(
                    self.engine.calculate_decision,
                    nibe_data,
                    price_data,
                    weather_data,
                    self.peak_this_month,  # Monthly peak threshold to protect
                    current_power_for_decision,  # Current whole-house power consumption
                    dhw_is_active,  # DHW heating active - skip weather comp
                    self.dhw_heating_end,  # When DHW last stopped - for cooldown
                    self.compressor_risk,  # Do not ask a saturated compressor for more heat
                )

                # Startup grace period: lockout + observation cycles
                now = dt_util.now()

                if explicit_command and decision.is_manual_override:
                    _LOGGER.info("Explicit user command bypasses startup observation")
                elif now < self._startup_grace_timeout:
                    # Phase 1: Time-based lockout
                    secs_left = int((self._startup_grace_timeout - now).total_seconds())
                    is_grace_period = True
                    _LOGGER.info(
                        "Startup Lockout (%ds): Observing. Decision: %.1f°C (%s)",
                        secs_left,
                        decision.offset,
                        decision.reasoning,
                    )
                    decision.reasoning = f"[Startup Lockout] {decision.reasoning}"
                elif self._startup_update_count < self._startup_grace_updates:
                    # Phase 2: Observation cycles (one per update interval)
                    self._startup_update_count += 1
                    is_grace_period = True
                    cycles_left = self._startup_grace_updates - self._startup_update_count
                    _LOGGER.info(
                        "Startup Observation (%d cycle%s): Observing. Decision: %.1f°C (%s)",
                        cycles_left,
                        "s" if cycles_left != 1 else "",
                        decision.offset,
                        decision.reasoning,
                    )
                    decision.reasoning = f"[Startup Observation] {decision.reasoning}"
                elif not getattr(self, "_grace_period_ended_logged", False):
                    _LOGGER.info("Startup complete - active control enabled")
                    self._grace_period_ended_logged = True
                    # Initialize volatility tracker with actual NIBE offset
                    # This prevents false "reversal" detection on first active cycle
                    nibe_offset = nibe_data.current_offset if nibe_data else 0.0
                    self._offset_volatility_tracker.record_change(
                        nibe_offset, "startup_init_from_nibe"
                    )
                    _LOGGER.debug(
                        "Volatility tracker initialized with NIBE offset: %.1f°C", nibe_offset
                    )

                _LOGGER.info(
                    "Decision: offset %.2f°C, reasoning: %s",
                    decision.offset,
                    decision.reasoning,
                )
            except (AttributeError, KeyError, ValueError, TypeError, ZeroDivisionError) as err:
                _LOGGER.error("Optimization failed: %s", err)
                # Fall back to safe operation (no offset)
                decision = get_safe_default_decision()

        # Check for volatile offset reversal before applying
        # Prevents rapid back-and-forth that the heat pump can't follow
        # (same logic as price volatility)
        # Skip volatility tracking during startup - we're not applying offsets yet
        #
        # Anti-windup bypass (Feb 2026): When anti-windup is active, it means the system
        # detected a self-induced DM spiral (raising S1 makes BT25-S1 gap grow → DM drops
        # faster). The volatile blocker must NOT block anti-windup reductions — doing so
        # keeps the harmful high offset active for 45 minutes while DM plunges.
        # Jan 31 2026 incident: volatile blocked anti-windup reduction 10→3°C for 45 min,
        # DM dropped from -1566 to -1855.
        if is_grace_period:
            # During startup, don't track decisions - they're not being applied
            # The tracker will be initialized with actual NIBE offset when startup completes
            pass
        elif decision.is_manual_override:
            # User-commanded offsets (force_offset/boost services) are
            # authoritative — the volatile blocker must never defer an
            # explicit user command for 45 minutes
            _LOGGER.info(
                "Manual override: bypassing volatile check (offset → %.1f°C)",
                decision.offset,
            )
            self._offset_volatility_tracker.record_change(decision.offset, decision.reasoning)
        elif decision.is_emergency:
            # Absolute safety path: indoor below MIN_TEMP_LIMIT, or DM past the aux limit.
            # The volatile blocker damps price-driven flip-flopping; it must never defer a safety
            # recovery. Blocking here would hold the previous (often negative) offset for up to
            # 45 minutes while DM keeps falling and the immersion heater runs.
            _LOGGER.warning(
                "Emergency decision: bypassing volatile check (offset %.1f°C → %.1f°C) - %s",
                (
                    self._offset_volatility_tracker.last_offset
                    if self._offset_volatility_tracker.last_offset is not None
                    else 0.0
                ),
                decision.offset,
                decision.reasoning,
            )
            self._offset_volatility_tracker.record_change(decision.offset, decision.reasoning)
        elif decision.anti_windup_active:
            # Anti-windup is a safety mechanism — always apply immediately
            # Record the change so volatile tracker knows the new baseline
            _LOGGER.info(
                "Anti-windup active: bypassing volatile check (offset %.1f°C → %.1f°C)",
                (
                    self._offset_volatility_tracker.last_offset
                    if self._offset_volatility_tracker.last_offset is not None
                    else 0.0
                ),
                decision.offset,
            )
            self._offset_volatility_tracker.record_change(decision.offset, decision.reasoning)
        elif self._offset_volatility_tracker.is_reversal_volatile(decision.offset):
            volatile_reason = self._offset_volatility_tracker.get_volatile_reason(decision.offset)
            _LOGGER.info(
                "Offset change blocked: %s (keeping %.1f°C)",
                volatile_reason,
                self._offset_volatility_tracker.last_offset,
            )
            # Keep the previous offset (last_offset is always set when is_reversal_volatile is True)
            last_offset = self._offset_volatility_tracker.last_offset or 0.0
            decision = OptimizationDecision(
                offset=last_offset,
                reasoning=f"[{volatile_reason}] {decision.reasoning}",
            )
        else:
            # Record the new offset for volatility tracking
            self._offset_volatility_tracker.record_change(decision.offset, decision.reasoning)

        self.last_decision_time = dt_util.utcnow()

        # Apply offset to the NIBE heating curve offset number entity
        # (register 47011 on F-series; MyUplink, nibe_heatpump, or template number)
        # Rate limiting (5 min) handled in nibe_adapter to prevent excessive API calls
        #
        # Accumulation logic: We track fractional offsets but only write to NIBE when
        # the integer part changes. This prevents oscillation when calculated offsets
        # hover around boundaries (e.g., 0.48 ↔ 0.52 both stay at 0°C in NIBE).
        if not apply:
            # A read, not a control cycle. Decide, publish, write nothing.
            _LOGGER.debug("Read-only refresh: decided %.2f°C, not applying", decision.offset)
        elif is_grace_period:
            _LOGGER.info("Skipping offset application during startup grace period")
        else:
            try:
                # Through the one guarded door: a coordinator whose entry has unloaded mid-refresh
                # must not get the last word on the pump. See _write_curve_offset.
                applied_offset = await self._write_curve_offset(
                    decision.offset,
                    force_write=explicit_command,
                )
                if applied_offset is not None:
                    _LOGGER.info(
                        "Applied offset %.2f°C as %d°C on NIBE",
                        decision.offset,
                        applied_offset,
                    )
                    self.last_applied_offset = float(applied_offset)
                    self.current_offset = float(applied_offset)
                    nibe_data.current_offset = float(applied_offset)
                    self.last_offset_timestamp = dt_util.utcnow()
                    self._learned_data_changed = True  # Trigger save on shutdown
                    if decision.is_manual_override:
                        self.engine.consume_manual_override()
                else:
                    _LOGGER.debug(
                        "Offset %.2f°C unchanged (NIBE offset not changed)",
                        decision.offset,
                    )
                    if explicit_command:
                        raise HomeAssistantError(
                            "The explicit heating offset did not reach the NIBE register"
                        )
            except (HomeAssistantError, AttributeError, OSError, ValueError) as err:
                _LOGGER.error("Failed to apply offset to NIBE: %s", err)
                if explicit_command:
                    raise
                # Automatic control retries on the next aligned cycle.

        self._accumulate_spot_savings(nibe_data, price_data)

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

            # A new day may also be a new MONTH. The effect tariff bills a monthly peak, so
            # last month's peaks must not carry over into this one - an instance that stays up
            # across a month boundary would otherwise bill against a stale threshold.
            month_changed = (now.year, now.month) != (
                self._last_update_date.year,
                self._last_update_date.month,
            )
            if month_changed:
                self.effect.prune_peaks_for_current_month()
                self.peak_this_month = self.effect.get_monthly_peak_summary()["highest"]
                _LOGGER.info(
                    "Month change detected: pruned previous month's peaks, "
                    "monthly peak reset to %.2f kW",
                    self.peak_this_month,
                )

            # Reset daily peak for new day
            self.peak_today = 0.0
            self.peak_today_time = None
            self.peak_today_source = "unknown"
            self.peak_today_period = None
            self._last_update_date = now.date()

        # Update peak tracking
        await self._update_peak_tracking(nibe_data)

        # Record observations for learning (Phase 6)
        await self._record_learning_observations(nibe_data, weather_data, self.current_offset)

        # Save state periodically
        await self.effect.async_save()

        # Save learned data if changed (every hour to avoid excessive writes)
        if self._learned_data_changed:
            now = dt_util.now()
            if (
                self._last_learning_save is None
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
        current_quarter = price_data.get_period_index(now_time) if price_data else None
        current_classification = (
            self.engine.price.get_current_classification(current_quarter)
            if current_quarter is not None
            else None
        )

        # Calculate estimated savings
        current_price = 0.0
        if price_data and hasattr(price_data, "today") and price_data.today:
            # Get current price
            if current_quarter is not None:
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

            # Use unified DHW active state (tracked earlier in update cycle)
            # dhw_was_active combines is_hot_water sensor + temp_lux switch
            is_actively_heating = self.dhw_was_active

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
            dhw_result: DHWRecommendation | None = None

            try:
                dhw_result = await self._calculate_dhw_recommendation(
                    nibe_data, price_data, weather_data, current_dhw_temp, now_time
                )
                dhw_recommendation = dhw_result.recommendation
                dhw_planning_summary = dhw_result.summary
                dhw_planning_details = dhw_result.details

                # Use the optimizer's recommended start time (timezone-aware from spot price)
                # When should_heat=True, recommended_start_time is None (heating now)
                # When should_heat=False, recommended_start_time is a future timestamp
                #   (guaranteed by DHWScheduleDecision dataclass validation)
                dhw_next_boost = dhw_planning_details.get("recommended_start_time")

                # Set schedule_status for UI display
                if dhw_result.decision:
                    if dhw_result.decision.should_heat:
                        dhw_planning_details["schedule_status"] = "heating_now"
                    else:
                        # Dataclass validation guarantees recommended_start_time exists
                        dhw_planning_details["schedule_status"] = "scheduled"
                else:
                    # Edge case: no decision (shouldn't happen in normal operation)
                    dhw_planning_details["schedule_status"] = "unknown"
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
            if (
                optimization_enabled
                and self.entry.data.get("enable_hot_water_optimization", False)
                and dhw_result is not None
                and dhw_result.decision is not None
            ):
                if not apply:
                    _LOGGER.debug("Read-only refresh: not applying DHW control")
                elif is_grace_period:
                    _LOGGER.info("Skipping DHW control during startup grace period")
                else:
                    await self._apply_dhw_control(
                        dhw_result.decision,
                        current_dhw_temp,
                        now_time,
                    )
        else:
            # DHW sensor not available - provide basic recommendation
            if nibe_data:
                _LOGGER.warning(
                    "DHW sensor (BT7) not found - ensure your NIBE integration "
                    "exposes the BT7/40013 sensor (enable the entity in "
                    "nibe_heatpump/MyUplink) or select it manually via "
                    "Reconfigure in EffektGuard"
                )
                dhw_recommendation = "DHW sensor not found - check NIBE integration"
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

                # Log airflow decision details
                _LOGGER.debug(
                    "Airflow decision: %s (enhance=%s, gain=%.2f kW, indoor=%.1f°C, "
                    "compressor=%d Hz, outdoor=%.1f°C)",
                    airflow_decision.reason,
                    airflow_decision.should_enhance,
                    airflow_decision.expected_gain_kw,
                    nibe_data.indoor_temp,
                    nibe_data.compressor_hz or 0,
                    nibe_data.outdoor_temp,
                )

                # Apply control only if airflow optimization is enabled (like DHW)
                airflow_enabled = self.entry.data.get(CONF_ENABLE_AIRFLOW_OPTIMIZATION, False)
                if airflow_enabled:
                    if not optimization_enabled:
                        _LOGGER.debug("Optimization disabled - not applying airflow control")
                    elif not apply:
                        _LOGGER.debug("Read-only refresh: not applying airflow control")
                    elif is_grace_period:
                        _LOGGER.info("Skipping airflow control during startup grace period")
                    else:
                        await self._apply_airflow_decision(airflow_decision)
                else:
                    _LOGGER.debug("Airflow optimization disabled - not applying control")

            except (AttributeError, ValueError, TypeError) as err:
                _LOGGER.warning("Airflow evaluation failed: %s", err)
                # Continue without airflow optimization

        # Ensure aligned scheduling is always active
        # This handles both first update AND event-triggered refreshes (e.g., power sensor)
        # that could otherwise break the scheduling chain
        self._schedule_aligned_refresh()

        return {
            "nibe": nibe_data,
            "price": price_data,
            "weather": weather_data,
            "thermal": self.engine.thermal,  # Thermal model for predictions
            "thermal_trend": temperature_trend_data,  # Temperature trend from predictor
            "outdoor_trend": outdoor_trend_data,  # Outdoor temperature trend
            "decision": decision,
            "offset": self.current_offset,
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
    ) -> DHWRecommendation:
        """Calculate DHW heating recommendation with detailed planning.

        Thin wrapper that gathers HA-specific data and delegates to
        dhw_optimizer.calculate_recommendation() for pure logic.

        Args:
            nibe_data: Current NIBE state
            price_data: Spot price data
            weather_data: Weather forecast
            current_dhw_temp: Current DHW temperature (°C)
            now_time: Current datetime

        Returns:
            DHWRecommendation with recommendation text and detailed planning info
        """
        # Gather HA-specific data
        # CRITICAL: Check options first (runtime changes), fall back to data then default
        target_indoor = self.entry.options.get(
            "target_indoor_temp",
            self.entry.data.get("target_indoor_temp", DEFAULT_INDOOR_TEMP),
        )
        hours_since_last = await self._calculate_hours_since_last_dhw()

        # Get thermal trend from predictor
        thermal_trend = self.thermal_predictor.get_current_trend() if self.thermal_predictor else {}
        trend_rate = thermal_trend.get("rate_per_hour", 0.0)

        # Get price classification and volatility
        current_quarter = price_data.get_period_index(now_time) if price_data else None
        price_classification = (
            self.engine.price.get_current_classification(current_quarter)
            if current_quarter is not None
            else "normal"
        )

        # Get price periods
        price_periods = []
        if price_data:
            price_periods = price_data.today + price_data.tomorrow

        # Extract basic data from NIBE state
        thermal_debt = nibe_data.degree_minutes if nibe_data else DM_THRESHOLD_START
        space_heating_demand = (
            nibe_data.power_kw if nibe_data and nibe_data.power_kw is not None else 0.0
        )
        outdoor_temp = nibe_data.outdoor_temp if nibe_data else 0.0
        indoor_temp = nibe_data.indoor_temp if nibe_data else DEFAULT_INDOOR_TEMP

        # Get climate zone name
        climate_zone_name = None
        if self.engine.climate_detector:
            climate_zone_name = self.engine.climate_detector.zone_info.name
        else:
            # `hass.components` was removed from Home Assistant; the type: ignore that used to sit
            # here claimed a stubs gap and was hiding a real AttributeError (audit F-068).
            async_create(
                self.hass,
                "EffektGuard could not detect your climate zone. "
                "Using balanced thermal debt thresholds. "
                "Configure latitude in integration settings for optimal climate-aware operation.",
                title="EffektGuard Configuration Recommended",
                notification_id="effektguard_climate_detection_missing",
            )

        # Get weather current temp for opportunity detection
        weather_current_temp = None
        if weather_data and hasattr(weather_data, "current_temp"):
            weather_current_temp = weather_data.current_temp

        # Get DHW amount from NIBE state for scheduled amount check (RULE 0)
        dhw_amount_minutes = nibe_data.dhw_amount_minutes if nibe_data else None

        # Get weather forecast for predictive pre-heating
        weather_forecast = None
        if weather_data and hasattr(weather_data, "forecast_hours"):
            weather_forecast = weather_data.forecast_hours

        # Call pure logic in dhw_optimizer (volatility calculated internally)
        result = self.dhw_optimizer.calculate_recommendation(
            current_dhw_temp=current_dhw_temp,
            thermal_debt=thermal_debt,
            space_heating_demand=space_heating_demand,
            outdoor_temp=outdoor_temp,
            indoor_temp=indoor_temp,
            target_indoor=target_indoor,
            price_classification=price_classification,
            current_time=now_time,
            price_periods=price_periods,
            hours_since_last_dhw=hours_since_last,
            thermal_trend_rate=trend_rate,
            climate_zone_name=climate_zone_name,
            weather_current_temp=weather_current_temp,
            price_data=price_data,
            dhw_amount_minutes=dhw_amount_minutes,
            weather_forecast=weather_forecast,
        )

        return result

    async def _get_last_dhw_heating_time(self) -> datetime | None:
        """Get timestamp when temporary lux was last activated.

        Reads from MyUplink entity's last_changed attribute (source of truth).
        Falls back to Home Assistant history API if entity is currently OFF.

        Returns:
            datetime: Last time DHW heating was activated (UTC)
            None: If no history available
        """
        if not self.temp_lux_entity:
            _LOGGER.debug("No temporary lux entity configured")
            return None

        # Try to read from current entity state
        state = self.hass.states.get(self.temp_lux_entity)
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
                # (get_instance/history are runtime attributes not in HA type stubs)
                rec = recorder.get_instance(self.hass)  # type: ignore[attr-defined]
                if rec is None:
                    _LOGGER.warning("Recorder not available, cannot check DHW history")
                    return None

                # Get state history using recorder instance executor
                states = await rec.async_add_executor_job(
                    recorder.history.state_changes_during_period,  # type: ignore[attr-defined]
                    self.hass,
                    start_time,
                    end_time,
                    self.temp_lux_entity,
                )

                if self.temp_lux_entity in states:
                    entity_states = states[self.temp_lux_entity]

                    # Find most recent ON state
                    for state_obj in reversed(entity_states):
                        if state_obj.state == STATE_ON:
                            _LOGGER.debug(
                                "Last DHW heating from history: %s", state_obj.last_changed
                            )
                            return state_obj.last_changed

                _LOGGER.debug("No ON state in history for %s", self.temp_lux_entity)

            except (AttributeError, KeyError, ValueError, OSError) as err:
                _LOGGER.error("Failed to read DHW heating history: %s", err)

        else:
            _LOGGER.warning("Temporary lux entity %s not found", self.temp_lux_entity)

        # Fall back to stored value (if any)
        if self._last_dhw_control_time is not None:
            _LOGGER.debug("Using fallback stored DHW time: %s", self._last_dhw_control_time)
            return self._last_dhw_control_time

        return None

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
        # Check if enhanced ventilation is currently active
        is_enhanced = await self.nibe.is_enhanced_ventilation_active()

        if is_enhanced is None:
            _LOGGER.debug("Ventilation control unavailable - no ventilation switch found")
            return

        now = dt_util.utcnow()

        # THE FAN COULD CYCLE FOREVER. The minimum-enhanced-duration guard was 5 minutes - exactly
        # one coordinator tick - so it permitted a turn-off on the very next cycle, and NOTHING at
        # all guarded the turn-ON. A decision oscillating around its threshold, which is what a
        # marginal COP gain does, produced twelve fan state changes an hour, indefinitely. On an
        # exhaust-air F750 each one perturbs the source air the compressor is drawing from.
        #
        # The optimizer already computes how long the enhancement should run - `duration_minutes`,
        # 15 to 60 min depending on the deficit - and that number was LOGGED and thrown away. It is
        # now the minimum run time, and a minimum rest at normal speed bounds the other direction.
        if decision.should_enhance:
            if is_enhanced:
                _LOGGER.debug("Ventilation already enhanced - %s", decision.reason)
                return

            resting_since = self._airflow_normal_since
            if resting_since is not None:
                rested = (now - resting_since).total_seconds() / 60
                if rested < NIBE_VENTILATION_MIN_REST_DURATION:
                    _LOGGER.debug(
                        "Ventilation: at normal for only %d of %d min - not re-enhancing yet",
                        int(rested),
                        NIBE_VENTILATION_MIN_REST_DURATION,
                    )
                    return

            if await self._write_enhanced_ventilation(True):
                self._airflow_enhance_start = now
                self._airflow_normal_since = None
                # The optimizer's own recommendation, floored so a decision that carries no
                # duration still cannot produce a five-minute burst.
                self._airflow_enhance_minutes = max(
                    decision.duration_minutes, NIBE_VENTILATION_MIN_ENHANCED_DURATION
                )
                _LOGGER.info(
                    "🌀 Ventilation ENHANCED: ON for at least %d min (+%.2f kW gain) - %s",
                    self._airflow_enhance_minutes,
                    decision.expected_gain_kw,
                    decision.reason,
                )
            return

        if not is_enhanced:
            _LOGGER.debug("Ventilation at normal - %s", decision.reason)
            return

        if self._airflow_enhance_start is not None:
            elapsed = (now - self._airflow_enhance_start).total_seconds() / 60
            if elapsed < self._airflow_enhance_minutes:
                _LOGGER.debug(
                    "Ventilation: keeping enhanced for %d more min (the optimizer asked for %d)",
                    int(self._airflow_enhance_minutes - elapsed),
                    self._airflow_enhance_minutes,
                )
                return

        if await self._write_enhanced_ventilation(False):
            self._airflow_enhance_start = None
            self._airflow_normal_since = now
            _LOGGER.info("🌀 Ventilation NORMAL: OFF - %s", decision.reason)

    def _is_dhw_start_rate_limited(self, now_time: datetime) -> bool:
        """True if a DHW boost was started or stopped too recently to start another.

        Guards STARTS only. A stop must never be deferred - see _apply_dhw_control.

        Args:
            now_time: Current datetime

        Returns:
            True when a new DHW boost must not be started yet
        """
        if self._last_dhw_control_time is None:
            return False

        minutes_since_last = (now_time - self._last_dhw_control_time).total_seconds() / 60
        if minutes_since_last < DHW_CONTROL_MIN_INTERVAL_MINUTES:
            _LOGGER.debug(
                "DHW start rate limited: %.1f min since last change (min %d min)",
                minutes_since_last,
                DHW_CONTROL_MIN_INTERVAL_MINUTES,
            )
            return True

        return False

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
        # Check temporary lux entity
        if not self.temp_lux_entity:
            # A _LOGGER.debug is not telling anyone, and this is a whole feature silently doing
            # nothing. Home Assistant's NIBE integration maps the temporary-lux register (50004)
            # for the F-series only, so an S-series pump exposes no such entity - while the UI goes
            # on showing a hot-water status, a recommendation and a SCHEDULED START TIME that can
            # never fire. Exactly the case the price-source repair issue was created for.
            self._raise_dhw_control_issue()
            return

        self._clear_dhw_control_issue()

        # Get current state of temporary lux switch
        temp_lux_state = self.hass.states.get(self.temp_lux_entity)
        if not temp_lux_state:
            _LOGGER.warning("Temporary lux entity %s not found", self.temp_lux_entity)
            return

        is_lux_on = temp_lux_state.state == "on"

        # A user-commanded boost (boost_dhw) opens a window ordinary optimization must not
        # close. Expiry closes it here, through the owned door; the safety abort below closes
        # it too - only safety outranks the user.
        if self._service_boost_until is not None:
            if not is_lux_on:
                # NIBE's own lux timeout, or the household, ended it first.
                self._service_boost_until = None
            elif now_time >= self._service_boost_until:
                _LOGGER.info("User DHW boost window ended - returning temporary lux to normal")
                if await self._set_temporary_lux(False):
                    self._last_dhw_control_time = now_time
                self._service_boost_until = None
                return

        # Use pre-calculated decision from _calculate_dhw_recommendation()
        # This avoids duplicate optimizer calls and log spam
        # Get thermal_debt and indoor_temp from coordinator data for abort conditions
        thermal_debt = (
            self.data.get("dhw_planning", {}).get("thermal_debt", 0.0)
            if self.last_update_success and hasattr(self, "data") and self.data is not None
            else 0.0
        )
        indoor_temp = (
            self.data.get("dhw_planning", {}).get("indoor_temperature", DEFAULT_INDOOR_TEMP)
            if self.last_update_success and hasattr(self, "data") and self.data is not None
            else DEFAULT_INDOOR_TEMP
        )
        # CRITICAL: Check options first (runtime changes), fall back to data then default
        target_indoor = self.entry.options.get(
            "target_indoor_temp",
            self.entry.data.get("target_indoor_temp", DEFAULT_INDOOR_TEMP),
        )

        # ABORT MONITORING: If DHW is currently heating, check abort conditions
        # This allows us to stop DHW early if conditions deteriorate (thermal debt, indoor temp)
        # thermal_debt and indoor_temp from coordinator data (same as used in decision calculation)
        if is_lux_on and decision.abort_conditions:
            should_abort, abort_reason = self.dhw_optimizer.check_abort_conditions(
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
                # The safety stop was the THIRD place that reached the lux switch on its own, and it
                # left `_lux_boost_is_ours` set after switching the boost off. Through the door.
                # Safety also outranks a user boost - the window closes with the switch.
                self._service_boost_until = None
                if await self._set_temporary_lux(False):
                    self._last_dhw_control_time = now_time
                return  # Exit early - abort handled

        # Apply control decision.
        #
        # RATE LIMITING APPLIES TO STARTS ONLY - never to stops.
        #
        # Rate-limiting a stop would strand an in-progress DHW cycle: every `should_heat=False`
        # path in should_start_dhw() returns an EMPTY abort_conditions list, so the abort branch
        # above is skipped, and the limiter's clock is started by the turn-ON - meaning the
        # interval runs from the beginning of the very cycle being stopped. DHW would hold the
        # compressor away from space heating
        # while thermal debt deepened.
        #
        # Stopping the lux boost cannot harm the pump - it only stops an EffektGuard-
        # initiated boost. Throttling it has no safety benefit and a real safety cost.
        # Oscillation stays bounded because the next START is still rate limited.
        if decision.should_heat and not is_lux_on:
            if self._is_dhw_start_rate_limited(now_time):
                return

            # Turn ON temporary lux to boost DHW
            _LOGGER.info(
                "DHW control: Activating temporary lux - %s (DHW: %.1f°C, DM: %.0f)",
                decision.priority_reason,
                current_dhw_temp,
                thermal_debt,
            )
            # Through the one door, which is what records that this boost is ours - and therefore
            # what lets the unload cleanup find it again. See _set_temporary_lux.
            if await self._set_temporary_lux(True):
                self._last_dhw_control_time = now_time

        elif not decision.should_heat and is_lux_on:
            if self._service_boost_until is not None:
                _LOGGER.debug(
                    "User DHW boost active until %s - price optimization does not cancel it",
                    self._service_boost_until,
                )
                return
            # Turn OFF temporary lux to block/stop DHW
            _LOGGER.info(
                "DHW control: Deactivating temporary lux - %s (DHW: %.1f°C, DM: %.0f)",
                decision.priority_reason,
                current_dhw_temp,
                thermal_debt,
            )
            if await self._set_temporary_lux(False):
                self._last_dhw_control_time = now_time
        else:
            # No change needed
            _LOGGER.debug(
                "DHW control: No change needed (should_heat=%s, lux_on=%s, reason=%s)",
                decision.should_heat,
                is_lux_on,
                decision.priority_reason,
            )

    def _accumulate_spot_savings(self, nibe_data, price_data) -> None:
        """Add this cycle's spot-price savings to the running daily total.

        Savings are reported to the owner as money, so they may only be computed from power that was
        MEASURED. Without a power sensor, `power_kw` holds a curve fit of the supply and outdoor
        temperatures, floored at 1.0 kW even with the compressor off - and it arrives in the same
        field as a real reading. It used to be passed straight to `actual_power_kw`, under a comment
        saying "using ACTUAL power consumption", and the resulting kronor were indistinguishable from
        earned ones.

        The coordinator already refuses to bill an estimated PEAK, and says so three times. This is
        the same rule, applied to the other number the owner is asked to trust.
        """
        if (
            not price_data
            or not getattr(price_data, "today", None)
            or not nibe_data
            or nibe_data.power_kw is None
            or nibe_data.power_is_estimated
        ):
            return

        current_quarter = price_data.get_period_index(dt_util.now())
        if current_quarter is None:
            return

        prices_today = [quarter.price for quarter in price_data.today]
        average_price = sum(prices_today) / len(prices_today)
        current_price = price_data.today[current_quarter].price

        self.savings_calculator.price_unit = getattr(self.gespot, "price_unit", None)
        cycle_savings = self.savings_calculator.calculate_spot_savings_per_cycle(
            actual_power_kw=nibe_data.power_kw,
            current_price=current_price,
            average_price_today=average_price,
            cycle_minutes=UPDATE_INTERVAL_MINUTES,
        )

        if self.savings_calculator.is_sek_price_unit:
            self._daily_spot_savings += cycle_savings
        else:
            _LOGGER.debug(
                "Skipping non-SEK spot-savings aggregation for price unit %s",
                self.savings_calculator.price_unit,
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
                self.nibe.power_sensor_entity
            )

            # Where this cycle's power reading came from. The billing guard at the end asks THIS,
            # and nothing else. It used to ask whether a power entity was configured, which says
            # nothing about whether the entity answered: a meter that dropped out left the estimate
            # from PRIORITY 3 to be recorded as a tariff peak, stamped as a meter reading.
            power_source = POWER_SOURCE_NONE

            # PRIORITY 1: External power meter (whole house including NIBE)
            # This is MOST IMPORTANT for peak billing - measures total house consumption
            # Used for: Monthly peak tracking (effect tariff billing)
            current_power = None
            if has_external_power_sensor:
                power_entity_id = self.nibe.power_sensor_entity
                power_state = self.hass.states.get(power_entity_id)

                # Unit handling lives in one place, shared with the NIBE adapter, which reads the
                # same entity: an unrecognised or absent unit is refused rather than guessed.
                current_power = power_kw_from_state(power_state)

                if current_power is not None:
                    power_source = POWER_SOURCE_EXTERNAL_METER
                    _LOGGER.debug(
                        "📊 External power meter (whole house): %.3f kW from %s",
                        current_power,
                        power_entity_id,
                    )
                    # Mark sensor as available (in case event listener hasn't fired yet)
                    if not self._power_sensor_available:
                        self._power_sensor_available = True
                        _LOGGER.debug("External power sensor marked as available")
                elif not self._power_sensor_available:
                    # Never seen alive - wait for the listener rather than tracking peaks on nothing
                    _LOGGER.debug(
                        "External power sensor %s not yet available (state: %s) - "
                        "skipping peak tracking (listener active: %s)",
                        power_entity_id,
                        power_state.state if power_state else "None",
                        self._power_sensor_listener is not None,
                    )
                    return  # Exit early, event listener will trigger refresh when ready
                else:
                    # It has answered before and is not answering now. Everything below still runs -
                    # the decision layers need SOME power figure - but the source stays unbillable,
                    # so nothing invented here reaches the tariff record.
                    # This used to say "Peak billing is suspended until it does", and it was not: no
                    # sample is taken from an estimate, which is what that sentence was guarding, but
                    # the billing hour carried on regardless and was billed when it closed, using
                    # whatever the meter last said before it went quiet, stretched across the whole
                    # silence. Now the hour is genuinely refused if the silence is long enough - see
                    # MAX_BILLING_OBSERVATION_GAP_MINUTES - so the log can say what the code does.
                    _LOGGER.warning(
                        "External power meter %s did not yield a reading (state: %s). This cycle is "
                        "not billable, and if the silence exceeds %d minutes the whole hour is "
                        "refused rather than billed on a stale reading.",
                        power_entity_id,
                        power_state.state if power_state else "None",
                        MAX_BILLING_OBSERVATION_GAP_MINUTES,
                    )

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
                    power_source = POWER_SOURCE_NIBE_CURRENTS
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
                current_power = self.effect.estimate_power_from_compressor(
                    nibe_data.compressor_hz, nibe_data.outdoor_temp
                )
                power_source = POWER_SOURCE_ESTIMATE
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
                is_heating = getattr(nibe_data, "is_heating", False)
                outdoor_temp = getattr(nibe_data, "outdoor_temp", 0.0)
                current_power = self.effect.estimate_power_consumption(is_heating, outdoor_temp)
                power_source = POWER_SOURCE_ESTIMATE
                _LOGGER.warning(
                    "⚠️  Power estimation fallback: %.2f kW (no real data available) "
                    "[ESTIMATE ONLY - not used for peak billing]",
                    current_power,
                )

            # A meter reading low while the compressor runs hard used to be overridden here: the code
            # assumed solar was masking the grid import and substituted an ESTIMATE of the
            # compressor's draw. It billed that estimate.
            #
            # The grid operator bills grid IMPORT, and the import is exactly what the meter saw. If
            # solar covers 4.7 kW of a 5.0 kW compressor, the house imported 0.3 kW and 0.3 kW is
            # what is charged. Recording ~5.5 kW instead inflated the month's peak by an order of
            # magnitude, in the owner's disfavour, and effect tariffs bill the top three quarters of
            # the month, so it stood for weeks.
            #
            # The meter is the truth. There is nothing to override.

            # Publish the instantaneous reading for the effect layer.
            #
            # The decision engine needs CURRENT power to judge how close this quarter is
            # to the monthly peak. It must never be given `peak_today`: that value is a
            # monotonically non-decreasing daily MAXIMUM (see below) which is reset only at
            # midnight, so a single morning spike would pin the effect layer to CRITICAL
            # for the rest of the day even with the compressor idle.
            #
            # This method runs after the decision within a cycle, so the engine reads the
            # previous cycle's value - at most UPDATE_INTERVAL_MINUTES old, and a genuine
            # measurement rather than a daily high-water mark.
            self.current_power_kw = current_power

            # The source was recorded where the value was produced. It used to be reconstructed here,
            # after the fact, from the config entry and the magnitude of the number - so a compressor
            # estimate above 0.5 kW was filed as "external_meter", and a peak that had been invented
            # became indistinguishable from one that had been measured.
            measurement_source = power_source

            # Get current timestamp for peak tracking
            now = dt_util.now()
            billing_period = get_current_billing_period(now)

            # Update daily peak (always track for display, even if estimated)
            if current_power > self.peak_today:
                self.peak_today = current_power
                self.peak_today_time = now
                self.peak_today_source = measurement_source
                # The billing PERIOD this peak fell in - an hour. The sensor weights it through
                # effective_tariff_power_kw, which now takes an hour, so it must be given one.
                self.peak_today_period = billing_period

                _LOGGER.info(
                    "New daily peak: %.2f kW at %s (billing hour %d, source: %s)",
                    current_power,
                    now.strftime("%H:%M:%S"),
                    billing_period,
                    measurement_source,
                )

            # CRITICAL: Only record monthly peaks with REAL measurements
            # Monthly peak billing requires accurate whole-house power measurement
            # Estimates are NEVER used for monthly peak tracking - billing must be accurate
            #
            # This asks where THIS cycle's number came from. It used to ask whether a power entity
            # was configured, which a meter that has gone unavailable still satisfies - so the
            # estimate that replaced it was billed anyway, in the same cycle the log said it must
            # never be.
            if power_source not in PEAK_CONTROL_POWER_SOURCES:
                _LOGGER.debug(
                    "Skipping monthly peak recording: %.2f kW came from %s, which is not a "
                    "measurement. Peak protection must not be driven by a guess.",
                    current_power,
                    power_source,
                )
                return

            # THE TARIFF BILLS THE HOURLY MEAN, AND WHAT THAT MEANS IS DEFINED IN ONE PLACE.
            #
            # This block used to carry its own copy of the arithmetic - a time-weighted mean over an
            # hour, on the absolute time line - and the simulator carried a DIFFERENT copy, an
            # arithmetic mean over the samples. Two implementations of the single most consequential
            # number this integration computes, and the harness was validating the one nobody runs.
            #
            # They were both wrong on the night the clocks go back, independently, so neither could
            # see the other's bug: the coordinator merged the repeated hour and deleted a 9 kW
            # billing peak. Now there is one definition, in billing_period.py, and the harness runs
            # THAT - so breaking it fails the simulation too, which is the property that was missing.
            completed = self._billing_period.add(now, current_power, power_source)

            peak_event = None
            if completed is not None:
                peak_event = await self.effect.record_period_measurement(
                    power_kw=completed.mean_power_kw,
                    period=completed.billing_hour,
                    timestamp=completed.started_at,
                    # The hour's OWN provenance - every sample votes, not the closing cycle.
                    source=completed.source,
                )

            if (
                peak_event
                and not self.entry.data.get("enable_optimization", True)
                and peak_event.is_billable
            ):
                # THE UNOPTIMISED BASELINE, MEASURED RATHER THAN ASSUMED.
                #
                # With optimization switched off the coordinator holds the curve offset at 0.0 and
                # the pump runs on its own heating curve - so the quarters recorded now are, by
                # definition, what this house does WITHOUT EffektGuard. That is precisely what
                # `update_baseline_peak` was written for ("Call this when you observe what the peak
                # would have been without optimization"), and nothing had ever called it: the
                # savings calculator fell back on `baseline = peak * 1.176` every single time, so a
                # higher peak reported more "savings" and the sensor could never read zero.
                #
                # IT MUST BE `effective_power`, NOT `actual_power`. The other side of the
                # comparison is `peak_this_month`, which is get_monthly_peak_summary()["highest"],
                # which is the EFFECTIVE (tariff-weighted) peak. Feeding the baseline the unweighted
                # number compared the same quarter against itself: one 6.0 kW quarter at 02:00, with
                # the optimiser doing nothing at all, reported 150 SEK/month of "savings" - all of
                # it the night weighting - and flagged it as MEASURED. That is the very bug this
                # block was written to kill, re-introduced by the block itself.
                #
                # AND IT MUST COME FROM A BILLABLE SOURCE. Peak RECORDING accepts nibe_currents,
                # because the pump is the dominant controllable load and throttling against a
                # NIBE-only history is coherent. But this figure is MONEY, and the effect tariff
                # bills WHOLE-HOUSE grid import. A baseline built from a sensor that cannot see the
                # oven, the EV or the water heater is not a baseline for anything the owner pays.
                self.savings_calculator.update_baseline_peak(peak_event.effective_power)

            if peak_event:
                # The HIGHEST of the tracked peaks, never peak_event.effective_power:
                # record_quarter_measurement returns an event for ANY new entry while the top-3
                # list is still filling, so a 6.0 kW peak followed by a 2.0 kW quarter would
                # drop the monthly peak to 2.0 and weaken the threshold for the rest of the month.
                self.peak_this_month = self.effect.get_monthly_peak_summary()["highest"]
                _LOGGER.info("New monthly peak: %.2f kW", self.peak_this_month)

        except (AttributeError, KeyError, ValueError, TypeError) as err:
            _LOGGER.warning("Failed to update peak tracking: %s", err)

    async def _write_curve_offset(self, offset: float, *, force_write: bool = False) -> int | None:
        """The ONE way this integration reaches the heat pump. Return the applied integer.

        A coordinator that has been shut down is not a writer. `_do_aligned_refresh` runs on
        `hass.async_create_task`, NOT `entry.async_create_task`, so HA cannot cancel it on unload -
        and it is mid-flight for seconds, awaiting the weather forecast over the network. It used to
        run to the end and drive the pump anyway: the shutdown flag guarded the timer re-arm, and
        nothing consulted it here.

        The entry unloads on the reconfigure flow (swapping the power meter), a manual reload, a
        removal, or a restart - NOT on an options change, which hot-reloads.

        tests/unit/coordinator/test_an_unloaded_integration_does_not_drive_the_heat_pump.py
        tests/unit/test_which_things_actually_unload_the_entry.py
        """
        if self._shutdown_requested:
            _LOGGER.debug(
                "Coordinator is shut down - refusing to write offset %.2f°C to the pump. The entry "
                "is unloaded; an in-flight refresh does not get the last word.",
                offset,
            )
            return None

        return await self.nibe.set_curve_offset(offset, force_write=force_write)

    async def _write_enhanced_ventilation(
        self, enabled: bool, *, force_write: bool = False
    ) -> bool:
        """The ONE way this integration commands the exhaust fan. Returns whether it wrote.

        Same race as the curve offset: written from the control loop, so it rides the same in-flight
        refresh. Worse on a reload - the dead coordinator can switch the fan ON while the new one
        starts up believing it is off, and nothing is left that will ever turn it off again.
        """
        if self._shutdown_requested:
            _LOGGER.debug(
                "Coordinator is shut down - refusing to set enhanced ventilation to %s. The entry "
                "is unloaded; the fan is not its to command.",
                enabled,
            )
            return False

        return await self.nibe.set_enhanced_ventilation(enabled, force_write=force_write)

    async def async_set_offset(self, offset: float, *, force_write: bool = False) -> int | None:
        """Apply heating curve offset to NIBE system.

        Args:
            offset: Offset value in °C (-10 to +10)
            force_write: Bypass ordinary write suppression for a safety transition.

        Returns:
            Integer applied to NIBE, or None when no write reached the pump.
        """
        try:
            applied_offset = await self._write_curve_offset(offset, force_write=force_write)
            if applied_offset is None:
                return None
            self.current_offset = float(applied_offset)
            self.last_applied_offset = float(applied_offset)
            self.last_offset_timestamp = dt_util.utcnow()
            self._learned_data_changed = True  # Trigger save on shutdown
            _LOGGER.info("Applied offset: %d°C", applied_offset)
            return applied_offset
        except (HomeAssistantError, AttributeError, OSError, ValueError) as err:
            _LOGGER.error("Failed to apply offset: %s", err)
            raise

    async def set_optimization_enabled(self, enabled: bool) -> None:
        """Enable or disable optimization.

        Args:
            enabled: True to enable optimization, False to disable
        """
        if self._shutdown_requested:
            _LOGGER.debug("Coordinator is shut down - ignoring optimization mode change")
            return

        if enabled:
            _LOGGER.info("Optimization enabled")
            previous_data = dict(self.entry.data)
            enabled_data = dict(previous_data)
            enabled_data[CONF_ENABLE_OPTIMIZATION] = True
            self.hass.config_entries.async_update_entry(self.entry, data=enabled_data)
            try:
                # Resume normal optimization now rather than waiting for the next aligned tick.
                await self.async_refresh_and_apply()
            except Exception:
                self.hass.config_entries.async_update_entry(self.entry, data=previous_data)
                raise
        else:
            _LOGGER.info("Optimization disabled - resetting offset to neutral")
            async with self._control_lock:
                applied_offset = await self.async_set_offset(0.0, force_write=True)
                await self._cancel_our_dhw_boost()
                is_enhanced = await self.nibe.is_enhanced_ventilation_active()
                if is_enhanced is None and self.nibe.has_ventilation_control:
                    # The fan may be enhanced and the switch cannot say. OFF must not be
                    # displayed until every owned actuator is KNOWN neutral.
                    raise HomeAssistantError(
                        "Cannot confirm the NIBE ventilation is back to normal - "
                        "the ventilation switch is unavailable"
                    )
                if is_enhanced:
                    ventilation_stopped = await self._write_enhanced_ventilation(
                        False,
                        force_write=True,
                    )
                    if not ventilation_stopped:
                        raise HomeAssistantError(
                            "Could not return NIBE ventilation to its normal setting"
                        )
                    self._airflow_enhance_start = None
                    self._airflow_normal_since = dt_util.utcnow()
            if applied_offset != 0:
                raise HomeAssistantError("Could not reset the NIBE heating offset to neutral")

            disabled_data = dict(self.entry.data)
            disabled_data[CONF_ENABLE_OPTIMIZATION] = False
            self.hass.config_entries.async_update_entry(self.entry, data=disabled_data)

    async def async_start_dhw_boost(self, duration_minutes: int, now_time: datetime) -> None:
        """Start a user-commanded hot-water boost that price optimization may not cancel.

        Only safety outranks the user: the thermal-debt abort still stops it, and so does
        unload. The duration is real - EffektGuard turns the switch back off when it expires,
        through the same owned door the cleanup uses - so the service argument means what it
        says instead of being validated and discarded. NIBE's own lux timeout still applies
        underneath; whichever ends first wins.
        """
        if not self.optimization_enabled:
            raise HomeAssistantError(
                "EffektGuard is OFF. Turn optimization on before boosting hot water."
            )

        if not await self._set_temporary_lux(True):
            raise HomeAssistantError(
                f"Could not start the hot-water boost on {self.temp_lux_entity}"
            )
        self._service_boost_until = now_time + timedelta(minutes=duration_minutes)

    async def async_apply_manual_override(self, offset: float, duration_minutes: int) -> None:
        """Apply one explicit user heating command through the locked control path."""
        if not self.entry.data.get("enable_optimization", True):
            raise HomeAssistantError(
                "EffektGuard is OFF. Turn optimization on before commanding a heating offset."
            )

        self.engine.set_manual_override(offset, duration_minutes)
        try:
            await self.async_refresh_and_apply(explicit_command=True)
        except Exception:
            self.engine.clear_manual_override()
            raise

    async def async_update_config(self, options: "EffektGuardConfigDict") -> None:
        """Update configuration without full reload.

        Allows hot-reload of runtime options like target temperature,
        thermal mass, and DHW settings without restarting the integration.

        Args:
            options: Dictionary of updated option values
        """
        _LOGGER.debug("Updating configuration: %s", options)

        # Update decision engine cached configuration values
        # CRITICAL: Decision engine caches these at init, must update them here
        if "target_indoor_temp" in options:
            self.engine.target_temp = float(options["target_indoor_temp"])
            _LOGGER.debug("Updated target temperature: %.1f°C", self.engine.target_temp)

        if "tolerance" in options:
            self.engine.tolerance = float(options["tolerance"])
            # The constant, not a second copy of 0.4. A number that lives in two places is a
            # number that will eventually disagree with itself.
            self.engine.tolerance_range = self.engine.tolerance * TOLERANCE_RANGE_MULTIPLIER
            _LOGGER.debug(
                "Updated tolerance: %.1f (range: %.1f°C)",
                self.engine.tolerance,
                self.engine.tolerance_range,
            )

        # Update thermal model parameters
        if "thermal_mass" in options:
            self.engine.thermal.thermal_mass = options["thermal_mass"]
            _LOGGER.debug("Updated thermal mass: %.2f", options["thermal_mass"])

        if "insulation_quality" in options:
            self.engine.thermal.insulation_quality = options["insulation_quality"]
            _LOGGER.debug("Updated insulation quality: %.2f", options["insulation_quality"])

        # Update optimization mode and recalculate mode config
        if "optimization_mode" in options:
            self.engine.config["optimization_mode"] = options["optimization_mode"]
            self.engine.update_mode_config()  # Recalculate mode-specific settings
            _LOGGER.debug("Updated optimization mode: %s", options["optimization_mode"])

        # Update control priority (note: stored in config dict, not cached)
        if "control_priority" in options:
            self.engine.config["control_priority"] = options["control_priority"]
            _LOGGER.debug("Updated control priority: %s", options["control_priority"])

        # Update switch states (stored in config dict, checked by layers)
        switch_keys = {
            "enable_optimization",
            "enable_peak_protection",
            "enable_price_optimization",
            "enable_weather_prediction",
            "enable_hot_water_optimization",
        }
        for key in switch_keys:
            if key in options:
                self.engine.config[key] = options[key]
                _LOGGER.debug("Updated switch %s: %s", key, options[key])

        # Update peak protection margin (note: stored in config dict, not cached)
        if "peak_protection_margin" in options:
            self.engine.config["peak_protection_margin"] = options["peak_protection_margin"]
            _LOGGER.debug("Updated peak protection margin: %.2f", options["peak_protection_margin"])

        # Update DHW settings
        dhw_config_changed = False
        dhw_keys = {
            "dhw_morning_hour",
            "dhw_morning_enabled",
            "dhw_evening_hour",
            "dhw_evening_enabled",
            "dhw_target_temp",
            CONF_DHW_MIN_AMOUNT,  # Include min_amount to trigger rebuild when changed
        }

        if any(key in options for key in dhw_keys):
            dhw_config_changed = True

        if dhw_config_changed:
            # Rebuild DHW demand periods
            dhw_target = float(options.get("dhw_target_temp", DEFAULT_DHW_TARGET_TEMP))
            # Get user-configured minimum hot water amount (default 5 minutes)
            dhw_min_amount = int(options.get(CONF_DHW_MIN_AMOUNT, DHW_MIN_AMOUNT_DEFAULT))
            demand_periods = []

            if options.get("dhw_morning_enabled", True):
                morning_hour = int(options.get("dhw_morning_hour", DEFAULT_DHW_MORNING_HOUR))
                demand_periods.append(
                    DHWDemandPeriod(
                        availability_hour=morning_hour,
                        target_temp=dhw_target,
                        duration_hours=2,
                        min_amount_minutes=dhw_min_amount,  # Apply user's configured min amount
                    )
                )

            if options.get("dhw_evening_enabled", True):
                evening_hour = int(options.get("dhw_evening_hour", DEFAULT_DHW_EVENING_HOUR))
                demand_periods.append(
                    DHWDemandPeriod(
                        availability_hour=evening_hour,
                        target_temp=dhw_target,
                        duration_hours=3,
                        min_amount_minutes=dhw_min_amount,  # Apply user's configured min amount
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

        # Handle airflow optimization toggle (F750/F730 exhaust air heat pumps)
        # Note: Optimizer is always created, this just controls whether it applies changes
        if CONF_ENABLE_AIRFLOW_OPTIMIZATION in options:
            airflow_enabled = options[CONF_ENABLE_AIRFLOW_OPTIMIZATION]
            _LOGGER.debug("Updated airflow optimization: %s", airflow_enabled)

            # If disabling, turn off enhanced ventilation if active
            if not airflow_enabled:
                self._airflow_enhance_start = None
                try:
                    if await self.nibe.is_enhanced_ventilation_active():
                        await self._write_enhanced_ventilation(False)
                        _LOGGER.info("Disabled enhanced ventilation on airflow optimizer disable")
                except (AttributeError, ValueError, OSError) as err:
                    _LOGGER.warning("Failed to disable enhanced ventilation: %s", err)

        # Update airflow rates if changed
        if self.airflow_optimizer:
            if CONF_AIRFLOW_STANDARD_RATE in options:
                self.airflow_optimizer.flow_standard = float(options[CONF_AIRFLOW_STANDARD_RATE])
                _LOGGER.debug(
                    "Updated airflow standard rate: %.0f m³/h", options[CONF_AIRFLOW_STANDARD_RATE]
                )
            if CONF_AIRFLOW_ENHANCED_RATE in options:
                self.airflow_optimizer.flow_enhanced = float(options[CONF_AIRFLOW_ENHANCED_RATE])
                _LOGGER.debug(
                    "Updated airflow enhanced rate: %.0f m³/h", options[CONF_AIRFLOW_ENHANCED_RATE]
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
    def optimization_enabled(self) -> bool:
        """Whether the master control gate permits explicit and automatic writes."""
        return self.entry.data.get(CONF_ENABLE_OPTIMIZATION, True)

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

            # Learning observes on its own clock, not the control loop's.
            #
            # The BT1 indoor sensor reports to 0.1 C. A house warming at a brisk 0.6 C/h moves
            # 0.05 C in five minutes - half a sensor tick - so an observation per control cycle
            # records the quantisation, not the building. The thermal PREDICTOR below still wants
            # every cycle: it tracks short-term trend, where five-minute resolution is the point.
            # The learner is asking a different question on a different timescale, and a building's
            # time constant is hours (LEARNING_OBSERVATION_INTERVAL_MINUTES, audit F-132).
            since_last = (
                None
                if self._last_learning_observation is None
                else now - self._last_learning_observation
            )
            if since_last is None or since_last >= timedelta(
                minutes=LEARNING_OBSERVATION_INTERVAL_MINUTES
            ):
                self.adaptive_learning.record_observation(
                    timestamp=now,
                    indoor_temp=nibe_data.indoor_temp,
                    outdoor_temp=nibe_data.outdoor_temp,
                    heating_offset=current_offset,
                )
                self._last_learning_observation = now

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

                if self._last_weather_record_date != current_date:
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
                "version": LEARNING_STORAGE_VERSION,
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

            # Save DHW optimizer state (critical for Legionella safety, heating rate learning)
            if self.dhw_optimizer:
                dhw_state = self.dhw_optimizer.get_dhw_state_for_persistence()
                if dhw_state:
                    learned_data["dhw_state"] = dhw_state

            await self.learning_store.async_save(learned_data)
            _LOGGER.debug("Saved learned data to storage")

        except (OSError, ValueError, KeyError, AttributeError) as err:
            _LOGGER.error("Failed to save learned data: %s", err, exc_info=True)

    async def _load_learned_data(self) -> dict[str, object] | None:
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

            _LOGGER.debug("No learned data in storage yet")
            return None

        except (OSError, ValueError, KeyError) as err:
            _LOGGER.warning("Failed to load learned data: %s", err)
            return None

    async def _save_thermal_predictor_immediate(self) -> None:
        """Save thermal predictor state immediately (throttled to avoid excessive disk writes).

        Called after each temperature trend update to persist state_history across reboots.
        Uses throttling to limit disk writes to UPDATE_INTERVAL_MINUTES
        (same as coordinator updates).
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
                existing_data["version"] = existing_data.get("version", LEARNING_STORAGE_VERSION)

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
                existing_data["version"] = existing_data.get("version", LEARNING_STORAGE_VERSION)

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
