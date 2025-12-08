"""DHW (Domestic Hot Water) optimizer for EffektGuard.

Intelligent DHW scheduling that coordinates with space heating to maximize
efficiency while preventing thermal debt accumulation.

Based on POST_PHASE_5_ROADMAP.md Phase 8 research and Forum_Summary.md findings.

Priority order:
1. Space heating comfort (indoor temp > target - 0.5°C)
2. DHW safety minimum (≥30°C for price optimization, ≥20°C critical)
3. Thermal debt prevention (climate-aware DM thresholds)
4. Space heating target (±0.3°C)
5. DHW comfort (50°C normal)
6. NIBE's automatic Legionella prevention (MONITOR ONLY - we don't trigger it)

DHW Heating Speed (enoch85):
- Tank heat-up: 1-2 hours (MUCH faster than UFH space heating!)
- No thermal mass delays like concrete slab UFH (6+ hours)
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict

from ..const import (
    COMPRESSOR_MIN_CYCLE_MINUTES,
    DEFAULT_DHW_TARGET_TEMP,
    DHW_COOLING_RATE,
    DHW_EXTENDED_RUNTIME_MINUTES,
    DHW_HEATING_TIME_HOURS,
    DHW_LEGIONELLA_DETECT,
    DHW_LEGIONELLA_MAX_DAYS,
    DHW_LEGIONELLA_PREVENT_TEMP,
    DHW_MAX_TEMP_VALIDATION,
    DHW_MAX_WAIT_HOURS,
    DHW_NORMAL_RUNTIME_MINUTES,
    DHW_PREHEAT_TARGET_OFFSET,
    DHW_SAFETY_CRITICAL,
    DHW_SAFETY_MIN,
    DHW_SAFETY_RUNTIME_MINUTES,
    DHW_SCHEDULING_WINDOW_MAX,
    DHW_SCHEDULING_WINDOW_MIN,
    DHW_SPACE_HEATING_DEFICIT_THRESHOLD,
    DHW_SPACE_HEATING_OUTDOOR_THRESHOLD,
    DHW_TARGET_HIGH_DEMAND,
    DHW_TREND_DEFICIT_THRESHOLD,
    DHW_TREND_RATE_THRESHOLD,
    DHW_URGENT_DEMAND_HOURS,
    DHW_URGENT_RUNTIME_MINUTES,
    DM_DHW_ABORT_FALLBACK,
    DM_DHW_BLOCK_FALLBACK,
    DM_RECOVERY_SAFETY_BUFFER,
    DM_THRESHOLD_START,
    MIN_DHW_TARGET_TEMP,
    SPACE_HEATING_DEMAND_DROP_HOURS,
    SPACE_HEATING_DEMAND_HIGH_THRESHOLD,
    SPACE_HEATING_DEMAND_LOW_THRESHOLD,
    SPACE_HEATING_DEMAND_MODERATE_THRESHOLD,
)
from .thermal_layer import estimate_dm_recovery_time
from .price_layer import CheapestWindowResult, PriceAnalyzer

_LOGGER = logging.getLogger(__name__)


class DemandPeriodInfoDict(TypedDict):
    """Information about an upcoming demand period."""

    hours_until: float
    target_temp: float


class DHWScheduleWindowDict(TypedDict):
    """Recommended DHW heating window."""

    start_time: datetime
    end_time: datetime
    reason: str


@dataclass
class DHWScheduleDecision:
    """DHW scheduling decision with safety conditions.

    IMPORTANT: max_runtime_minutes is for MONITORING/LOGGING only, not auto-off control.
    NIBE controls actual DHW completion based on BT7 reaching target temperature.
    We monitor abort_conditions (thermal debt, indoor temp) to decide if we should
    request NIBE to stop heating early via temp lux switch.

    Workflow:
    1. Decision says should_heat=True → Turn on temp lux switch
    2. NIBE heats DHW until BT7 reaches target OR coordinator detects abort condition
    3. If abort condition hit → Turn off temp lux switch early
    4. max_runtime_minutes is reference for logging/diagnostics, not enforced timer
    5. After heating completes, DHW_CONTROL_MIN_INTERVAL_MINUTES cooldown applies

    The coordinator respects DHW_CONTROL_MIN_INTERVAL_MINUTES (60 min) between
    control actions to avoid switch spam and allow NIBE to complete heating cycles.
    """

    should_heat: bool
    priority_reason: str
    target_temp: float
    max_runtime_minutes: int  # Reference for logging - not enforced auto-off timer
    abort_conditions: list[str]  # Monitor these to decide early abort
    recommended_start_time: datetime | None = None

    def __post_init__(self):
        """Validate decision consistency.

        Ensures user always knows when DHW will be available.
        Prevents regression to "Unknown" sensor state.
        """
        # If not heating, MUST provide next opportunity
        if not self.should_heat and self.recommended_start_time is None:
            raise ValueError(
                f"DHW decision '{self.priority_reason}' blocks heating "
                f"but doesn't provide recommended_start_time. "
                f"User needs to know when DHW will resume!"
            )

        # If heating now, recommended_start_time should be current time or None
        # (None is acceptable for immediate heating as sensor shows "pending")


@dataclass
class DHWRecommendation:
    """Complete DHW recommendation result.

    Contains all information needed for display and control:
    - Human-readable recommendation and summary
    - Machine-readable planning details
    - Raw decision for control logic

    Moved from coordinator._calculate_dhw_recommendation for shared reuse.
    """

    recommendation: str  # Human-readable recommendation text
    summary: str  # Multi-line planning summary for display
    details: dict  # Machine-readable planning details
    decision: DHWScheduleDecision | None  # Raw decision for control logic


@dataclass
class DHWDemandPeriod:
    """User-defined high DHW demand period."""

    start_hour: int  # 0-23
    target_temp: float  # Target DHW temp by this time
    duration_hours: int  # How long high demand lasts

    def __post_init__(self):
        """Validate types and ranges after initialization.

        Catches configuration errors early before they cause runtime failures.
        """
        # Type validation
        if not isinstance(self.start_hour, int):
            raise TypeError(
                f"start_hour must be int, got {type(self.start_hour).__name__}. "
                f"Check config flow conversion."
            )

        if not isinstance(self.duration_hours, int):
            raise TypeError(f"duration_hours must be int, got {type(self.duration_hours).__name__}")

        # Range validation
        if not 0 <= self.start_hour <= 23:
            raise ValueError(f"start_hour must be 0-23, got {self.start_hour}")

        if not 0 < self.duration_hours <= 24:
            raise ValueError(f"duration_hours must be 1-24, got {self.duration_hours}")

        if self.target_temp < DHW_SAFETY_MIN or self.target_temp > DHW_MAX_TEMP_VALIDATION:
            raise ValueError(
                f"target_temp must be {DHW_SAFETY_MIN}-{DHW_MAX_TEMP_VALIDATION}°C (safety range), "
                f"got {self.target_temp}"
            )


class IntelligentDHWScheduler:
    """Intelligent DHW scheduling with thermal debt prevention.

    IMPORTANT: This class MONITORS NIBE's automatic Legionella protection,
    it does NOT trigger Legionella boosts itself to avoid waste:
    - NIBE has fixed weekly 65°C schedule (can't be changed via API)
    - If we trigger boost on day 6, NIBE still runs on day 7 = WASTE
    - Instead: We detect when NIBE does it, track timing, avoid conflicts

    User should manually set NIBE's schedule to cheap hours (e.g., Sunday 3 AM).

    CLIMATE-AWARE THRESHOLDS:
    DHW blocking thresholds are now dynamic based on climate zone and outdoor temperature.
    Uses ClimateZoneDetector to determine appropriate DM thresholds for current conditions.
    Fallback constants from const.py used only if climate detector unavailable.

    SHARED LAYER REUSE (Phase 10):
    When emergency_layer is provided, uses EmergencyLayer.should_block_dhw() for
    consistent thermal debt blocking logic with space heating optimization.
    """

    def __init__(
        self,
        demand_periods: list[DHWDemandPeriod] | None = None,
        climate_detector=None,  # Climate zone detector for dynamic thresholds
        user_target_temp: float | None = None,  # User-configured DHW target temperature
        emergency_layer=None,  # Optional EmergencyLayer for shared DM blocking logic
        price_analyzer: PriceAnalyzer | None = None,  # Optional for shared price logic
    ):
        """Initialize DHW scheduler.

        Args:
            demand_periods: User-defined high demand periods (e.g., morning showers)
            climate_detector: Optional ClimateZoneDetector for dynamic DM thresholds
            user_target_temp: User-configured DHW target temperature (°C)
            emergency_layer: Optional EmergencyLayer for shared thermal debt blocking logic.
                If provided, uses emergency_layer.should_block_dhw() for consistent
                behavior with space heating optimization.
            price_analyzer: Optional PriceAnalyzer for shared price forecast logic.
                If provided, uses price_analyzer.find_cheapest_window() for consistent
                window search with space heating optimization.
        """
        self.demand_periods = demand_periods or []
        self.climate_detector = climate_detector
        self.emergency_layer = emergency_layer
        self.price_analyzer = price_analyzer
        self.last_legionella_boost: datetime | None = None
        self.bt7_history: deque = deque(maxlen=48)  # 12 hours @ 15-min intervals

        # Set user target temperature
        # Priority: 1) Explicit parameter, 2) First demand period, 3) DEFAULT_DHW_TARGET_TEMP
        if user_target_temp is not None:
            self.user_target_temp = user_target_temp
        elif self.demand_periods:
            self.user_target_temp = self.demand_periods[0].target_temp
        else:
            self.user_target_temp = DEFAULT_DHW_TARGET_TEMP

        if self.emergency_layer:
            _LOGGER.info(
                "DHW optimizer initialized with shared EmergencyLayer for thermal debt blocking"
            )
        elif self.climate_detector:
            _LOGGER.info(
                "DHW optimizer initialized with climate-aware thresholds: %s zone",
                self.climate_detector.zone_info.name,
            )
        else:
            _LOGGER.warning(
                "DHW optimizer initialized without climate detector, using fallback thresholds "
                "(DM block: %.0f, abort: %.0f)",
                DM_DHW_BLOCK_FALLBACK,
                DM_DHW_ABORT_FALLBACK,
            )

        if self.price_analyzer:
            _LOGGER.info("DHW optimizer initialized with shared PriceAnalyzer for window search")

    def update_bt7_temperature(self, temp: float, timestamp: datetime) -> None:
        """Update BT7/BT6 temperature history and detect NIBE's Legionella boost.

        Call this every coordinator update to track DHW temperature and
        automatically detect when NIBE's automatic Legionella boost runs.

        Note: Despite the name, this can accept either BT7 (top) or BT6 (charging)
        temperature. BT7 is preferred for Legionella detection (peaks at 65°C),
        but BT6 works as fallback if BT7 not available.

        Args:
            temp: Current DHW temperature (BT7 top preferred, BT6 charging acceptable)
            timestamp: Current timestamp
        """
        self.bt7_history.append((timestamp, temp))

        # Detect if NIBE just completed Legionella boost
        if self._detect_legionella_boost_completion():
            self.last_legionella_boost = timestamp
            _LOGGER.info(
                "NIBE Legionella boost detected (automatic): peaked at %.1f°C at %s",
                max(t for _, t in self.bt7_history),
                timestamp,
            )

    def _detect_legionella_boost_completion(self) -> bool:
        """Detect when NIBE's automatic Legionella boost just completed.

        Detection logic:
        - BT7 peaked at ≥56°C recently (real-world max observed with immersion heater)
        - Now cooling down (dropped 3°C from peak)

        Returns:
            True if boost detected, False otherwise
        """
        if len(self.bt7_history) < 10:
            return False

        # Get recent temperatures
        recent_temps = [temp for _, temp in self.bt7_history]
        max_temp = max(recent_temps)
        current_temp = recent_temps[-1]

        # Check if peaked above Legionella threshold and now cooling
        if max_temp >= DHW_LEGIONELLA_DETECT and current_temp < (max_temp - 3.0):
            # Make sure we haven't already recorded this boost
            if self.last_legionella_boost:
                time_since_last = (
                    self.bt7_history[-1][0] - self.last_legionella_boost
                ).total_seconds() / 3600
                if time_since_last < 6.0:  # Same boost event
                    return False
            return True

        return False

    async def initialize_from_history(self, hass, bt7_entity_id: str) -> None:
        """Initialize BT7 temperature history from Home Assistant recorder.

        On startup/restart, load past 14 days of BT7 temperature data to:
        1. Check if a high-temp cycle (≥56°C) occurred recently
        2. Auto-detect last Legionella boost without manual tracking
        3. Populate recent bt7_history (last 12h) for real-time detection

        This makes the system resilient to restarts - we don't lose track of
        Legionella cycles just because Home Assistant rebooted.

        Args:
            hass: Home Assistant instance
            bt7_entity_id: Entity ID for BT7 sensor (e.g., sensor.f750_hot_water_top_bt7)
        """
        try:
            from homeassistant.components import recorder
            from homeassistant.components.recorder import history
            import homeassistant.util.dt as dt_util

            # Get recorder instance
            if not recorder.is_entity_recorded(hass, bt7_entity_id):
                _LOGGER.warning(
                    "BT7 sensor %s not recorded by Home Assistant - cannot load history",
                    bt7_entity_id,
                )
                return

            # Load past 14 days of BT7 temperature data (Legionella tracking window)
            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=DHW_LEGIONELLA_MAX_DAYS)

            _LOGGER.debug(
                "Loading BT7 temperature history from %s to %s (%.0f days)",
                start_time,
                end_time,
                DHW_LEGIONELLA_MAX_DAYS,
            )

            # Get state history
            states = await recorder.get_instance(hass).async_add_executor_job(
                history.state_changes_during_period,
                hass,
                start_time,
                end_time,
                bt7_entity_id,
            )

            if not states or bt7_entity_id not in states:
                _LOGGER.info("No BT7 history available for Legionella detection")
                return

            # Process states and populate bt7_history
            state_list = states[bt7_entity_id]
            max_temp_seen = 0.0
            max_temp_time = None

            for state in state_list:
                try:
                    temp = float(state.state)
                    timestamp = state.last_changed

                    # Add to recent history for Legionella detection
                    # (limited to last 48 entries = 12 hours @ 15min for peak detection)
                    self.bt7_history.append((timestamp, temp))

                    # Track maximum temperature seen
                    if temp > max_temp_seen:
                        max_temp_seen = temp
                        max_temp_time = timestamp

                except (ValueError, TypeError):
                    continue

            _LOGGER.info(
                "Loaded %d BT7 temperature readings from past %.0f days (max: %.1f°C at %s)",
                len(state_list),
                DHW_LEGIONELLA_MAX_DAYS,
                max_temp_seen,
                max_temp_time,
            )

            # If we saw a high-temp cycle (≥56°C) in the past 14 days, record it
            if max_temp_seen >= DHW_LEGIONELLA_DETECT and max_temp_time:
                # Check if this is recent enough (within 14 days tracking window)
                hours_ago = (end_time - max_temp_time).total_seconds() / 3600.0

                if hours_ago <= 24.0 * DHW_LEGIONELLA_MAX_DAYS:  # Within tracking window
                    self.last_legionella_boost = max_temp_time
                    _LOGGER.info(
                        "Detected previous Legionella boost from history: %.1f°C at %s (%.1f hours ago)",
                        max_temp_seen,
                        max_temp_time,
                        hours_ago,
                    )

        except Exception as err:
            _LOGGER.error(
                "Failed to load BT7 history for Legionella detection: %s", err, exc_info=True
            )

    def should_start_dhw(
        self,
        current_dhw_temp: float,
        space_heating_demand_kw: float,
        thermal_debt_dm: float,
        indoor_temp: float,
        target_indoor_temp: float,
        outdoor_temp: float,
        price_classification: str,  # "cheap", "normal", "expensive", "peak"
        current_time: datetime,
        price_periods: list | None = None,  # QuarterPeriod list for window scheduling
        hours_since_last_dhw: float | None = None,  # Hours since last DHW heating
    ) -> DHWScheduleDecision:
        """Decide if DHW heating should start based on priority rules.

        Uses climate-aware DM thresholds that adapt to local conditions and current temperature.

        Args:
            current_dhw_temp: Current DHW temperature (°C)
            space_heating_demand_kw: Current space heating demand (kW)
            thermal_debt_dm: Current degree minutes
            indoor_temp: Current indoor temperature (°C)
            target_indoor_temp: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            price_classification: Current electricity price classification
            current_time: Current datetime
            price_periods: Optional list of QuarterPeriod for window-based scheduling
            hours_since_last_dhw: Optional hours since last DHW heating (for max wait check)

        Returns:
            DHWScheduleDecision with recommendation and abort conditions
        """
        indoor_deficit = target_indoor_temp - indoor_temp

        # Determine DM blocking using shared EmergencyLayer or fallback to local logic
        # Phase 10: EmergencyLayer provides consistent thermal mass-adjusted thresholds
        should_block_for_thermal_debt = False
        dm_block_threshold: float
        dm_abort_threshold: float

        if self.emergency_layer:
            # Use shared EmergencyLayer for consistent thermal debt blocking
            should_block_for_thermal_debt = self.emergency_layer.should_block_dhw(
                thermal_debt_dm, outdoor_temp
            )
            # Get thresholds from emergency layer's climate detector for logging/abort conditions
            if self.emergency_layer.climate_detector:
                dm_thresholds = self.emergency_layer.climate_detector.get_expected_dm_range(
                    outdoor_temp
                )
                dm_block_threshold = dm_thresholds["warning"]
                dm_abort_threshold = dm_thresholds["warning"] - 80
            else:
                dm_block_threshold = DM_DHW_BLOCK_FALLBACK
                dm_abort_threshold = DM_DHW_ABORT_FALLBACK

            _LOGGER.debug(
                "DHW using shared EmergencyLayer: should_block=%s, DM=%.0f, outdoor=%.1f°C",
                should_block_for_thermal_debt,
                thermal_debt_dm,
                outdoor_temp,
            )
        elif self.climate_detector:
            # Fallback to local climate detector
            dm_thresholds = self.climate_detector.get_expected_dm_range(outdoor_temp)
            dm_block_threshold = dm_thresholds["warning"]  # Use warning threshold for blocking
            dm_abort_threshold = dm_thresholds["warning"] - 80  # 80 DM buffer before critical
            should_block_for_thermal_debt = thermal_debt_dm <= dm_block_threshold

            _LOGGER.debug(
                "DHW DM thresholds for %.1f°C (zone: %s): block=%.0f, abort=%.0f",
                outdoor_temp,
                self.climate_detector.zone_info.name,
                dm_block_threshold,
                dm_abort_threshold,
            )
        else:
            # Fallback to fixed thresholds from const.py if climate detector unavailable
            dm_block_threshold = DM_DHW_BLOCK_FALLBACK
            dm_abort_threshold = DM_DHW_ABORT_FALLBACK
            should_block_for_thermal_debt = thermal_debt_dm <= dm_block_threshold

        # === RULE 1: CRITICAL THERMAL DEBT - NEVER START DHW ===
        if should_block_for_thermal_debt:
            _LOGGER.warning(
                "DHW blocked by RULE 1 (thermal debt): DM=%.0f ≤ threshold (zone: %s, outdoor: %.1f°C)",
                thermal_debt_dm,
                (
                    self.emergency_layer.climate_detector.zone_info.name
                    if self.emergency_layer and self.emergency_layer.climate_detector
                    else (
                        self.climate_detector.zone_info.name if self.climate_detector else "unknown"
                    )
                ),
                outdoor_temp,
            )
            # Find next opportunity using centralized logic
            next_opportunity = self._find_next_dhw_opportunity(
                current_time=current_time,
                current_dhw_temp=current_dhw_temp,
                thermal_debt_dm=thermal_debt_dm,
                outdoor_temp=outdoor_temp,
                price_periods=price_periods,
                blocking_reason="CRITICAL_THERMAL_DEBT",
                dm_block_threshold=dm_block_threshold,
            )

            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="CRITICAL_THERMAL_DEBT",
                target_temp=self.user_target_temp,
                max_runtime_minutes=0,
                abort_conditions=[],
                recommended_start_time=next_opportunity,
            )

        # === RULE 2: SPACE HEATING EMERGENCY - HOUSE TOO COLD ===
        if (
            indoor_deficit > DHW_SPACE_HEATING_DEFICIT_THRESHOLD
            and outdoor_temp < DHW_SPACE_HEATING_OUTDOOR_THRESHOLD
        ):
            # Find next opportunity using centralized logic
            next_opportunity = self._find_next_dhw_opportunity(
                current_time=current_time,
                current_dhw_temp=current_dhw_temp,
                thermal_debt_dm=thermal_debt_dm,
                outdoor_temp=outdoor_temp,
                price_periods=price_periods,
                blocking_reason="SPACE_HEATING_EMERGENCY",
                dm_block_threshold=dm_block_threshold,
            )

            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="SPACE_HEATING_EMERGENCY",
                target_temp=self.user_target_temp,
                max_runtime_minutes=0,
                abort_conditions=[],
                recommended_start_time=next_opportunity,
            )

        # === RULE 3: DHW SAFETY MINIMUM - MUST HEAT (Limited) ===
        # Safety minimum: Heat below 30°C BUT defer if:
        # - Price is expensive/peak AND thermal debt is NOT concerning
        # - This prevents peak billing hits when DHW can wait and space heating is healthy
        if current_dhw_temp < DHW_SAFETY_MIN:
            # Check if we should defer due to peak pricing + thermal debt
            # Only defer if temp is still safe (20-30°C range) and not critically low
            can_defer_for_peak = (
                current_dhw_temp >= DHW_SAFETY_CRITICAL  # Not critically low (>= 20°C)
                and price_classification in ["expensive", "peak"]  # High cost period
                and thermal_debt_dm > (dm_block_threshold + 20)  # DM healthy enough to defer
            )

            if can_defer_for_peak:
                # Find optimal window to heat during cheaper period
                # This prevents just "waiting and hoping" - actively schedule for best price
                if price_periods and self.price_analyzer:
                    lookahead_hours = self._get_lookahead_hours(current_time)
                    optimal_window = self.price_analyzer.find_cheapest_window(
                        current_time=current_time,
                        price_periods=price_periods,
                        duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                        lookahead_hours=lookahead_hours,
                    )

                    if optimal_window and optimal_window.hours_until <= 0.17:  # Within 10 min
                        # Optimal window is NOW - heat immediately despite expensive current price
                        _LOGGER.info(
                            "DHW safety: In optimal window at %s (%.1före/kWh) - heating now",
                            optimal_window.start_time.strftime("%H:%M"),
                            optimal_window.avg_price,
                        )
                        return DHWScheduleDecision(
                            should_heat=True,
                            priority_reason=f"DHW_SAFETY_WINDOW_Q{optimal_window.quarters[0]}_@{optimal_window.avg_price:.1f}öre",
                            target_temp=DHW_SAFETY_MIN,
                            max_runtime_minutes=DHW_SAFETY_RUNTIME_MINUTES,
                            abort_conditions=[
                                f"thermal_debt < {dm_abort_threshold:.0f}",
                                f"indoor_temp < {target_indoor_temp - 0.5}",
                            ],
                            recommended_start_time=optimal_window.start_time,
                        )
                    elif optimal_window:
                        # Wait for upcoming cheap window
                        _LOGGER.info(
                            "DHW safety defer: temp %.1f°C (safe >= %.1f°C), next window at %s (%.1fh, %.1före/kWh)",
                            current_dhw_temp,
                            DHW_SAFETY_CRITICAL,
                            optimal_window.start_time.strftime("%H:%M"),
                            optimal_window.hours_until,
                            optimal_window.avg_price,
                        )
                        return DHWScheduleDecision(
                            should_heat=False,
                            priority_reason=f"DHW_SAFETY_WAITING_WINDOW_IN_{optimal_window.hours_until:.1f}H_@{optimal_window.avg_price:.1f}",
                            target_temp=self.user_target_temp,
                            max_runtime_minutes=0,
                            abort_conditions=[],
                            recommended_start_time=optimal_window.start_time,
                        )

                # No optimal window found - just defer (fallback to old behavior)
                _LOGGER.info(
                    "DHW safety minimum defer: temp %.1f°C (safe >= %.1f°C), price=%s, DM=%.0f. "
                    "Waiting for better price to avoid peak billing.",
                    current_dhw_temp,
                    DHW_SAFETY_CRITICAL,
                    price_classification,
                    thermal_debt_dm,
                )
                # Find next opportunity using centralized logic
                next_opportunity = self._find_next_dhw_opportunity(
                    current_time=current_time,
                    current_dhw_temp=current_dhw_temp,
                    thermal_debt_dm=thermal_debt_dm,
                    outdoor_temp=outdoor_temp,
                    price_periods=price_periods,
                    blocking_reason="DHW_SAFETY_DEFERRED_PEAK_PRICE",
                    dm_block_threshold=dm_block_threshold,
                )

                return DHWScheduleDecision(
                    should_heat=False,
                    priority_reason="DHW_SAFETY_DEFERRED_PEAK_PRICE",
                    target_temp=self.user_target_temp,
                    max_runtime_minutes=0,
                    abort_conditions=[],
                    recommended_start_time=next_opportunity,
                )

            # Otherwise heat immediately (critically low or good price)
            # SMART TWO-TIER STRATEGY:
            # - Emergency heating: Heat to DHW_SAFETY_MIN (30°C) to get safe quickly
            # - Full heating: Wait for cheap prices to heat to user target
            # This prevents wasting money heating to full temp during expensive periods
            return DHWScheduleDecision(
                should_heat=True,
                priority_reason="DHW_SAFETY_MINIMUM",
                target_temp=DHW_SAFETY_MIN,  # Only heat to safe level, not full target
                max_runtime_minutes=DHW_SAFETY_RUNTIME_MINUTES,
                abort_conditions=[
                    f"thermal_debt < {dm_abort_threshold:.0f}",
                    f"indoor_temp < {target_indoor_temp - 0.5}",
                ],
                recommended_start_time=current_time,  # Heat NOW
            )

        # === RULE 2.3: HYGIENE BOOST (HIGH-TEMP CYCLE FOR LEGIONELLA PREVENTION) ===
        # If DHW hasn't been above 56°C in past 14 days, heat to 56°C during cheapest period
        # This prevents Legionella bacteria growth in the low-temp range (20-45°C)
        # with the new lower safety thresholds (10°C/20°C).
        #
        # REQUIRES DHW IMMERSION HEATER (Swedish: elpatron):
        # - Heat pump compressor can only reach ~50-55°C max (COP limitation)
        # - NIBE automatically engages DHW tank immersion heater for high-temp cycles
        # - Real-world observation: Max 56°C achieved with compressor + immersion heater
        # - This is normal operation for Legionella prevention in all NIBE systems
        # - Scheduling during cheap periods minimizes immersion heater cost
        #
        # NOTE: This is the DHW tank's built-in immersion heater (elpatron), NOT the
        # space heating auxiliary heater. They are separate electrical heating systems.
        #
        # References:
        # - Boverket.se: Water heaters should maintain ≥60°C (ideal), bacteria killed at high temps
        # - User observation: System reaches max 56°C with electrical boost (real-world constraint)
        # - Swedish forum: "Vp klarar inte 60°C, därför elpatron för legionella"
        #   (Heat pump can't reach 60°C, therefore immersion heater for Legionella)
        # - NIBE Menu 4.9.5: Built-in Legionella function uses immersion heater weekly/bi-weekly
        #
        # PRIORITY: Higher than emergency completion (bacteria prevention critical)
        days_since_legionella = None
        if self.last_legionella_boost:
            try:
                days_since_legionella = (
                    current_time - self.last_legionella_boost
                ).total_seconds() / 86400.0
            except TypeError:
                # Handle timezone mismatch between naive and aware datetimes
                # Convert both to naive for comparison
                current_naive = (
                    current_time.replace(tzinfo=None) if current_time.tzinfo else current_time
                )
                last_boost_naive = (
                    self.last_legionella_boost.replace(tzinfo=None)
                    if self.last_legionella_boost.tzinfo
                    else self.last_legionella_boost
                )
                days_since_legionella = (current_naive - last_boost_naive).total_seconds() / 86400.0

        if (
            days_since_legionella is None  # Never had high-temp cycle
            or days_since_legionella >= DHW_LEGIONELLA_MAX_DAYS  # 14+ days
        ) and price_classification == "cheap":  # Only during cheap periods
            _LOGGER.warning(
                "DHW hygiene boost needed: %s days since last high-temp cycle (limit %.0f days). "
                "Heating to %.0f°C for Legionella prevention (requires DHW immersion heater - "
                "compressor max ~50-55°C, real-world max observed %.0f°C with immersion heater).",
                "Never" if days_since_legionella is None else f"{days_since_legionella:.1f}",
                DHW_LEGIONELLA_MAX_DAYS,
                DHW_LEGIONELLA_PREVENT_TEMP,
                DHW_LEGIONELLA_DETECT,
            )
            _LOGGER.info(
                "Hygiene boost scheduled during cheap electricity period to minimize "
                "immersion heater cost. NIBE will use compressor to ~50-55°C, "
                "then DHW tank immersion heater to reach %.0f°C target.",
                DHW_LEGIONELLA_PREVENT_TEMP,
            )
            return DHWScheduleDecision(
                should_heat=True,
                priority_reason="DHW_HYGIENE_BOOST",
                target_temp=DHW_LEGIONELLA_PREVENT_TEMP,  # 60°C kills bacteria
                max_runtime_minutes=DHW_EXTENDED_RUNTIME_MINUTES,  # May take longer
                abort_conditions=[
                    f"thermal_debt < {dm_abort_threshold:.0f}",
                    f"indoor_temp < {target_indoor_temp - 0.5}",
                ],
                recommended_start_time=current_time,
            )

        # === RULE 2.5: COMPLETE EMERGENCY HEATING TO COMFORT LEVEL ===
        # After emergency heating reached DHW_SAFETY_MIN (30°C), complete to comfort level
        # during cheap prices. This is the second phase of two-tier emergency heating.
        if (
            DHW_SAFETY_MIN <= current_dhw_temp < MIN_DHW_TARGET_TEMP  # In 30-45°C range
            and price_classification == "cheap"  # Wait for cheap prices
        ):
            _LOGGER.info(
                "DHW completing emergency heating: %.1f°C → %.1f°C during cheap period",
                current_dhw_temp,
                self.user_target_temp,
            )
            return DHWScheduleDecision(
                should_heat=True,
                priority_reason="DHW_COMPLETE_EMERGENCY_HEATING",
                target_temp=self.user_target_temp,
                max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                abort_conditions=[
                    f"thermal_debt < {dm_abort_threshold:.0f}",
                    f"indoor_temp < {target_indoor_temp - 0.5}",
                ],
                recommended_start_time=current_time,
            )

        # === RULE 3.5: MAXIMUM WAIT TIME EXCEEDED ===
        # Must heat every 36 hours (hygiene, comfort)
        if hours_since_last_dhw and hours_since_last_dhw > DHW_MAX_WAIT_HOURS:
            # Only heat during cheap/normal prices (avoid expensive/peak even with max wait)
            if price_classification in ["cheap", "normal"]:
                _LOGGER.warning(
                    "DHW maximum wait exceeded: %.1f hours (limit %.0fh)",
                    hours_since_last_dhw,
                    DHW_MAX_WAIT_HOURS,
                )
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason=f"DHW_MAX_WAIT_EXCEEDED_{hours_since_last_dhw:.1f}H",
                    target_temp=self.user_target_temp,
                    max_runtime_minutes=DHW_EXTENDED_RUNTIME_MINUTES,
                    abort_conditions=[
                        f"thermal_debt < {dm_abort_threshold:.0f}",
                    ],
                    recommended_start_time=current_time,
                )

        # === RULE 4: HIGH SPACE HEATING DEMAND - DELAY DHW ===
        # NOTE: Legionella prevention is handled by NIBE automatically
        # We monitor when it happens (update_bt7_temperature) but don't trigger it
        # to avoid waste (NIBE runs on fixed schedule regardless of our triggers)
        if (
            space_heating_demand_kw > SPACE_HEATING_DEMAND_HIGH_THRESHOLD
            and thermal_debt_dm < DM_THRESHOLD_START
        ):
            # Find next opportunity using centralized logic
            next_opportunity = self._find_next_dhw_opportunity(
                current_time=current_time,
                current_dhw_temp=current_dhw_temp,
                thermal_debt_dm=thermal_debt_dm,
                outdoor_temp=outdoor_temp,
                price_periods=price_periods,
                blocking_reason="HIGH_SPACE_HEATING_DEMAND",
                dm_block_threshold=dm_block_threshold,
            )

            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="HIGH_SPACE_HEATING_DEMAND",
                target_temp=self.user_target_temp,
                max_runtime_minutes=0,
                abort_conditions=[],
                recommended_start_time=next_opportunity,
            )

        # === RULE 5: HIGH DEMAND PERIOD - TARGET TEMP BY START TIME ===
        # Check if we're approaching a user-defined high demand period
        # URGENT heating overrides price classification (except peak)
        upcoming_demand = self._check_upcoming_demand_period(current_time)
        if upcoming_demand:
            hours_until = upcoming_demand["hours_until"]
            target = upcoming_demand["target_temp"]

            # IMMEDIATE: Within 30 min of demand - heat regardless of price (except peak)
            if hours_until < DHW_URGENT_DEMAND_HOURS and current_dhw_temp < target:
                if price_classification != "peak":
                    return DHWScheduleDecision(
                        should_heat=True,
                        priority_reason=f"URGENT_DEMAND_IN_{hours_until * 60:.0f}MIN",
                        target_temp=target,
                        max_runtime_minutes=DHW_URGENT_RUNTIME_MINUTES,
                        abort_conditions=[
                            f"thermal_debt < {dm_abort_threshold:.0f}",
                            f"indoor_temp < {target_indoor_temp - 0.5}",
                        ],
                        recommended_start_time=current_time,
                    )

        # === RULE 6: SMART WINDOW-BASED SCHEDULING ===
        # If DHW is adequate, only heat during CHEAP periods
        # This mirrors space heating logic: adequate = no urgency = wait for cheap
        # Adequate temp = one shower guaranteed (morning/evening usage pattern)
        if current_dhw_temp >= MIN_DHW_TARGET_TEMP and price_classification != "cheap":
            _LOGGER.info(
                "DHW: Adequate (%.1f°C ≥ %.1f°C), price %s - no heating needed",
                current_dhw_temp,
                MIN_DHW_TARGET_TEMP,
                price_classification,
            )
            # Find next cheap opportunity
            next_opportunity = self._find_next_dhw_opportunity(
                current_time=current_time,
                current_dhw_temp=current_dhw_temp,
                thermal_debt_dm=thermal_debt_dm,
                outdoor_temp=outdoor_temp,
                price_periods=price_periods,
                blocking_reason=f"DHW_ADEQUATE_WAITING_CHEAP_{price_classification.upper()}",
                dm_block_threshold=dm_block_threshold,
            )
            return DHWScheduleDecision(
                should_heat=False,
                priority_reason=f"DHW_ADEQUATE_WAITING_CHEAP_{price_classification.upper()}",
                target_temp=self.user_target_temp,
                max_runtime_minutes=0,
                abort_conditions=[],
                recommended_start_time=next_opportunity,
            )

        # Find the absolute cheapest window first, then check if we should heat
        optimal_window = None  # Initialize to None for fallback logic

        if current_dhw_temp < (self.user_target_temp + 5.0):
            # Use window-based scheduling if price data available
            if price_periods and self.price_analyzer:
                lookahead_hours = self._get_lookahead_hours(current_time)
                optimal_window = self.price_analyzer.find_cheapest_window(
                    current_time=current_time,
                    price_periods=price_periods,
                    duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                    lookahead_hours=lookahead_hours,
                )

                if optimal_window is None:
                    # No price data - only heat during cheap periods if needed
                    if (
                        price_classification == "cheap"
                        and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
                    ):
                        return DHWScheduleDecision(
                            should_heat=True,
                            priority_reason="CHEAP_NO_WINDOW_DATA",
                            target_temp=self.user_target_temp + DHW_PREHEAT_TARGET_OFFSET,
                            max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                            abort_conditions=[
                                f"thermal_debt < {dm_abort_threshold:.0f}",
                                f"indoor_temp < {target_indoor_temp - 0.5}",
                            ],
                            recommended_start_time=current_time,
                        )

                # STEP 1: Check if we're in the optimal window (within 15 min of start)
                # Use 15 min buffer to accommodate 15-minute coordinator cycle
                # This ensures we catch the window even if update happens at 3:45, 3:50, or 3:55
                elif optimal_window.hours_until <= 0.25:  # 15 minutes
                    # This is the optimal time! Heat regardless of spare capacity
                    # Trust the window optimizer - it found the absolute cheapest period
                    # RULE 1 already protects against critical thermal debt
                    _LOGGER.info(
                        "DHW: Heating in optimal window at %s (%.1före/kWh), DM=%.0f",
                        optimal_window.start_time.strftime("%H:%M"),
                        optimal_window.avg_price,
                        thermal_debt_dm,
                    )
                    return DHWScheduleDecision(
                        should_heat=True,
                        priority_reason=f"OPTIMAL_WINDOW_Q{optimal_window.quarters[0]}_@{optimal_window.avg_price:.1f}öre",
                        target_temp=min(self.user_target_temp + DHW_PREHEAT_TARGET_OFFSET, 60.0),
                        max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                        abort_conditions=[
                            f"thermal_debt < {dm_abort_threshold:.0f}",
                            f"indoor_temp < {target_indoor_temp - 0.5}",
                        ],
                        recommended_start_time=optimal_window.start_time,
                    )

                # Wait for better window ahead (if DHW still comfortable)
                # Require window to be at least 10 min away to prevent premature activation
                elif current_dhw_temp >= MIN_DHW_TARGET_TEMP:  # ✅ FIXED - Changed from > to >=
                    _LOGGER.info(
                        "DHW: Comfortable (%.1f°C ≥ %.1f°C), waiting for optimal window at %s (%.1fh, %.1före/kWh)",
                        current_dhw_temp,
                        MIN_DHW_TARGET_TEMP,
                        optimal_window.start_time.strftime("%H:%M"),
                        optimal_window.hours_until,
                        optimal_window.avg_price,
                    )
                    return DHWScheduleDecision(
                        should_heat=False,
                        priority_reason=f"WAITING_OPTIMAL_WINDOW_IN_{optimal_window.hours_until:.1f}H_@{optimal_window.avg_price:.1f}",
                        target_temp=self.user_target_temp,
                        max_runtime_minutes=0,
                        abort_conditions=[],
                        recommended_start_time=optimal_window.start_time,
                    )

            # Fallback: No price data AND no optimal window ahead
            # Heat now if DHW getting low and price is cheap
            # Only use this fallback when we don't have a better window to wait for
            if (
                not price_periods  # No price data available
                and price_classification == "cheap"
                and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
            ):
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
                    target_temp=min(self.user_target_temp + DHW_PREHEAT_TARGET_OFFSET, 60.0),
                    max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                    abort_conditions=[
                        f"thermal_debt < {dm_abort_threshold:.0f}",
                        f"indoor_temp < {target_indoor_temp - 0.5}",
                    ],
                    recommended_start_time=current_time,
                )

        # === RULE 7: COMFORT HEATING (DHW GETTING LOW) ===
        # Heat below minimum user target during cheap prices (no window check - urgent enough)
        if current_dhw_temp < MIN_DHW_TARGET_TEMP and price_classification == "cheap":
            return DHWScheduleDecision(
                should_heat=True,
                priority_reason="DHW_COMFORT_LOW_CHEAP",
                target_temp=self.user_target_temp,
                max_runtime_minutes=DHW_SAFETY_RUNTIME_MINUTES,
                abort_conditions=[
                    f"thermal_debt < {dm_abort_threshold:.0f}",
                    f"indoor_temp < {target_indoor_temp - 0.5}",
                ],
                recommended_start_time=current_time,
            )

        # === RULE 8: ALL CONDITIONS FAIL - DON'T HEAT ===
        # Find next opportunity using centralized logic
        next_opportunity = self._find_next_dhw_opportunity(
            current_time=current_time,
            current_dhw_temp=current_dhw_temp,
            thermal_debt_dm=thermal_debt_dm,
            outdoor_temp=outdoor_temp,
            price_periods=price_periods,
            blocking_reason="DHW_ADEQUATE",
            dm_block_threshold=dm_block_threshold,
        )

        return DHWScheduleDecision(
            should_heat=False,
            priority_reason="DHW_ADEQUATE",
            target_temp=self.user_target_temp,
            max_runtime_minutes=0,
            abort_conditions=[],
            recommended_start_time=next_opportunity,
        )

    def _get_lookahead_hours(self, current_time: datetime) -> float:
        """Calculate how far ahead to look for cheap DHW windows.

        Uses shared PriceAnalyzer.calculate_lookahead_hours() for consistent
        behavior with space heating optimization.

        Args:
            current_time: Current datetime

        Returns:
            Hours to look ahead (1-24h range)
        """
        next_demand = self._check_upcoming_demand_period(current_time)
        next_demand_hours = next_demand["hours_until"] if next_demand else None

        if self.price_analyzer:
            return self.price_analyzer.calculate_lookahead_hours(
                heating_type="dhw",
                next_demand_hours=next_demand_hours,
            )

        # Fallback if no price_analyzer
        if next_demand_hours is not None:
            return max(1.0, min(next_demand_hours, 24.0))
        return 24.0

    def _estimate_dm_recovery_time(
        self, current_dm: float, target_dm: float, outdoor_temp: float
    ) -> float:
        """Estimate hours until DM recovers to target level.

        Delegates to shared estimate_dm_recovery_time() in thermal_layer.py
        for consistent behavior with space heating optimization.

        Args:
            current_dm: Current degree minutes value (negative)
            target_dm: Target DM to reach (less negative)
            outdoor_temp: Current outdoor temperature for recovery rate

        Returns:
            Estimated hours until recovery (minimum 0.5h, maximum 12h)
        """
        return estimate_dm_recovery_time(current_dm, target_dm, outdoor_temp)

    def _find_next_dhw_opportunity(
        self,
        current_time: datetime,
        current_dhw_temp: float,
        thermal_debt_dm: float,
        outdoor_temp: float,
        price_periods: list | None,
        blocking_reason: str,
        dm_block_threshold: float,
    ) -> datetime:
        """Find next time DHW can safely heat.

        Combines multiple factors to determine earliest safe opportunity:
        - Thermal debt recovery time
        - Cheap price windows
        - Temperature cooling rate
        - Blocking condition resolution

        Args:
            current_time: Current datetime
            current_dhw_temp: Current DHW temperature
            thermal_debt_dm: Current degree minutes
            outdoor_temp: Current outdoor temperature
            price_periods: Price data for window finding
            blocking_reason: Why DHW is currently blocked
            dm_block_threshold: DM threshold for DHW blocking

        Returns:
            Earliest safe opportunity datetime
        """
        # Calculate different constraint times
        opportunities = []

        # 1. Thermal debt recovery time
        if thermal_debt_dm < dm_block_threshold:
            recovery_hours = self._estimate_dm_recovery_time(
                current_dm=thermal_debt_dm,
                target_dm=dm_block_threshold + DM_RECOVERY_SAFETY_BUFFER,
                outdoor_temp=outdoor_temp,
            )
            opportunities.append(current_time + timedelta(hours=recovery_hours))

        # 2. Next cheap price window
        if price_periods and self.price_analyzer:
            lookahead_hours = self._get_lookahead_hours(current_time)
            cheap_window = self.price_analyzer.find_cheapest_window(
                current_time=current_time,
                price_periods=price_periods,
                duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                lookahead_hours=lookahead_hours,
            )
            if cheap_window:
                opportunities.append(cheap_window.start_time)

        # 3. Temperature cooling estimate (when will DHW need heating)
        temp_margin = current_dhw_temp - MIN_DHW_TARGET_TEMP
        if temp_margin > 0:
            hours_until_low = temp_margin / DHW_COOLING_RATE
            opportunities.append(current_time + timedelta(hours=hours_until_low))

        # Return earliest opportunity (or default to 2 hours if no constraints)
        if opportunities:
            next_opportunity = min(opportunities)
            _LOGGER.debug(
                "Next DHW opportunity for %s: %s (from %d constraints)",
                blocking_reason,
                next_opportunity.strftime("%Y-%m-%d %H:%M"),
                len(opportunities),
            )
            return next_opportunity
        else:
            return current_time + timedelta(hours=SPACE_HEATING_DEMAND_DROP_HOURS)

    def format_planning_summary(
        self,
        recommendation: str,
        current_temp: float,
        target_temp: float,
        thermal_debt: float,
        dm_thresholds: dict,
        space_heating_demand: float,
        price_classification: str,
        weather_opportunity: str | None,
    ) -> str:
        """Format human-readable DHW planning summary.

        Moved from coordinator._format_dhw_planning_summary for shared reuse.

        Args:
            recommendation: Base recommendation text
            current_temp: Current DHW temperature
            target_temp: Target DHW temperature
            thermal_debt: Current thermal debt (DM)
            dm_thresholds: Thermal debt thresholds (block, abort)
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
        if thermal_debt < dm_thresholds.get("abort", DM_DHW_ABORT_FALLBACK):
            status_text = f"CRITICAL (DM {thermal_debt:.0f})"
        elif thermal_debt < dm_thresholds.get("block", DM_DHW_BLOCK_FALLBACK):
            status_text = f"WARNING (DM {thermal_debt:.0f})"
        else:
            status_text = f"OK (DM {thermal_debt:.0f})"
        lines.append(f"Thermal Debt: {status_text}")

        # Space heating status
        if space_heating_demand > SPACE_HEATING_DEMAND_HIGH_THRESHOLD:
            lines.append(f"Heating Demand: HIGH ({space_heating_demand:.1f} kW)")
        elif space_heating_demand > SPACE_HEATING_DEMAND_MODERATE_THRESHOLD:
            lines.append(f"Heating Demand: MODERATE ({space_heating_demand:.1f} kW)")
        elif space_heating_demand > SPACE_HEATING_DEMAND_LOW_THRESHOLD:
            lines.append(f"Heating Demand: LOW ({space_heating_demand:.1f} kW)")
        else:
            lines.append(f"Heating Demand: MINIMAL ({space_heating_demand:.1f} kW)")

        # Weather opportunity
        if weather_opportunity:
            lines.append(f"Weather: {weather_opportunity}")

        lines.append("")
        lines.append(f"Recommendation: {recommendation}")

        return "\n".join(lines)

    def check_abort_conditions(
        self,
        abort_conditions: list[str],
        thermal_debt: float,
        indoor_temp: float,
        target_indoor: float,
    ) -> tuple[bool, str | None]:
        """Check if any DHW abort conditions are triggered.

        Moved from coordinator._check_dhw_abort_conditions for shared reuse.

        Abort conditions are returned by DHW optimizer to monitor during active heating.
        If triggered, we should stop DHW heating early to prioritize space heating.

        Args:
            abort_conditions: List of condition strings from DHW decision
                Examples: ["thermal_debt < -500", "indoor_temp < 21.5"]
            thermal_debt: Current degree minutes (DM) value
            indoor_temp: Current indoor temperature
            target_indoor: Target indoor temperature (currently unused but kept for future)

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

    def calculate_recommendation(
        self,
        current_dhw_temp: float,
        thermal_debt: float,
        space_heating_demand: float,
        outdoor_temp: float,
        indoor_temp: float,
        target_indoor: float,
        price_classification: str,
        current_time: datetime,
        price_periods: list | None,
        hours_since_last_dhw: float,
        thermal_trend_rate: float = 0.0,
        climate_zone_name: str | None = None,
        weather_current_temp: float | None = None,
    ) -> DHWRecommendation:
        """Calculate complete DHW heating recommendation.

        Pure logic moved from coordinator._calculate_dhw_recommendation.
        Takes all data as parameters (no HA dependencies), returns structured result.

        The coordinator should:
        1. Gather all HA-specific data (entry.data, notifications, history)
        2. Call this method with gathered data
        3. Handle HA-specific side effects (notifications)

        Args:
            current_dhw_temp: Current DHW temperature (°C)
            thermal_debt: Current degree minutes (DM)
            space_heating_demand: Current heating power (kW)
            outdoor_temp: Current outdoor temperature (°C)
            indoor_temp: Current indoor temperature (°C)
            target_indoor: Target indoor temperature (°C)
            price_classification: Price classification (cheap/normal/expensive/peak)
            current_time: Current datetime
            price_periods: Price period list for window scheduling
            hours_since_last_dhw: Hours since last DHW heating
            thermal_trend_rate: Indoor temp change rate (°C/hour)
            climate_zone_name: Climate zone name for display
            weather_current_temp: Current weather temp for opportunity detection

        Returns:
            DHWRecommendation with recommendation, summary, details, and decision
        """
        indoor_deficit = max(0.0, target_indoor - indoor_temp)

        # Get DM thresholds from climate detector or use fallback
        if self.climate_detector:
            dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
            dm_thresholds = {
                "block": dm_range["warning"],
                "abort": dm_range["critical"],
            }
            zone_name = climate_zone_name or self.climate_detector.zone_info.name
        else:
            dm_thresholds = {
                "block": DM_DHW_BLOCK_FALLBACK,
                "abort": DM_DHW_ABORT_FALLBACK,
            }
            zone_name = climate_zone_name or "Unknown"

        # Block DHW if indoor cooling rapidly AND below target
        if (
            indoor_deficit > DHW_TREND_DEFICIT_THRESHOLD
            and thermal_trend_rate < DHW_TREND_RATE_THRESHOLD
        ):
            planning_summary = (
                f"⚠️ DHW Blocked - Space Heating Priority\n"
                f"Indoor: {indoor_temp:.1f}°C (target {target_indoor:.1f}°C)\n"
                f"Trend: Cooling {abs(thermal_trend_rate):.2f}°C/hour\n"
                f"Reason: Prevent further indoor temperature drop"
            )

            recommendation = (
                f"Block DHW - Indoor temp falling rapidly ({thermal_trend_rate:.2f}°C/h), "
                f"{indoor_deficit:.1f}°C below target. Prioritize space heating."
            )

            return DHWRecommendation(
                recommendation=recommendation,
                summary=planning_summary,
                details={
                    "should_heat": False,
                    "priority_reason": "INDOOR_COOLING_RAPIDLY",
                    "current_temperature": current_dhw_temp,
                    "target_temperature": 50.0,
                    "indoor_temp": indoor_temp,
                    "indoor_trend": thermal_trend_rate,
                    "indoor_deficit": indoor_deficit,
                },
                decision=None,
            )

        # Get decision from should_start_dhw
        decision = self.should_start_dhw(
            current_dhw_temp=current_dhw_temp,
            space_heating_demand_kw=space_heating_demand,
            thermal_debt_dm=thermal_debt,
            indoor_temp=indoor_temp,
            target_indoor_temp=target_indoor,
            outdoor_temp=outdoor_temp,
            price_classification=price_classification,
            current_time=current_time,
            price_periods=price_periods,
            hours_since_last_dhw=hours_since_last_dhw,
        )

        # Build detailed planning attributes
        from .thermal_layer import get_thermal_debt_status

        planning_details = {
            "should_heat": decision.should_heat,
            "priority_reason": decision.priority_reason,
            "current_temperature": current_dhw_temp,
            "target_temperature": decision.target_temp,
            "thermal_debt": thermal_debt,
            "thermal_debt_threshold_block": dm_thresholds["block"],
            "thermal_debt_threshold_abort": dm_thresholds["abort"],
            "thermal_debt_status": get_thermal_debt_status(thermal_debt, dm_thresholds),
            "space_heating_demand_kw": round(space_heating_demand, 2),
            "current_price_classification": price_classification,
            "outdoor_temperature": outdoor_temp,
            "indoor_temperature": indoor_temp,
            "climate_zone": zone_name,
            "recommended_start_time": decision.recommended_start_time,
        }

        # Check for weather opportunity
        weather_opportunity = None
        if weather_current_temp is not None and self.climate_detector:
            zone_avg = self.climate_detector.zone_info.winter_avg_low
            temp_deviation = outdoor_temp - zone_avg
            if temp_deviation > 5.0:
                weather_opportunity = (
                    f"Unusually warm (+{temp_deviation:.1f}°C), good for DHW heating"
                )
                planning_details["weather_opportunity"] = weather_opportunity

        # Convert decision to human-readable recommendation
        if not decision.should_heat:
            if decision.priority_reason == "CRITICAL_THERMAL_DEBT":
                recommendation = (
                    f"Block DHW - Thermal debt warning (DM: {thermal_debt:.0f}, zone: {zone_name})"
                )
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
        planning_summary = self.format_planning_summary(
            recommendation=recommendation,
            current_temp=current_dhw_temp,
            target_temp=decision.target_temp,
            thermal_debt=thermal_debt,
            dm_thresholds=dm_thresholds,
            space_heating_demand=space_heating_demand,
            price_classification=price_classification,
            weather_opportunity=weather_opportunity,
        )

        return DHWRecommendation(
            recommendation=recommendation,
            summary=planning_summary,
            details=planning_details,
            decision=decision,
        )

    def _check_upcoming_demand_period(self, current_time: datetime) -> DemandPeriodInfoDict | None:
        """Check if approaching a high demand period (up to 24h ahead).

        Extended window: We can preheat up to 24h ahead since DHW tank
        cools slowly and heating is fast (1-2 hours).

        Smart fallback: Uses whatever forecast data available (min 1h ahead).

        Args:
            current_time: Current datetime

        Returns:
            Dict with hours_until and target_temp, or None if no upcoming period
        """
        for period in self.demand_periods:
            # Calculate hours until period start
            today_start = current_time.replace(
                hour=period.start_hour, minute=0, second=0, microsecond=0
            )

            if today_start < current_time:
                # Period is tomorrow
                today_start += timedelta(days=1)

            hours_until = (today_start - current_time).total_seconds() / 3600

            # Extended window: 24h ahead (DHW tank doesn't cool much in 24h)
            # Minimum: 1h ahead (need some time for optimization)
            if DHW_SCHEDULING_WINDOW_MIN <= hours_until <= DHW_SCHEDULING_WINDOW_MAX:
                return {
                    "hours_until": hours_until,  # Keep as float for precise decisions
                    "target_temp": period.target_temp,
                    "period_start": today_start,
                }

        return None

    def get_recommended_dhw_schedule(
        self,
        price_data,  # Spot price data with 96 quarters
        weather_data,  # Weather forecast
        current_dhw_temp: float,
        thermal_debt_dm: float,
    ) -> list[DHWScheduleWindowDict]:
        """Calculate recommended DHW heating schedule for next 24 hours.

        Finds optimal windows based on:
        - Price (prioritize CHEAP periods)
        - Thermal debt safety
        - High demand periods
        - Weather (avoid heating before cold spells)

        Args:
            price_data: Spot price data
            weather_data: Weather forecast
            current_dhw_temp: Current DHW temperature
            thermal_debt_dm: Current thermal debt

        Returns:
            List of recommended heating windows with start/end times
        """
        # TODO: Implement smart scheduling
        # This will analyze next 24 hours and return optimal windows
        # For now, return empty list (manual control only)
        return []
