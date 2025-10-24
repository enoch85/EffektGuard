"""DHW (Domestic Hot Water) optimizer for EffektGuard.

Intelligent DHW scheduling that coordinates with space heating to maximize
efficiency while preventing thermal debt accumulation.

Based on POST_PHASE_5_ROADMAP.md Phase 8 research and Forum_Summary.md findings.

Priority order:
1. Space heating comfort (indoor temp > target - 0.5°C)
2. DHW safety minimum (≥20°C for price optimization, ≥10°C critical)
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
from typing import Any

from ..const import (
    DEBUG_FORCE_OUTDOOR_TEMP,
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
    DHW_SPARE_CAPACITY_PERCENT,
    DHW_TARGET_HIGH_DEMAND,
    DHW_URGENT_DEMAND_HOURS,
    DHW_URGENT_RUNTIME_MINUTES,
    DM_DHW_ABORT_FALLBACK,
    DM_DHW_BLOCK_FALLBACK,
    DM_DHW_SPARE_CAPACITY_FALLBACK,
    MIN_DHW_TARGET_TEMP,
)

_LOGGER = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        demand_periods: list[DHWDemandPeriod] | None = None,
        climate_detector=None,  # Climate zone detector for dynamic thresholds
    ):
        """Initialize DHW scheduler.

        Args:
            demand_periods: User-defined high demand periods (e.g., morning showers)
            climate_detector: Optional ClimateZoneDetector for dynamic DM thresholds
        """
        self.demand_periods = demand_periods or []
        self.climate_detector = climate_detector
        self.last_legionella_boost: datetime | None = None
        self.bt7_history: deque = deque(maxlen=48)  # 12 hours @ 15-min intervals

        # Extract user-configured target temperature from demand periods
        # Use first demand period's target, or fall back to const.py constant
        if self.demand_periods:
            self.user_target_temp = self.demand_periods[0].target_temp
        else:
            self.user_target_temp = DEFAULT_DHW_TARGET_TEMP

        if self.climate_detector:
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

    def _has_spare_compressor_capacity(self, thermal_debt_dm: float, outdoor_temp: float) -> bool:
        """Check if heat pump has spare capacity for DHW without risking thermal debt.

        Uses climate-aware calculation: requires thermal debt to be at least
        DHW_SPARE_CAPACITY_PERCENT above the warning threshold for current conditions.

        Example calculations:
        - Stockholm at -10°C: warning=-700, spare capacity threshold = -700 * 0.8 = -560
          * DM -400: Has spare capacity (✓)
          * DM -650: No spare capacity (✗)
        - Kiruna at -30°C: warning=-1200, spare capacity threshold = -1200 * 0.8 = -960
          * DM -800: Has spare capacity (✓)
          * DM -1000: No spare capacity (✗)

        Args:
            thermal_debt_dm: Current degree minutes
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            True if has spare capacity, False if space heating needs all capacity
        """
        if self.climate_detector:
            dm_range = self.climate_detector.get_expected_dm_range(outdoor_temp)
            warning_threshold = dm_range["warning"]

            # Calculate spare capacity threshold (20% buffer above warning)
            spare_capacity_threshold = warning_threshold * (
                1.0 - DHW_SPARE_CAPACITY_PERCENT / 100.0
            )

            has_capacity = thermal_debt_dm > spare_capacity_threshold

            _LOGGER.debug(
                "DHW spare capacity check: DM=%.0f, warning=%.0f, threshold=%.0f, has_capacity=%s",
                thermal_debt_dm,
                warning_threshold,
                spare_capacity_threshold,
                has_capacity,
            )

            return has_capacity
        else:
            # Fallback: Conservative fixed threshold if climate detector unavailable
            # Use fallback constant from const.py
            return thermal_debt_dm > DM_DHW_SPARE_CAPACITY_FALLBACK

    def _detect_legionella_boost_completion(self) -> bool:
        """Detect when NIBE's automatic Legionella boost just completed.

        Detection logic:
        - BT7 peaked at ≥63°C recently
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

        # Get climate-aware DM thresholds based on current outdoor temperature
        if self.climate_detector:
            dm_thresholds = self.climate_detector.get_expected_dm_range(outdoor_temp)
            dm_block_threshold = dm_thresholds["warning"]  # Use warning threshold for blocking
            dm_abort_threshold = dm_thresholds["warning"] - 80  # 80 DM buffer before critical

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

        # === RULE 1: CRITICAL THERMAL DEBT - NEVER START DHW ===
        if thermal_debt_dm <= dm_block_threshold:
            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="CRITICAL_THERMAL_DEBT",
                target_temp=0.0,
                max_runtime_minutes=0,
                abort_conditions=[],
            )

        # === RULE 2: SPACE HEATING EMERGENCY - HOUSE TOO COLD ===
        if indoor_deficit > 0.5 and outdoor_temp < 0:
            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="SPACE_HEATING_EMERGENCY",
                target_temp=0.0,
                max_runtime_minutes=0,
                abort_conditions=[],
            )

        # === RULE 3: DHW SAFETY MINIMUM - MUST HEAT (Limited) ===
        # Safety minimum: Heat below 20°C BUT defer if:
        # - Price is expensive/peak AND thermal debt is NOT concerning
        # - This prevents peak billing hits when DHW can wait and space heating is healthy
        if current_dhw_temp < DHW_SAFETY_MIN:
            # Check if we should defer due to peak pricing + thermal debt
            # Only defer if temp is still safe (10-20°C range) and not critically low
            can_defer_for_peak = (
                current_dhw_temp >= DHW_SAFETY_CRITICAL  # Not critically low (>= 10°C)
                and price_classification in ["expensive", "peak"]  # High cost period
                and thermal_debt_dm > (dm_block_threshold + 20)  # DM healthy enough to defer
            )

            if can_defer_for_peak:
                _LOGGER.info(
                    "DHW safety minimum defer: temp %.1f°C (safe >= %.1f°C), price=%s, DM=%.0f. "
                    "Waiting for better price to avoid peak billing.",
                    current_dhw_temp,
                    DHW_SAFETY_CRITICAL,
                    price_classification,
                    thermal_debt_dm,
                )
                return DHWScheduleDecision(
                    should_heat=False,
                    priority_reason="DHW_SAFETY_DEFERRED_PEAK_PRICE",
                    target_temp=0.0,
                    max_runtime_minutes=0,
                    abort_conditions=[],
                )

            # Otherwise heat immediately (critically low or good price)
            # SMART TWO-TIER STRATEGY:
            # - Emergency heating: Heat to DHW_SAFETY_MIN (20°C) to get safe quickly
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
            )

        # === RULE 2.3: HYGIENE BOOST (HIGH-TEMP CYCLE FOR LEGIONELLA PREVENTION) ===
        # If DHW hasn't been above 60°C in past 14 days, heat to 60°C during cheapest period
        # This prevents Legionella bacteria growth in the low-temp range (20-45°C)
        # with the new lower safety thresholds (10°C/20°C).
        #
        # REQUIRES DHW IMMERSION HEATER (Swedish: elpatron):
        # - Heat pump compressor can only reach ~50-55°C max (COP limitation)
        # - NIBE automatically engages DHW tank immersion heater to complete 60°C target
        # - This is normal operation for Legionella prevention in all NIBE systems
        # - Scheduling during cheap periods minimizes immersion heater cost
        #
        # NOTE: This is the DHW tank's built-in immersion heater (elpatron), NOT the
        # space heating auxiliary heater. They are separate electrical heating systems.
        #
        # References:
        # - Boverket.se: Water heaters should maintain ≥60°C, bacteria killed at high temps
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
                "compressor max ~55°C, NIBE will automatically use immersion heater to complete).",
                "Never" if days_since_legionella is None else f"{days_since_legionella:.1f}",
                DHW_LEGIONELLA_MAX_DAYS,
                DHW_LEGIONELLA_PREVENT_TEMP,
            )
            _LOGGER.info(
                "Hygiene boost scheduled during cheap electricity period to minimize "
                "immersion heater cost. NIBE will use compressor to ~50-55°C, "
                "then DHW tank immersion heater to complete 60°C target."
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
            )

        # === RULE 2.5: COMPLETE EMERGENCY HEATING TO COMFORT LEVEL ===
        # After emergency heating reached DHW_SAFETY_MIN (20°C), complete to comfort level
        # during cheap prices. This is the second phase of two-tier emergency heating.
        if (
            DHW_SAFETY_MIN <= current_dhw_temp < MIN_DHW_TARGET_TEMP  # In 20-45°C range
            and price_classification == "cheap"  # Wait for cheap prices
            and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)
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
                )

        # === RULE 4: HIGH SPACE HEATING DEMAND - DELAY DHW ===
        # NOTE: Legionella prevention is handled by NIBE automatically
        # We monitor when it happens (update_bt7_temperature) but don't trigger it
        # to avoid waste (NIBE runs on fixed schedule regardless of our triggers)
        if space_heating_demand_kw > 6.0 and thermal_debt_dm < -60:
            return DHWScheduleDecision(
                should_heat=False,
                priority_reason="HIGH_SPACE_HEATING_DEMAND",
                target_temp=0.0,
                max_runtime_minutes=0,
                abort_conditions=[],
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
        # Find the absolute cheapest window first, then check if we should heat
        optimal_window = None  # Initialize to None for fallback logic

        if current_dhw_temp < (self.user_target_temp + 5.0):
            # Use window-based scheduling if price data available
            if price_periods:
                lookahead_hours = self.get_lookahead_hours(current_time)
                optimal_window = self.find_cheapest_dhw_window(
                    current_time=current_time,
                    lookahead_hours=lookahead_hours,
                    dhw_duration_minutes=45,
                    price_periods=price_periods,
                )

                if optimal_window is None:
                    # No price data - only heat during cheap periods if needed
                    if (
                        price_classification == "cheap"
                        and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
                        and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)
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
                        )

                # STEP 1: Check if we're in the optimal window (within 10 min of start)
                # Use 10 min buffer (not 15 min) to prevent edge-case activations
                # when quarters transition from future to current
                elif optimal_window["hours_until"] <= 0.17:  # 10 minutes
                    # This is the optimal time! Heat if we have spare capacity
                    # Trust the window optimizer - it found the absolute cheapest period
                    if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):
                        _LOGGER.info(
                            "DHW: Heating in optimal window at %s (%.1före/kWh)",
                            optimal_window["start_time"].strftime("%H:%M"),
                            optimal_window["avg_price"],
                        )
                        return DHWScheduleDecision(
                            should_heat=True,
                            priority_reason=f"OPTIMAL_WINDOW_Q{optimal_window['quarters'][0]}_@{optimal_window['avg_price']:.1f}öre",
                            target_temp=min(
                                self.user_target_temp + DHW_PREHEAT_TARGET_OFFSET, 60.0
                            ),
                            max_runtime_minutes=DHW_NORMAL_RUNTIME_MINUTES,
                            abort_conditions=[
                                f"thermal_debt < {dm_abort_threshold:.0f}",
                                f"indoor_temp < {target_indoor_temp - 0.5}",
                            ],
                            recommended_start_time=optimal_window["start_time"],
                        )

                # Wait for better window ahead (if DHW still comfortable)
                # Require window to be at least 10 min away to prevent premature activation
                elif current_dhw_temp > MIN_DHW_TARGET_TEMP:
                    _LOGGER.info(
                        "DHW: Next window at %s (%.1fh, %.1före/kWh)",
                        optimal_window["start_time"].strftime("%H:%M"),
                        optimal_window["hours_until"],
                        optimal_window["avg_price"],
                    )
                    return DHWScheduleDecision(
                        should_heat=False,
                        priority_reason=f"WAITING_OPTIMAL_WINDOW_IN_{optimal_window['hours_until']:.1f}H_@{optimal_window['avg_price']:.1f}",
                        target_temp=0.0,
                        max_runtime_minutes=0,
                        abort_conditions=[],
                        recommended_start_time=optimal_window["start_time"],
                    )

            # Fallback: No price data AND no optimal window ahead
            # Heat now if DHW getting low and price is cheap
            # Only use this fallback when we don't have a better window to wait for
            if (
                not price_periods  # No price data available
                and price_classification == "cheap"
                and current_dhw_temp < DEFAULT_DHW_TARGET_TEMP
                and self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp)
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
                )

        # === RULE 7: COMFORT HEATING (DHW GETTING LOW) ===
        # Heat below minimum user target during cheap prices (no window check - urgent enough)
        if current_dhw_temp < MIN_DHW_TARGET_TEMP and price_classification == "cheap":
            if self._has_spare_compressor_capacity(thermal_debt_dm, outdoor_temp):
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason="DHW_COMFORT_LOW_CHEAP",
                    target_temp=self.user_target_temp,
                    max_runtime_minutes=DHW_SAFETY_RUNTIME_MINUTES,
                    abort_conditions=[
                        f"thermal_debt < {dm_abort_threshold:.0f}",
                        f"indoor_temp < {target_indoor_temp - 0.5}",
                    ],
                )

        # === RULE 8: ALL CONDITIONS FAIL - DON'T HEAT ===
        return DHWScheduleDecision(
            should_heat=False,
            priority_reason="DHW_ADEQUATE",
            target_temp=0.0,
            max_runtime_minutes=0,
            abort_conditions=[],
        )

    def get_lookahead_hours(self, current_time: datetime) -> int:
        """Calculate how far ahead to look for cheap DHW windows.

        Returns hours until next demand period (capped at 24h), or 24h if no demand period.

        Args:
            current_time: Current datetime

        Returns:
            Hours to look ahead (max 24h)
        """
        from ..const import DHW_SCHEDULING_WINDOW_MAX, DHW_SCHEDULING_WINDOW_MIN

        next_demand = self._check_upcoming_demand_period(current_time)

        if next_demand:
            hours_until = next_demand["hours_until"]
            # Return hours until demand, but ensure it's at least 1h and max 24h
            return max(DHW_SCHEDULING_WINDOW_MIN, min(int(hours_until), DHW_SCHEDULING_WINDOW_MAX))

        # No upcoming demand period - look full 24 hours ahead
        return DHW_SCHEDULING_WINDOW_MAX

    def find_cheapest_dhw_window(
        self,
        current_time: datetime,
        lookahead_hours: int,
        dhw_duration_minutes: int,
        price_periods: list,
    ) -> dict | None:
        """Find cheapest continuous window for DHW heating.

        Uses sliding window algorithm to find the absolute cheapest continuous
        period for DHW heating within the lookahead window. DHW heating is
        time-shiftable (can wait for cheaper prices) but requires a continuous
        window since we can't split heating across gaps.

        Implementation:
        - Typical duration: 45 minutes (3 quarters, faster than space heating)
        - Shifts window within lookahead period (up to next demand period)
        - Must be continuous 15-minute periods (no gaps)
        - Finds absolute cheapest average price (not just "cheap" classification)

        Args:
            current_time: Current timestamp
            lookahead_hours: How far ahead to search
            dhw_duration_minutes: DHW heating duration (typically 45 min)
            price_periods: Combined list of today + tomorrow QuarterPeriod objects

        Returns:
            {
                "start_time": datetime,
                "end_time": datetime,
                "quarters": [Q30, Q31, Q32],
                "avg_price": float,
                "hours_until": float,
            }
            or None if insufficient data
        """
        from math import ceil

        if not price_periods:
            _LOGGER.warning("No price data available for DHW window search")
            return None

        # Convert duration to quarters (45 min = 3 quarters)
        quarters_needed = ceil(dhw_duration_minutes / 15)

        # Filter to lookahead window
        end_time = current_time + timedelta(hours=lookahead_hours)

        # Build available quarters from price periods
        # QuarterPeriod has: quarter_of_day, hour, minute, price, is_daytime, start_time
        # Use the actual datetime from GE-Spot (already timezone-aware) instead of reconstructing
        available_quarters = []

        for period in price_periods:
            # Use actual datetime from GE-Spot (already handles timezone and date correctly)
            period_time = period.start_time

            # Check if within lookahead window
            if period_time >= current_time and period_time < end_time:
                # Calculate quarter ID based on actual datetime to distinguish duplicates
                days_ahead = (period_time.date() - current_time.date()).days
                quarter_id = period.quarter_of_day + (days_ahead * 96)

                available_quarters.append(
                    {
                        "start": period_time,
                        "end": period_time + timedelta(minutes=15),
                        "quarter": quarter_id,
                        "price": period.price,
                    }
                )

        if len(available_quarters) < quarters_needed:
            _LOGGER.warning(
                "Not enough price data for DHW window search: %d quarters available, %d needed",
                len(available_quarters),
                quarters_needed,
            )
            return None

        _LOGGER.debug(
            "DHW window search: %d quarters available, need %d quarters (%d min), lookahead %.1fh",
            len(available_quarters),
            quarters_needed,
            dhw_duration_minutes,
            lookahead_hours,
        )

        # Sliding window to find cheapest continuous period
        lowest_price = None
        lowest_index = None
        window_candidates = []  # Track all valid windows for debugging

        for i in range(len(available_quarters) - quarters_needed + 1):
            window = available_quarters[i : i + quarters_needed]

            # Verify continuity (15-min gaps)
            is_continuous = True
            for j in range(len(window) - 1):
                time_gap = (window[j + 1]["start"] - window[j]["end"]).total_seconds()
                if abs(time_gap) > 1:  # Allow 1 second tolerance
                    is_continuous = False
                    break

            if not is_continuous:
                continue

            window_avg = sum(q["price"] for q in window) / quarters_needed

            # Track this candidate
            window_candidates.append(
                {
                    "start": window[0]["start"],
                    "avg_price": window_avg,
                    "hours_until": (window[0]["start"] - current_time).total_seconds() / 3600,
                }
            )

            if lowest_price is None or window_avg < lowest_price:
                lowest_price = window_avg
                lowest_index = i

        # Log all candidates for debugging
        if window_candidates:
            _LOGGER.debug(
                "DHW window candidates found: %d windows evaluated",
                len(window_candidates),
            )
            # Show top 5 cheapest
            sorted_candidates = sorted(window_candidates, key=lambda x: x["avg_price"])
            for idx, candidate in enumerate(sorted_candidates[:5], 1):
                _LOGGER.debug(
                    "  #%d: %.1före/kWh at %s (%.1fh away)%s",
                    idx,
                    candidate["avg_price"],
                    candidate["start"].strftime("%H:%M"),
                    candidate["hours_until"],
                    " ← SELECTED" if idx == 1 else "",
                )

        if lowest_index is None:
            _LOGGER.error("Could not find continuous DHW window (unexpected)")
            return None

        optimal_window = available_quarters[lowest_index : lowest_index + quarters_needed]

        result = {
            "start_time": optimal_window[0]["start"],
            "end_time": optimal_window[-1]["end"],
            "quarters": [q.get("quarter", i + lowest_index) for i, q in enumerate(optimal_window)],
            "avg_price": lowest_price,
            "hours_until": (optimal_window[0]["start"] - current_time).total_seconds() / 3600,
        }

        _LOGGER.info(
            "DHW optimal window selected: %s (Q%d-Q%d) @ %.1före/kWh, %.1fh away",
            result["start_time"].strftime("%H:%M"),
            result["quarters"][0],
            result["quarters"][-1],
            result["avg_price"],
            result["hours_until"],
        )

        return result

    def _check_upcoming_demand_period(self, current_time: datetime) -> dict[str, Any] | None:
        """Check if approaching a high demand period (up to 24h ahead).

        Extended window: We can preheat up to 24h ahead since DHW tank
        cools slowly and heating is fast (1-2 hours).

        Smart fallback: Uses whatever forecast data available (min 1h ahead).

        Args:
            current_time: Current datetime

        Returns:
            Dict with hours_until and target_temp, or None if no upcoming period
        """
        from ..const import DHW_SCHEDULING_WINDOW_MAX, DHW_SCHEDULING_WINDOW_MIN

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

    def record_legionella_boost(self, boost_time: datetime) -> None:
        """Record successful Legionella boost completion.

        DEPRECATED: Use update_bt7_temperature() instead for automatic detection.
        This method kept for backward compatibility.

        Args:
            boost_time: When boost was completed
        """
        self.last_legionella_boost = boost_time
        _LOGGER.info("Legionella boost manually recorded at %s", boost_time)

    def get_recommended_dhw_schedule(
        self,
        price_data,  # GE-Spot data with 96 quarters
        weather_data,  # Weather forecast
        current_dhw_temp: float,
        thermal_debt_dm: float,
    ) -> list[dict[str, Any]]:
        """Calculate recommended DHW heating schedule for next 24 hours.

        Finds optimal windows based on:
        - Price (prioritize CHEAP periods)
        - Thermal debt safety
        - High demand periods
        - Weather (avoid heating before cold spells)

        Args:
            price_data: GE-Spot price data
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
