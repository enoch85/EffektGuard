"""DHW (Domestic Hot Water) optimizer for EffektGuard.

Intelligent DHW scheduling that coordinates with space heating to maximize
efficiency while preventing thermal debt accumulation.

Based on POST_PHASE_5_ROADMAP.md Phase 8 research and Forum_Summary.md findings.

Priority order:
1. Space heating comfort (indoor temp > target - 0.5°C)
2. DHW safety minimum (≥35°C for health)
3. Thermal debt prevention (DM > -240)
4. Space heating target (±0.3°C)
5. DHW comfort (50°C normal)
6. NIBE's automatic Legionella prevention (MONITOR ONLY - we don't trigger it)

DHW Heating Speed (enoch95):
- Tank heat-up: 1-2 hours (MUCH faster than UFH space heating!)
- No thermal mass delays like concrete slab UFH (6+ hours)
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class DHWScheduleDecision:
    """DHW scheduling decision with safety conditions."""

    should_heat: bool
    priority_reason: str
    target_temp: float
    max_runtime_minutes: int
    abort_conditions: list[str]
    recommended_start_time: datetime | None = None


@dataclass
class DHWDemandPeriod:
    """User-defined high DHW demand period."""

    start_hour: int  # 0-23
    target_temp: float  # Target DHW temp by this time
    duration_hours: int  # How long high demand lasts


class IntelligentDHWScheduler:
    """Intelligent DHW scheduling with thermal debt prevention.

    IMPORTANT: This class MONITORS NIBE's automatic Legionella protection,
    it does NOT trigger Legionella boosts itself to avoid waste:
    - NIBE has fixed weekly 65°C schedule (can't be changed via API)
    - If we trigger boost on day 6, NIBE still runs on day 7 = WASTE
    - Instead: We detect when NIBE does it, track timing, avoid conflicts

    User should manually set NIBE's schedule to cheap hours (e.g., Sunday 3 AM).
    """

    # Temperature thresholds (from research)
    DHW_SAFETY_MIN = 35.0  # °C - health/safety minimum (Legionella risk below this)
    DHW_TARGET_NORMAL = 50.0  # °C - comfortable target
    DHW_TARGET_HIGH = 55.0  # °C - extra comfort for high demand periods
    DHW_LEGIONELLA_DETECT = 63.0  # °C - threshold to detect NIBE's Legionella boost

    # Thermal debt limits (from Forum_Summary.md)
    DM_DHW_BLOCK = -240  # Never start DHW below this (stevedvo threshold)
    DM_DHW_ABORT = -400  # Abort DHW if reached during run (approaching catastrophic)

    def __init__(
        self,
        demand_periods: list[DHWDemandPeriod] | None = None,
    ):
        """Initialize DHW scheduler.

        Args:
            demand_periods: User-defined high demand periods (e.g., morning showers)
        """
        self.demand_periods = demand_periods or []
        self.last_legionella_boost: datetime | None = None
        self.bt7_history: deque = deque(maxlen=48)  # 12 hours @ 15-min intervals

    def update_bt7_temperature(self, temp: float, timestamp: datetime) -> None:
        """Update BT7 temperature history and detect NIBE's Legionella boost.

        Call this every coordinator update to track DHW temperature and
        automatically detect when NIBE's automatic Legionella boost runs.

        Args:
            temp: Current BT7 (hot water top) temperature
            timestamp: Current timestamp
        """
        self.bt7_history.append((timestamp, temp))

        # Detect if NIBE just completed Legionella boost
        if self._detect_legionella_boost_completion():
            self.last_legionella_boost = timestamp
            _LOGGER.info(
                "NIBE Legionella boost detected (automatic): peaked at %s°C at %s",
                max(t for _, t in self.bt7_history),
                timestamp,
            )

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
        if max_temp >= self.DHW_LEGIONELLA_DETECT and current_temp < (max_temp - 3.0):
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
    ) -> DHWScheduleDecision:
        """Decide if DHW heating should start based on priority rules.

        Args:
            current_dhw_temp: Current DHW temperature (°C)
            space_heating_demand_kw: Current space heating demand (kW)
            thermal_debt_dm: Current degree minutes
            indoor_temp: Current indoor temperature (°C)
            target_indoor_temp: Target indoor temperature (°C)
            outdoor_temp: Current outdoor temperature (°C)
            price_classification: Current electricity price classification
            current_time: Current datetime

        Returns:
            DHWScheduleDecision with recommendation and abort conditions
        """
        indoor_deficit = target_indoor_temp - indoor_temp

        # === RULE 1: CRITICAL THERMAL DEBT - NEVER START DHW ===
        if thermal_debt_dm <= self.DM_DHW_BLOCK:
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
        if current_dhw_temp < self.DHW_SAFETY_MIN:
            return DHWScheduleDecision(
                should_heat=True,
                priority_reason="DHW_SAFETY_MINIMUM",
                target_temp=self.DHW_TARGET_NORMAL,
                max_runtime_minutes=30,  # Limited to prevent thermal debt
                abort_conditions=[
                    f"thermal_debt < {self.DM_DHW_ABORT}",
                    f"indoor_temp < {target_indoor_temp - 0.5}",
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
        # SMART SCHEDULING: Heat during cheapest hours before demand, not just last-minute
        # Extended window: Up to 24h ahead (DHW tank heat-up is fast: 1-2 hours)
        upcoming_demand = self._check_upcoming_demand_period(current_time)
        if upcoming_demand:
            hours_until = upcoming_demand["hours_until"]
            target = upcoming_demand["target_temp"]

            # Determine if we should heat now based on price classification
            # Strategy: Use cheapest available hours before demand period
            should_heat_now = False
            priority_reason = ""

            # IMMEDIATE: Less than 2 hours until demand - heat regardless of price
            if hours_until < 2 and current_dhw_temp < target:
                should_heat_now = True
                priority_reason = f"URGENT_DEMAND_IN_{hours_until:.1f}H"

            # OPTIMAL: 2-24 hours until demand - heat during cheap/normal prices
            # Note: DHW tank cools slowly, can preheat up to 24h ahead
            elif 2 <= hours_until <= 24 and current_dhw_temp < target:
                if price_classification in ["cheap", "normal"]:
                    should_heat_now = True
                    priority_reason = (
                        f"OPTIMAL_PREHEAT_DEMAND_{hours_until:.0f}H_{price_classification.upper()}"
                    )

            if should_heat_now:
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason=priority_reason,
                    target_temp=target,
                    max_runtime_minutes=90,  # Longer for high demand
                    abort_conditions=[
                        f"thermal_debt < {self.DM_DHW_ABORT}",
                        f"indoor_temp < {target_indoor_temp - 0.5}",
                    ],
                    recommended_start_time=current_time,
                )

        # === RULE 6: CHEAP ELECTRICITY - OPPORTUNISTIC HEATING ===
        if price_classification == "cheap" and current_dhw_temp < self.DHW_TARGET_HIGH:
            # Only if space heating is satisfied
            if indoor_deficit < 0.3 and thermal_debt_dm > -100:
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason="CHEAP_ELECTRICITY_OPPORTUNITY",
                    target_temp=self.DHW_TARGET_HIGH,  # Extra buffer
                    max_runtime_minutes=45,
                    abort_conditions=[
                        f"thermal_debt < {self.DM_DHW_ABORT}",
                        f"indoor_temp < {target_indoor_temp - 0.5}",
                    ],
                )

        # === RULE 7: NORMAL DHW HEATING - TEMPERATURE LOW ===
        if current_dhw_temp < (self.DHW_TARGET_NORMAL - 5.0):
            # Only if indoor comfortable and thermal debt OK
            if indoor_deficit < 0.3 and thermal_debt_dm > -100:
                return DHWScheduleDecision(
                    should_heat=True,
                    priority_reason="NORMAL_DHW_HEATING",
                    target_temp=self.DHW_TARGET_NORMAL,
                    max_runtime_minutes=30,
                    abort_conditions=[
                        f"thermal_debt < {self.DM_DHW_ABORT}",
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
            if 1 <= hours_until <= 24:
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
