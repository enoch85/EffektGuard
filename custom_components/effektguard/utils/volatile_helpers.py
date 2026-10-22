"""Volatile price period detection and offset change smoothing.

This module contains common logic for handling volatile price periods
and preventing rapid offset reversals that the heat pump can't follow.

All layers (price, thermal, decision_engine, dhw) should use these helpers
to ensure consistent behavior.

Volatility is determined by the current price run length:
- If current classification (CHEAP, NORMAL, etc.) runs for < 3 quarters (45 min),
  it's considered volatile
- 45 min is based on compressor dynamics: 30 min ramp-up + 15 min cool-down
- Volatility scanning always checks the full day (96 quarters)

Offset volatility tracking:
- Applied at coordinator level AFTER all layers have contributed
- Blocks large offset reversals (>2°C) within the min cycle time (45 min)
- Generic for ALL layer decisions - catches any jumpy offset regardless of source
- The gate's POLICY - which proposals go past the tracker, what a refused one publishes - is
  `gate_offset`, shared by the coordinator and the simulator, so the bench measures what the
  pump gets (F-142)

PEAK cluster detection: Short EXPENSIVE/NORMAL periods sandwiched between
PEAK periods are treated as part of the PEAK cluster (not volatile).
"""

import time

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..const import (
    BENEFICIAL_CLASSIFICATIONS,
    MINUTES_PER_QUARTER,
    QuarterClassification,
    VOLATILE_MIN_DURATION_MINUTES,
    VOLATILE_MIN_DURATION_QUARTERS,
    VOLATILE_TIMING_TOLERANCE_SECONDS,
)
from .time_utils import resolve_period_index

# Time conversion constant (for offset tracker)
SECONDS_PER_MINUTE: int = 60


@dataclass
class VolatileInfo:
    """Volatility detection result with details for logging."""

    is_volatile: bool
    run_length: int  # Total run length (backwards + forwards)
    remaining_quarters: int  # Quarters remaining forward with same classification
    classification_name: str
    reason: str  # Human-readable reason for logging
    in_peak_cluster: bool  # True if sandwiched between PEAK periods
    is_ending_soon: bool  # True if favorable period ending soon (< 3 quarters remain)


def get_volatile_info(
    price_analyzer,
    price_data,
    current_quarter: int | None = None,
) -> VolatileInfo:
    """Get detailed volatility info for current period.

    Counts total run length (backwards + forwards) with the same classification.
    If < 3 quarters (45 min) total, period is volatile (unless in PEAK cluster).

    PEAK cluster: Short EXPENSIVE/NORMAL periods between PEAK periods are
    not volatile - they should be treated as part of the expensive cluster.

    Args:
        price_analyzer: PriceAnalyzer instance with classification methods
        price_data: Current price data (may be None)
        current_quarter: Position in price_data.today to check. If None,
            resolves the interval containing the current time.

    Returns:
        VolatileInfo with is_volatile flag, run length, cluster info, and reason
    """
    if not price_data or not price_data.today:
        return VolatileInfo(
            is_volatile=False,
            run_length=0,
            remaining_quarters=0,
            classification_name="UNKNOWN",
            reason="",
            in_peak_cluster=False,
            is_ending_soon=False,
        )

    if current_quarter is None:
        current_quarter = resolve_period_index(price_data)

    if current_quarter is None or current_quarter >= len(price_data.today):
        return VolatileInfo(
            is_volatile=False,
            run_length=0,
            remaining_quarters=0,
            classification_name="UNKNOWN",
            reason="",
            in_peak_cluster=False,
            is_ending_soon=False,
        )

    # Get current classification
    current_classification = price_analyzer.get_current_classification(current_quarter)
    classification_name = current_classification.name

    # Count total run length (current quarter = 1)
    run_length = 1
    remaining_quarters = 1  # Track forward quarters separately

    today_length = len(price_data.today)
    tomorrow_length = len(price_data.tomorrow)

    # Scan backwards
    for offset in range(1, today_length):
        check_quarter = current_quarter - offset
        if check_quarter < 0:
            break
        if price_analyzer.get_current_classification(check_quarter) == current_classification:
            run_length += 1
        else:
            break

    # Scan forwards (count both run_length and remaining_quarters)
    for offset in range(1, today_length + tomorrow_length):
        check_quarter = current_quarter + offset
        if check_quarter < today_length:
            check_class = price_analyzer.get_current_classification(check_quarter)
        elif price_data.has_tomorrow:
            tomorrow_quarter = check_quarter - today_length
            if tomorrow_quarter < tomorrow_length:
                check_class = price_analyzer.get_tomorrow_classification(tomorrow_quarter)
            else:
                break
        else:
            break

        if check_class == current_classification:
            run_length += 1
            remaining_quarters += 1
        else:
            # Classification changed, count complete
            break

    # Initial volatility check (short run, not PEAK)
    is_brief_run = run_length < VOLATILE_MIN_DURATION_QUARTERS
    is_volatile = is_brief_run and current_classification != QuarterClassification.PEAK

    # Check if CHEAP period is ending soon (v0.4.9 logic)
    # Only CHEAP periods should trigger "ending soon" to allow gradual cooldown before expensive.
    # VERY_CHEAP continues boosting, NORMAL doesn't have special ending logic.
    # This prevents oscillation at CHEAP→NORMAL transitions.
    is_ending_soon = (
        current_classification == QuarterClassification.CHEAP
        and remaining_quarters < VOLATILE_MIN_DURATION_QUARTERS
    )

    # PEAK cluster detection - only for volatile EXPENSIVE/NORMAL periods
    in_peak_cluster = False

    if (
        current_classification in [QuarterClassification.EXPENSIVE, QuarterClassification.NORMAL]
        and is_volatile
    ):
        # Check for PEAK before (within min duration window)
        has_peak_before = False
        for back_offset in range(1, VOLATILE_MIN_DURATION_QUARTERS + 1):
            check_quarter = current_quarter - back_offset
            if check_quarter < 0:
                break
            check_class = price_analyzer.get_current_classification(check_quarter)
            if check_class == QuarterClassification.PEAK:
                has_peak_before = True
                break
            if check_class in BENEFICIAL_CLASSIFICATIONS:
                break

        # Check for PEAK after (only if found before)
        has_peak_after = False
        if has_peak_before:
            for fwd_offset in range(1, VOLATILE_MIN_DURATION_QUARTERS + 1):
                check_quarter = current_quarter + fwd_offset
                if check_quarter < today_length:
                    check_class = price_analyzer.get_current_classification(check_quarter)
                elif price_data.has_tomorrow:
                    tomorrow_quarter = check_quarter - today_length
                    if tomorrow_quarter < tomorrow_length:
                        check_class = price_analyzer.get_tomorrow_classification(tomorrow_quarter)
                    else:
                        break
                else:
                    break

                if check_class is None:
                    break
                if check_class == QuarterClassification.PEAK:
                    has_peak_after = True
                    break
                if check_class in BENEFICIAL_CLASSIFICATIONS:
                    break

        if has_peak_before and has_peak_after:
            in_peak_cluster = True
            is_volatile = False  # Not volatile when part of PEAK cluster

    # Build reason string for logging
    reason = ""
    if is_volatile:
        run_min = run_length * MINUTES_PER_QUARTER
        min_required = VOLATILE_MIN_DURATION_MINUTES
        reason = f"Price volatile: {classification_name} {run_min}min<{min_required}min"
    elif is_ending_soon:
        remaining_min = remaining_quarters * MINUTES_PER_QUARTER
        min_required = VOLATILE_MIN_DURATION_MINUTES
        reason = f"Period ending soon: {classification_name} {remaining_min}min<{min_required}min"

    return VolatileInfo(
        is_volatile=is_volatile,
        run_length=run_length,
        remaining_quarters=remaining_quarters,
        classification_name=classification_name,
        reason=reason,
        in_peak_cluster=in_peak_cluster,
        is_ending_soon=is_ending_soon,
    )


def should_skip_volatile_boost(
    is_volatile: bool, offset: float, is_ending_soon: bool = False
) -> bool:
    """Check if heating boost should be skipped during volatile or ending period.

    Skip positive offsets (heating boosts) when:
    1. Period is volatile (too short to benefit)
    2. Favorable period is ending soon (allow DM recovery before expensive period)

    Args:
        is_volatile: True if in volatile period (short run)
        offset: The calculated offset (positive = heating boost)
        is_ending_soon: True if favorable period ending in < 3 quarters

    Returns:
        True if the boost should be skipped (offset should be set to 0.0)
    """
    return (is_volatile or is_ending_soon) and offset > 0


@dataclass
class OffsetChangeInfo:
    """Tracks offset changes to detect volatile reversals."""

    offset: float
    timestamp: float  # time.time() value
    reason: str


class OffsetVolatilityTracker:
    """Track offset changes and detect volatile reversals.

    Prevents rapid back-and-forth offset changes that the heat pump can't
    follow. When a large offset change is made, block reversals for a
    minimum duration (same as compressor min cycle: 45 minutes).

    This is applied at the coordinator level AFTER all layers have contributed,
    so it catches jumpy offsets from ANY source (comfort, price, thermal, etc.).

    Callers do not consult the tracker directly: `gate_offset` decides which proposals
    reach it and what a refused one publishes, and both the coordinator and the simulator
    go through it.
    """

    # Threshold for "large" offset change (°C difference)
    LARGE_CHANGE_THRESHOLD: float = 2.0

    def __init__(
        self,
        min_duration_minutes: int | None = None,
        time_source: Callable[[], float] | None = None,
    ):
        """Initialize tracker.

        Args:
            min_duration_minutes: Minimum time between reversals.
                                  Default uses VOLATILE_MIN_DURATION_MINUTES (45 min).
            time_source: Epoch-seconds callable, for a caller whose clock is not the wall
                         clock. Defaults to `time.time`, which is what production uses.
                         The simulation harness drives this with SIMULATED time: the harness
                         monkeypatches `dt_util`, not `time`, so against the real clock every
                         reversal in a fast run looks like it happened within the window and
                         the blocker would reject all of them. Injecting the clock is how the
                         harness exercises THIS code rather than a transcription of it.
        """
        self._last_change: OffsetChangeInfo | None = None
        self._time_source: Callable[[], float] | None = time_source
        # Use same min duration as price volatility (based on compressor dynamics)
        self._min_duration_minutes = (
            min_duration_minutes
            if min_duration_minutes is not None
            else VOLATILE_MIN_DURATION_MINUTES
        )

    def _now(self) -> float:
        """Epoch seconds. Resolved per call, not bound in __init__, so that a test patching
        `time.time` still reaches it."""
        return self._time_source() if self._time_source is not None else time.time()

    @property
    def last_offset(self) -> float | None:
        """Get the last recorded offset."""
        return self._last_change.offset if self._last_change else None

    @property
    def last_change(self) -> OffsetChangeInfo | None:
        """The last recorded change - for a caller that must take back one that never applied."""
        return self._last_change

    @last_change.setter
    def last_change(self, change: OffsetChangeInfo | None) -> None:
        self._last_change = change

    def record_change(self, offset: float, reason: str = "") -> None:
        """Record an offset change.

        Args:
            offset: The new offset value
            reason: Reason for the change (for logging)
        """

        self._last_change = OffsetChangeInfo(
            offset=offset,
            timestamp=self._now(),
            reason=reason,
        )

    def is_reversal_volatile(self, new_offset: float) -> bool:
        """Check if proposed offset change is a volatile reversal.

        A reversal is volatile if:
        1. We have a previous offset recorded
        2. The change is "large" (> 2°C difference)
        3. It's in the opposite direction from last change
        4. It's within the minimum duration window

        Args:
            new_offset: Proposed new offset

        Returns:
            True if this would be a volatile reversal (should be blocked)
        """
        if self._last_change is None:
            return False

        last_offset = self._last_change.offset
        change_magnitude = abs(new_offset - last_offset)

        # Not a large change - allow it
        if change_magnitude < self.LARGE_CHANGE_THRESHOLD:
            return False

        # Check if it's a reversal (direction change)
        # A reversal is when the offset crosses zero OR moves back toward zero significantly
        # Examples:
        #   -5 → +1: reversal (crosses zero)
        #   +3 → -2: reversal (crosses zero)
        #   -2 → -5: NOT reversal (same direction, away from zero)
        #   +2 → +5: NOT reversal (same direction, away from zero)
        #   -5 → -2: reversal (same sign but moving back toward zero)
        #   +5 → +2: reversal (same sign but moving back toward zero)
        crosses_zero = (last_offset < 0 < new_offset) or (new_offset < 0 < last_offset)
        moves_toward_zero = abs(new_offset) < abs(last_offset)

        is_reversal = crosses_zero or moves_toward_zero

        if not is_reversal:
            return False

        # Check time since last change
        time_since_last = self._now() - self._last_change.timestamp
        min_duration_seconds = self._min_duration_minutes * SECONDS_PER_MINUTE

        # Tolerance prevents floating-point rounding from blocking at the displayed boundary
        # (e.g., 2699.8s displays as "45min" but is still < 2700s without tolerance)
        if time_since_last < (min_duration_seconds - VOLATILE_TIMING_TOLERANCE_SECONDS):
            return True  # Volatile reversal - block it

        return False

    def get_volatile_reason(self, new_offset: float) -> str:
        """Get reason string for volatile reversal (for logging).

        Args:
            new_offset: Proposed new offset

        Returns:
            Human-readable reason string
        """
        if self._last_change is None:
            return ""

        time_since_last = self._now() - self._last_change.timestamp
        minutes_since = time_since_last / SECONDS_PER_MINUTE

        return (
            f"Offset volatile: {self._last_change.offset:.1f}→{new_offset:.1f} "
            f"reversal blocked ({minutes_since:.0f}min < {self._min_duration_minutes}min)"
        )


class OffsetProposal(Protocol):
    """What the gate reads off a proposal: the engine's OptimizationDecision, by shape.

    Named here, not imported: `decision_engine` imports this module, and a type-only import the
    other way is still a cycle to the import-structure test.
    """

    offset: float
    reasoning: str
    is_manual_override: bool
    is_emergency: bool
    anti_windup_active: bool
    saturation_capped: bool


@dataclass(frozen=True)
class GatedOffset:
    """What the volatility gate lets through to the register, and why."""

    offset: float  # the proposal, or the offset the tracker held on to
    reasoning: str  # the proposal's, with the refusal in front of it when held
    blocked: bool  # the tracker refused the proposal and kept its last offset
    bypass: str | None  # the path that took the proposal past the tracker, else None


def gate_offset(
    tracker: OffsetVolatilityTracker,
    decision: OffsetProposal,
    *,
    optimization_enabled: bool = True,
    track: bool = True,
    record: bool = True,
) -> GatedOffset:
    """The coordinator's gate on a proposal: what reaches the register, what the tracker records.

    One policy for the coordinator and the simulator. The tracker alone is not the policy: the
    simulator once shared the class and ran every proposal through it, while the coordinator
    took the paths below past it, so the bench measured a gate the pump never had.

    - `track=False`: observe only. The proposal is neither held nor recorded, as during start-up
      observation, where nothing is applied and the tracker is seeded from the pump's own offset
      when observation ends.
    - `record=False`: a read, not a control cycle. The policy below decides what is published, as
      it would for a control cycle now, but nothing is recorded. A recorded read became the
      baseline the next control cycle's reversal check compared against, and when that check
      refused, the published - and written - offset was the read's, which no cycle had applied.
    - A user command is authoritative and is recorded, never held: a manual override
      (force_offset, boost), or optimization switched OFF, whose neutral offset must not be
      swapped for the last automatic one behind an OFF display.
    - Three safety paths are recorded without consulting the tracker, because holding them for
      45 minutes is the harm they exist to stop. An EMERGENCY decision (indoor below
      MIN_TEMP_LIMIT, or DM past the aux limit): a held cut leaves DM falling with the immersion
      heater running. An ANTI-WINDUP decision: on 31 January 2026 a blocked reduction 10 -> 3 C
      held the spiral for 45 minutes while DM fell from -1566 to -1855. The wear guard's
      WITHDRAWAL of a raise the saturated compressor cannot follow (F-124) - only the withdrawal;
      the cap stops at the pump's own curve, so a cut onto a saturated compressor is still held.
    - Otherwise a volatile reversal is refused: the tracker's last offset is published, with the
      refusal in front of the proposal's reasoning, and nothing is recorded. Anything else is
      recorded as the new baseline.

    Args:
        tracker: The caller's tracker; its state is what the policy reads and records.
        decision: The engine's proposal, with the flags its layers set.
        optimization_enabled: The master switch. OFF is a user command, not the engine's.
        track: False while the decision is only observed, never applied.
        record: False for a read-only refresh: gated as a control cycle, recorded as nothing.

    Returns:
        The offset and reasoning to publish, whether the proposal was held, and which path - if
        any - took it past the tracker.
    """
    if not track:
        return GatedOffset(decision.offset, decision.reasoning, blocked=False, bypass=None)

    if decision.is_manual_override:
        bypass = "Manual override"
    elif not optimization_enabled:
        bypass = "Optimization off"
    elif decision.is_emergency:
        bypass = "Emergency decision"
    elif decision.anti_windup_active:
        bypass = "Anti-windup active"
    elif decision.saturation_capped and decision.offset < (tracker.last_offset or 0.0):
        bypass = "Compressor saturated"
    else:
        bypass = None

    if bypass is None and tracker.is_reversal_volatile(decision.offset):
        refusal = tracker.get_volatile_reason(decision.offset)
        return GatedOffset(
            tracker.last_offset or 0.0,
            f"[{refusal}] {decision.reasoning}",
            blocked=True,
            bypass=None,
        )

    if record:
        tracker.record_change(decision.offset, decision.reasoning)
    return GatedOffset(decision.offset, decision.reasoning, blocked=False, bypass=bypass)
