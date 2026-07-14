"""The billed quantity, defined once: the time-weighted mean power over a billing hour.

The coordinator and the simulator both call this. They used to compute it separately, with different
formulas, and both were wrong on the DST fall-back - so neither could catch the other.

Three things the arithmetic must not lose:

  * TIME-WEIGHTED, not sample-counted. HA's update cycle jitters, so the samples in an hour are not
    evenly spaced and their arithmetic mean is not the hour's mean power.
  * THE HOUR IS ABSOLUTE. Wall-clock 02:00 happens twice on the last Sunday of October, and PEP 495
    ignores `fold` when comparing two aware datetimes with the same tzinfo - so a local-datetime
    boundary check merges two separately-billable hours into one.
  * THE LABEL AND STAMP STAY LOCAL. The night discount is a wall-clock window and peaks are bucketed
    by calendar month; 00:00 on 1 Nov local is 23:00 on 31 Oct in UTC.

No Home Assistant imports, so the simulator runs this rather than a lookalike.

tests/unit/optimization/test_one_definition_of_the_billed_quantity.py
tests/unit/coordinator/test_the_billing_hour_survives_the_clocks_going_back.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..const import (
    BILLING_PERIOD_MINUTES,
    MAX_BILLING_OBSERVATION_GAP_MINUTES,
    PEAK_CONTROL_POWER_SOURCES,
    POWER_SOURCE_EXTERNAL_METER,
    POWER_SOURCE_NIBE_CURRENTS,
    POWER_SOURCE_NONE,
)

BILLING_PERIOD = timedelta(minutes=BILLING_PERIOD_MINUTES)
MAX_BILLING_OBSERVATION_GAP_SECONDS = MAX_BILLING_OBSERVATION_GAP_MINUTES * 60


@dataclass(frozen=True)
class CompletedBillingPeriod:
    """One whole billing hour, measured. This is the thing the grid charges for."""

    mean_power_kw: float
    billing_hour: int  # the LOCAL hour of the day, 0-23 - what the night discount reads
    started_at: datetime  # LOCAL and aware - what the calendar month is taken from
    sample_sources: frozenset[str]  # every source that contributed a sample to this hour

    @property
    def source(self) -> str:
        """What this hour may be recorded AS - decided by every sample, not the closing one.

        The coordinator used to stamp the hour with the CURRENT cycle's source, so an hour
        whose middle was measured at the pump's phase currents became a billable meter hour
        the moment the meter answered again at the boundary. The tariff bills whole-house
        grid import; an hour is a meter measurement only if the meter measured all of it.
        """
        if self.sample_sources == {POWER_SOURCE_EXTERNAL_METER}:
            return POWER_SOURCE_EXTERNAL_METER
        if self.sample_sources <= PEAK_CONTROL_POWER_SOURCES:
            return POWER_SOURCE_NIBE_CURRENTS
        return POWER_SOURCE_NONE


class BillingPeriodAccumulator:
    """Accumulates power samples into completed billing hours."""

    def __init__(self) -> None:
        self._absolute_start: datetime | None = None
        self._local_start: datetime | None = None
        self._billing_hour: int = 0
        # True when the current hour began before observation did - it was never fully measured, so
        # it is not a bill. Only the first hour after startup can be partial.
        self._partial: bool = False
        self._samples: list[tuple[datetime, float]] = []
        self._sources: set[str] = set()

    def add(self, now: datetime, power_kw: float, source: str) -> CompletedBillingPeriod | None:
        """Record a sample. Returns the previous hour if this sample closed it.

        `now` is local and aware, as `dt_util.now()` gives it - `fold` included, which is the only
        thing distinguishing the two 02:00s on the night the clocks go back. `source` is where the
        reading came from; the completed hour's provenance is the set of them.
        """
        local_start = now.replace(minute=0, second=0, microsecond=0)
        # Converting the local hour boundary to UTC IS fold-aware, so the two 02:00s resolve to two
        # instants an hour apart. Comparing the local datetimes directly would not - see PEP 495.
        absolute_start = local_start.astimezone(timezone.utc)
        absolute_now = now.astimezone(timezone.utc)

        if absolute_start == self._absolute_start:
            self._samples.append((absolute_now, power_kw))
            self._sources.add(source)
            return None

        completed = self._close()

        # An hour is partial only if the very first sample ever seen arrives after its boundary.
        self._partial = self._absolute_start is None and absolute_now != absolute_start
        self._absolute_start = absolute_start
        self._local_start = local_start
        self._billing_hour = now.hour
        self._samples = [(absolute_start, power_kw)]
        self._sources = {source}
        return completed

    def flush(self) -> CompletedBillingPeriod | None:
        """Close the hour in progress and return it.

        The SIMULATOR calls this; production does not. An hour cut short by a shutdown was never
        measured and is not a bill.
        """
        completed = self._close()
        self._absolute_start = None
        self._local_start = None
        self._samples = []
        self._sources = set()
        return completed

    def _close(self) -> CompletedBillingPeriod | None:
        """The time-weighted mean of the hour just ended, or None if there is nothing to bill."""
        if self._absolute_start is None or not self._samples or self._partial:
            return None

        period_end = self._absolute_start + BILLING_PERIOD
        previous_time, previous_power = self._samples[0]
        weighted = 0.0
        longest_gap = 0.0
        for sample_time, sample_power in self._samples[1:]:
            span = (sample_time - previous_time).total_seconds()
            longest_gap = max(longest_gap, span)
            weighted += previous_power * span
            previous_time = sample_time
            previous_power = sample_power
        # The last reading stands until the boundary. It counts as a gap too: a meter that dies at
        # 10:05 and never returns leaves 55 minutes resting on one reading, which is as unmeasured as
        # a hole in the middle - and that is the ORDINARY shape of a dropout.
        final_span = (period_end - previous_time).total_seconds()
        longest_gap = max(longest_gap, final_span)
        weighted += previous_power * final_span

        # An hour the meter slept through is not a measurement of an hour. Weighting a reading by how
        # long it stood extrapolates it, which is right at the five-minute cadence and absurd across a
        # blackout - it invents a peak, and the tariff defends the month's top three for weeks.
        # tests/unit/coordinator/test_an_hour_the_meter_slept_through_is_not_a_bill.py
        if longest_gap > MAX_BILLING_OBSERVATION_GAP_SECONDS:
            return None

        return CompletedBillingPeriod(
            mean_power_kw=weighted / (period_end - self._absolute_start).total_seconds(),
            billing_hour=self._billing_hour,
            started_at=self._local_start,
            sample_sources=frozenset(self._sources),
        )
