"""The billed quantity, defined once.

The Swedish effect tariff bills the MEAN POWER OVER A BILLING HOUR. That number decides whether the
heat pump is throttled for the rest of the month, so it is the most consequential figure this
integration computes - and it used to be computed twice, by two different pieces of code, using two
different formulas:

    coordinator.py      a time-weighted mean, each sample weighted by how long it stood.
    sim_harness.py      `sum(samples) / len(samples)` - a plain arithmetic mean.

They agree when samples are evenly spaced, and the simulator steps a uniform five minutes, so the
harness's numbers were never wrong. They were something worse: they were produced by code that ships
to nobody. Every tariff figure the simulator printed - every SEK, every kW of peak, every claim about
the feature this integration is named for - came from an implementation no user runs.

That is not a theoretical complaint. The daylight-saving defect lived in the coordinator's
accumulator: on the night the clocks go back it merged the repeated hour and deleted a 9 kW billing
peak, recording it as 1 kW. The simulator had the SAME BUG, INDEPENDENTLY, in its own copy - so it
could not see it. Two implementations of one quantity, both broken, each blind to the other.

There is now one. The coordinator uses it; the harness uses it; breaking it fails both.

WHAT THE ARITHMETIC HAS TO GET RIGHT, and why each part is there:

  * TIME-WEIGHTED, not sample-counted. Home Assistant's update cycle is not a metronome - it jitters,
    it is delayed under load, and a restart drops samples. 1 kW standing for 55 minutes and 9 kW for
    the last five is a 1.67 kW hour; counting samples calls it 5.0 and bills three times the truth.

  * THE HOUR IS COUNTED ON THE ABSOLUTE TIME LINE. On the last Sunday of October the wall-clock hour
    02 happens twice, and PEP 495 says `fold` is IGNORED when two aware datetimes with the SAME
    tzinfo are compared - so `02:00 CEST == 02:00 CET`, and a local-datetime boundary check merges
    two real, separately-metered, separately-billable hours into one.

  * THE LABEL AND THE STAMP STAY LOCAL. The tariff's night discount (22:00-06:00) is a wall-clock
    window, and the effect layer buckets peaks by calendar month - and the hour 00:00-01:00 on
    1 November is 23:00-00:00 on 31 October in UTC, which would file a November peak against a month
    already billed.

Deliberately free of Home Assistant imports: it is pure datetime arithmetic, so the simulator can run
the real thing rather than a lookalike, which was the whole problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..const import BILLING_PERIOD_MINUTES, MAX_BILLING_OBSERVATION_GAP_MINUTES

BILLING_PERIOD = timedelta(minutes=BILLING_PERIOD_MINUTES)
MAX_BILLING_OBSERVATION_GAP_SECONDS = MAX_BILLING_OBSERVATION_GAP_MINUTES * 60


@dataclass(frozen=True)
class CompletedBillingPeriod:
    """One whole billing hour, measured. This is the thing the grid charges for."""

    mean_power_kw: float
    billing_hour: int  # the LOCAL hour of the day, 0-23 - what the night discount reads
    started_at: datetime  # LOCAL and aware - what the calendar month is taken from


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

    def add(self, now: datetime, power_kw: float) -> CompletedBillingPeriod | None:
        """Record a sample. Returns the previous hour if this sample closed it.

        `now` is the local, timezone-aware time - exactly what `dt_util.now()` hands over, `fold` and
        all. That `fold` is load-bearing: it is the only thing distinguishing the two 02:00s on the
        night the clocks go back.
        """
        local_start = now.replace(minute=0, second=0, microsecond=0)
        # Converting the local hour boundary to UTC IS fold-aware, so the two 02:00s resolve to two
        # instants an hour apart. Comparing the local datetimes directly would not - see PEP 495.
        absolute_start = local_start.astimezone(timezone.utc)
        absolute_now = now.astimezone(timezone.utc)

        if absolute_start == self._absolute_start:
            self._samples.append((absolute_now, power_kw))
            return None

        completed = self._close()

        # An hour is partial only if the very first sample ever seen arrives after its boundary.
        self._partial = self._absolute_start is None and absolute_now != absolute_start
        self._absolute_start = absolute_start
        self._local_start = local_start
        self._billing_hour = now.hour
        # Anchored at the boundary. A partial hour is discarded unbilled, so its anchor cannot reach
        # a number anybody sees; every other hour genuinely starts there.
        self._samples = [(absolute_start, power_kw)]
        return completed

    def flush(self) -> CompletedBillingPeriod | None:
        """Close the hour in progress and return it.

        The SIMULATOR calls this: its run ends on an hour boundary, and that final hour is complete
        in sim-time. Production does not - Home Assistant keeps running, and an hour cut short by a
        shutdown was never measured and is not a bill.
        """
        completed = self._close()
        self._absolute_start = None
        self._local_start = None
        self._samples = []
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
        # The last reading stands until the boundary, mirroring the way the first one is anchored to
        # it. Both spans are absolute, so a repeated DST hour is 3600 seconds like any other. This
        # span counts as a gap too: a meter that dies at 10:05 and never returns leaves 55 minutes of
        # the hour resting on one reading, and that is exactly as unmeasured as a gap in the middle.
        final_span = (period_end - previous_time).total_seconds()
        longest_gap = max(longest_gap, final_span)
        weighted += previous_power * final_span

        # AN HOUR THE METER SLEPT THROUGH IS NOT A MEASUREMENT OF AN HOUR.
        #
        # Weighting a reading by how long it stood silently extrapolates it forward, which is right
        # at the five-minute cadence and absurd across a blackout. A meter reading 9 kW at 10:00,
        # going `unavailable`, and returning at 10:55 reading 1 kW had that 9 kW stretched over fifty
        # unwatched minutes: the hour was billed at 8.33 kW, from two samples, while the log said
        # "Peak billing is suspended until it does" ten times over. It was not suspended.
        #
        # The tariff bills the mean of the month's three highest hours and this integration throttles
        # the pump to defend that record, so an invented peak holds the heat back for weeks. The rule
        # already existed for the first hour after startup - "it began before we could watch it" -
        # and simply was not applied to an hour whose middle nobody watched either.
        if longest_gap > MAX_BILLING_OBSERVATION_GAP_SECONDS:
            return None

        return CompletedBillingPeriod(
            mean_power_kw=weighted / (period_end - self._absolute_start).total_seconds(),
            billing_hour=self._billing_hour,
            started_at=self._local_start,
        )
