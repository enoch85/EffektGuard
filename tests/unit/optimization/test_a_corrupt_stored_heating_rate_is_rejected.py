"""Corrupt stored state must not poison the DHW scheduler.

The heating rate is used as a divisor in `estimate_heating_time`, so a rate restored from storage
must pass the same plausibility band (DHW_HEATING_RATE_MIN..MAX) as a learned one - a truncated or
hand-edited .storage file could otherwise load 0.0 or 0.1 and make the scheduler panic-heat forever.
"""

from datetime import datetime, timezone

import pytest

from custom_components.effektguard.const import (
    DHW_DEFAULT_HEATING_RATE,
    DHW_HEATING_RATE_MAX,
    DHW_HEATING_RATE_MIN,
)


class TestCorruptStoredHeatingRateIsRejected:
    """Storage is untrusted input. It must not become a divisor."""

    @staticmethod
    def _optimizer():
        from custom_components.effektguard.optimization.dhw_optimizer import (
            IntelligentDHWScheduler,
        )

        return IntelligentDHWScheduler()

    @pytest.mark.parametrize(
        "corrupt",
        [0.0, 0.1, -5.0, 900.0, "fourteen", None, True],
        ids=["zero", "near_zero", "negative", "absurd", "string", "none", "bool"],
    )
    def test_implausible_stored_rate_is_ignored(self, corrupt):
        optimizer = self._optimizer()
        before = optimizer.learned_heating_rate

        optimizer.restore_from_persistence({"learned_heating_rate": corrupt})

        assert optimizer.learned_heating_rate == before, (
            f"A stored heating rate of {corrupt!r} was accepted. It is used as a divisor in "
            "estimate_heating_time: 0.0 raises ZeroDivisionError, and 0.1 yields a 200-hour "
            "heat-up estimate that makes the scheduler panic-heat forever."
        )

        # Whatever it falls back to must itself be usable as a divisor.
        effective = optimizer.learned_heating_rate or DHW_DEFAULT_HEATING_RATE
        assert DHW_HEATING_RATE_MIN <= effective <= DHW_HEATING_RATE_MAX

    def test_a_plausible_stored_rate_is_still_restored(self):
        """Do not over-correct: a legitimate learned rate must survive a restart."""
        optimizer = self._optimizer()

        optimizer.restore_from_persistence(
            {"learned_heating_rate": 18.0, "heating_rate_observations": 7}
        )

        assert optimizer.learned_heating_rate == pytest.approx(18.0)
        assert optimizer.heating_rate_observations == 7

    def test_corrupt_legionella_timestamp_does_not_abort_the_restore(self):
        """A bad timestamp used to raise and abort the rest of learning initialization."""
        optimizer = self._optimizer()

        optimizer.restore_from_persistence(
            {"last_legionella_boost": "not-a-timestamp", "learned_heating_rate": 18.0}
        )

        # The heating rate after it in the same method must still have been restored.
        assert optimizer.learned_heating_rate == pytest.approx(18.0)

    def test_a_legionella_timestamp_without_an_offset_is_ignored(self):
        """Without an offset it cannot be measured against Home Assistant's aware clock: restored,
        it raised TypeError in every hot-water decision."""
        optimizer = self._optimizer()

        optimizer.restore_from_persistence({"last_legionella_boost": "2026-10-20T10:00:00"})

        optimizer.should_start_dhw(
            current_dhw_temp=46.0,
            thermal_debt_dm=-200.0,
            space_heating_demand_kw=0.0,
            outdoor_temp=5.0,
            indoor_temp=21.0,
            target_indoor_temp=21.0,
            price_classification="normal",
            current_time=datetime(2026, 10, 22, 10, tzinfo=timezone.utc),
        )
        assert optimizer.last_legionella_boost is None

    def test_estimate_heating_time_never_divides_by_a_bad_rate(self):
        """Defence in depth: the divisor itself is guarded."""
        optimizer = self._optimizer()

        hours = optimizer.estimate_heating_time(
            current_temp=30.0, target_temp=50.0, heating_rate=0.0
        )

        expected = 20.0 / DHW_DEFAULT_HEATING_RATE
        assert hours == pytest.approx(expected), (
            "estimate_heating_time must fall back to the default rate rather than dividing "
            "by zero."
        )
