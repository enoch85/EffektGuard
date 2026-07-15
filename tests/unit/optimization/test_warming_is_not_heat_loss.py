"""Solar gain is not heat loss, and corrupt stored state must not poison the scheduler.

Comfort layer: `indoor_rate` is a SIGNED °C/h trend. The effective heat-loss rate must be
`max(-indoor_rate, 0.0)`, not `max(abs(indoor_rate), ...)` - taking the absolute value reads a warming
house as losing heat fast, shrinking buffer_hours and triggering a pre-heat while it overheats.

DHW heating rate: the rate is used as a divisor in `estimate_heating_time`, so a rate restored from
storage must pass the same plausibility band (DHW_HEATING_RATE_MIN..MAX) as a learned one - a
truncated or hand-edited .storage file could otherwise load 0.0 or 0.1 and make the scheduler
panic-heat forever.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    DHW_DEFAULT_HEATING_RATE,
    DHW_HEATING_RATE_MAX,
    DHW_HEATING_RATE_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
)
from custom_components.effektguard.optimization.comfort_layer import ComfortLayer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel


class TestWarmingIsNotHeatLoss:
    """The thermal buffer grows when the house warms. It must not read as draining."""

    @staticmethod
    def _layer(indoor_rate: float) -> ComfortLayer:
        return ComfortLayer(
            get_thermal_trend=lambda: {
                "trend": "warming" if indoor_rate > 0 else "cooling",
                "rate_per_hour": indoor_rate,
                "confidence": 1.0,
            },
            thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
            mode_config=MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED],
            tolerance_range=0.2,
            target_temp=21.0,
        )

    @staticmethod
    def _state(indoor_temp: float) -> MagicMock:
        state = MagicMock()
        state.indoor_temp = indoor_temp
        state.outdoor_temp = -4.0
        state.supply_temp = 35.0
        state.degree_minutes = -100.0
        state.current_offset = 0.0
        state.timestamp = datetime(2026, 1, 15, 9, 0)
        state.indoor_temp_valid = True
        return state

    def _effective_heat_loss(self, indoor_rate: float) -> float:
        """Extract the loss rate the layer computed, from its own reason string.

        The layer reports `... @ {effective_heat_loss:.2f}°C/h ...`, which is the value
        under test. `_analyze_expensive_periods` is stubbed so the arithmetic under test is
        isolated from price-data plumbing: an upcoming spike 2 h out, lasting 2 h.
        """
        layer = self._layer(indoor_rate)
        layer._analyze_expensive_periods = lambda price_data, thermal_mass: (2.0, 2.0, 60.0)

        decision = layer._evaluate_thermal_aware_overshoot(
            nibe_state=self._state(21.9),
            weather_data=None,
            price_data=MagicMock(),
            overshoot=0.9,
            temp_deviation=0.9,
        )
        assert decision is not None, "Expected the thermal-aware branch to engage"
        # "... = 1.5h @ 0.60°C/h loss | ..."
        tail = decision.reason.split("@ ", 1)[1]
        return float(tail.split("°C/h", 1)[0])

    def test_a_warming_house_is_not_counted_as_losing_heat(self):
        """+0.6 °C/h of solar gain must NOT be read as 0.6 °C/h of heat loss."""
        warming = self._effective_heat_loss(indoor_rate=+0.6)
        still = self._effective_heat_loss(indoor_rate=0.0)

        assert warming == pytest.approx(still), (
            f"A house warming at +0.6 °C/h reported {warming:.2f} °C/h of heat loss, versus "
            f"{still:.2f} °C/h when static. abs() turned solar gain into heat loss, shrinking "
            "the thermal buffer and triggering a pre-heat while the house was OVERHEATING."
        )

    def test_a_cooling_house_still_counts_as_losing_heat(self):
        """Do not over-correct: real cooling must still drive the loss rate."""
        cooling = self._effective_heat_loss(indoor_rate=-0.6)
        still = self._effective_heat_loss(indoor_rate=0.0)

        assert cooling > still, (
            "A house cooling at -0.6 °C/h must report a HIGHER heat-loss rate than a static "
            "one - that is the case the `max()` exists for."
        )
        assert cooling == pytest.approx(0.6, abs=0.01)


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
