"""A saturated compressor is asked for no more than its own curve (F-124).

The offset raises the pump's supply setpoint S1. A compressor with no frequency left cannot follow
it: a higher S1 only deepens the degree-minute deficit (DM = integral(BT25 - S1)) and brings the
immersion heater in sooner. Saturation is what the compressor DOES, not a fixed frequency, so the
trigger is fed one five-minute reading after another, as the coordinator feeds it.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    COMPRESSOR_HZ_MAX,
    MIN_TARGET_TEMP,
    MIN_TEMP_LIMIT,
    PRICE_TOLERANCE_MAX,
    SAFETY_EMERGENCY_OFFSET,
    TOLERANCE_RANGE_MULTIPLIER,
)
from custom_components.effektguard.models.nibe import NibeF750Profile
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.utils.compressor_monitor import CompressorSaturation

START = datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=5)
TARGET = 21.0
TOLERANCE = 0.5
BAND = TOLERANCE * TOLERANCE_RANGE_MULTIPLIER  # the comfort band the engine hands its layers
CEILING_HZ = 82  # a ceiling the code is never told - it has to find it


def _reading(at: datetime, hz, dm: float, **state) -> NibeState:
    """One five-minute reading."""
    return NibeState(
        outdoor_temp=state.get("outdoor_temp", -12.0),
        indoor_temp=state.get("indoor_temp", TARGET),
        supply_temp=40.0,
        return_temp=35.0,
        degree_minutes=dm,
        current_offset=state.get("current_offset", 0.0),
        is_heating=True,
        is_hot_water=False,
        timestamp=at,
        compressor_hz=hz,
        power_kw=3.0,
        indoor_temp_valid=state.get("indoor_temp_valid", True),
    )


def _feed(readings, airflow=lambda i: False, tracker=None) -> list[bool]:
    """Run one tracker through (Hz, DM, hot water) readings five minutes apart; the verdict after
    each. A ("clock", minutes) entry moves the clock without a reading."""
    tracker, at, verdicts = tracker or CompressorSaturation(), START, []
    for i, entry in enumerate(readings):
        if entry[0] == "clock":
            at += timedelta(minutes=entry[1])
            continue
        hz, dm, hot_water = entry
        verdicts.append(
            tracker.update(_reading(at, hz, dm), hot_water=hot_water, airflow_enhanced=airflow(i))
        )
        at += CYCLE
    return verdicts


def _climb_to(ceiling: int, cycles: int = 12, dm: float = -300.0) -> list:
    """A compressor answering falling DM: from its minimum up to `ceiling` in an hour."""
    return [
        (min(20 + (ceiling - 20) * i // (cycles - 1), ceiling), dm - 10.0 * i, False)
        for i in range(cycles)
    ]


def _hold(hz: int, cycles: int, dm_from: float, dm_step: float = -10.0) -> list:
    return [(hz, dm_from + dm_step * i, False) for i in range(cycles)]


CLIMBED = _climb_to(CEILING_HZ)  # DM ends at -410
SATURATED = CLIMBED + _hold(CEILING_HZ, 24, -420.0)  # two hours flat out while DM falls
MILD_DAY = [(20 if (i // 6) % 2 else 26, -60.0, False) for i in range(48)]  # 20-26 Hz, DM holding


class TestTheTrigger:
    """CompressorSaturation: the compressor holds a ceiling it has earned while DM keeps falling."""

    def test_a_compressor_holding_its_ceiling_while_dm_falls_is_saturated(self):
        assert _feed(SATURATED)[-1] is True

    @pytest.mark.parametrize(
        "after",
        [
            pytest.param(_hold(CEILING_HZ, 6, -660.0, +10.0), id="dm-recovering-at-the-ceiling"),
            pytest.param(
                [(100, -660.0, True)] * 3 + _hold(CEILING_HZ, 3, -700.0),
                id="a-short-hot-water-run-above-the-ceiling",
            ),
            pytest.param(
                [(CEILING_HZ - 15, -660.0, False)] + _hold(CEILING_HZ, 2, -670.0),
                id="one-stray-reading",
            ),
            pytest.param(
                [(CEILING_HZ + 15, -660.0, False)] + _hold(CEILING_HZ, 2, -670.0),
                id="one-stray-reading-above",
            ),
            pytest.param(_hold(CEILING_HZ, 12 * 30, -660.0), id="thirty-hours-flat-out"),
            pytest.param(
                [(100, -660.0, True)] * 7 + _hold(CEILING_HZ, 4, -700.0),
                id="a-hot-water-run-longer-than-a-window",
            ),
            pytest.param(
                [(100, -660.0, True)] * 7
                + [(CEILING_HZ - 20, -700.0, False)]
                + _hold(CEILING_HZ, 3, -710.0),
                id="ramping-back-after-hot-water",
            ),
        ],
    )
    def test_it_stays_saturated_while_the_compressor_stays_at_its_ceiling(self, after):
        assert all(_feed(SATURATED + after)[len(SATURATED) :])

    @pytest.mark.parametrize(
        "after",
        [
            pytest.param(_hold(CEILING_HZ + 12, 6, -660.0), id="a-climb"),
            pytest.param(
                [(CEILING_HZ + i // 3, -660.0 - 10.0 * i, False) for i in range(18)],
                id="a-climb-too-slow-to-leave-the-band",
            ),
            pytest.param(_hold(CEILING_HZ - 12, 6, -660.0), id="a-back-off"),
            pytest.param(
                _hold(CEILING_HZ, 12 * 24, -660.0, 0.0)
                + [(0, -660.0, False)]
                + _hold(CEILING_HZ, 24, -660.0),
                id="a-new-run-at-the-ceiling-a-day-later",
            ),
            pytest.param([(0, -660.0, False)], id="a-stop"),
            pytest.param([(None, -660.0, False)], id="a-dropout"),
            pytest.param([(0, -660.0, True)], id="a-stop-while-hot-water-is-asked-for"),
            pytest.param(
                [(100, -660.0, True)] * 7 + _hold(CEILING_HZ - 12, 4, -700.0),
                id="headroom-after-a-long-hot-water-run",
            ),
            pytest.param([("clock", 20), (CEILING_HZ, -670.0, False)], id="a-gap-in-readings"),
            pytest.param([("clock", -60), (CEILING_HZ, -670.0, False)], id="the-clock-going-back"),
        ],
    )
    def test_it_lets_go_when_the_compressor_shows_headroom_or_the_run_ends(self, after):
        assert _feed(SATURATED + after)[-1] is False

    @pytest.mark.parametrize(
        "readings",
        [
            pytest.param(_hold(20, 36, -260.0), id="start-up-hold-at-the-minimum"),
            pytest.param(
                MILD_DAY + [(0, -60.0, False)] * 2 + _hold(22, 8, -60.0, -14.0),
                id="start-up-hold-after-a-mild-day",
            ),
            pytest.param(_hold(55, 36, -260.0), id="a-frozen-reading"),
            pytest.param(
                CLIMBED
                + [
                    (CEILING_HZ - 15 if i % 6 == 5 else CEILING_HZ, -420.0 - 10.0 * i, False)
                    for i in range(24)
                ],
                id="a-reading-off-the-ceiling-in-every-window",
            ),
            pytest.param(
                CLIMBED
                + _hold(CEILING_HZ, 12, -410.0, 0.0)
                + [(100, -410.0, True)] * 3
                + _hold(CEILING_HZ, 2, -440.0, -30.0),
                id="dm-falling-over-too-few-readings",
            ),
            pytest.param(_hold(CEILING_HZ, 36, -260.0), id="a-run-that-began-at-the-ceiling"),
            pytest.param(
                CLIMBED + _hold(CEILING_HZ, 24, -420.0, 0.0), id="flat-out-but-holding-its-dm"
            ),
            pytest.param(
                CLIMBED
                + [
                    (100 if i % 6 == 5 else CEILING_HZ, -420.0 - 40.0 * (i // 6), i % 6 == 5)
                    for i in range(24)
                ],
                id="dm-falling-only-for-hot-water",
            ),
        ],
    )
    def test_without_that_evidence_it_is_not_saturated(self, readings):
        assert not any(_feed(readings))

    def test_a_compressor_climbing_past_its_old_highs_is_not_saturated(self):
        """At the old high it may look stuck for a while; climbing on, it plainly is not."""
        readings = (
            _climb_to(60)
            + _hold(60, 24, -400.0, 0.0)
            + [(60 + i // 2, -400.0 - 10.0 * i, False) for i in range(36)]
        )

        assert _feed(readings)[-1] is False

    def test_back_at_its_ceiling_after_backing_off_it_is_saturated_again(self):
        """Not climbing: the reference is the highest level an hour or more old, not the newest."""
        readings = (
            SATURATED + _hold(CEILING_HZ - 12, 8, -660.0, +5.0) + _hold(CEILING_HZ, 9, -620.0)
        )

        assert _feed(readings)[-1] is True

    def test_diagnostics_show_no_level_once_the_compressor_stops(self):
        tracker = CompressorSaturation()

        _feed(SATURATED + [(0, -660.0, False)], tracker=tracker)

        assert (tracker.level, tracker.ceiling) == (None, None)

    def test_a_ceiling_ages_out_of_the_memory(self):
        """Yesterday's higher ceiling must not hide today's: after a day it no longer counts."""
        day = 12 * 24
        readings = (
            _climb_to(95)
            + _hold(95, 12, -400.0)
            + [(20 if i % 24 < 6 else 50, -300.0, False) for i in range(day)]
            + SATURATED
        )

        assert _feed(readings)[-1] is True

    def test_enhanced_airflow_keeps_its_own_ceiling(self):
        """An exhaust-air pump tops out higher with enhanced ventilation; neither ceiling may
        hide the other, or the guard is blind for a day after every boost."""
        enhanced = range(len(SATURATED), len(SATURATED) + 36)
        readings = (
            SATURATED
            + _climb_to(110, dm=-660.0)
            + _hold(110, 24, -780.0)
            + _hold(CEILING_HZ, 18, -1020.0)
        )

        verdicts = _feed(readings, airflow=lambda i: i in enhanced)

        assert verdicts[len(SATURATED) + 35] is True, "saturated at the enhanced ceiling"
        assert verdicts[-1] is True, "saturated again at the normal ceiling after the boost"

    @pytest.mark.parametrize("dm_step, saturated", [(0.0, True), (+10.0, False)])
    def test_at_the_inverters_maximum_it_needs_no_falling_dm(self, dm_step, saturated):
        """At COMPRESSOR_HZ_MAX there is no frequency left whatever the memory holds - unless the
        compressor is gaining on its degree minutes."""
        assert _feed(_hold(COMPRESSOR_HZ_MAX - 3, 7, -300.0, dm_step))[-1] is saturated


def _engine(target: float = TARGET, tolerance: float = TOLERANCE, predictor=None) -> DecisionEngine:
    return DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={
            "target_indoor_temp": target,
            "tolerance": tolerance,
            "optimization_mode": "balanced",
            "latitude": 59.33,
            "heating_type": "radiator",
            "heat_loss_coefficient": 150.0,
            "thermal_mass": 0.7,
            "insulation_quality": 1.0,
        },
        heat_pump_model=NibeF750Profile(),
        thermal_predictor=predictor,
    )


def _decision(saturated: bool, dm: float = -600.0, engine=None, **state):
    return (engine or _engine()).calculate_decision(
        nibe_state=_reading(START, CEILING_HZ, dm, **state),
        price_data=None,
        weather_data=None,
        current_peak=0.0,
        current_power=3.0,
        compressor_saturated=saturated,
    )


def _offset(saturated: bool, dm: float = -600.0, engine=None, **state) -> float:
    return _decision(saturated, dm, engine, **state).offset


class TestTheCap:
    """Saturated, the engine asks for the pump's own curve - unless the room needs the heat."""

    @pytest.mark.parametrize("above", [0.0, 0.1])
    @pytest.mark.parametrize("current_offset", [0.0, 3.0])
    def test_a_saturated_compressor_gets_its_own_curve(self, current_offset, above):
        """At the top of its comfort band or above a raise is declined, and one already on the
        register is withdrawn - never turned into a cut."""
        top = {"indoor_temp": TARGET + BAND + above, "current_offset": current_offset}

        assert _offset(False, **top) > 0.5, "precondition: a raise"
        assert _offset(True, **top) == 0.0

    @pytest.mark.parametrize(
        "saturated, indoor, capped",
        [
            pytest.param(False, TARGET + BAND, False, id="not-saturated"),
            pytest.param(True, TARGET + BAND, True, id="a-raise-withdrawn"),
            pytest.param(True, TARGET + 1.5, False, id="a-cut-standing"),
        ],
    )
    def test_the_decision_says_when_it_withdrew_a_raise(self, saturated, indoor, capped):
        """The coordinator's volatility gate lets the withdrawal through on this, and only that:
        every other reversal it still holds for its 45 minutes."""
        assert _decision(saturated, indoor_temp=indoor).saturation_capped is capped

    @pytest.mark.parametrize("share", [0.25, 0.5, 1.0, 1.5])
    def test_lower_in_the_band_a_share_of_the_raise_passes(self, share):
        """Half at the target, all of it at the band's lower edge: there, immersion heat is the heat
        the house is short of."""
        indoor = TARGET + BAND - share * 2 * BAND
        asked = _offset(False, indoor_temp=indoor)

        assert asked > 0.5, "precondition: a raise"
        assert _offset(True, indoor_temp=indoor) == pytest.approx(asked * min(share, 1.0))

    def test_at_the_bands_lower_edge_nothing_is_capped(self):
        """Read in tenths, as a sensor reports it: 21.0 + 0.2 - 20.8 falls a hair short of 0.4 in
        floating point, and must not pass for a cap."""
        edge = round(TARGET - BAND, 1)

        assert _offset(True, indoor_temp=edge) == _offset(False, indoor_temp=edge)

    def test_a_cut_stands(self):
        cut = _offset(False, indoor_temp=TARGET + 1.5)

        assert cut < 0.0, "precondition: a cut"
        assert _offset(True, indoor_temp=TARGET + 1.5) == cut

    def test_past_the_aux_limit_it_is_neither_cut_nor_raised(self):
        top = {"dm": -1600.0, "indoor_temp": TARGET + BAND}

        assert _offset(False, **top) > 0.5, "precondition: a raise"
        assert _offset(True, **top) == 0.0

    def test_without_a_room_sensor_nothing_says_the_house_is_short(self):
        assert _offset(False, indoor_temp_valid=False) > 0.5, "precondition: a raise"
        assert _offset(True, indoor_temp_valid=False) == 0.0

    def test_the_absolute_safety_floor_still_wins(self):
        """With the lowest target and the widest slider the band's edge sits under MIN_TEMP_LIMIT,
        so the comfort share alone would let only part of the safety offset through."""
        engine = _engine(target=MIN_TARGET_TEMP, tolerance=PRICE_TOLERANCE_MAX)

        offset = _offset(True, engine=engine, indoor_temp=MIN_TEMP_LIMIT - 0.1)

        assert offset == SAFETY_EMERGENCY_OFFSET

    def test_a_damped_safety_offset_is_left_alone(self, monkeypatch):
        """A house below MIN_TEMP_LIMIT that is warming fast gets a damped safety offset. Nothing
        here may take it lower - the cap only ever trims a raise above the safety floor."""
        monkeypatch.setattr(dt_util, "now", lambda time_zone=None: START)
        monkeypatch.setattr(dt_util, "utcnow", lambda: START)
        predictor = ThermalStatePredictor()
        for k in range(12):  # 16.0 -> 17.9 C in the last hour
            predictor.record_state(
                timestamp=START - CYCLE * (12 - k),
                indoor_temp=16.0 + 0.17 * k,
                outdoor_temp=-10.0,
                heating_offset=10.0,
                flow_temp=45.0,
                degree_minutes=-300.0,
            )
        engine = _engine(predictor=predictor)
        cold = {"engine": engine, "indoor_temp": MIN_TEMP_LIMIT - 0.1, "dm": -300.0}
        damped = _offset(False, **cold)

        assert 0.0 < damped < SAFETY_EMERGENCY_OFFSET, "precondition: a damped safety offset"
        assert _offset(True, **cold) == damped
