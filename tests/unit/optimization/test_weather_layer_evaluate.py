"""Tests for WeatherPredictionLayer.evaluate_layer() method.

Phase 4 of layer refactor: Tests for the extracted weather prediction layer evaluation logic.
"""

from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import MagicMock

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.adapters.weather_adapter import WeatherData, WeatherForecastHour
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.prediction_layer import ThermalStatePredictor
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel
from custom_components.effektguard.optimization.weather_layer import (
    WeatherPredictionLayer,
    WeatherLayerDecision,
)
from custom_components.effektguard.const import (
    LAYER_WEIGHT_WEATHER_PREDICTION,
    UFH_CONCRETE_PREDICTION_HORIZON,
    UPDATE_INTERVAL_MINUTES,
    WEATHER_DAILY_MEAN_HOURS,
    WEATHER_FORECAST_HORIZON,
    WEATHER_FORECAST_PERIOD_HOURS,
    WEATHER_GENTLE_OFFSET,
    WEATHER_WEIGHT_CAP,
)


@pytest.fixture
def weather_layer():
    """Create a WeatherPredictionLayer instance with default thermal mass."""
    return WeatherPredictionLayer(thermal_mass=1.0)


@pytest.fixture
def heavy_thermal_mass_layer():
    """Create a WeatherPredictionLayer with heavy thermal mass (concrete slab)."""
    return WeatherPredictionLayer(thermal_mass=1.5)


@pytest.fixture
def mock_nibe_state():
    """Create mock NIBE state."""
    state = MagicMock()
    state.indoor_temp = 21.0
    state.outdoor_temp = 0.0
    state.degree_minutes = -100.0
    return state


@pytest.fixture
def mock_weather_data_dropping():
    """Create mock weather data with temperature drop forecast."""
    weather = MagicMock()
    # Twelve hours as now, then a day 10 °C colder: the daily mean drops 5 °C within 12 h
    weather.forecast_hours = [MagicMock(temperature=t) for t in [0.0] * 12 + [-10.0] * 36]
    return weather


@pytest.fixture
def mock_weather_data_stable():
    """Create mock weather data with stable forecast."""
    weather = MagicMock()
    # Temperature stable around 0°C
    weather.forecast_hours = [
        MagicMock(temperature=0.0),
        MagicMock(temperature=0.5),
        MagicMock(temperature=-0.5),
        MagicMock(temperature=0.0),
    ]
    return weather


class TestEvaluateLayerDisabled:
    """Test evaluate_layer when weather prediction is disabled."""

    def test_disabled_returns_zero_offset(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """When disabled, returns offset=0.0, weight=0.0."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=False,
        )

        assert isinstance(result, WeatherLayerDecision)
        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "Disabled by user" in result.reason

    def test_disabled_returns_correct_name(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Disabled layer still returns proper name."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=False,
        )

        assert result.name == "Weather Pre-heat"


class TestEvaluateLayerNoData:
    """Test evaluate_layer when weather data unavailable."""

    def test_no_weather_data_returns_zero(self, weather_layer, mock_nibe_state):
        """When weather_data is None, returns zero."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=None,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "No weather data" in result.reason

    def test_empty_forecast_returns_zero(self, weather_layer, mock_nibe_state):
        """When forecast_hours is empty, returns zero."""
        empty_weather = MagicMock()
        empty_weather.forecast_hours = []
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=empty_weather,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        # Could be "No weather data" or "No forecast data" depending on check order
        assert "no" in result.reason.lower() or "data" in result.reason.lower()


class TestEvaluateLayerForecastTrigger:
    """Test evaluate_layer forecast-triggered pre-heating."""

    def test_large_temp_drop_triggers_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Temperature drop >= 5°C triggers pre-heating."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == WEATHER_GENTLE_OFFSET  # +0.5°C
        assert result.weight > 0
        assert "proactive" in result.reason.lower() or "drop" in result.reason.lower()

    def test_stable_weather_no_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_stable
    ):
        """Stable weather doesn't trigger pre-heating."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_stable,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == 0.0
        assert result.weight == 0.0
        assert "stable" in result.reason.lower()


@pytest.mark.parametrize("horizon", [WEATHER_FORECAST_HORIZON, UFH_CONCRETE_PREDICTION_HORIZON])
@pytest.mark.parametrize("past_day_mean", [None, 12.0])
def test_a_cold_night_is_not_a_cold_snap(mock_nibe_state, past_day_mean, horizon):
    """May: 16 °C afternoons, 8 °C nights. The night is 8 °C below now, but no day is colder.

    A slab's horizon reaches days the forecast covers only partly; those must not read a night as
    a colder day either.
    """
    weather_layer = WeatherPredictionLayer(thermal_mass=1.0, forecast_horizon=horizon)
    mock_nibe_state.outdoor_temp = 16.0
    weather = MagicMock()
    weather.forecast_hours = [
        MagicMock(temperature=16.0 if 8 <= hour % 24 < 20 else 8.0) for hour in range(48)
    ]

    result = weather_layer.evaluate_layer(
        nibe_state=mock_nibe_state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        enable_weather_prediction=True,
        past_day_mean=past_day_mean,
    )

    assert result.offset == 0.0, f"pre-heated for a cold night: {result.reason}"


def test_a_slab_sees_a_colder_day_beyond_the_first_forecast_day(mock_nibe_state):
    """48 forecast hours hold a whole day only up to 24 h ahead, and a slab looks 48 h ahead.

    A day 4 °C colder from hour 36, under a 6 °C day-night swing, is a cold snap while it is still
    36 h away. Judged on whole days alone, it waited until 24 h (review of #38).
    """
    slab = WeatherPredictionLayer(
        thermal_mass=1.5, forecast_horizon=UFH_CONCRETE_PREDICTION_HORIZON
    )
    weather = MagicMock()
    weather.forecast_hours = [
        MagicMock(temperature=(3.0 if 8 <= hour % 24 < 20 else -3.0) - (4.0 if hour >= 36 else 0.0))
        for hour in range(48)
    ]

    result = slab.evaluate_layer(
        nibe_state=mock_nibe_state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        enable_weather_prediction=True,
        past_day_mean=0.0,
    )

    assert result.offset == WEATHER_GENTLE_OFFSET, result.reason


def test_a_front_hours_away_is_a_cold_snap_for_a_radiator_house(mock_nibe_state):
    """A fast house still pre-heats for a front three hours out, measured against the past day."""
    radiators = WeatherPredictionLayer(thermal_mass=0.5)
    weather = MagicMock()
    weather.forecast_hours = [MagicMock(temperature=t) for t in [0.0] * 3 + [-6.0] * 45]

    result = radiators.evaluate_layer(
        nibe_state=mock_nibe_state,
        weather_data=weather,
        thermal_trend={"rate_per_hour": 0.0, "confidence": 0.8},
        enable_weather_prediction=True,
        past_day_mean=0.0,
    )

    assert result.offset == WEATHER_GENTLE_OFFSET, result.reason


class TestEvaluateLayerIndoorCoolingTrigger:
    """Test evaluate_layer indoor cooling confirmation trigger."""

    def test_indoor_cooling_triggers_preheat(
        self, weather_layer, mock_nibe_state, mock_weather_data_stable
    ):
        """Indoor cooling >= 0.5°C/h triggers pre-heating even with stable forecast."""
        # Rapid indoor cooling confirms heat loss
        thermal_trend = {"rate_per_hour": -0.6, "confidence": 0.8}

        result = weather_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_stable,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        assert result.offset == WEATHER_GENTLE_OFFSET
        assert result.weight > 0
        assert "cooling" in result.reason.lower()


class TestEvaluateLayerThermalMassWeight:
    """Test weight scaling by thermal mass."""

    def test_heavy_thermal_mass_higher_weight(
        self, heavy_thermal_mass_layer, mock_nibe_state, mock_weather_data_dropping
    ):
        """Heavy thermal mass (1.5) gets higher weight."""
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = heavy_thermal_mass_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        # Weight = LAYER_WEIGHT_WEATHER_PREDICTION * 1.5 (capped at WEATHER_WEIGHT_CAP)
        expected_weight = min(LAYER_WEIGHT_WEATHER_PREDICTION * 1.5, WEATHER_WEIGHT_CAP)
        assert result.weight == expected_weight

    def test_light_thermal_mass_lower_weight(self, mock_nibe_state, mock_weather_data_dropping):
        """Light thermal mass (0.5) gets lower weight."""
        light_layer = WeatherPredictionLayer(thermal_mass=0.5)
        thermal_trend = {"rate_per_hour": 0.0, "confidence": 0.8}

        result = light_layer.evaluate_layer(
            nibe_state=mock_nibe_state,
            weather_data=mock_weather_data_dropping,
            thermal_trend=thermal_trend,
            enable_weather_prediction=True,
        )

        expected_weight = LAYER_WEIGHT_WEATHER_PREDICTION * 0.5
        assert result.weight == expected_weight


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
CYCLE = timedelta(minutes=UPDATE_INTERVAL_MINUTES)
DAY = timedelta(hours=WEATHER_DAILY_MEAN_HOURS)
PERIOD = timedelta(hours=WEATHER_FORECAST_PERIOD_HOURS)
TWO_DAYS = [
    CYCLE * k for k in range(1, 2 * DAY // CYCLE + 1)
]  # a reading a cycle, the last one ago


def _restored(ages: list[timedelta]) -> ThermalStatePredictor:
    """The predictor as the coordinator restores it at start-up: a 15 C reading at each age."""
    return ThermalStatePredictor.from_dict(
        {
            "state_history": [
                {
                    "timestamp": (NOW - age).isoformat(),
                    "indoor_temp": 21.0,
                    "outdoor_temp": 15.0,
                    "heating_offset": 0.0,
                    "flow_temp": 30.0,
                    "degree_minutes": -100.0,
                }
                for age in sorted(ages, reverse=True)
            ]
        }
    )


def _unread(ages: list[timedelta], stretch: timedelta) -> list[timedelta]:
    """The same readings with `stretch` between two of them, half a day ago."""
    return [age for age in ages if not DAY / 2 < age < DAY / 2 + stretch]


@pytest.mark.parametrize(
    "ages, measured",
    [
        pytest.param(TWO_DAYS, True, id="readings-every-cycle"),
        pytest.param(
            [age + timedelta(days=3) for age in TWO_DAYS], False, id="restored-three-days-late"
        ),
        pytest.param(
            [age + timedelta(hours=2) for age in TWO_DAYS], False, id="restored-two-hours-late"
        ),
        pytest.param(TWO_DAYS[: len(TWO_DAYS) // 4], False, id="half-a-day-since-install"),
        pytest.param(_unread(TWO_DAYS, PERIOD + CYCLE), False, id="an-hour-unread"),
        pytest.param(_unread(TWO_DAYS, 3 * CYCLE), True, id="a-restart"),
    ],
)
def test_the_past_day_is_measured_only_while_readings_cover_it(ages, measured):
    """Measured at 15 C, the past day makes a flat 5 C forecast a colder day. A history that does
    not cover that day - restored days late, or with hours of it unread - is some other day's mean,
    so the next 24 forecast hours stand in for it, and no day ahead is colder than they are."""
    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(0.7, 1.0),
        config={"target_indoor_temp": 21.0, "latitude": 59.33, "heating_type": "radiator"},
        thermal_predictor=_restored(ages),
    )
    flat = [WeatherForecastHour(datetime=NOW + PERIOD * h, temperature=5.0) for h in range(48)]

    decision = engine.calculate_decision(
        nibe_state=NibeState(
            outdoor_temp=5.0,
            indoor_temp=21.0,
            supply_temp=30.0,
            return_temp=26.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=NOW,
        ),
        price_data=None,
        weather_data=WeatherData(current_temp=5.0, forecast_hours=flat),
        current_peak=0.0,
        current_power=1.5,
    )

    preheat = next(layer for layer in decision.layers if layer.name == "Weather Pre-heat")
    assert (preheat.offset == WEATHER_GENTLE_OFFSET) is measured, preheat.reason
