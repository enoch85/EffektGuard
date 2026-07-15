"""A system with no room sensor must still get thermal-debt protection.

With no BT50 the adapter reports DEFAULT_INDOOR_TEMP (21.0) as a placeholder. It equals the
usual target, so `temp_deviation` is exactly 0.0 - which two gates in the emergency layer read
as "at target": `temp_deviation > tolerance_range` is False, and `temp_deviation >= 0` is always
True, returning weight 0.0 unless the price is cheap. That disabled the whole thermal-debt layer
on exactly the sensorless systems that depend on degree minutes most. The safety layer had the
mirror failure: it fires below MIN_TEMP_LIMIT (18.0), which the placeholder 21.0 sits above.

Correct behaviour: comfort-reasoning layers ABSTAIN when the indoor reading is not a
measurement, and the degree-minute tiers run normally, as NIBE runs without a sensor.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.const import (
    DEFAULT_INDOOR_TEMP,
    DM_RECOVERY_TIERS,
    LAYER_WEIGHT_SAFETY,
)
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer, ThermalModel

STOCKHOLM_LATITUDE = 59.33

# Deep thermal debt, well past the climate-aware warning threshold for Stockholm at -15 C.
DEEP_DEBT_DM = -1200.0


def sensorless_state(degree_minutes: float = DEEP_DEBT_DM) -> NibeState:
    """Exactly what the adapter produces when there is no room sensor."""
    return NibeState(
        outdoor_temp=-15.0,
        indoor_temp=DEFAULT_INDOOR_TEMP,  # placeholder, not a measurement
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=degree_minutes,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 1, 15, 6, 0),
        indoor_temp_valid=False,
    )


class TestEmergencyLayerStillProtectsSensorlessSystems:
    @staticmethod
    def _layer() -> EmergencyLayer:
        return EmergencyLayer(
            climate_detector=ClimateZoneDetector(STOCKHOLM_LATITUDE),
            heating_type="radiator",
        )

    def test_deep_thermal_debt_still_triggers_recovery_without_a_room_sensor(self):
        """Case 2 saw deviation 0.0, called it "at target", and abstained without a sensor."""
        decision = self._layer().evaluate_layer(
            nibe_state=sensorless_state(),
            weather_data=None,
            price_data=None,  # price not cheap -> the old Case 2 would return weight 0.0
            target_temp=21.0,
            tolerance_range=0.2,
        )

        assert decision.tier in DM_RECOVERY_TIERS, (
            f"Thermal-debt recovery did not engage at DM {DEEP_DEBT_DM} on a system with no "
            f"room sensor - got tier={decision.tier!r}, weight={decision.weight}. The "
            "placeholder indoor temperature made the layer believe it was at target."
        )
        assert decision.weight > 0.0
        assert decision.offset > 0.0

    def test_a_real_room_sensor_at_target_still_suppresses_recovery(self):
        """Do not over-correct: with a MEASURED reading at target, Case 2 must still work."""
        measured_at_target = NibeState(
            outdoor_temp=-15.0,
            indoor_temp=21.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=DEEP_DEBT_DM,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime(2026, 1, 15, 6, 0),
            indoor_temp_valid=True,
        )

        decision = self._layer().evaluate_layer(
            nibe_state=measured_at_target,
            weather_data=None,
            price_data=None,
            target_temp=21.0,
            tolerance_range=0.2,
        )

        assert decision.tier == "OK"
        assert decision.weight == 0.0


class TestSafetyLayerAbstainsWithoutAMeasurement:
    @pytest.fixture
    def engine(self):
        return DecisionEngine(
            price_analyzer=PriceAnalyzer(),
            effect_manager=EffectManager(MagicMock()),
            thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
            config={
                "target_indoor_temp": 21.0,
                "tolerance": 0.5,
                "latitude": STOCKHOLM_LATITUDE,
            },
        )

    def test_safety_layer_abstains_rather_than_reporting_ok(self, engine):
        """A placeholder of 21.0 must not be read as "comfortably above 18.0"."""
        decision = engine._safety_layer(sensorless_state())

        assert decision.weight == 0.0
        assert "abstain" in decision.reason.lower()

    def test_absolute_safety_floor_ignores_a_placeholder_indoor_reading(self, engine):
        """The floor must not be driven by a value that was never measured."""
        healthy_dm = sensorless_state(degree_minutes=-100.0)

        assert engine._absolute_safety_floor(healthy_dm) is None

    def test_absolute_safety_floor_still_engages_on_degree_minutes(self, engine):
        """Sensorless systems are protected by DM, and that path must remain live."""
        at_aux_limit = sensorless_state(degree_minutes=-1600.0)

        floor = engine._absolute_safety_floor(at_aux_limit)
        assert floor is not None

    def test_safety_layer_still_fires_on_a_real_cold_reading(self, engine):
        """Do not over-correct: a MEASURED 17 C must still trigger the floor."""
        cold = NibeState(
            outdoor_temp=-15.0,
            indoor_temp=17.0,
            supply_temp=35.0,
            return_temp=30.0,
            degree_minutes=-100.0,
            current_offset=0.0,
            is_heating=True,
            is_hot_water=False,
            timestamp=datetime(2026, 1, 15, 6, 0),
            indoor_temp_valid=True,
        )

        decision = engine._safety_layer(cold)
        assert decision.weight == pytest.approx(LAYER_WEIGHT_SAFETY)
        assert decision.offset > 0.0
