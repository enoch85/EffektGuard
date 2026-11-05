"""The log names the climate zone, never the latitude it was detected from.

Home Assistant's latitude is the home's location, and logs get pasted into public issues. Building
the decision engine logged it to two decimals - about a kilometre - in three places.
"""

import logging
from unittest.mock import MagicMock

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.optimization.effect_layer import EffectManager
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import ThermalModel

# A home in Gothenburg, to four decimals.
_HOME_LATITUDE = 57.7089
# The far side of every zone: abs() of it is past the pole.
_NO_ZONE_LATITUDE = -95.4321


def _logged(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def _assert_location_absent(text: str, latitude: float) -> None:
    for shown in (f"{abs(latitude):.1f}", f"{abs(latitude):.2f}", f"{abs(latitude):.4f}"):
        assert shown not in text, f"the log holds the home's latitude ({shown}):\n{text}"


def test_building_the_engine_logs_the_zone_and_not_the_latitude(caplog):
    caplog.set_level(logging.DEBUG)

    engine = DecisionEngine(
        price_analyzer=PriceAnalyzer(),
        effect_manager=EffectManager(MagicMock()),
        thermal_model=ThermalModel(thermal_mass=1.0, insulation_quality=1.0),
        config={"latitude": _HOME_LATITUDE},
    )

    assert engine.climate_detector.zone_info.name in _logged(caplog)
    _assert_location_absent(_logged(caplog), _HOME_LATITUDE)


def test_a_latitude_outside_every_zone_is_not_logged_either(caplog):
    caplog.set_level(logging.DEBUG)

    detector = ClimateZoneDetector(latitude=_NO_ZONE_LATITUDE)

    assert detector.zone_key == "standard"
    assert "outside every zone" in _logged(caplog)
    _assert_location_absent(_logged(caplog), _NO_ZONE_LATITUDE)
