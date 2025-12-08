"""Climate zone detection and configuration for heat pump heating optimization.

DESIGN PHILOSOPHY:
This module provides climate-aware configuration for heat pump heating systems.
Focus is on regions where heating is the primary use case and thermal debt (DM)
management is critical.

Climate zones are based on winter heating needs, not geographic classification.
If you're in a location with minimal winter heating needs, you're in "Standard"
zone and probably don't need aggressive optimization anyway.

Separation of Concerns:
- This module: Climate zone detection and DM thresholds
- weather_compensation.py: Flow temperature calculations
- decision_engine.py: Uses climate data for emergency layer

References:
    - IMPLEMENTATION_PLAN/FUTURE/CLIMATE_ZONE_DM_INTEGRATION.md
    - Swedish_NIBE_Forum_Findings.md - DM -1500 absolute maximum validation
    - Forum_Summary.md - Real-world thermal debt case studies
"""

import logging
from dataclasses import dataclass
from typing import Final

from ..const import (
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)

_LOGGER = logging.getLogger(__name__)


# Climate zones focused on heating needs (coldest to mildest)
HEATING_CLIMATE_ZONES: Final = {
    "extreme_cold": {
        "name": "Extreme Cold",
        "description": "Severe winter heating demands",
        "latitude_range": (66.5, 90.0),  # Arctic Circle and above
        "winter_avg_low": CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
        "dm_normal_range": (-800, -1200),  # Expected DM range
        "dm_warning_threshold": -1200,  # Warning if deeper than this
        "safety_margin_base": 2.5,  # °C extra flow temp margin
        "examples": ["Kiruna (SWE)", "Tromsø (NOR)", "Fairbanks (USA)"],
    },
    "very_cold": {
        "name": "Very Cold",
        "description": "Heavy winter heating demands",
        "latitude_range": (60.5, 66.5),  # Northern inland regions
        "winter_avg_low": CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
        "dm_normal_range": (-600, -1000),
        "dm_warning_threshold": -1000,
        "safety_margin_base": 1.5,  # °C
        "examples": ["Luleå (SWE)", "Umeå (SWE)", "Oulu (FIN)", "Trondheim (NOR)"],
    },
    "cold": {
        "name": "Cold",
        "description": "Substantial winter heating demands",
        "latitude_range": (56.0, 60.5),  # Southern Nordics (includes Helsinki despite latitude)
        "winter_avg_low": CLIMATE_ZONE_COLD_WINTER_AVG,
        "dm_normal_range": (-450, -700),
        "dm_warning_threshold": -700,
        "safety_margin_base": 1.0,  # °C
        "examples": ["Stockholm (SWE)", "Oslo (NOR)", "Göteborg (SWE)", "Helsinki (FIN)"],
    },
    "moderate_cold": {
        "name": "Moderate Cold",
        "description": "Moderate winter heating demands",
        "latitude_range": (54.5, 56.0),  # Øresund region (Denmark + Skåne)
        "winter_avg_low": CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
        "dm_normal_range": (-300, -500),
        "dm_warning_threshold": -500,
        "safety_margin_base": 0.5,  # °C
        "examples": ["Copenhagen (DEN)", "Malmö (SWE)", "Aarhus (DEN)", "Helsingborg (SWE)"],
    },
    "standard": {
        "name": "Standard",
        "description": "Minimal winter heating demands",
        "latitude_range": (0.0, 54.5),  # Everything else
        "winter_avg_low": CLIMATE_ZONE_STANDARD_WINTER_AVG,
        "dm_normal_range": (-200, -350),
        "dm_warning_threshold": -350,
        "safety_margin_base": 0.0,  # No extra margin needed
        "examples": ["Default for mild climates"],
    },
}

# Zone order for detection (coldest to mildest)
ZONE_ORDER: Final = ["extreme_cold", "very_cold", "cold", "moderate_cold", "standard"]

# Absolute safety limit - NEVER EXCEED regardless of climate
# Source: Swedish NIBE forums - validated in real-world Nordic conditions
DM_ABSOLUTE_MAXIMUM: Final = -1500


@dataclass
class ClimateZoneInfo:
    """Climate zone information for a location."""

    zone_key: str
    name: str
    description: str
    winter_avg_low: float
    dm_normal_min: float  # Shallow end of normal range
    dm_normal_max: float  # Deep end of normal range
    dm_warning_threshold: float
    safety_margin_base: float
    examples: list[str]


class ClimateZoneDetector:
    """Detect and provide climate zone information for heat pump optimization.

    USAGE:
        detector = ClimateZoneDetector(latitude=59.33)  # Stockholm
        zone_info = detector.zone_info
        expected_dm = detector.get_expected_dm_range(outdoor_temp=-10.0)

    DESIGN:
        - Simple latitude-based detection (uses Home Assistant's built-in latitude)
        - No configuration needed beyond existing HA settings
        - Automatic adaptation from Arctic to Mediterranean
        - Context-aware DM thresholds based on climate zone + current temperature
    """

    def __init__(self, latitude: float):
        """Initialize climate zone detector.

        Args:
            latitude: Home location latitude in degrees (positive = North, negative = South)
        """
        self.latitude = abs(latitude)  # Use absolute for hemisphere independence
        self.zone_key = self._detect_zone()
        self.zone_info = self._get_zone_info()

        _LOGGER.info(
            "Climate zone detected: %s (%s) at latitude %.2f°N - "
            "Winter avg: %.1f°C, DM normal range: %.0f to %.0f",
            self.zone_info.name,
            self.zone_info.description,
            latitude,
            self.zone_info.winter_avg_low,
            self.zone_info.dm_normal_min,
            self.zone_info.dm_normal_max,
        )

    def _detect_zone(self) -> str:
        """Detect climate zone based on latitude.

        NORDIC FOCUS:
        Nordic countries (SWE/NOR/FIN/DNK, lat 54.5°-71°) get fine-grained zones.
        For rest of world, broader zones are sufficient.

        BOUNDARY DECISIONS:
        - 56.0° boundary: Copenhagen/Malmö (55.6-55.7°N) in moderate_cold, Stockholm in cold
        - 60.5° boundary: Helsinki (60.17°N) in cold despite latitude, Umeå in very_cold
        - 66.5° boundary: Arctic Circle - dramatic climate change

        Returns:
            Zone key from ZONE_ORDER
        """
        for zone_key in ZONE_ORDER:
            zone_data = HEATING_CLIMATE_ZONES[zone_key]
            lat_min, lat_max = zone_data["latitude_range"]

            if lat_min <= self.latitude <= lat_max:
                _LOGGER.debug(
                    "Latitude %.2f matches zone '%s' (%.1f-%.1f)",
                    self.latitude,
                    zone_key,
                    lat_min,
                    lat_max,
                )
                return zone_key

        # Default to standard for edge cases
        _LOGGER.warning(
            "Latitude %.2f outside all zone ranges, defaulting to 'standard'", self.latitude
        )
        return "standard"

    def _get_zone_info(self) -> ClimateZoneInfo:
        """Get structured zone information.

        Returns:
            ClimateZoneInfo dataclass with all zone parameters
        """
        zone_data = HEATING_CLIMATE_ZONES[self.zone_key]
        dm_min, dm_max = zone_data["dm_normal_range"]

        return ClimateZoneInfo(
            zone_key=self.zone_key,
            name=zone_data["name"],
            description=zone_data["description"],
            winter_avg_low=zone_data["winter_avg_low"],
            dm_normal_min=dm_min,
            dm_normal_max=dm_max,
            dm_warning_threshold=zone_data["dm_warning_threshold"],
            safety_margin_base=zone_data["safety_margin_base"],
            examples=zone_data["examples"],
        )

    def get_expected_dm_range(self, outdoor_temp: float) -> dict[str, float]:
        """Calculate expected DM range for current outdoor temperature.

        CONTEXT-AWARE THRESHOLDS:
        Base DM expectations come from climate zone, then adjust based on how much
        colder/warmer current temperature is compared to zone's winter average.

        EXAMPLES:
        - Kiruna (Extreme Cold, winter avg -30°C):
          * At -30°C: DM -800 to -1200 is normal
          * At -20°C: DM -600 to -1000 is normal (10°C warmer = shallower)

        - Stockholm (Cold, winter avg -10°C):
          * At -10°C: DM -450 to -700 is normal
          * At 0°C: DM -250 to -450 is normal (10°C warmer = shallower)

        Args:
            outdoor_temp: Current outdoor temperature (°C)

        Returns:
            Dict with 'normal_min', 'normal_max', 'warning', 'critical'
        """
        # Calculate temperature delta from zone's winter average
        # Positive = warmer than average, Negative = colder than average
        temp_delta = outdoor_temp - self.zone_info.winter_avg_low

        # Adjustment: -20 DM per degree colder than average
        # Warmer = less heat needed = shallower DM expected
        # Colder = more heat needed = deeper DM expected
        adjustment = temp_delta * 20  # More positive = less negative DM

        # Calculate adjusted thresholds
        normal_min = self.zone_info.dm_normal_min + adjustment
        normal_max = self.zone_info.dm_normal_max + adjustment
        warning = self.zone_info.dm_warning_threshold + adjustment

        # Ensure we never expect DM beyond absolute maximum
        # Leave 100 DM buffer before critical limit
        normal_min = max(normal_min, DM_ABSOLUTE_MAXIMUM + 100)
        normal_max = max(normal_max, DM_ABSOLUTE_MAXIMUM + 50)
        warning = max(warning, DM_ABSOLUTE_MAXIMUM + 50)

        # Debug logging removed to reduce spam - this is called multiple times per update

        return {
            "normal_min": normal_min,
            "normal_max": normal_max,
            "warning": warning,
            "critical": DM_ABSOLUTE_MAXIMUM,  # Always -1500
        }

    def get_safety_margin(self) -> float:
        """Get flow temperature safety margin for this climate zone.

        Used by weather compensation to add extra safety margin in colder climates.

        Returns:
            Safety margin in °C to add to calculated flow temperature
        """
        return self.zone_info.safety_margin_base
