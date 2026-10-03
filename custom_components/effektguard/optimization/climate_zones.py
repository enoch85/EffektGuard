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
    DM_NORMAL_MIN_BUFFER,
    DM_THRESHOLD_AUX_LIMIT,
    DM_THRESHOLD_START,
    DM_WARNING_BUFFER,
    CLIMATE_ZONE_EXTREME_COLD_WINTER_AVG,
    CLIMATE_ZONE_VERY_COLD_WINTER_AVG,
    CLIMATE_ZONE_COLD_WINTER_AVG,
    CLIMATE_ZONE_MODERATE_COLD_WINTER_AVG,
    CLIMATE_ZONE_STANDARD_WINTER_AVG,
)

_LOGGER = logging.getLogger(__name__)


def keep_triggers_clear_of_the_compressor_band(
    thresholds: dict[str, float],
) -> dict[str, float]:
    """No degree-minute TRIGGER may reach into the band the compressor cycles through anyway.

    NIBE starts the compressor at DM_THRESHOLD_START (-60) and stops it at 0, so degree minutes
    traverse that band on EVERY NORMAL CYCLE, in every season, on every heat pump. A threshold
    inside it fires on healthy operation rather than on trouble. DM_WARNING_BUFFER keeps a margin
    below the band for the ordinary undershoot that follows a compressor start.

    THE THRESHOLDS ARE BUILT IN TWO STEPS AND BOTH CAN BREACH IT, which is why this is a function
    and not two lines inside one of them:

    1. `get_expected_dm_range` shifts the zone thresholds by `temp_delta * 20` and that shift was
       unbounded above. In Stockholm the warning threshold climbed to -40 at +25 C - inside the
       band - and to +60 at +30 C, i.e. POSITIVE, where every possible reading is a warning.

    2. `apply_thermal_mass_buffer` then DIVIDES by up to 1.3. Clamping only in step 1 left the
       invariant true of a number production never uses: -110 / 1.3 = -85, back inside the band,
       and a concrete-slab house 0.2 C under target on a +25 C July morning was commanded +4.0 C
       of curve offset by the T1 recovery tier. Clamping in step 1 alone is the bug I shipped and
       then wrote a test for that could not see it, because the test called step 1.

    Winter is untouched: a slab's warning threshold of -415 is far below the ceiling and `min`
    leaves it exactly where it is. This is a ceiling on mild days, never a floor on cold ones.

    `normal_min` is deliberately NOT clamped. Mind the naming - it is the SHALLOW end of the
    normal band (`normal_min` > `normal_max`, both negative) and it describes where degree minutes
    are ALLOWED to sit, rather than triggering anything. In mild weather they really do reach 0,
    because that is where NIBE stops the compressor. `critical` is the hardware auxiliary-heat
    limit and is nowhere near the band.
    """
    warm_ceiling = DM_THRESHOLD_START - DM_WARNING_BUFFER

    return {
        **thresholds,
        "normal_max": min(thresholds["normal_max"], warm_ceiling),
        "warning": min(thresholds["warning"], warm_ceiling),
    }


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

# The absolute safety limit is imported from const.py (DM_THRESHOLD_AUX_LIMIT), never restated here:
# a second copy could drift from the EMERGENCY tier and the `critical` threshold published here,
# which must agree on when the house is in danger (F-112 is open because the number itself may be
# wrong; audit F-076). The buffers below keep the expected band clear of that floor, so the edge of
# "normal" is not also the emergency trigger.


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

        EXAMPLES (computed, not remembered - check them against the code if you change it):
        - Kiruna (Extreme Cold, winter avg -20°C):
          * At -30°C: DM -1000 to -1400 is normal
          * At -20°C: DM  -800 to -1200 is normal (at the zone average, so the base range)

        - Stockholm (Cold, winter avg -8°C):
          * At -10°C: DM -490 to -740 is normal
          * At   0°C: DM -290 to -540 is normal (8°C warmer than average = shallower)

        The examples are the ADJUSTED ranges this method returns, not the base ranges before the
        temperature adjustment; docs/CLIMATE_ZONES.md must match them row for row (checked by
        tests/validation/test_climate_zones_doc_matches_the_code.py).

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

        # COLD SIDE: never expect degree minutes beyond the absolute limit.
        normal_min = max(normal_min, DM_THRESHOLD_AUX_LIMIT + DM_NORMAL_MIN_BUFFER)
        normal_max = max(normal_max, DM_THRESHOLD_AUX_LIMIT + DM_WARNING_BUFFER)
        warning = max(warning, DM_THRESHOLD_AUX_LIMIT + DM_WARNING_BUFFER)

        # WARM SIDE: see keep_triggers_clear_of_the_compressor_band. There was no clamp at all.
        return keep_triggers_clear_of_the_compressor_band(
            {
                "normal_min": normal_min,
                "normal_max": normal_max,
                "warning": warning,
                "critical": DM_THRESHOLD_AUX_LIMIT,
            }
        )

    def get_safety_margin(self) -> float:
        """Get flow temperature safety margin for this climate zone.

        Used by weather compensation to add extra safety margin in colder climates.

        Returns:
            Safety margin in °C to add to calculated flow temperature
        """
        return self.zone_info.safety_margin_base
