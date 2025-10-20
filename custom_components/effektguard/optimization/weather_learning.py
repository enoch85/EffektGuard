"""Weather pattern learning for seasonal prediction enhancement.

Learns historical weather patterns to improve long-term predictions:
- Multi-year pattern database
- Seasonal typical weather (percentiles)
- Unusual weather detection
- Swedish SMHI climate integration

Based on POST_PHASE_5_ROADMAP.md Phase 6.3 and Swedish_Climate_Adaptations.md.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from .learning_types import UnusualWeatherAlert, WeatherExpectation, WeatherPattern

_LOGGER = logging.getLogger(__name__)


class WeatherPatternLearner:
    """Learn seasonal weather patterns for better predictions.

    Supplements weather forecast with historical context:
    - Learns typical weather for each time of year
    - Detects unusual patterns (extreme cold/warm)
    - Improves multi-day predictions
    - Adapts to local micro-climate

    Swedish climate awareness:
    - Handles -30°C to +5°C range
    - SMHI historical data compatible
    - Malmö/Stockholm/Kiruna patterns
    """

    def __init__(self, climate_zone_info=None):
        """Initialize weather pattern learner.

        Args:
            climate_zone_info: ClimateZoneInfo object for seasonal defaults
        """
        self.pattern_db: dict[str, list[WeatherPattern]] = defaultdict(list)
        # Key format: "month-week" (e.g., "1-2" = January, week 2)
        self.climate_zone = climate_zone_info

    def get_seasonal_default(self, month: int) -> tuple[float, float, float]:
        """Get seasonal temperature defaults based on climate zone and month.

        Provides realistic defaults for when historical data is not yet available.
        Uses climate zone winter average and adjusts for seasonal variation.

        Args:
            month: Month number (1-12)

        Returns:
            Tuple of (typical_low, typical_avg, typical_high) in °C
        """
        if self.climate_zone is None:
            # Global fallback if no climate zone
            winter_avg = 0.0
        else:
            winter_avg = self.climate_zone.winter_avg_low

        # Seasonal temperature adjustment relative to winter baseline
        # Based on SMHI Climate Normals 1991-2020 from representative Swedish stations
        # Stations: Kiruna (67.8°N), Arvidsjaur (65.6°N), Östersund (63.2°N),
        #           Stockholm (59.4°N), Southern Sweden (56.9°N)
        # Values represent typical deviation from Jan-Feb average temperatures
        seasonal_adjustment = {
            1: -0.3,  # January (slightly below baseline)
            2: 0.3,  # February (slightly above baseline)
            3: 4.1,  # March (early spring warming)
            4: 9.3,  # April (spring)
            5: 15.6,  # May (late spring)
            6: 20.4,  # June (early summer)
            7: 22.9,  # July (peak summer)
            8: 21.5,  # August (late summer)
            9: 16.5,  # September (early autumn)
            10: 10.8,  # October (autumn) - From real SMHI data
            11: 5.3,  # November (late autumn)
            12: 1.4,  # December (early winter)
        }

        adjustment = seasonal_adjustment.get(month, 0.0)
        typical_avg = winter_avg + adjustment

        # Daily temperature range (diurnal variation)
        # Larger in spring/autumn, smaller in winter
        if month in [3, 4, 5, 9, 10, 11]:  # Transitional seasons
            temp_range = 8.0  # °C
        elif month in [6, 7, 8]:  # Summer
            temp_range = 10.0  # °C
        else:  # Winter
            temp_range = 5.0  # °C

        typical_low = typical_avg - (temp_range / 2)
        typical_high = typical_avg + (temp_range / 2)

        return typical_low, typical_avg, typical_high

    def record_weather_pattern(
        self,
        date: datetime,
        daily_temps: list[float],
        temp_volatility: float | None = None,
    ) -> None:
        """Record weather pattern for learning.

        Args:
            date: Date of pattern
            daily_temps: Hourly temperatures for the day (or sampled)
            temp_volatility: Optional pre-calculated volatility
        """
        if not daily_temps:
            _LOGGER.warning("No temperature data provided for %s", date)
            return

        # Calculate pattern statistics
        avg_temp = np.mean(daily_temps)
        min_temp = np.min(daily_temps)
        max_temp = np.max(daily_temps)
        temp_range = max_temp - min_temp

        if temp_volatility is None:
            temp_volatility = np.std(daily_temps)

        # Create pattern key (month and week of month)
        week_of_month = (date.day - 1) // 7 + 1
        pattern_key = f"{date.month}-{week_of_month}"

        # Store pattern
        pattern = WeatherPattern(
            date=date,
            avg_temp=float(avg_temp),
            min_temp=float(min_temp),
            max_temp=float(max_temp),
            temp_range=float(temp_range),
            volatility=float(temp_volatility),
            year=date.year,
        )

        self.pattern_db[pattern_key].append(pattern)

        _LOGGER.debug(
            "Recorded weather pattern %s: avg=%.1f°C, range=%.1f°C, volatility=%.1f",
            pattern_key,
            avg_temp,
            temp_range,
            temp_volatility,
        )

    def predict_typical_weather(
        self,
        current_date: datetime,
        days_ahead: int = 3,
    ) -> WeatherExpectation:
        """Predict typical weather for this time of year.

        Based on historical patterns from same period in previous years.

        Args:
            current_date: Current date
            days_ahead: Days to look ahead (affects which week to check)

        Returns:
            Expected typical weather with confidence
        """
        # Calculate target period
        target_date = current_date + timedelta(days=days_ahead)
        week_of_month = (target_date.day - 1) // 7 + 1
        period_key = f"{target_date.month}-{week_of_month}"

        # Get historical patterns for this period
        patterns = self.pattern_db.get(period_key, [])

        if not patterns:
            # No historical data - use seasonal defaults based on climate zone
            typical_low, typical_avg, typical_high = self.get_seasonal_default(target_date.month)

            _LOGGER.debug(
                "No historical data for %s, using seasonal defaults: "
                "avg=%.1f°C (range: %.1f to %.1f°C)",
                period_key,
                typical_avg,
                typical_low,
                typical_high,
            )

            return WeatherExpectation(
                period_key=period_key,
                typical_low=typical_low,
                typical_avg=typical_avg,
                typical_high=typical_high,
                confidence=0.0,  # Low confidence - no historical data
                years_of_data=0,
            )

        # Calculate percentiles from historical data
        avg_temps = [p.avg_temp for p in patterns]
        min_temps = [p.min_temp for p in patterns]

        # Use actual historical data for percentiles
        typical_low = float(np.percentile(min_temps, 25))  # 25th percentile
        typical_avg = float(np.median(avg_temps))  # Median
        typical_high = float(np.percentile(avg_temps, 75))  # 75th percentile

        # Calculate confidence based on data consistency
        years_of_data = len(set(p.year for p in patterns))
        data_variance = np.std(avg_temps)

        # More years and low variance = higher confidence
        year_confidence = min(years_of_data / 5.0, 1.0)  # 5+ years = full confidence
        consistency_confidence = 1.0 - min(data_variance / 10.0, 1.0)
        confidence = year_confidence * 0.6 + consistency_confidence * 0.4

        return WeatherExpectation(
            period_key=period_key,
            typical_low=typical_low,
            typical_avg=typical_avg,
            typical_high=typical_high,
            confidence=confidence,
            years_of_data=years_of_data,
        )

    def detect_unusual_weather(
        self,
        current_date: datetime,
        forecast: list[float],
    ) -> UnusualWeatherAlert:
        """Detect if forecasted weather is unusual for this time of year.

        Compares forecast to historical patterns to identify extremes.

        Args:
            current_date: Current date
            forecast: Forecasted temperatures (hourly or daily)

        Returns:
            Unusual weather alert with recommendations
        """
        if not forecast:
            return UnusualWeatherAlert(
                is_unusual=False,
                severity="normal",
                forecast_temps=forecast,
                deviation_from_typical=0.0,
                recommendation="No forecast data",
            )

        # Get typical weather for this period
        typical = self.predict_typical_weather(current_date)

        # Calculate forecast statistics
        forecast_avg = np.mean(forecast)
        forecast_min = np.min(forecast)

        # Compare to typical
        avg_deviation = forecast_avg - typical.typical_avg
        min_deviation = forecast_min - typical.typical_low

        # Determine severity
        # Swedish context: Consider temperature, not just deviation
        if abs(avg_deviation) > 10.0 or abs(min_deviation) > 15.0:
            severity = "extreme"
            is_unusual = True
        elif abs(avg_deviation) > 5.0 or abs(min_deviation) > 8.0:
            severity = "unusual"
            is_unusual = True
        else:
            severity = "normal"
            is_unusual = False

        # Generate recommendation
        if is_unusual:
            if avg_deviation < -5.0:
                # Unusually cold
                if forecast_min < -15.0:
                    # Extreme cold for Swedish conditions
                    recommendation = (
                        f"Extreme cold forecast ({forecast_min:.0f}°C). "
                        f"Aggressive pre-heating recommended. "
                        f"Monitor DM -1500 limit closely. "
                        f"Auxiliary heating may be needed."
                    )
                else:
                    recommendation = (
                        f"Unusually cold ({forecast_avg:.0f}°C vs typical {typical.typical_avg:.0f}°C). "
                        f"Pre-heat 6-12 hours ahead. "
                        f"Reduce DHW priority."
                    )
            elif avg_deviation > 5.0:
                # Unusually warm
                recommendation = (
                    f"Unusually warm ({forecast_avg:.0f}°C). "
                    f"Reduce heating, opportunity for DHW heating."
                )
            else:
                recommendation = "Unusual weather pattern detected, monitor closely."
        else:
            recommendation = "Weather within normal range for this time of year."

        return UnusualWeatherAlert(
            is_unusual=is_unusual,
            severity=severity,
            forecast_temps=forecast,
            deviation_from_typical=avg_deviation,
            recommendation=recommendation,
        )

    def get_seasonal_statistics(self, month: int) -> dict[str, Any]:
        """Get statistical summary for a specific month.

        Args:
            month: Month number (1-12)

        Returns:
            Dictionary with monthly statistics
        """
        # Collect all patterns for this month
        month_patterns = []
        for week in range(1, 6):  # Up to 5 weeks per month
            period_key = f"{month}-{week}"
            month_patterns.extend(self.pattern_db.get(period_key, []))

        if not month_patterns:
            return {
                "month": month,
                "patterns": 0,
                "avg_temp": None,
                "min_temp": None,
                "max_temp": None,
            }

        avg_temps = [p.avg_temp for p in month_patterns]
        min_temps = [p.min_temp for p in month_patterns]
        max_temps = [p.max_temp for p in month_patterns]

        return {
            "month": month,
            "patterns": len(month_patterns),
            "years": len(set(p.year for p in month_patterns)),
            "avg_temp": float(np.mean(avg_temps)),
            "min_temp": float(np.min(min_temps)),
            "max_temp": float(np.max(max_temps)),
            "avg_range": float(np.mean([p.temp_range for p in month_patterns])),
            "volatility": float(np.mean([p.volatility for p in month_patterns])),
        }

    def get_pattern_database_summary(self) -> dict[str, Any]:
        """Get summary of pattern database.

        Returns:
            Dictionary with database statistics
        """
        total_patterns = sum(len(patterns) for patterns in self.pattern_db.values())

        if total_patterns == 0:
            return {
                "total_patterns": 0,
                "periods_covered": 0,
                "years_covered": 0,
                "oldest_pattern": None,
                "newest_pattern": None,
            }

        all_patterns = [p for patterns in self.pattern_db.values() for p in patterns]
        years = set(p.year for p in all_patterns)
        dates = [p.date for p in all_patterns]

        return {
            "total_patterns": total_patterns,
            "periods_covered": len(self.pattern_db),
            "years_covered": len(years),
            "oldest_pattern": min(dates).isoformat(),
            "newest_pattern": max(dates).isoformat(),
            "year_range": f"{min(years)}-{max(years)}",
        }

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert pattern database to dictionary for storage.

        Returns:
            Dictionary representation of pattern database
        """
        return {
            period_key: [
                {
                    "date": p.date.isoformat(),
                    "avg_temp": p.avg_temp,
                    "min_temp": p.min_temp,
                    "max_temp": p.max_temp,
                    "temp_range": p.temp_range,
                    "volatility": p.volatility,
                    "year": p.year,
                }
                for p in patterns
            ]
            for period_key, patterns in self.pattern_db.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[dict[str, Any]]]) -> "WeatherPatternLearner":
        """Create WeatherPatternLearner from dictionary.

        Args:
            data: Dictionary representation of pattern database

        Returns:
            Reconstructed WeatherPatternLearner instance
        """
        learner = cls()

        for period_key, patterns_data in data.items():
            learner.pattern_db[period_key] = [
                WeatherPattern(
                    date=datetime.fromisoformat(p["date"]),
                    avg_temp=p["avg_temp"],
                    min_temp=p["min_temp"],
                    max_temp=p["max_temp"],
                    temp_range=p["temp_range"],
                    volatility=p["volatility"],
                    year=p["year"],
                )
                for p in patterns_data
            ]

        return learner


# Import for timedelta
from datetime import timedelta
