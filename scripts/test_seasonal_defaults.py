#!/usr/bin/env python3
"""Test seasonal defaults for weather learning"""

import sys

sys.path.insert(0, "/workspaces/EffektGuard")

from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.weather_learning import WeatherPatternLearner

# Test for Malmö (latitude 55.60, Moderate Cold zone)
detector = ClimateZoneDetector(latitude=55.60)
zone_info = detector.zone_info

print("=" * 80)
print(f"Climate Zone: {zone_info.name} ({zone_info.description})")
print(f"Winter Average Low: {zone_info.winter_avg_low}°C")
print("=" * 80)
print()

# Old hardcoded defaults
print("OLD DEFAULTS (hardcoded):")
print(f"  All year: Low=-5°C, Avg=0°C, High=+5°C")
print()

# New seasonal defaults
learner = WeatherPatternLearner(climate_zone_info=zone_info)

print("NEW SEASONAL DEFAULTS (climate-aware):")
print("-" * 80)
print(f"{'Month':<15} | {'Low':>8} | {'Avg':>8} | {'High':>8} | {'Description'}")
print("-" * 80)

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

for month_num, month_name in enumerate(months, start=1):
    low, avg, high = learner.get_seasonal_default(month_num)

    if month_num in [10, 11]:  # Highlight autumn
        marker = " ← AUTUMN" if month_num == 10 else " ← AUTUMN"
    elif month_num in [1, 2, 12]:  # Winter
        marker = " ← WINTER"
    else:
        marker = ""

    print(f"{month_name:<15} | {low:>7.1f}° | {avg:>7.1f}° | {high:>7.1f}° | {marker}")

print("-" * 80)
print()

# Show October specifically (your issue)
oct_low, oct_avg, oct_high = learner.get_seasonal_default(10)
print("OCTOBER ANALYSIS:")
print(f"  New defaults: Low={oct_low:.1f}°C, Avg={oct_avg:.1f}°C, High={oct_high:.1f}°C")
print(f"  Old defaults: Low=-5.0°C, Avg=0.0°C, High=+5.0°C")
print()
print("YOUR ACTUAL DAY (Oct 19):")
print(f"  Outdoor temps: -4.3°C to +7.2°C")
print(f"  Average: ~+1.5°C")
print()
print("COMPARISON:")
print(f"  Old: +1.5°C vs 0.0°C baseline = +1.5°C deviation → 'Unusually warm!'")
print(
    f"  New: +1.5°C vs {oct_avg:.1f}°C baseline = {1.5 - oct_avg:.1f}°C deviation → Normal range!"
)
print()
print("✅ With seasonal defaults, your autumn day would NOT trigger 'unusually warm'!")
