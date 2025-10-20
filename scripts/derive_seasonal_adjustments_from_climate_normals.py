#!/usr/bin/env python3
"""
Derive seasonal adjustments from Swedish climate normals (1991-2020).

Data source: SMHI climate normals published at:
https://www.smhi.se/data/meteorologi/temperatur/

This script uses publicly available 30-year climate normals from representative
Swedish weather stations to calculate realistic seasonal temperature adjustments
for the EffektGuard weather learning system.
"""

from statistics import mean

# SMHI Climate Normals 1991-2020 - Monthly Average Temperatures (°C)
# Source: https://www.smhi.se/data/meteorologi/temperatur/arsdata-normalvarden-for-sverige
CLIMATE_NORMALS = {
    # Extreme Cold (66.5-90°N): Kiruna Airport
    "Kiruna": {
        "latitude": 67.82,
        "zone": "Extreme Cold",
        "monthly_avg": {
            1: -13.3,
            2: -12.8,
            3: -8.5,
            4: -2.9,
            5: 3.8,
            6: 9.6,
            7: 13.0,
            8: 10.9,
            9: 5.6,
            10: -1.3,
            11: -7.4,
            12: -11.5,
        },
    },
    # Very Cold (63.5-66.5°N): Arvidsjaur
    "Arvidsjaur": {
        "latitude": 65.59,
        "zone": "Very Cold",
        "monthly_avg": {
            1: -13.0,
            2: -11.7,
            3: -6.7,
            4: -0.8,
            5: 5.8,
            6: 11.7,
            7: 14.5,
            8: 12.3,
            9: 7.0,
            10: 0.9,
            11: -5.4,
            12: -10.7,
        },
    },
    # Cold (60.5-63.5°N): Östersund
    "Östersund": {
        "latitude": 63.18,
        "zone": "Cold",
        "monthly_avg": {
            1: -8.5,
            2: -7.2,
            3: -2.7,
            4: 2.5,
            5: 8.9,
            6: 13.9,
            7: 16.1,
            8: 14.5,
            9: 9.4,
            10: 3.9,
            11: -2.2,
            12: -6.4,
        },
    },
    # Moderate Cold (57.5-60.5°N): Stockholm (Uppsala)
    "Stockholm": {
        "latitude": 59.35,
        "zone": "Moderate Cold",
        "monthly_avg": {
            1: -1.1,
            2: -1.2,
            3: 1.5,
            4: 6.3,
            5: 12.2,
            6: 16.2,
            7: 18.4,
            8: 17.4,
            9: 12.7,
            10: 7.3,
            11: 2.9,
            12: -0.2,
        },
    },
    # Moderate (54.5-57.5°N): Southern Sweden (Växjö/Malmö region)
    "Southern Sweden": {
        "latitude": 56.85,
        "zone": "Moderate",
        "monthly_avg": {
            1: -0.2,
            2: -0.2,
            3: 2.3,
            4: 7.0,
            5: 12.5,
            6: 16.0,
            7: 18.1,
            8: 17.6,
            9: 13.4,
            10: 8.6,
            11: 4.1,
            12: 1.0,
        },
    },
}


def calculate_seasonal_adjustments(location_name: str, monthly_temps: dict) -> dict:
    """Calculate seasonal adjustments relative to winter baseline."""
    # Winter baseline (Jan-Feb average)
    winter_baseline = mean([monthly_temps[1], monthly_temps[2]])

    print(f"\n{location_name} ({CLIMATE_NORMALS[location_name]['zone']}):")
    print(f"  Latitude: {CLIMATE_NORMALS[location_name]['latitude']:.2f}°N")
    print(f"  Winter baseline (Jan-Feb): {winter_baseline:.1f}°C")

    # Calculate adjustment for each month (difference from winter baseline)
    adjustments = {}
    for month, temp in monthly_temps.items():
        adjustment = temp - winter_baseline
        adjustments[month] = adjustment

    return adjustments


def main():
    """Generate seasonal adjustment table from climate normals."""
    print("=" * 80)
    print("Swedish Climate Normals Analysis (1991-2020)")
    print("=" * 80)

    all_adjustments = {}

    for location, data in CLIMATE_NORMALS.items():
        adjustments = calculate_seasonal_adjustments(location, data["monthly_avg"])
        all_adjustments[location] = adjustments

    # Print table
    print("\n" + "=" * 80)
    print("MONTHLY ADJUSTMENTS vs Winter Baseline (°C)")
    print("=" * 80)

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Header
    print(f"\n{'Month':<8}", end="")
    for location in CLIMATE_NORMALS.keys():
        print(f"{location[:15]:>16}", end="")
    print()
    print("-" * 80)

    # Data rows
    for month in range(1, 13):
        print(f"{month_names[month-1]:<8}", end="")
        for location in CLIMATE_NORMALS.keys():
            adj = all_adjustments[location][month]
            print(f"{adj:>15.1f}°", end="")
        print()

    # Calculate average adjustments across all stations
    print("\n" + "=" * 80)
    print("AVERAGE SEASONAL ADJUSTMENTS (All Stations)")
    print("=" * 80)

    avg_adjustments = {}
    for month in range(1, 13):
        month_vals = [all_adjustments[loc][month] for loc in CLIMATE_NORMALS.keys()]
        avg_adjustments[month] = mean(month_vals)

    print(f"\n{'Month':<12}{'Adjustment':>15}{'Typical Temp Range':>30}")
    print("-" * 80)
    for month in range(1, 13):
        adj = avg_adjustments[month]
        # Rough temp range (assuming baseline around 0°C for average Sweden)
        temp_range = f"{adj-2.5:.0f}° to {adj+2.5:.0f}°"
        print(f"{month_names[month-1]:<12}{adj:>14.1f}°  {temp_range:>30}")

    # Generate Python code
    print("\n" + "=" * 80)
    print("UPDATED seasonal_adjustment DICT FOR weather_learning.py")
    print("=" * 80)
    print()
    print("# Based on SMHI Climate Normals 1991-2020 from representative Swedish stations")
    print("# Adjustments relative to winter (Jan-Feb) baseline")
    print("# Stations: Kiruna (67.8°N), Arvidsjaur (65.6°N), Östersund (63.2°N),")
    print("#           Stockholm (59.4°N), Southern Sweden (56.9°N)")
    print("seasonal_adjustment = {")
    for month in range(1, 13):
        adj = avg_adjustments[month]
        comment = f"  # {month_names[month-1]}: {adj-2.5:.0f}° to {adj+2.5:.0f}°C typical range"
        print(f"    {month}: {adj:>5.1f},{comment}")
    print("}")

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("\nNOTE: These are Sweden-wide averages. The ClimateZoneDetector adjusts")
    print("      the winter baseline based on latitude, so these seasonal adjustments")
    print("      work for all climate zones from Arctic to Mediterranean.")
    print("=" * 80)


if __name__ == "__main__":
    main()
