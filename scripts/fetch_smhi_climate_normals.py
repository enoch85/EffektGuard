#!/usr/bin/env python3
"""
Fetch SMHI climate data to derive realistic seasonal temperature adjustments.

This script queries SMHI's open data API once to get historical temperature data
from representative stations across Sweden, then calculates seasonal adjustments
for the hardcoded weather learning defaults.

Usage: python3 scripts/fetch_smhi_climate_normals.py
"""

import json
import urllib.request
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean

# Representative stations across Sweden's climate zones
STATIONS = {
    # Extreme Cold (66.5-90°N): Kiruna/Abisko region
    "Abisko Aut": {"id": 188790, "lat": 68.35, "zone": "Extreme Cold"},
    # Very Cold (63.5-66.5°N): Arvidsjaur/Arjeplog region
    "Arvidsjaur A": {"id": 159880, "lat": 65.59, "zone": "Very Cold"},
    # Cold (60.5-63.5°N): Östersund region
    "Östersund-Frösön Flygplats": {"id": 134110, "lat": 63.20, "zone": "Cold"},
    # Moderate Cold (57.5-60.5°N): Uppsala/Stockholm region
    "Uppsala Aut": {"id": 97510, "lat": 59.85, "zone": "Moderate Cold"},
    # Moderate (54.5-57.5°N): Växjö region (southern Sweden)
    "Växjö A": {"id": 64510, "lat": 56.85, "zone": "Moderate"},
}


def fetch_station_data(station_id: int, station_name: str) -> dict:
    """Fetch recent temperature data from SMHI station."""
    print(f"Fetching data for {station_name} (ID: {station_id})...")

    monthly_temps = defaultdict(list)

    # Get the data URL (it's a link in the response)
    period_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/{station_id}/period/latest-months.json"

    try:
        # First, get the period info which contains link to actual data
        with urllib.request.urlopen(period_url, timeout=30) as response:
            period_info = json.loads(response.read())

        # Extract the actual data URL
        data_url = None
        for item in period_info.get("data", []):
            for link in item.get("link", []):
                if link.get("type") == "application/json":
                    data_url = link.get("href")
                    break
            if data_url:
                break

        if not data_url:
            print(f"  ⚠️  Could not find data URL")
            return {}

        # Now fetch the actual temperature data
        with urllib.request.urlopen(data_url, timeout=30) as response:
            data = json.loads(response.read())

        # Extract temperature values by month
        for entry in data.get("value", []):
            # Parse the reference date (format: YYYY-MM-DD)
            date_str = entry["ref"]
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            temp = float(entry["value"])
            monthly_temps[dt.month].append(temp)

        print(f"  ✓ Fetched {len(data.get('value', []))} temperature records")

    except Exception as e:
        print(f"  ⚠️  Error fetching data: {e}")
        return {}

    # Calculate monthly averages
    monthly_avg = {}
    for month in range(1, 13):
        if monthly_temps[month]:
            monthly_avg[month] = mean(monthly_temps[month])
        else:
            monthly_avg[month] = None

    return monthly_avg


def calculate_seasonal_adjustments(station_data: dict) -> dict:
    """Calculate seasonal adjustments relative to winter baseline."""
    # Winter baseline (Jan-Feb average)
    winter_temps = [t for m, t in station_data.items() if m in [1, 2] and t is not None]
    if not winter_temps:
        return {}

    winter_baseline = mean(winter_temps)

    # Calculate adjustment for each month (difference from winter baseline)
    adjustments = {}
    for month, temp in station_data.items():
        if temp is not None:
            adjustments[month] = temp - winter_baseline

    return adjustments


def main():
    """Fetch SMHI data and generate seasonal adjustment table."""
    print("=" * 70)
    print("SMHI Climate Data Scraper for EffektGuard")
    print("=" * 70)
    print()

    all_results = {}

    for station_name, info in STATIONS.items():
        monthly_data = fetch_station_data(info["id"], station_name)

        if monthly_data:
            adjustments = calculate_seasonal_adjustments(monthly_data)
            all_results[station_name] = {
                "info": info,
                "monthly_temps": monthly_data,
                "adjustments": adjustments,
            }

            print(
                f"  ✓ Got data for {len([v for v in monthly_data.values() if v is not None])} months"
            )
            winter_temps = [t for m, t in monthly_data.items() if m in [1, 2] and t is not None]
            if winter_temps:
                print(f"  Winter baseline: {mean(winter_temps):.1f}°C")
            print()
        else:
            print(f"  ✗ No data available")
            print()

    # Print summary table
    print("\n" + "=" * 70)
    print("MONTHLY TEMPERATURE ADJUSTMENTS (vs Winter Baseline)")
    print("=" * 70)
    print()

    print(f"{'Month':<12}", end="")
    for name in STATIONS.keys():
        print(f"{name[:18]:>20}", end="")
    print()
    print("-" * 70)

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

    for month in range(1, 13):
        print(f"{month_names[month-1]:<12}", end="")
        for name in STATIONS.keys():
            if name in all_results and month in all_results[name]["adjustments"]:
                adj = all_results[name]["adjustments"][month]
                print(f"{adj:>19.1f}°", end="")
            else:
                print(f"{'---':>19}", end="")
        print()

    # Generate Python code for updated seasonal_adjustment dict
    print("\n" + "=" * 70)
    print("SUGGESTED SEASONAL_ADJUSTMENT DICT FOR weather_learning.py")
    print("=" * 70)
    print()

    # Calculate average adjustments across all stations
    avg_adjustments = {}
    for month in range(1, 13):
        month_vals = []
        for name, results in all_results.items():
            if month in results["adjustments"]:
                month_vals.append(results["adjustments"][month])
        if month_vals:
            avg_adjustments[month] = mean(month_vals)

    print("# Based on SMHI climate data 2023-2024 from representative stations")
    print("# Adjustments relative to winter (Jan-Feb) baseline")
    print("seasonal_adjustment = {")
    for month in range(1, 13):
        if month in avg_adjustments:
            adj = avg_adjustments[month]
            comment = f"  # {month_names[month-1]}"
            print(f"    {month}: {adj:>5.1f},{comment}")
    print("}")

    print("\n" + "=" * 70)
    print("✓ Data fetch complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
