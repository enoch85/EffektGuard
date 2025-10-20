#!/usr/bin/env python3
"""
Real-Life Day Simulation Tool

Extracts actual decision data from debug.log and simulates what would happen
with the new climate-aware logic using the test_decision_scenarios.py script
which already uses production constants.

Provides a comprehensive comparison of:
- What actually happened (old logic)
- What would have happened (new climate-aware logic with production constants)
"""

import re
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RealDataPoint:
    """Real data extracted from debug.log"""

    time: str
    dm: float
    outdoor_temp: float
    indoor_temp: float
    flow_temp: float
    actual_offset: float
    price: float
    # Additional context from logs
    predicted_drop: Optional[float] = None
    warming_rate: Optional[float] = None
    current_power: Optional[float] = None
    monthly_peak: Optional[float] = None
    dhw_temp: Optional[float] = None
    unusual_weather: Optional[str] = None
    price_quarter: Optional[str] = None


@dataclass
class TestResult:
    """Result from test_decision_scenarios.py"""

    offset: float
    reasoning: str


def parse_debug_log(log_file: str) -> List[RealDataPoint]:
    data_points = []

    with open(log_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for decision lines
        if "Decision: offset " in line:
            # Extract timestamp
            ts_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if not ts_match:
                i += 1
                continue

            timestamp = ts_match.group(1)
            time_only = timestamp.split()[1]

            # Extract offset from decision line
            offset_match = re.search(r"Decision: offset (-?\d+\.?\d*)°C", line)
            if not offset_match:
                i += 1
                continue

            offset = float(offset_match.group(1))

            # Now look BACKWARDS to find the most recent values before this decision
            dm = None
            outdoor = None
            indoor = None
            flow = None
            price = 70.0
            predicted_drop = None
            warming_rate = None
            current_power = None
            monthly_peak = None
            dhw_temp = None
            unusual_weather = None
            price_quarter = None

            # Search backwards up to 50 lines
            for j in range(max(0, i - 50), i):
                prev_line = lines[j]

                # Get DM (closest one before decision)
                if dm is None and "degree_minutes = " in prev_line:
                    dm_match = re.search(r"degree_minutes = (-?\d+\.?\d*)", prev_line)
                    if dm_match:
                        dm = float(dm_match.group(1))

                # Get outdoor temp
                if outdoor is None and ("outdoor -" in prev_line or "outdoor=" in prev_line):
                    outdoor_match = re.search(r"outdoor[=\s]+(-?\d+\.?\d*)°?C", prev_line)
                    if outdoor_match:
                        outdoor = float(outdoor_match.group(1))

                # Get indoor temp
                if indoor is None:
                    if "Multi-sensor indoor temp:" in prev_line:
                        indoor_match = re.search(
                            r"Multi-sensor indoor temp: (\d+\.?\d*)°C", prev_line
                        )
                        if indoor_match:
                            indoor = float(indoor_match.group(1))
                    elif "indoor=" in prev_line:
                        indoor_match = re.search(r"indoor[=\s]+(\d+\.?\d*)°?C", prev_line)
                        if indoor_match:
                            indoor = float(indoor_match.group(1))

                # Get flow temp - look for "Current: XX°C"
                if flow is None and "Current:" in prev_line and "°C" in prev_line:
                    flow_match = re.search(r"Current: (\d+\.?\d*)°C", prev_line)
                    if flow_match:
                        flow = float(flow_match.group(1))

                # Get price
                if "öre/kWh" in prev_line:
                    price_match = re.search(r"(\d+\.?\d*) öre/kWh", prev_line)
                    if price_match:
                        price = float(price_match.group(1))
                    # Get price quarter
                    quarter_match = re.search(r"Q(\d+)", prev_line)
                    if quarter_match:
                        price_quarter = f"Q{quarter_match.group(1)}"

                # Get predicted drop
                if predicted_drop is None and "predicted drop" in prev_line:
                    drop_match = re.search(r"predicted drop (-?\d+\.?\d*)°C", prev_line)
                    if drop_match:
                        predicted_drop = float(drop_match.group(1))

                # Get warming rate
                if warming_rate is None and "warming" in prev_line and "°C/h" in prev_line:
                    warm_match = re.search(r"warming (-?\d+\.?\d*)°C/h", prev_line)
                    if warm_match:
                        warming_rate = float(warm_match.group(1))

                # Get current power
                if current_power is None and "External power meter" in prev_line:
                    power_match = re.search(r"(\d+\.?\d*) kW", prev_line)
                    if power_match:
                        current_power = float(power_match.group(1))
                elif current_power is None and "NIBE 3-phase power:" in prev_line:
                    power_match = re.search(r"= (\d+\.?\d*) kW", prev_line)
                    if power_match:
                        current_power = float(power_match.group(1))

                # Get monthly peak
                if monthly_peak is None and "monthly peak" in prev_line.lower():
                    peak_match = re.search(r"peak[:\s]+(\d+\.?\d*) kW", prev_line, re.IGNORECASE)
                    if peak_match:
                        monthly_peak = float(peak_match.group(1))

                # Get DHW temp
                if dhw_temp is None and "DHW top temperature" in prev_line:
                    dhw_match = re.search(r"(\d+\.?\d*)°C", prev_line)
                    if dhw_match:
                        dhw_temp = float(dhw_match.group(1))

                # Get unusual weather
                if unusual_weather is None and "Unusual weather detected:" in prev_line:
                    unusual_match = re.search(r"Unusual weather detected: (.+)", prev_line)
                    if unusual_match:
                        unusual_weather = unusual_match.group(1).strip()

            # Only add if we have the critical values
            if dm is not None and outdoor is not None:
                data_points.append(
                    RealDataPoint(
                        time=time_only,
                        dm=dm,
                        outdoor_temp=outdoor,
                        indoor_temp=indoor if indoor is not None else 22.0,
                        flow_temp=flow if flow is not None else 40.0,
                        actual_offset=offset,
                        price=price,
                        predicted_drop=predicted_drop,
                        warming_rate=warming_rate,
                        current_power=current_power,
                        monthly_peak=monthly_peak,
                        dhw_temp=dhw_temp,
                        unusual_weather=unusual_weather,
                        price_quarter=price_quarter,
                    )
                )

        i += 1

    return data_points


def run_decision_test(dp: RealDataPoint) -> TestResult:
    """Run test_decision_scenarios.py with real data point"""
    cmd = [
        "python3",
        "/workspaces/EffektGuard/scripts/test_decision_scenarios.py",
        "--scenario",
        "custom",
        "--indoor",
        str(dp.indoor_temp),
        "--outdoor",
        str(dp.outdoor_temp),
        "--dm",
        str(dp.dm),
        "--flow-temp",
        str(dp.flow_temp),
        "--price",
        str(dp.price),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        # Parse output for decision - look for "🏆 RESULT: +2.00°C"
        for line in result.stdout.split("\n"):
            if "🏆 RESULT:" in line or "RESULT:" in line:
                offset_match = re.search(r"([+-]?\d+\.?\d*)°C", line)
                if offset_match:
                    return TestResult(offset=float(offset_match.group(1)), reasoning="Success")

        return TestResult(offset=0.0, reasoning="No result found")
    except Exception as e:
        return TestResult(offset=0.0, reasoning=f"Error: {e}")

        return 0.0, "No decision found"
    except Exception as e:
        return 0.0, f"Error: {e}"


def print_comparison(data_points: List[RealDataPoint], test_results: List[TestResult]):
    """Print side-by-side comparison"""
    print("\n" + "=" * 180)
    print("REAL-LIFE DAY SIMULATION - ACTUAL vs NEW CLIMATE-AWARE LOGIC")
    print("=" * 180)
    print("Data extracted from: /workspaces/EffektGuard/IMPLEMENTATION_PLAN/debug.log")
    print("Testing with: test_decision_scenarios.py (uses production constants)")
    print(f"Total decision points: {len(data_points)}")
    print(f"Time range: {data_points[0].time} to {data_points[-1].time}")
    print("=" * 180)
    print()

    more_aggressive = 0
    similar = 0
    less_aggressive = 0

    actual_total = 0.0
    new_total = 0.0

    # Calculate stats for ALL points
    for dp, result in zip(data_points, test_results):
        diff = result.offset - dp.actual_offset

        if abs(diff) < 0.1:
            similar += 1
        elif diff > 0:
            more_aggressive += 1
        else:
            less_aggressive += 1

        actual_total += dp.actual_offset
        new_total += result.offset

    # Show key time periods with context
    print(
        f"{'Time':<8} | {'DM':>6} | {'Out°C':>5} | {'In°C':>5} | {'Flow°C':>6} | {'ACTUAL':>8} | {'NEW':>8} | {'Δ':>8} | Context"
    )
    print("-" * 180)

    # Show every Nth point to give overview of whole day (show ~50 samples across the full range)
    step = max(1, len(data_points) // 50)
    shown_count = 0

    for i in range(0, len(data_points), step):
        dp = data_points[i]
        result = test_results[i]
        diff = result.offset - dp.actual_offset

        if abs(diff) < 0.1:
            status = "≈"
        elif diff > 0:
            status = "⬆️"
        else:
            status = "⬇️"

        # Build context string
        context_parts = []
        if dp.predicted_drop:
            context_parts.append(f"Drop:{dp.predicted_drop:+.1f}°C")
        if dp.warming_rate:
            context_parts.append(f"Warm:{dp.warming_rate:+.2f}°C/h")
        if dp.current_power:
            context_parts.append(f"Pwr:{dp.current_power:.1f}kW")
        if dp.monthly_peak:
            context_parts.append(f"Peak:{dp.monthly_peak:.1f}kW")
        if dp.unusual_weather:
            # Shorten the unusual weather message
            if "Unusually warm" in dp.unusual_weather:
                context_parts.append("☀️Warm")
            elif "Unusually cold" in dp.unusual_weather:
                context_parts.append("❄️Cold")
        if dp.price_quarter:
            context_parts.append(dp.price_quarter)

        context_str = " ".join(context_parts) if context_parts else "-"

        print(
            f"{dp.time:<8} | {dp.dm:>6.0f} | {dp.outdoor_temp:>5.1f} | {dp.indoor_temp:>5.1f} | {dp.flow_temp:>6.1f} | "
            f"{dp.actual_offset:>+7.2f}° | {result.offset:>+7.2f}° | {diff:>+7.2f}° {status} | {context_str}"
        )

        shown_count += 1
        # Add separator every 10 rows
        if shown_count % 10 == 0 and i < len(data_points) - step:
            print("-" * 180)

    print()
    print("=" * 180)
    print()
    print("SUMMARY")
    print("-" * 180)
    avg_actual = actual_total / len(data_points)
    avg_new = new_total / len(test_results)
    print(
        f"Average offset - Actual: {avg_actual:+.2f}°C, New: {avg_new:+.2f}°C, Difference: {avg_new - avg_actual:+.2f}°C"
    )
    print(
        f"Decisions: More aggressive: {more_aggressive}, Similar: {similar}, Less aggressive: {less_aggressive}"
    )
    print()
    dm_min = min(d.dm for d in data_points)
    dm_max = max(d.dm for d in data_points)
    print(f"DM Range: {dm_min:.0f} (worst) to {dm_max:.0f} (best)")

    # Show context statistics
    has_predicted = sum(1 for d in data_points if d.predicted_drop is not None)
    has_warming = sum(1 for d in data_points if d.warming_rate is not None)
    has_power = sum(1 for d in data_points if d.current_power is not None)
    has_peak = sum(1 for d in data_points if d.monthly_peak is not None)
    has_unusual = sum(1 for d in data_points if d.unusual_weather is not None)

    print()
    print(f"Context data extracted:")
    print(
        f"  - Predicted drops: {has_predicted}/{len(data_points)} ({has_predicted*100//len(data_points)}%)"
    )
    print(
        f"  - Warming rates: {has_warming}/{len(data_points)} ({has_warming*100//len(data_points)}%)"
    )
    print(f"  - Current power: {has_power}/{len(data_points)} ({has_power*100//len(data_points)}%)")
    print(f"  - Monthly peak: {has_peak}/{len(data_points)} ({has_peak*100//len(data_points)}%)")
    print(
        f"  - Unusual weather: {has_unusual}/{len(data_points)} ({has_unusual*100//len(data_points)}%)"
    )

    print()
    if more_aggressive > len(data_points) * 0.7:
        print("✅ CONCLUSION: New logic would have been MORE AGGRESSIVE in recovery")
        print(f"   → Likely would have prevented DM from reaching {dm_min:.0f}")
    elif less_aggressive > len(data_points) * 0.7:
        print("⚠️  CONCLUSION: New logic would have been LESS AGGRESSIVE")
        print("   → May not have prevented the thermal debt crisis")
    else:
        print("ℹ️  CONCLUSION: New logic shows MIXED behavior")
        print("   → Impact on DM crisis uncertain")
    print("=" * 180)


def main():
    """Main simulation function"""

    log_file = "/workspaces/EffektGuard/IMPLEMENTATION_PLAN/debug.log"

    print("Extracting real decision data from debug.log...")
    data_points = parse_debug_log(log_file)

    if not data_points:
        print("❌ No decision points found in debug.log")
        return 1

    print(f"✓ Found {len(data_points)} decision points")
    print(f"  Time range: {data_points[0].time} to {data_points[-1].time}")
    print(
        f"  DM range: {min(d.dm for d in data_points):.0f} to {max(d.dm for d in data_points):.0f}"
    )

    print("\nSimulating with new climate-aware logic (using production constants)...")

    # Run test for each data point
    test_results = []
    for i, dp in enumerate(data_points):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(data_points)}...")
        result = run_decision_test(dp)
        test_results.append(result)

    print_comparison(data_points, test_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
