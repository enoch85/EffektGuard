#!/usr/bin/env python3
"""Test that DHW next boost time uses GE-Spot datetime directly (clean solution).

This test verifies that the DHW next boost time sensor displays the correct
timezone-aware datetime from the DHW optimizer, without any manual reconstruction.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler

# Create timezone-aware test times
stockholm_tz = ZoneInfo("Europe/Stockholm")
now = datetime(2025, 10, 24, 0, 37, 0, tzinfo=stockholm_tz)  # 00:37

print("=" * 80)
print("TEST: DHW Next Boost Time - Clean Solution")
print("=" * 80)
print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Current time ISO: {now.isoformat()}")

# Create price periods with actual GE-Spot datetime format
price_periods = []

# Generate quarters from 00:00 to 03:00 (just for testing)
base_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
prices = {
    0: 23.36,  # 00:00
    1: 21.11,  # 00:15
    2: 13.59,  # 00:30
    3: 9.84,  # 00:45
    4: 14.96,  # 01:00
    5: 10.91,  # 01:15
    6: 9.72,  # 01:30
    7: 6.91,  # 01:45
    8: 10.91,  # 02:00 ← Should be selected (cheapest continuous 3-quarter window)
    9: 7.49,  # 02:15
    10: 5.50,  # 02:30
    11: 4.64,  # 02:45
}

for quarter, price in prices.items():
    hour = quarter // 4
    minute = (quarter % 4) * 15
    start_time = base_date.replace(hour=hour, minute=minute)
    price_periods.append(QuarterPeriod(start_time=start_time, price=price))

print(f"\nCreated {len(price_periods)} price periods with timezone-aware datetimes")
print("\nSample periods:")
for i in [0, 5, 8, 11]:
    period = price_periods[i]
    print(f"  Q{i}: {period.start_time.strftime('%H:%M')} - {period.price:.2f}öre/kWh")
    print(f"       ISO: {period.start_time.isoformat()}")

# Initialize DHW optimizer (no arguments needed, uses defaults)
scheduler = IntelligentDHWScheduler()

# Test DHW decision
print("\n" + "-" * 80)
print("Testing DHW decision...")
print("-" * 80)

decision = scheduler.should_start_dhw(
    current_dhw_temp=48.0,  # Below target+5 (50+5=55), will search for window
    space_heating_demand_kw=1.5,
    thermal_debt_dm=-50,  # OK
    indoor_temp=21.5,
    target_indoor_temp=21.0,
    outdoor_temp=11.7,
    price_classification="normal",
    current_time=now,
    price_periods=price_periods,
    hours_since_last_dhw=3.5,
)

print(f"\nDecision: should_heat={decision.should_heat}")
print(f"Priority reason: {decision.priority_reason}")
print(f"Target temp: {decision.target_temp}°C")

# Check recommended_start_time
print("\n" + "=" * 80)
print("CRITICAL CHECK: recommended_start_time")
print("=" * 80)

if decision.recommended_start_time:
    rec_time = decision.recommended_start_time
    print(f"\n✓ recommended_start_time is set!")
    print(f"  Value: {rec_time}")
    print(f"  Type: {type(rec_time)}")
    print(f"  Formatted: {rec_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  ISO: {rec_time.isoformat()}")

    # Calculate time until
    time_until = rec_time - now
    hours_until = time_until.total_seconds() / 3600
    print(f"  Time until: {hours_until:.2f}h ({int(hours_until * 60)} minutes)")

    # Verify it's timezone-aware
    if rec_time.tzinfo is not None:
        print(f"\n✓ CORRECT: Timezone-aware datetime (tzinfo={rec_time.tzinfo})")
    else:
        print(f"\n✗ ERROR: Naive datetime (no timezone info)")

    # Verify it matches expected time (around 02:00)
    expected_hour = 2
    if rec_time.hour == expected_hour:
        print(f"✓ CORRECT: Hour is {expected_hour} as expected")
    else:
        print(f"✗ WARNING: Hour is {rec_time.hour}, expected {expected_hour}")

    # Verify it's on the correct day (today since it's after 00:37)
    if rec_time.date() == now.date():
        print(f"✓ CORRECT: Same day (2025-10-24)")
    else:
        print(f"✗ ERROR: Wrong day - {rec_time.date()} vs {now.date()}")

else:
    print("\n✗ ERROR: recommended_start_time is None!")

print("\n" + "=" * 80)
print("SIMULATED SENSOR VALUE")
print("=" * 80)

# Simulate what the sensor would display
if decision.recommended_start_time:
    sensor_value = decision.recommended_start_time.isoformat()
    print(f"\nSensor would display: {sensor_value}")
    print(
        f"Home Assistant UI would show: {decision.recommended_start_time.strftime('%B %d, %Y at %H:%M')}"
    )

    # This is what the user sees in their screenshot
    expected_display = "October 24, 2025 at 02:00"
    actual_display = decision.recommended_start_time.strftime("%B %d, %Y at %H:%M")

    if actual_display == expected_display:
        print(f"\n✓ SUCCESS: Display matches expected!")
        print(f"  Expected: {expected_display}")
        print(f"  Actual:   {actual_display}")
    else:
        print(f"\n✗ MISMATCH:")
        print(f"  Expected: {expected_display}")
        print(f"  Actual:   {actual_display}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
