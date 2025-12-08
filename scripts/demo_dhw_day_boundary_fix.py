#!/usr/bin/env python3
"""Test script to verify DHW day boundary fix."""
from datetime import datetime, timedelta
from custom_components.effektguard.const import DHW_NORMAL_RUNTIME_MINUTES
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.adapters.gespot_adapter import QuarterPeriod


def create_test_quarters():
    """Create test price data spanning midnight."""
    quarters = []

    # Today's last quarter (23:45-00:00): Q95 at 60.0 öre
    quarters.append(
        QuarterPeriod(
            quarter_of_day=95,
            hour=23,
            minute=45,
            price=60.0,
            is_daytime=False,
        )
    )

    # Tomorrow's early morning quarters (00:00-03:00): Q0-Q11 at cheap prices
    for q in range(12):  # Q0-Q11
        hour = q // 4
        minute = (q % 4) * 15
        price = 5.0 + (q * 0.5)  # 5.0 to 10.5 öre (CHEAP!)
        quarters.append(
            QuarterPeriod(
                quarter_of_day=q,
                hour=hour,
                minute=minute,
                price=price,
                is_daytime=False,
            )
        )

    # Add some expensive morning quarters Q12-Q23 (03:00-06:00)
    for q in range(12, 24):
        hour = q // 4
        minute = (q % 4) * 15
        price = 70.0 + (q * 0.5)  # Expensive!
        quarters.append(
            QuarterPeriod(
                quarter_of_day=q,
                hour=hour,
                minute=minute,
                price=price,
                is_daytime=False,
            )
        )

    return quarters


def main():
    """Run the test."""
    print("=" * 80)
    print("DHW DAY BOUNDARY FIX VERIFICATION")
    print("=" * 80)

    from custom_components.effektguard.optimization.price_layer import PriceAnalyzer

    analyzer = PriceAnalyzer()
    current_time = datetime(2025, 10, 23, 23, 45)  # 23:45 today

    quarters = create_test_quarters()

    print(f"\nCurrent time: {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Price periods: {len(quarters)} quarters")
    print("\nPrice data:")
    for q in quarters:
        print(f"  Q{q.quarter_of_day:2d} ({q.hour:02d}:{q.minute:02d}): {q.price:5.1f} öre/kWh")

    # Find cheapest window using PriceAnalyzer directly
    print(f"\nSearching for cheapest {DHW_NORMAL_RUNTIME_MINUTES}-minute DHW window...")
    optimal_window = analyzer.find_cheapest_window(
        current_time=current_time,
        price_periods=quarters,
        duration_minutes=DHW_NORMAL_RUNTIME_MINUTES,
        lookahead_hours=8.0,
    )

    if optimal_window:
        print(f"\n✓ Optimal window found:")
        print(f"  Start time: {optimal_window.start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  End time: {optimal_window.end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Quarters: {optimal_window.quarters}")
        print(f"  Average price: {optimal_window.avg_price:.2f} öre/kWh")
        print(f"  Hours until: {optimal_window.hours_until:.2f}h")

        # Verify it found tomorrow's cheap prices (should be around Q0-Q2, price ~5-6 öre)
        if optimal_window.avg_price < 10.0 and optimal_window.hours_until > 0:
            print("\n✅ SUCCESS! Found tomorrow's cheap window (not today's expensive Q95)")
            print("   The day boundary fix is working correctly!")
            return 0
        else:
            print("\n❌ FAILED! Window seems wrong")
            print(f"   Expected: price < 10 öre, hours_until > 0")
            print(
                f"   Got: price = {optimal_window.avg_price:.1f} öre, hours_until = {optimal_window.hours_until:.1f}h"
            )
            return 1
    else:
        print("\n❌ FAILED! No window found")
        return 1


if __name__ == "__main__":
    exit(main())
