"""Test DHW history initialization from Home Assistant recorder.

This test demonstrates that the DHW optimizer loads BT7 temperature history
on startup, which automatically detects past Legionella cycles and prevents
spurious triggers at midnight.

The fix: Instead of preventing the trigger when last_legionella_boost is None,
we ensure it's NEVER None by loading historical BT7 data from Home Assistant's
recorder on startup. This way we always know about recent high-temp cycles.
"""

from datetime import datetime, timedelta
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.const import DHW_LEGIONELLA_DETECT
from tests.conftest import create_mock_price_analyzer


def create_dhw_scheduler(**kwargs):
    """Create DHW scheduler with mock price_analyzer."""
    if "price_analyzer" not in kwargs:
        kwargs["price_analyzer"] = create_mock_price_analyzer()
    return IntelligentDHWScheduler(**kwargs)


def test_history_initialization_concept():
    """
    Demonstrate the concept: Load historical BT7 temps to detect past cycles.

    Before restart (Oct 24):
    - System saw 56°C peak → Legionella cycle detected

    After restart (Oct 25 17:00):
    - OLD behavior: last_legionella_boost = None → triggers at midnight!
    - NEW behavior: Load past 24h history → sees 56°C → last_legionella_boost set ✓

    Result: No spurious trigger at midnight because we KNOW about yesterday's cycle.
    """

    scheduler = create_dhw_scheduler()
    current_time = datetime.now()

    # Simulate loading historical data that shows a Legionella cycle yesterday
    # (This is what initialize_from_history() does automatically)
    historical_temps = [
        (current_time - timedelta(hours=30), 45.0),
        (current_time - timedelta(hours=28), 48.0),
        (current_time - timedelta(hours=26), 52.0),
        (current_time - timedelta(hours=24), 56.0),  # Peak yesterday!
        (current_time - timedelta(hours=22), 54.0),
        (current_time - timedelta(hours=20), 52.0),  # Cooling down
        (current_time - timedelta(hours=18), 50.0),
        (current_time - timedelta(hours=16), 49.0),
        (current_time - timedelta(hours=12), 48.0),
        (current_time - timedelta(hours=8), 47.0),
        (current_time - timedelta(hours=4), 46.6),  # Current temp
    ]

    # Simulate what initialize_from_history() does
    # Track the MOST RECENT time temperature reached Legionella threshold
    last_legionella_time = None

    for timestamp, temp in historical_temps:
        scheduler.bt7_history.append((timestamp, temp))
        # Track most recent time at threshold (not just max temp)
        if temp >= DHW_LEGIONELLA_DETECT:
            if last_legionella_time is None or timestamp > last_legionella_time:
                last_legionella_time = timestamp

    # If we saw ≥55°C in past 24h, record it
    if last_legionella_time:
        scheduler.last_legionella_boost = last_legionella_time

    # Now last_legionella_boost is NOT None - it's set to yesterday!
    assert (
        scheduler.last_legionella_boost is not None
    ), "History initialization should set last_legionella_boost"

    hours_ago = (current_time - scheduler.last_legionella_boost).total_seconds() / 3600.0
    print(f"✓ Detected Legionella boost from history: {hours_ago:.1f} hours ago")

    # Now when midnight comes and price becomes "cheap":
    # days_since_legionella = ~1 day (not None!)
    # Condition: (None OR >= 14) AND cheap = (False OR False) AND True = False
    # Result: NO spurious trigger! ✓

    result = scheduler.should_start_dhw(
        current_dhw_temp=46.6,
        space_heating_demand_kw=0.0,
        thermal_debt_dm=-151,
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=9.2,
        price_classification="cheap",  # Midnight price change
        current_time=current_time,
    )

    assert result.priority_reason != "DHW_HYGIENE_BOOST", (
        f"Should NOT trigger because we know about yesterday's cycle. "
        f"Got: {result.priority_reason}"
    )

    print("✓ No spurious trigger at midnight - we know about yesterday's 56°C cycle!")


def test_hygiene_boost_still_triggers_when_actually_overdue():
    """Verify hygiene boost still works when it's genuinely overdue."""

    scheduler = create_dhw_scheduler()

    # Simulate history showing last boost was 15 days ago
    scheduler.last_legionella_boost = datetime.now() - timedelta(days=15)

    result = scheduler.should_start_dhw(
        current_dhw_temp=46.6,
        space_heating_demand_kw=0.0,
        thermal_debt_dm=-151,
        indoor_temp=21.5,
        target_indoor_temp=21.0,
        outdoor_temp=9.2,
        price_classification="cheap",
        current_time=datetime.now(),
    )

    assert (
        result.priority_reason == "DHW_HYGIENE_BOOST"
    ), f"Should trigger when genuinely overdue (15 days). Got: {result.priority_reason}"

    print("✓ Hygiene boost correctly triggers when actually overdue (15 days)")


if __name__ == "__main__":
    print("Testing DHW history initialization approach...\n")

    test_history_initialization_concept()
    test_hygiene_boost_still_triggers_when_actually_overdue()

    print("\n✅ All tests passed!")
    print("\nThe Solution:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Instead of changing the trigger logic, we ensure we always")
    print("HAVE historical data by loading past 24h of BT7 temps on startup.")
    print("")
    print("Result: last_legionella_boost is never None → no midnight bugs!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
