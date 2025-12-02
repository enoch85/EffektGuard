#!/usr/bin/env python3
"""
Simulate the proposed Common Sense fix with real data from debug.log.

This script shows what the NEW behavior would produce vs the OLD behavior.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RealScenario:
    """Real scenario extracted from debug.log."""
    time: str
    overshoot: float  # °C above target
    dm: int  # Degree minutes
    flow_temp: float  # Current flow temp
    optimal_flow: float  # Kühne optimal
    old_decision: float  # What system decided
    wc_offset: float  # What WC layer wanted
    price_class: str  # NORMAL, PEAK, etc.


# Real scenarios from debug.log Dec 2, 2025
SCENARIOS = [
    RealScenario(
        time="10:42",
        overshoot=1.1,
        dm=-688,
        flow_temp=33.4,
        optimal_flow=27.6,
        old_decision=+1.70,
        wc_offset=-3.9,
        price_class="NORMAL",
    ),
    RealScenario(
        time="11:32",
        overshoot=1.3,
        dm=-632,
        flow_temp=33.4,
        optimal_flow=27.6,
        old_decision=+1.70,
        wc_offset=-3.9,
        price_class="NORMAL",
    ),
    RealScenario(
        time="12:17",
        overshoot=1.4,
        dm=-547,
        flow_temp=33.4,
        optimal_flow=27.5,
        old_decision=+0.72,
        wc_offset=-4.0,
        price_class="NORMAL",
    ),
    RealScenario(
        time="13:22",
        overshoot=1.5,
        dm=-476,
        flow_temp=32.9,
        optimal_flow=27.6,
        old_decision=+0.77,
        wc_offset=-3.5,
        price_class="NORMAL",
    ),
    RealScenario(
        time="14:02",
        overshoot=1.5,
        dm=-445,  # Approximate - DM finally dropping below -400
        flow_temp=31.4,
        optimal_flow=27.7,
        old_decision=-0.05,
        wc_offset=-2.4,
        price_class="NORMAL",
    ),
]


def old_common_sense(overshoot: float, weather_stable: bool = True) -> tuple[float, float, str]:
    """
    OLD Common Sense behavior - just returns 0.0 offset, 0.0 weight.
    Only BLOCKS proactive layer, doesn't force negative offset.
    """
    COMMON_SENSE_TEMP_ABOVE_TARGET = 1.0  # Current const.py value
    
    if overshoot >= COMMON_SENSE_TEMP_ABOVE_TARGET and weather_stable:
        return 0.0, 0.0, f"Common sense: blocked (overshoot {overshoot:.1f}°C) - but weight=0 so ignored!"
    return None, None, "Not triggered"


def new_common_sense(overshoot: float, weather_stable: bool = True) -> tuple[float, float, str]:
    """
    NEW Common Sense behavior - returns AGGRESSIVE negative offset with graduated weight.
    Forces the system to coast when above target.
    
    Graduated response (starts at 0.8°C, full at 1.5°C):
    - 0.8°C overshoot: offset -7°C, weight 0.5
    - 1.15°C overshoot: offset -8.5°C, weight 0.75
    - 1.5°C overshoot: offset -10°C, weight 1.0
    """
    OVERSHOOT_START = 0.8  # Start responding at 0.8°C above target
    OVERSHOOT_FULL = 1.5   # Full response at 1.5°C above target
    
    if overshoot >= OVERSHOOT_START and weather_stable:
        # Normalize overshoot to 0.0-1.0 range between start and full
        progress = min((overshoot - OVERSHOOT_START) / (OVERSHOOT_FULL - OVERSHOOT_START), 1.0)
        
        # Scale offset from -7 at 0.8°C to -10 at 1.5°C
        aggressive_offset = -7.0 + progress * -3.0
        
        # Scale weight from 0.5 at 0.8°C to 1.0 at 1.5°C
        weight = 0.5 + progress * 0.5
        
        return aggressive_offset, weight, f"Common sense: COAST at {aggressive_offset:.1f}°C, weight {weight:.2f} (overshoot {overshoot:.1f}°C)"
    
    return None, None, "Not triggered"


def simulate_scenario(scenario: RealScenario) -> None:
    """Simulate old vs new behavior for a scenario."""
    print(f"\n{'='*70}")
    print(f"Time: {scenario.time} | Overshoot: +{scenario.overshoot}°C | DM: {scenario.dm}")
    print(f"Flow: {scenario.flow_temp}°C (optimal: {scenario.optimal_flow}°C) | Price: {scenario.price_class}")
    print(f"-" * 70)
    
    # Old behavior
    old_offset, old_weight, old_reason = old_common_sense(scenario.overshoot)
    print(f"OLD Common Sense: {old_reason}")
    print(f"OLD Final Decision: {scenario.old_decision:+.2f}°C")
    print(f"OLD WC layer wanted: {scenario.wc_offset:+.1f}°C but weight only 0.20 (deferred)")
    
    # New behavior
    new_offset, new_weight, new_reason = new_common_sense(scenario.overshoot)
    print()
    print(f"NEW Common Sense: {new_reason}")
    if new_offset is not None:
        print(f"NEW Final Decision: {new_offset:+.1f}°C (weight {new_weight})")
        
        # Predict effect
        dm_recovery_rate = 30  # DM per hour at low offset (estimated)
        if new_offset <= -7:
            dm_recovery_rate = 100  # Much faster recovery when we actually stop heating!
        
        hours_to_recover = abs(scenario.dm) / dm_recovery_rate
        print(f"NEW Predicted: DM would recover from {scenario.dm} → 0 in ~{hours_to_recover:.1f}h")
    
    # Delta
    if new_offset is not None:
        delta = new_offset - scenario.old_decision
        print(f"\nDELTA: {delta:+.1f}°C (from {scenario.old_decision:+.2f} → {new_offset:+.1f})")


def main():
    print("=" * 70)
    print("SIMULATION: Common Sense Overshoot Fix")
    print("=" * 70)
    print("\nUsing REAL data from debug.log Dec 2, 2025")
    print("Target: 21.0°C | All scenarios had overshoot ≥1.0°C")
    print("\nNEW behavior: When overshoot ≥0.8°C and weather stable:")
    print("  - Offset scales from -7°C (at 0.8°C) to -10°C (at 1.5°C)")
    print("  - Weight scales from 0.5 (at 0.8°C) to 1.0 (at 1.5°C)")
    print("  - This gradually FORCES the system to coast as overshoot increases")
    
    for scenario in SCENARIOS:
        simulate_scenario(scenario)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n| Time  | Overshoot | OLD Decision | NEW Decision | Delta    |")
    print("|-------|-----------|--------------|--------------|----------|")
    for s in SCENARIOS:
        new_offset, _, _ = new_common_sense(s.overshoot)
        delta = new_offset - s.old_decision if new_offset else 0
        print(f"| {s.time} | +{s.overshoot}°C     | {s.old_decision:+.2f}°C      | {new_offset:+.1f}°C       | {delta:+.1f}°C    |")
    
    print("\nKEY INSIGHT:")
    print("When you manually set -10°C offset, DM recovered from -878 to +100.")
    print("The new Common Sense behavior automates exactly that response.")


if __name__ == "__main__":
    main()
