#!/usr/bin/env python3
"""Visualize price optimization behavior - Current vs Expected.

Based on real data from 2025-12-05 showing:
- Today prices: ~160-200 öre/kWh during day, peak around 16-18h
- Tomorrow prices: ~50-60 öre/kWh all day (65% cheaper)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

# Real price data approximated from screenshot (öre/kWh)
# Today: Dec 5, 2025 - prices from 00:00 to 23:45
today_prices_hourly = [
    # Night hours (cheap)
    80, 75, 70, 65, 60, 55,  # 00:00-05:00
    # Morning ramp
    90, 120, 150, 170,  # 06:00-09:00
    # Day (expensive)
    175, 180, 185, 175, 170, 176,  # 10:00-15:00
    # Peak hours
    190, 200, 185, 160,  # 16:00-19:00
    # Evening decline
    130, 110, 95, 85,  # 20:00-23:00
]

# Tomorrow: Dec 6, 2025 - cheap all day
tomorrow_prices_hourly = [
    52, 50, 48, 47, 46, 48,  # 00:00-05:00
    55, 58, 60, 60,  # 06:00-09:00
    58, 56, 55, 54, 55, 56,  # 10:00-15:00
    58, 60, 58, 55,  # 16:00-19:00
    52, 50, 48, 46,  # 20:00-23:00
]

# Expand to 15-min intervals
today_prices = []
for p in today_prices_hourly:
    today_prices.extend([p] * 4)

tomorrow_prices = []
for p in tomorrow_prices_hourly:
    tomorrow_prices.extend([p] * 4)

# Current time is ~16:00 (Q64)
current_quarter = 64  # 16:00

# Create time axis (hours from now, with current time = 0)
hours_from_now = np.arange(-current_quarter, 192 - current_quarter) / 4

# Combine prices
all_prices = today_prices + tomorrow_prices

# --- Calculate CURRENT behavior (broken) ---
# When price layer weight is 0.24 and prediction layer is 0.65
current_offset = []
for i, price in enumerate(all_prices):
    if i < current_quarter:
        # Past - no action
        current_offset.append(0)
    else:
        # Current buggy behavior:
        # - Prediction layer: +1.5°C (constant pre-heating for predicted cold)
        # - Price layer: reduced weight due to volatility
        # - Weather comp: -1.4°C
        
        if price > 180:  # Peak
            offset = -1.0  # Some reduction but prediction still fights it
        elif price > 160:  # Expensive
            offset = 1.0   # Prediction wins (current bug)
        elif price > 100:  # Normal
            offset = 0.5
        else:  # Cheap
            offset = 1.5   # Pre-heating correctly activates
        current_offset.append(offset)

# --- Calculate EXPECTED behavior (fixed) ---
# When price layer keeps full 0.8 weight during EXPENSIVE
expected_offset = []
for i, price in enumerate(all_prices):
    if i < current_quarter:
        # Past - no action
        expected_offset.append(0)
    else:
        hours_until_cheap = 0
        # Find hours until sustained cheap prices
        for j in range(i, len(all_prices)):
            if all_prices[j] < 80:  # Cheap threshold
                hours_until_cheap = (j - i) / 4
                break
        
        if price > 180:  # Peak
            offset = -3.0  # Maximum reduction
        elif price > 160:  # Expensive
            if hours_until_cheap < 12:  # Cheap coming soon
                offset = -1.8  # Reduce heating, wait for cheap
            else:
                offset = -0.5
        elif price > 100:  # Normal
            offset = 0.0
        else:  # Cheap
            # Pre-heat during cheap if cold coming
            offset = 1.5
        expected_offset.append(offset)

# --- Create visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('EffektGuard Price Optimization: Current Bug vs Expected Behavior\n(Based on Dec 5-6, 2025 real prices)', 
             fontsize=14, fontweight='bold')

# Color coding for price regions
def get_price_color(price):
    if price > 180:
        return '#ff4444'  # Red - Peak
    elif price > 160:
        return '#ff8844'  # Orange - Expensive  
    elif price > 100:
        return '#ffcc44'  # Yellow - Normal
    else:
        return '#44cc44'  # Green - Cheap

# Plot 1: Electricity prices
ax1 = axes[0]
colors = [get_price_color(p) for p in all_prices]
for i in range(len(all_prices) - 1):
    ax1.fill_between([hours_from_now[i], hours_from_now[i+1]], 
                     [all_prices[i], all_prices[i+1]], 
                     alpha=0.7, color=colors[i])
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Current time (16:00)')
ax1.set_ylabel('Price (öre/kWh)', fontsize=11)
ax1.set_ylim(0, 220)
ax1.grid(True, alpha=0.3)
ax1.set_title('Electricity Prices: Today (expensive) → Tomorrow (65% cheaper)', fontsize=11)

# Add price zone legend
peak_patch = mpatches.Patch(color='#ff4444', label='PEAK (>180 öre)')
expensive_patch = mpatches.Patch(color='#ff8844', label='EXPENSIVE (160-180 öre)')
normal_patch = mpatches.Patch(color='#ffcc44', label='NORMAL (100-160 öre)')
cheap_patch = mpatches.Patch(color='#44cc44', label='CHEAP (<100 öre)')
ax1.legend(handles=[peak_patch, expensive_patch, normal_patch, cheap_patch], 
           loc='upper right', fontsize=9)

# Add "Tomorrow" label
ax1.annotate('← TODAY', xy=(-4, 200), fontsize=10, fontweight='bold', color='gray')
ax1.annotate('TOMORROW →', xy=(12, 60), fontsize=10, fontweight='bold', color='green')

# Plot 2: Current (buggy) behavior
ax2 = axes[1]
# Only plot from current time onwards
future_hours = hours_from_now[current_quarter:]
future_current = current_offset[current_quarter:]
ax2.fill_between(future_hours, future_current, 0, 
                  where=[o > 0 for o in future_current], 
                  color='#ff6666', alpha=0.7, label='Heating (+offset)')
ax2.fill_between(future_hours, future_current, 0, 
                  where=[o <= 0 for o in future_current], 
                  color='#6666ff', alpha=0.7, label='Reducing (-offset)')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax2.set_ylabel('Offset (°C)', fontsize=11)
ax2.set_ylim(-4, 3)
ax2.grid(True, alpha=0.3)
ax2.set_title('CURRENT Behavior (Bug): Heating during expensive period!', fontsize=11, color='red')
ax2.legend(loc='upper right', fontsize=9)

# Annotate the problem
ax2.annotate('BUG: +1°C offset\n(prediction layer\noverrides price)', 
             xy=(1, 1.0), xytext=(3, 2.2),
             fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

# Plot 3: Expected (fixed) behavior
ax3 = axes[2]
future_expected = expected_offset[current_quarter:]
ax3.fill_between(future_hours, future_expected, 0, 
                  where=[o > 0 for o in future_expected], 
                  color='#ff6666', alpha=0.7, label='Heating (+offset)')
ax3.fill_between(future_hours, future_expected, 0, 
                  where=[o <= 0 for o in future_expected], 
                  color='#6666ff', alpha=0.7, label='Reducing (-offset)')
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_ylabel('Offset (°C)', fontsize=11)
ax3.set_xlabel('Hours from now', fontsize=11)
ax3.set_ylim(-4, 3)
ax3.grid(True, alpha=0.3)
ax3.set_title('EXPECTED Behavior (Fixed): Reduce now, pre-heat during cheap tomorrow', 
              fontsize=11, color='green')
ax3.legend(loc='upper right', fontsize=9)

# Annotate the fix
ax3.annotate('FIXED: -1.8°C offset\n(reduce heating,\nwait for cheap)', 
             xy=(1, -1.8), xytext=(3, -3.0),
             fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))

ax3.annotate('Pre-heat when\nprices are cheap', 
             xy=(10, 1.5), xytext=(14, 2.5),
             fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green'))

# Add hour markers on x-axis
ax3.set_xticks(range(-16, 32, 4))
ax3.set_xlim(-16, 32)

plt.tight_layout()
plt.savefig('/workspaces/EffektGuard/docs/dev/price_optimization_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Graph saved to: /workspaces/EffektGuard/docs/dev/price_optimization_comparison.png")
print("\nKey observations:")
print("  • Current (bug): +1°C offset during expensive period (176 öre)")
print("  • Expected (fix): -1.8°C offset during expensive, +1.5°C during cheap tomorrow")
print("  • Savings potential: ~65% by shifting heating to tomorrow")
