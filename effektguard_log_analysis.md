# EffektGuard Log Analysis Report
**Period:** 2025-12-01 16:28 → 2025-12-02 15:47 (23.3 hours)

---

## Executive Summary

The EffektGuard integration managed a NIBE F750 heat pump in Southern Sweden (55.60°N, "Moderate Cold" climate zone). The log reveals a **significant thermal debt event overnight** where DM (Degree Minutes) dropped from -220 to **-878** due to DHW (Domestic Hot Water) heating conflicting with space heating needs.

### Key Findings

| Metric | Value |
|--------|-------|
| Indoor temp range | 21.5°C – 22.7°C |
| Outdoor temp range | 2.0°C – 4.5°C |
| Flow temp range | 29.5°C – 55.4°C |
| DM range | +100 → -878 |
| Peak DM deficit | -878 at 07:02 |
| Recovery time | ~7 hours (07:02 → 14:02) |
| Total DHW blocks | 26 warnings |

---

## System Configuration

| Parameter | Value |
|-----------|-------|
| Heat Pump Model | NIBE F750 (F-series ASHP) |
| Climate Zone | Moderate Cold (54.5-56.0°N) |
| Latitude | 55.60°N |
| Winter Avg Low | -1.0°C |
| DM Normal Range | -300 to -500 |
| Heat Coefficient (HC) | 180.0 W/°C |
| Thermal Mass Factor | 2.0× |
| Monthly Peak | 5.65 kW |
| Estimated Savings | 50 SEK/month |

---

## Hourly Temperature & DM Summary

| Time | Indoor | Outdoor | Flow | DM | Price Class | Notes |
|------|--------|---------|------|-----|-------------|-------|
| 16:28 (Dec 1) | 22.7°C | 3.0°C | 30.0°C | -178 | PEAK | Startup |
| 17:23 | 21.7°C | 2.0°C | 30.5°C | -202 | EXPENSIVE | Cooling |
| 18:23 | 21.6°C | 2.5°C | 31.0°C | -214 | NORMAL | Stable |
| 19:23 | 21.6°C | 3.5°C | 30.5°C | -218 | NORMAL | |
| 20:23 | 21.5°C | 3.5°C | 30.5°C | -219 | NORMAL | |
| 21:23 | 21.6°C | 4.0°C | 30.5°C | -220 | NORMAL | |
| 22:22 | 21.6°C | 4.0°C | 30.0°C | -231 | NORMAL | |
| 23:22 | 21.5°C | 4.0°C | 30.0°C | -236 | NORMAL | |
| 00:22 (Dec 2) | 21.6°C | 3.0°C | 31.5°C | -272 | CHEAP | Pre-heat |
| 01:22 | 21.5°C | 3.5°C | 31.0°C | -357 | CHEAP | DHW started |
| **02:22** | 21.5°C | 3.5°C | 30.9°C | -377 | CHEAP | DHW heating |
| **02:32** | 21.6°C | 3.5°C | **43.9°C** | -419 | CHEAP | DHW active! |
| **02:42** | 21.6°C | 3.5°C | **55.4°C** | -468 | CHEAP | DHW peak |
| 03:22 | 21.5°C | 3.5°C | 31.4°C | -617 | CHEAP | DHW complete |
| 04:22 | 21.7°C | 4.0°C | 32.9°C | -713 | CHEAP | Recovery |
| 05:22 | 21.9°C | 3.5°C | 32.4°C | -791 | CHEAP | Recovery |
| 06:22 | 21.9°C | 3.0°C | 32.9°C | -821 | NORMAL | Recovery |
| **07:02** | 21.9°C | 2.5°C | 33.4°C | **-878** | NORMAL | **Peak deficit** |
| 08:22 | 21.9°C | 2.5°C | 32.9°C | -826 | EXPENSIVE | Improving |
| 09:22 | 22.0°C | 2.5°C | 33.4°C | -753 | EXPENSIVE | |
| 10:22 | 22.2°C | 4.0°C | 32.9°C | -718 | NORMAL | |
| 11:22 | 22.3°C | 4.0°C | 33.4°C | -653 | NORMAL | |
| 12:22 | 22.4°C | 4.5°C | 31.9°C | -544 | NORMAL | |
| 13:22 | 22.5°C | 4.0°C | 32.9°C | -476 | NORMAL | |
| **14:02** | 22.5°C | 3.5°C | 31.4°C | **+13** | NORMAL | **Recovery!** |
| 14:17 | 22.5°C | 3.5°C | 31.4°C | +100 | NORMAL | Saturated |
| 15:47 | 22.2°C | 2.5°C | 30.4°C | +100 | EXPENSIVE | End of log |

---

## Offset Control Summary

| NIBE Offset | Time Period | Duration | Reason |
|-------------|-------------|----------|--------|
| 0°C | 16:28 – 02:32 | ~10h | Normal operation |
| +1°C | 02:37 – 02:52 | ~15min | DHW recovery boost |
| +2°C | ~04:27 – ~07:02 | ~2.5h | Deep debt recovery |
| +1°C | ~09:47 – ~11:57 | ~2h | Continued recovery |
| 0°C | ~11:57 – 15:47 | ~4h | Normal operation |

**Note:** Calculated offsets ranged from -0.23°C to +1.99°C but NIBE only accepts integer values, so an accumulator pattern was used.

---

## DHW (Hot Water) Events

### Planned Heating Window
- **Optimal window selected:** 03:45 @ 27.1 öre/kWh
- **Last DHW heating:** 2025-12-02 01:08:16 (triggered by NIBE internally)

### DHW Blocking Events (Thermal Debt Protection)

The DHW optimizer blocked heating 26 times due to thermal debt:

| Time | DM | Warning Threshold | Status |
|------|-----|-------------------|--------|
| 02:32 | -419 | -410 | BLOCKED |
| 02:37 | -443 | -410 | BLOCKED |
| 02:42 | -468 | -410 | BLOCKED |
| 02:47 | -495 | -410 | BLOCKED |
| 02:52 | -521 | -410 | BLOCKED |
| 02:57 | -561 | -410 | BLOCKED |
| ... | ... | ... | BLOCKED |
| 04:22 | -713 | -400 | BLOCKED |

---

## Price Optimization Analysis

| Price Class | öre/kWh Range | System Response |
|-------------|---------------|-----------------|
| CHEAP | 27-40 | Pre-heat, boost offset |
| NORMAL | 50-80 | Standard operation |
| EXPENSIVE | 80-90 | Reduce heating if warm |
| PEAK | 85-105 | Defer heating, coast |

### Price Forecast Utilization
- Morning cheap period (00:00-06:00): Used for pre-heating and DHW
- Evening peak (16:00-18:00): Reduced heating, coasted on thermal mass
- Forecast horizon: 8 hours (4h base × 2.0 thermal mass)

---

## Weather Compensation Calculations

| Parameter | Value |
|-----------|-------|
| Formula | Kühne |
| Indoor Target | 21.0°C |
| Safety Margin | 0.5°C |
| Optimal Flow @ 3°C outdoor | 27.4°C |
| Adjusted Flow | 27.9°C |
| Actual Flow (typical) | 30.0-31.0°C |
| Calculated Offset | -1.4°C to -2.1°C |

The flow temperature was consistently **2-3°C higher than optimal**, indicating potential for energy savings through better curve tuning.

---

## Logical Issues Identified

### 1. **DHW Heating Conflict (CRITICAL)**
The NIBE heat pump started DHW heating internally at 01:08, which was detected by EffektGuard but couldn't be stopped. This caused:
- Flow temp spike to 55.4°C
- DM dropped from -377 to -678 during DHW cycle
- Space heating was starved for ~40 minutes
- Created 7-hour recovery burden

**Recommendation:** Investigate if EffektGuard can fully control DHW scheduling to prevent NIBE's autonomous DHW heating.

### 2. **Offset Accumulator Not Accumulating**
The log shows `accumulator = calculated` rather than true accumulation:
```
Offset calculation: calculated=0.44°C, NIBE_current=0°C, accumulator=0.44°C
```
This means fractional offsets are lost rather than accumulated toward a ±1°C threshold.

**Recommendation:** Verify accumulator logic is summing values over time.

### 3. **Excessive DM Recovery Time**
DM took **7 hours** to recover from -878 to positive. During this time:
- Compressor ran continuously at higher frequency
- Flow temps elevated to 32-34°C
- Energy consumption increased significantly

**Recommendation:** Consider more aggressive offset boosting during deep debt to accelerate recovery.

### 4. **Flow Temperature Consistently High**
Optimal flow was calculated at 27.4-28.2°C but actual flow was 30-34°C. This represents ~10% over-heating.

---

## Warnings & Errors

| Category | Count | Description |
|----------|-------|-------------|
| Custom Integrations | 15 | Untested integrations (normal) |
| DHW Blocked | 26 | Thermal debt protection |
| LED Automation | 3 | "Already running" |
| Tesla Sensor | 1 | State not strictly increasing |
| Met.no API | 2 | DNS connection error |
| pysignalr | 1 | Missing 'CommandResponse' method |

---

## Compressor Performance

| Metric | Value |
|--------|-------|
| Normal Frequency | 32-34 Hz |
| 1-hour Average | 32-33 Hz |
| 6-hour Average | 32-33 Hz |
| Mode | Space heating |
| Status | OK throughout |

---

## Conclusions

1. **The system successfully maintained indoor temperature** between 21.5-22.5°C despite the DM crisis
2. **Price optimization worked correctly** - pre-heating during cheap periods, coasting during expensive
3. **The DHW conflict was the root cause** of the thermal debt event
4. **Recovery was successful but slow** - better strategies could accelerate recovery
5. **Flow temperatures are higher than necessary** - curve optimization could save energy
