# Spot price optimization

`PriceAnalyzer` in `optimization/price_layer.py` turns GE-Spot's 15-minute prices (96 a day, 92 or
100 on DST days) into one vote per cycle.

## Classification

Each day is ranked against its own percentiles (`PRICE_PERCENTILE_*`); tomorrow is classified on
its own once published.

| Class | Price | Base offset |
|---|---|---|
| VERY_CHEAP | at or below P10 | +4.0 |
| CHEAP | at or below P25, and below P90 | +1.5 |
| NORMAL | between | 0 |
| EXPENSIVE | above P75 | -1.0, x1.5 by day (quarters 24-87, 06:00-22:00) |
| PEAK | above P90 | -10.0 (`MIN_OFFSET`) |

A flat day, where P90 minus P10 is under 5 % of the day's price scale
(`PRICE_FLAT_DAY_SPREAD_FRACTION`), is all NORMAL. Constants: `PRICE_OFFSET_*`,
`PRICE_DAYTIME_MULTIPLIER`.

## From base offset to vote

1. **PEAK cluster**: an EXPENSIVE quarter between PEAKs takes the PEAK offset.
2. **Pre-PEAK**: the quarter before a PEAK is capped at -6.0 (`PRICE_PRE_PEAK_OFFSET`) so the pump
   slows down in time.
3. **Volatile**: a run of the current class shorter than 3 quarters (45 min,
   `VOLATILE_MIN_DURATION_QUARTERS`), or a cheap run ending within 3 quarters, drops a positive
   offset to 0 (`should_skip_volatile_boost`).
4. **Tolerance**: the 0.5-3.0 slider maps to a factor of 0.2-1.0 (`PRICE_TOLERANCE_FACTOR_*`),
   times the mode multiplier (comfort 0.7, balanced 1.0, savings 1.3).
5. **Forecast**, not on PEAK: the look-ahead is 5 h times the thermal mass
   (`PRICE_FORECAST_BASE_HORIZON`, 2.5-10 h, running into tomorrow when published). A run at least
   3 quarters long and 3 quarters away under 50 % of the current price adds -1.5
   (`PRICE_FORECAST_REDUCTION_OFFSET`); one over 150 % adds +2.0 (`PRICE_FORECAST_PREHEAT_OFFSET`).
   With DM below -400 a negative adjustment becomes +0.3 (`PRICE_FORECAST_DM_DEBT_OFFSET`).
6. **Stored heat**: pre-heating into a room already above target by more than the mode allows
   (0.3 / 0.5 / 1.0 °C) is trimmed by the excess.
7. **No deeper than -2.0** (`PRICE_OFFSET_DEEPEST_CUT`), whatever the class, the slider and the
   mode. A deeper cut stops the compressor instead of slowing it: the slab drains, and the heat
   comes back from the immersion heater or not at all.

## Weight

0.8 (`LAYER_WEIGHT_PRICE`); 1.0 at PEAK or in a PEAK cluster, which makes it a critical cost layer
([08](08_layer_priority_system.md)); x0.3 when volatile (`VOLATILE_WEIGHT_REDUCTION`), unless the
look-ahead adds heat for the hours ahead. No price data, or the Price Optimization switch off: no
vote. Missing prices also raise a repair issue.

## Example

EXPENSIVE at 08:00, balanced mode, tolerance 3.0, no forecast adjustment: -1.0 x 1.5 x 1.0 = -1.5
at weight 0.8. At the default tolerance 0.5 the factor is 0.2, so -0.3.

`sensor.effektguard_price_period_classification` shows the current class and
`sensor.effektguard_quarter_of_day` the quarter; `effektguard.calculate_optimal_schedule` returns
the next 24 hours' classes and base offsets.
