# Effect tariff peak protection

The effect tariff bills the mean power of the highest 15-minute periods of the month. EffektGuard
records those periods and lowers the offset when the current period is about to set a new one.
Code: `EffectManager` in `optimization/effect_layer.py`, the power reading in
`coordinator._update_peak_tracking`, the period mean in `optimization/billing_period.py`.

## What is recorded

- Power per cycle, first source that answers: the external power meter, else the pump's phase
  currents (BE1-BE3), else an estimate from the compressor frequency. Only the first two are
  measurements; an estimate is shown but never recorded.
- `BillingPeriodAccumulator` builds the time-weighted mean of each 15-minute period
  (`BILLING_PERIOD_MINUTES`). A completed period from a measured source goes to
  `record_period_measurement`.
- Night periods (22:00-06:00) count half (`NIGHT_TARIFF_WEIGHT`); day is quarters 24-87.
- At most one period per day, the three highest of the month. Below 0.5 kW
  (`PEAK_RECORDING_MINIMUM`) is ignored, above 100 kW (`PEAK_RECORDING_MAXIMUM`) refused as a sensor
  fault. A month change drops last month's peaks.
- Only a meter peak counts as a billing figure (`is_billable`); a phase-current peak still sets the
  control threshold.

`sensor.effektguard_peak_today` and `sensor.effektguard_peak_this_month` show the day's high-water
mark and the month's highest recorded period.

## The vote

`EffectManager.evaluate_layer` gets the month's peak (the highest of the three), this cycle's power
projected to the period mean, and the indoor trend. The severity comes from the margin between the
tariff-weighted power and the lowest of the three peaks.

| Condition | Offset | Weight |
|---|---|---|
| At or within 0.5 kW of the lowest peak (`EFFECT_PEAK_MARGIN_CRITICAL`) | -3.0 (`EFFECT_OFFSET_CRITICAL`) | 1.0 |
| Power plus the trend term within 1.0 kW of the month's peak, trend rising (`EFFECT_MARGIN_PREDICTIVE`) | -1.5 (`EFFECT_OFFSET_PREDICTIVE`) | 0.85 |
| Within 1.0 kW of the lowest peak (`EFFECT_PEAK_MARGIN_WARNING`), demand rising | -1.0 | 0.75 |
| Within 1.0 kW, demand stable or falling | -0.5 | 0.65 |
| Otherwise, no peaks yet, or peak protection off | 0 | 0 |

The trend term: indoor falling faster than 0.4 °C/h adds 1.5 kW, faster than 0.2 °C/h adds 0.5 kW,
rising faster than 0.3 °C/h subtracts 0.5 kW.

At weight 1.0 the layer is a critical cost layer: it wins the aggregation unless the safety floor
or the EMERGENCY tier is active, and during T1-T3 recovery it only moderates the recovery down to
the tier's peak-aware offset ([08](08_layer_priority_system.md)).

Without a valid power reading this cycle the coordinator passes 0 kW rather than a guess.
`effektguard.reset_peak_tracking` clears the month's peaks.
