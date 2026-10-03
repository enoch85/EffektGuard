# Exhaust-air recovery: the same joules, counted twice

The airflow feature raises exhaust ventilation to pull more heat out of the outgoing air. The
question is whether that is worth the extra fresh air the building must then reheat.

The original answer had three terms:

```
Net Benefit = (Extra heat extracted) + (COP improvement) − (Ventilation penalty)
              +0.41 kW                 +1.32 kW            −0.70 kW      = +1.03 kW
```

**The middle term is the first term, written again.**

## The identity

In steady state, the first law across the heat pump gives

```
Q_cond = P_el + Q_evap
```

Differentiate at constant electrical input:

```
d(Q_cond) = d(Q_evap) = P_el · d(COP)
```

`P_el · d(COP)` **is** `d(Q_evap)`. It is not a separate benefit that arrives alongside the extra
extraction; it is a second name for it. Adding both counts the same heat twice.

Notice the direction of the error: it is not conservative. It inflates the benefit, so the feature
appears to pay when it does not.

## The manufacturer's own controlled experiment

This is not a theoretical objection. **The NIBE S735 installer manual publishes four operating points
at identical conditions — A20(12)W35, minimum compressor frequency — where the only variable is the
exhaust airflow.** A COP-versus-airflow experiment, run by the manufacturer, under EN 14511.

Taking the 90 → 252 m³/h step:

| quantity | value |
|---|---|
| ΔP_H (measured heat output) | **+0.410 kW** |
| P_el × (COP₂ − COP₁) — the code's `delta_cop_benefit` | **+0.387 kW** |
| ΔQ_evap = ΔP_H − ΔP_el — the code's `delta_extraction` | **+0.404 kW** |

**The "COP benefit" and the "extra extraction" are the same number**, to within the manual's own
rounding. The identity is confirmed by NIBE's published data, not merely argued from theory.

## What is left when the double-count is removed

```
net_gain = (extra heat extracted at the evaporator) − (extra fresh air the building must reheat)
```

`calculate_net_thermal_gain()` computes exactly this, and it is negative:

| outdoor | net gain |
|---|---|
| +10 °C | **+0.03 kW** |
| +5 °C | −0.14 kW |
| 0 °C | **−0.31 kW** |
| −5 °C | −0.48 kW |
| −10 °C | **−0.65 kW** |
| −15 °C | −0.82 kW |

It gets worse as it gets colder, because the ventilation penalty scales with (T_in − T_out) while
the extraction does not.

## ⚠️ The honest conclusion

**Enhanced airflow is a cold-weather recovery measure, and in cold weather it loses heat.** The only
outdoor temperature at which it shows a gain is **+10 °C**, where it gains 0.03 kW — and where
nobody needs it.

`airflow_optimizer` declines to enhance when the net gain is ≤ 0, so the feature is currently inert
in the conditions it exists for. That is the correct behaviour for the physics as written.

Whether the feature survives at all is **audit finding F-032, open with the owner**: the choice is
between deleting it and disabling it, and that is not a call to make on someone else's heat pump.
What is *not* an option is restoring the COP term to make the numbers look better.

## Sources

- **NIBE S735 installer manual**, EN 14511 performance tables — four points at A20(12)W35, minimum
  compressor frequency, exhaust airflow as the sole variable.
- First law of thermodynamics.

## Not sourced

The **"~20 % COP improvement from a warmer evaporator"** figure that produced the +1.32 kW. It is
described in the old docs as "empirical" with no citation, and the S735 data above shows that
whatever its magnitude, it is not additive.
