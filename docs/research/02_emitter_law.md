# Flow temperature: the EN 442 emitter law

This is the derivation behind `utils/emitter.py`. The fitted expression it replaced ("Kühne") **was
removed** — it was being fed the wrong quantity, and the end of this page shows exactly which.

## The model

```
φ      = (T_room − T_out) / (T_room − T_out_design)      dimensionless relative load
ΔT     = ΔT_design · φ^(1/n)                             invert the emitter law
spread = spread_design · φ                               constant mass flow
T_flow = T_room + ΔT_design · φ^(1/n) + spread_design · φ / 2
```

Every step from a published standard:

1. **EN 12831** — building heat loss is linear in the air-temperature difference. Hence
   `φ = Φ/Φ_design = (T_room − T_out) / (T_room − T_out_design)`.
2. **EN 442-1:2014 §3.31** ("characteristic equation") — an emitter's output follows
   `Φ/Φ_N = (ΔT/ΔT_N)^n`.
3. Set emitter output equal to the building load and **invert (2)**:
   `ΔT = ΔT_design · φ^(1/n)`. **The 1/n exponent enters exactly here**, as the inverse of the
   emitter exponent — it is not a fitted constant.
4. Constant mass flow: `Φ = ṁ·c·(T_V − T_R)`, so `spread = spread_design · φ`, linearly.
5. `T_V = T_room + ΔT_mean + spread/2`.

## The constants, checked against the standard's normative text

- **EN 442-1 §3.23**: *"excess temperature of **50 K** … inlet 75 °C, outlet 65 °C, reference air
  20 °C."* ⇒ `RADIATOR_RATED_DT = 50.0` ✅. Note this is the **arithmetic** mean, not the log-mean.
  ⚠️ The log-mean of the same reference point is 49.83 K. **Never mix a log-mean ΔT with the 50 K
  reference.**
- `RADIATOR_POWER_COEFFICIENT = 1.3` ✅ — panel radiators measure 1.26–1.33; sectional 1.30. (A real
  EN 442 conformity sheet, Global MIX 600, gives 1.32266.)
- `UFH_POWER_COEFFICIENT = 1.1` ✅ — **underfloor heating is not 1.3.** EN 1264 gives the UFH base
  equation as `q = 8.92 · (θ_floor − θ_room)^1.1` W/m², which reproduces Uponor's official design
  table on all 13 rows.

| emitter | n | 1/n |
|---|---|---|
| Underfloor (golvvärme) | **1.0 – 1.1** | 0.91 – 1.00 |
| Panel radiators | **1.26 – 1.33** (use 1.3) | 0.75 – 0.79 |
| Sectional / column radiators | 1.30 | 0.77 |
| Convectors | 1.25 – 1.45 | 0.69 – 0.80 |

EN 1264 also caps the occupied-floor **surface** temperature at 29 °C, which bounds what a UFH
system can deliver regardless of what the flow temperature is.

## Validation against NIBE's own curve

The model is checked against NIBE's published heating curves, digitised from the vector artwork in
the **FIGHTER 1225 Monterings- och skötselanvisning (MOS SE 0735-3, p.23)** — Bézier control points
extracted, axes calibrated, residuals < 0.11 °C. Validated three independent ways:

- re-digitised from the FIGHTER 1115 manual — agrees to **0.01 °C**;
- reproduces the F1155 manual's own screenshot (curve 9, offset 0, outdoor 0 °C → the display reads
  **41**; digitised value **41.0**, exact);
- reproduces NIBE's three official worked examples in VVM 225 IHB.

**The test that matters:** NIBE curve 9 at 0 °C outdoor reads **41.0 °C**. Reproduce it yourself —

```python
from custom_components.effektguard.utils.emitter import en442_flow_temp

en442_flow_temp(
    indoor_setpoint=21.0,
    outdoor_temp=0.0,
    design_outdoor_temp=-15.0, # DUT
    design_flow_temp=52.6,     # curve 9 at -15 °C, from the digitised artwork
    design_spread=5.0,         # the spread the CIRCULATOR holds, not EN 442's 75/65 rating
    emitter_exponent=1.3,      # panel radiators
    balance_point_temp=17.0,   # 21 - DEFAULT_BALANCE_POINT_OFFSET: bodies, appliances, sun
)   # -> 41.39
```

| model | flow temp at 0 °C | error vs NIBE |
|---|---|---|
| NIBE's published curve 9 | **41.0 °C** | — |
| **EN 442 + balance point** | **41.39 °C** | **0.39 °C** ✅ |
| EN 442, no gains (balance = 21 °C) | 42.72 °C | 1.72 °C ✗ |
| a straight line between the endpoints | 38.63 °C | 2.37 °C ✗ |

The emitter law tracks NIBE's own curve to under half a degree; a linear interpolation is out by
more than two. What it reproduces is the **curvature**, and that curvature is the `φ^(1/n)` term.
This is why the exponent matters and cannot be folded into a fitted slope.

### Two corrections this example used to hide

An earlier version of this page printed **40.80 °C, error 0.20 °C** — a better fit than the honest
model achieves. It was not better. It was **two bugs cancelling**, and the cancellation is why
neither was ever found:

* **`design_spread=10.0`**, captioned "EN 442 reference: 75/65". That 10 K is the **rating** spread
  that *defines* a radiator's ΔT50 output. It is not the spread a heat pump's circulator maintains,
  which is ~5 K. And the code then **scaled** it by load, modelling a fixed-speed pump.
* **No balance point.** Heat demand was taken as linear in `(indoor − outdoor)`, so the house was
  assumed to need heat at 20 °C outdoors.

The first made the curve too **cool** in mild weather; the second made it too **hot** in mild
weather. Same place, opposite signs. Together they matched NIBE to a fifth of a degree; fixing
either one alone made the fit *worse*, which is exactly the trap that keeps a pair of errors like
this alive. Both are fixed now, and the residual 0.39 °C is real.

(The exact figure moves a little with the assumed design spread and room setpoint — the inputs are
spelled out above precisely so that it is checkable rather than quotable. The ranking does not move
at all.)

Reference points, offset 0, outdoor −15 °C:

| system | NIBE curve | flow temp |
|---|---|---|
| Radiators (F1155 factory default) | curve **9** | **52.6 °C** |
| UFH concrete (F750 factory default) | curve **5** | **38.2 °C** |

NIBE's own definitions: a low-temperature radiator system *"needs a flow temperature of **55 °C on
the coldest day**"*; underfloor, *"about **35–40 °C**"*.

## What was wrong before

The previous implementation used a fitted expression whose exponent was **0.78**.

`1/n = 1/1.3 = 0.769 ≈ 0.77`. **The 0.78 was the inverse emitter exponent all along** — an
independent empirical fit of a real Vaillant curve landed on the same number, "consistent with a
radiator law of ΔT^1.3 or thereabouts". The *structure* was right.

What was wrong was the **input**: it was being fed a **heat-loss coefficient** where the derivation
requires a **dimensionless relative load φ**, and the emitter sizing was hidden inside another
fitted constant. A dimensionally inconsistent input to a structurally correct law produces numbers
that look plausible and are not.

The inputs the EN 442 model needs are all quantities an installer actually knows: the design outdoor
temperature (DUT), the design ΔT and spread (from the pump's curve, or NIBE's recommended starting
values), and the emitter exponent n. None of them is a reverse-engineered dimensionless label.

## Sources

- **EN 442-1:2014**, §3.23 (reference conditions), §3.31 (characteristic equation)
- **EN 12831** (heat load — linearity in ΔT_air)
- **EN 1264** (underfloor heating: base equation, surface-temperature limits)
- NIBE **FIGHTER 1225** MOS SE 0735-3 p.23; **FIGHTER 1115**; **F1155** IHB; **VVM 225** IHB
