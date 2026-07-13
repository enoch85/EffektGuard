# Flow temperature: the EN 442 emitter law

This is the derivation behind `utils/emitter.py`. The fitted expression it replaced ("Kühne") **was
removed** — it was being fed the wrong quantity, and the end of this page shows exactly which.

## The model

```
balance = T_room − internal_gains_W / heat_loss_W_per_K   the house heats itself to here
φ       = (balance − T_out) / (balance − T_out_design)     dimensionless relative load
ΔT      = ΔT_design · φ^(1/n)                              invert the emitter law
T_flow  = T_room + ΔT_design · φ^(1/n) + spread_design / 2 ← spread CONSTANT, not scaled
```

Every step from a published standard:

1. **EN 12831** — building heat loss is linear in the air-temperature difference, and it reaches
   zero at the **balance point**, not at room temperature: bodies, appliances and the sun cover the
   losses until several degrees below the setpoint. Hence
   `φ = Φ/Φ_design = (balance − T_out) / (balance − T_out_design)`.
2. **EN 442-1:2014 §3.31** ("characteristic equation") — an emitter's output follows
   `Φ/Φ_N = (ΔT/ΔT_N)^n`.
3. Set emitter output equal to the building load and **invert (2)**:
   `ΔT = ΔT_design · φ^(1/n)`. **The 1/n exponent enters exactly here**, as the inverse of the
   emitter exponent — it is not a fitted constant.
4. **The spread is CONSTANT.** `Φ = ṁ·c·(T_V − T_R)`, so a *fixed-speed* circulator gives constant
   mass flow and a spread proportional to load — that is a wet boiler. **A heat pump modulates its
   circulator** to hold the commissioned spread and varies the flow *rate* instead. Scaling the
   spread models the wrong machine, and the error pivots invisibly on the design point.
5. `T_V = T_room + ΔT_mean + spread/2`.

⚠️ **This page used to print `spread = spread_design · φ` right here**, in the headline equation,
after the code below it had already been fixed. A reader implementing from the old version rebuilt
the exact bug. Both halves of the model — the constant spread, and the balance point in `φ` — are
easy to get wrong in *cancelling* ways; see the two warnings further down.

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

### ⚠️ NIBE's published curve does not validate this model, and cannot

This page used to claim it did. That claim was wrong three times over, and the corrections matter
more than the original argument, so they are kept here rather than quietly deleted.

**1. Curve 9 is a straight line.** Least-squares through the six digitised points leaves a residual
of **0.19 °C**. Its successive slopes are −0.800, −0.740, −0.780, −0.820, −0.880 °C/°C — they wobble
*non-monotonically*, and a real emitter law steepens monotonically toward cold. The wobble is
digitisation noise and it is **larger than the curvature it was being used to detect**. Collinear
points confirm every model fitted to them.

The old table on this page claimed a straight line was out by **2.37 °C**. It is out by 0.19 °C. The
honest comparison at 0 °C outdoor:

| model | flow at 0 °C | error vs NIBE's 41.0 |
|---|---|---|
| least-squares straight line | 40.76 °C | **−0.24 °C** |
| EN 442 + derived gains | 41.64 °C | +0.64 °C |
| EN 442, no gains | 42.72 °C | +1.72 °C |

NIBE's controller **interpolates its curves linearly**. Ours follows EN 442. The gap between them is
not our error — **it is the trim**, which is the entire reason this layer exists.

**2. The gains term cannot be fitted to a heating curve at all.** A constant spread lifts the curve
by `(spread/2)·(1 − φ^(1/n))`; a balance point drops it by a term of the *same shape and opposite
sign*. Both vanish at the design point and grow in mild weather. They are **the same basis
function**, so they are not separately identifiable, and whatever spread you assume the fit hands
you a "gains" figure that absorbs it.

Proof, and it is run as a test: fit this law to **Kühne's Vaillant curve, which is a pure power law
with provably zero gains**, and a balance point appears anyway, tracking the spread you assumed —
0.3 K at spread 0, **2.6 K at spread 5**, 4.9 K at spread 10.

**3. So `DEFAULT_BALANCE_POINT_OFFSET = 4.0` was fitted to noise, through a degenerate basis.** It is
gone. Gains are **watts**, and the balance point is *derived*:

```
balance_point = indoor_setpoint − INTERNAL_GAINS_W / heat_loss_coefficient
```

600 W over 180 W/K → 3.33 K → a balance point of 17.7 °C for a 21 °C room. A fixed offset in
*degrees* would have made the free heat from your fridge scale with how leaky your house is, which
is backwards. The two measured sources both express it in watts: heatpumpmonitor.org's **583 W
median across 383 monitored systems**, and OpenEnergyMonitor's SCOP tool (15.5 °C base against a
19.3 °C room).

### The pair of errors this page used to hide

An earlier version printed **40.80 °C, error 0.20 °C** — a *better* fit than the honest model gets.
It was not better. It was **two bugs cancelling**:

* **`design_spread=10.0`**, captioned "EN 442 reference: 75/65". That 10 K is the **rating** spread
  that *defines* a radiator's ΔT50 output, not the spread a heat pump's circulator holds (~5 K). And
  the code then **scaled** it by load, modelling a fixed-speed pump.
* **No balance point**, so the house was assumed to need heat at 20 °C outdoors.

Same place, opposite signs. Together they matched NIBE to a fifth of a degree, and fixing *either
one alone made the fit worse* — which is exactly what keeps a pair of errors like this alive.

And then the fix repeated the mistake in a subtler form: the replacement pair (constant spread +
fitted balance point) is *also* a cancelling pair, which is how a curve fit could report a
triumphant RMS for a constant that was measuring nothing. **Agreement with a curve is not evidence
when your basis functions are degenerate.** That is the lesson worth keeping from this page.

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
