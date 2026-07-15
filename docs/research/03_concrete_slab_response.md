# A concrete slab: why the horizon is 24 hours — and why the pre-heat is still an open question

One constant comes from this analysis:

```python
UFH_CONCRETE_PREDICTION_HORIZON = 24.0   # hours
```

A second change this analysis argues for — a stronger pre-heat offset — has **not** been made,
and the last section says exactly where that stands.

## The thermal model

Two-node transient model of the owner's floor — 100 mm ground slab plus 60 mm screed:

| quantity | value |
|---|---|
| slab heat capacity `C_slab` | 7.04 kWh/K |
| room/air heat capacity `C_room` | 3.0 kWh/K |
| slab → room coupling `K` | 1614 W/K |
| building heat-loss coefficient `UA` | 150 W/K |
| thermal diffusivity `α` | 8.05 × 10⁻⁷ m²/s |

Which gives:

| | |
|---|---|
| conduction lag, pipe → floor surface | **0.9 h** |
| conduction through the full slab | **3.5 h** |
| room moves **+1.0 °C** | **2.4 – 4.6 h** |
| fast time constant (slab ↔ room) | **1.25 h** |
| slow time constant (fabric → outdoors) | **70 h** |
| slab charge under a constant heat input | **~9 % at 6 h · ~19 % at 14 h · ~29 % at 24 h** |

⚠️ An earlier version of this table claimed the slab "reaches 63 % of its response in ~14 h".
**That is not what this model computes.** 63 % is one time constant of a first-order system, and
the coupled system's slow constant is ~70 h, not 14 — the eigenvalues in this very table said so
while the row above them said otherwise. Integrating the documented matrix gives ~19 % at 14 h.
The corrected numbers make the conclusion *stronger*, not weaker.

## ⚠️ Six hours is the LAG. Twenty-four is the MINIMUM horizon.

These are different questions and the codebase conflated them.

"It takes about six hours to heat a concrete slab" is roughly right *as a lag* — the room begins
moving within 2.4–4.6 h. But the slab is only **about a fifth charged at fourteen hours, and
under a third charged at twenty-four**; the store's own filling time constant is measured in
days. If you are deciding *today* whether to start storing heat for a cold snap, six hours of
look-ahead tells you almost nothing — and even a day only begins the job. Which is what the
owner said before this model existed: *"we need to pre-heat super early … like DAYS ahead."*

## Why the pre-heat trigger never fired (fixed — the horizon follows thermal mass)

The trigger is *"a drop of at least 4 °C within the forecast horizon"*. The horizon was a **fixed
12 hours, for every house, whatever it was built of**:

| cold snap | drop within 12 h | fires? | drop within 24 h | fires? |
|---|---:|---|---:|---|
| 15 °C over 6 h — sudden plunge | −15.0 | **yes** | −15.0 | yes |
| 15 °C over 24 h | −7.5 | yes | −7.5 | yes |
| **15 °C over 48 h — a two-day slide** | **−3.8** | **NO** | −7.5 | yes |
| **20 °C over 72 h — a three-day slide** | **−3.3** | **NO** | −6.7 | yes |

**A slab does not get into thermal debt from a plunge.** The pump's own heating curve is reactive,
but it is fast, and it catches that. A slab gets into debt from a **slow, deep slide** — and within
any twelve hours of a two-day slide the temperature falls less than the four degrees needed to
trigger. So the pre-heat **never fired on the case that needed it**, while firing reliably on the
case that needed it least.

Guarded by `tests/unit/optimization/test_preheat_sees_the_cold_coming.py` — the horizon must
follow thermal mass, and a two-day slide must be visible to a slab.

## 🛑 OPEN — the pre-heat offset is too small to charge the store, and changing it is the owner's call

Production ships:

```python
WEATHER_GENTLE_OFFSET = 0.83   # °C - the pre-heat offset actually in const.py
```

The pre-heat's job is to fill the building's thermal store — `THERMAL_BATTERY_BAND`, ±1.0 °C —
*before* the cold arrives. The sizing rule is arithmetic, not taste:

```
energy to fill the band = C_fabric × THERMAL_BATTERY_BAND
surplus the offset buys = offset × DEFAULT_CURVE_SENSITIVITY × dQ/dFlow
time to fill            = energy / surplus        (must be ≤ the forecast horizon)
```

Against the simulator's plant models, `+0.83 °C` fills the band in **28.4 h** (radiator house,
12 h horizon) and **34.6 h** (slab, 24 h horizon): the cold always arrives first. The constant's
own history records the struggle without diagnosing it — *"tuned Oct 20, was 0.5 → 0.6 → 0.7 →
0.77"* — nudged in hundredths when the arithmetic wanted it tripled. At **+2.0 °C** the times are
9.6 h and 14.8 h, inside both horizons, bounded by `WEATHER_COMP_MAX_OFFSET` (3.0) and handed
over to the comfort layer at the edge of the band.

**This change has not been made.** It is a control-tuning decision on real heat pumps — a
stronger pre-heat buys comfort insurance with money — and an earlier version of this document
declared it as `WEATHER_PREHEAT_OFFSET = 2.0` as if it had shipped, citing a guard test that
does not exist. It had not shipped, there is no such constant, and the validation suite now
fails any fenced declaration of a constant the code does not have.

## Owner's words

> *"it takes around 6 hours to heat concrete slab … we need to pre-heat super early if we know a
> cold snap is coming, I mean like DAYS ahead."*

Then, on being shown the 6-hour figure:

> *"well, 6 hours was low. 24 hours is more correct actually"*

Both correct — and the corrected transient above says the second is a floor, not a ceiling.
