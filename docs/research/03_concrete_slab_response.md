# A concrete slab: why the horizon is 48 hours, and how big the pre-heat is

One constant comes from this analysis:

```python
UFH_CONCRETE_PREDICTION_HORIZON = 48.0   # hours
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

The trigger was *"a drop of at least 4 °C within the forecast horizon"* (since 2026-10-12 it compares
24-hour means, so a night's low cannot fire it). The horizon was a **fixed 12 hours, for every
house, whatever it was built of**:

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

## The pre-heat offset (decided 2026-10-11)

```python
WEATHER_GENTLE_OFFSET = 2.0   # °C
```

The old +0.83 was averaged with the other layers' votes to below the one whole step the offset
register needs, so in practice it never banked. On the owner's own winter 2025/26 data, the
average offset in the 24 h before every cold snap was negative. +2.0 fills the thermal-battery
band inside the horizon (9.6 h radiator, 14.8 h slab, against 28.4 and 34.6 h at +0.83 on the
simulator's plants).

**Why 48 hours.** An optimiser on the plant fitted to the owner's F750 slab gives the immersion
heat (kWh) each cold snap needed, by how many hours ahead pre-heating started. It used perfect
forecasts and the compressor only:

| snap | react | 24 h | 48 h | 72 h |
|---|---|---|---|---|
| 24 Dec | 8.8 | 8.8 | 3.9 | 0 |
| 29 Dec | 4.7 | 2.7 | 0.4 | 0 |
| 19 Jan | 8.3 | 6.8 | 3.6 | 1.3 |

6-24 h ahead does almost nothing. Most of the gain comes by 48 h, and 72 h adds little.
Heat is only banked cheaply while the weather is still mild enough for the compressor to have
headroom. 48 h is also as far as an hourly weather forecast reaches.

**Still open:** the comfort layer cuts any bank once the room is about 0.6 °C over target. That
is listed in the owner's TODO, not changed here.

## Owner's words

> *"it takes around 6 hours to heat concrete slab … we need to pre-heat super early if we know a
> cold snap is coming, I mean like DAYS ahead."*

Then, on being shown the 6-hour figure:

> *"well, 6 hours was low. 24 hours is more correct actually"*

Both correct — and the corrected transient above says the second is a floor, not a ceiling.
