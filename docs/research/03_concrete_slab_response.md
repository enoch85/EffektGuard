# A concrete slab: why 24 hours, and why +2.0 °C

Two constants come from this analysis:

```python
UFH_CONCRETE_PREDICTION_HORIZON = 24.0   # hours
WEATHER_PREHEAT_OFFSET          = 2.0    # °C
```

Both used to be wrong, and both were wrong in a way that made the pre-heat useless without making
it look useless.

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
| slab reaches **63 %** of its response | **~14 h** |
| fast time constant (slab ↔ room) | 1.25 h |
| slow time constant (fabric → outdoors) | 70 h |

## ⚠️ Six hours is the LAG. Twenty-four is the HORIZON.

These are different questions and the codebase conflated them.

"It takes about six hours to heat a concrete slab" is roughly right *as a lag* — the room begins
moving within 2.4–4.6 h. But the **slab is only 63 % charged at fourteen hours.** If you are
deciding *today* whether to start storing heat for a cold snap, six hours of look-ahead tells you
almost nothing. You have to plan over the time it takes the store to actually fill, and that is a
day.

## Why the pre-heat trigger never fired

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

This is the same inversion as the degree-minute ladder: the mechanism is backwards relative to when
it is wanted.

## Why +0.83 °C could not charge the battery

The pre-heat's job is to fill the building's thermal store — `THERMAL_BATTERY_BAND`, ±1.0 °C —
*before* the cold arrives. The sizing rule is not a matter of taste:

```
energy to fill the band = C_fabric × THERMAL_BATTERY_BAND
surplus the offset buys = offset × DEFAULT_CURVE_SENSITIVITY × dQ/dFlow
time to fill            = energy / surplus        (must be ≤ the forecast horizon)
```

Against the simulator's validated plant models, the old `+0.83 °C`:

| house | time to fill the ±1 °C band | horizon | verdict |
|---|---|---|---|
| radiator (τ 30 h, C 4.5 kWh/K) | **28.4 h** | 12 h | never |
| concrete slab (τ 80 h, C 14.4 kWh/K) | **34.6 h** | 24 h | never |

The cold always arrived first. The constant's own history records the struggle without ever
diagnosing it — *"tuned Oct 20, was 0.5 → 0.6 → 0.7 → 0.77"*. **It was being nudged in hundredths
when it needed to be tripled.**

At **+2.0 °C**: 9.6 h (radiator) and 14.8 h (slab). Both inside their horizons.

It cannot overheat the house: the comfort layer takes charge at the edge of the storage band, so a
strong pre-heat is bounded by construction — it charges the fabric quickly and hands over. And
+2.0 sits inside `WEATHER_COMP_MAX_OFFSET` (3.0), the bound on every weather-driven correction.

## Guarded by

- `tests/unit/optimization/test_preheat_sees_the_cold_coming.py` — the horizon must follow thermal
  mass, and a two-day slide must be visible to a slab.
- `tests/unit/optimization/test_preheat_can_actually_charge_the_house.py` — the fabric must reach
  the edge of the storage band **within** the horizon the house is given, on both plant models.

## Owner's words

> *"it takes around 6 hours to heat concrete slab … we need to pre-heat super early if we know a
> cold snap is coming, I mean like DAYS ahead."*

Then, on being shown the 6-hour figure:

> *"well, 6 hours was low. 24 hours is more correct actually"*

Both correct. The transient model above is what settles it.
