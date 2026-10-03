# Degree minutes, and the number that actually governs an F750

## What a degree minute is

NIBE integrates the gap between the measured supply temperature and the calculated setpoint:

```
DM = ∫ (BT25 − S1) dt          BT25 = measured supply, S1 = calculated supply setpoint
```

Units are °C·minutes. DM goes **negative** when the pump is not keeping up, and NIBE uses it as the
single scalar that decides when to start the compressor and when to call for auxiliary heat.

## NIBE menu 4.9.3 — the primary source

**NIBE F750 Installer Manual, IHB GB 1301-1 (part 231236), menu 4.9.3:**

| setting | default | range |
|---|---|---|
| `start compressor` | **−60** | −1000 … −30 |
| `start addition` | **−700** | −2000 … −30 |

`DM_THRESHOLD_START = -60` in `const.py` is this number.

## ⚠️ The consequence the codebase was built without

**`start addition` is −700. The immersion heater (elpatron) engages there, by design, and works DM
back UP.** On a healthy pump DM therefore *asymptotes* near −700; it does not run away toward
−1500.

`DM_THRESHOLD_AUX_LIMIT = -1500` is treated throughout the code as the emergency threshold — and on
the owner's F750, **the pump's own auxiliary heat has already been running for 800 degree-minutes
by the time DM gets there.** EffektGuard's climate-zone warning threshold for Stockholm is **−740**,
which is *deeper* than −700: the elpatron fires **first**, every time.

This is not a small calibration point. It means:

- The EMERGENCY tier (DM ≤ −1500) has, as far as can be determined, **never fired in the product's
  life.**
- Reasoning about "what happens as DM approaches −1500" describes a régime a healthy F750 never
  enters.
- The elpatron is **not the enemy.** NIBE placed it at −700 deliberately, to spare the compressor
  from grinding at full frequency for hours. Fighting it with a bigger curve offset trades cheap
  kWh for expensive compressor life.

Audit findings F-112 and F-129 both turn on this. F-112 (rescaling the recovery ladder to the real
aux-start) is **open with the owner** — the −700 figure is sourced, but the ladder redesign is not,
and 68 existing safety tests encode the −1500 world.

## ⚠️ Why the DM safety net is structurally blind to EffektGuard's own under-heating

Read the identity again:

```
DM = ∫ (BT25 − S1) dt
```

EffektGuard's only actuator is the **curve offset**, and lowering the offset lowers **S1**. So when
EffektGuard cuts heat to save money:

- S1 falls,
- `BT25 − S1` becomes *less negative*,
- **DM improves** — while the house gets colder.

**The degree-minute safety net cannot see under-heating that EffektGuard itself causes.** DM is a
measure of whether the pump is meeting *its own setpoint*, not whether the house is warm. Lower the
setpoint and the pump is meeting it comfortably, by construction. (Audit F-120.)

The mirror case (F-124): raising the offset on a compressor that is already saturated raises S1
while BT25 cannot follow, so **DM gets worse** — the boost that was meant to recover the debt
deepens it, and buys nothing but compressor hours. This is why the compressor-wear guard holds the
offset at HIGH frequency risk rather than raising it (F-129).

**The comfort floor, not DM, is what protects the occupant.**

## Compressor wear

`CompressorHealthMonitor.assess_risk()`:

| risk | condition |
|---|---|
| HIGH | above 100 Hz for more than 15 minutes — at maximum, nothing left to give |
| ELEVATED | above 80 Hz for more than 2 hours |

At HIGH risk the decision engine **holds** the offset: it may decline to ask for more, never for
less, and it stands aside entirely for the absolute safety floor. This costs no comfort, and that is
not a judgement but an identity — the extra offset was not producing heat, because the compressor
had no frequency left to give it with.

## Sources

- NIBE F750 Installer Manual **IHB GB 1301-1** (231236), menu 4.9.3 — `start compressor`,
  `start addition` defaults and ranges.
- NIBE's degree-minute definition is consistent across the F-series and S-series manuals.

## Not sourced

The **−1500** figure itself. It is attributed in the code to "Swedish forums" and to a `stevedvo`
F2040 case study, neither of which is in this repository. It functions as an absolute backstop and
is far below the pump's own aux-start, so it is harmless as a floor — but it should not be
described as research-validated until someone can produce the research.
