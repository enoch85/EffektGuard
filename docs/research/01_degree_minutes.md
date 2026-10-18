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
deepens it, and buys nothing but compressor hours and immersion heat. This is why the
compressor-wear guard caps raises while the compressor is saturated (F-124, F-129).

**The comfort floor, not DM, is what protects the occupant.**

## Compressor wear

`CompressorSaturation` (in `utils/compressor_monitor.py`) decides when the compressor has no
frequency left; the coordinator feeds it from every cycle that drives the pump, never from a
read-only refresh. No fixed frequency can decide it. In a year of the owner's F750, an hourly mean
above 100 Hz came 112 times with EffektGuard's airflow boost on and 5 times without, while its
hardest hours at normal airflow averaged 75-90 Hz; a ground-source pump at 60 Hz may have headroom
to spare. What a NIBE inverter does with headroom is
answer falling degree minutes with more frequency - the owner's F750 adds about 7 Hz in the next
five-minute cycle after losing 6-10 DM.

Many publish how far up that goes. The F750, F730 and F1155/F1255 report their current maximum
frequency in register 43123 ("Compr. current max.freq."), the S1155/S1255 in 31043; the SMO 40 and
VVM modules an F2040 runs behind report none (the register lists of the `nibe` library that Home
Assistant's `nibe_heatpump` integration reads). On the owner's F750 it moves between 70 and 120 Hz,
often several times an hour (October 2026). On the night of 2026-10-13 it went from 120 to 92, 83,
91 and 86 Hz in two hours, the compressor at each while DM fell from -626 to -940. Where the pump
publishes it, the compressor is saturated when DM falls at `ANTI_WINDUP_DM_DROPPING_RATE` or faster
through a window in which every reading sits within the band of the maximum published with it, and
it stays saturated while the frequency stays at that maximum. Of the 10 stretches of 40 minutes or
more the owner's F750 spent at its maximum in October 2026, DM fell at 50/h or faster through 2
(130/h and 88/h, the night of 10-13); through the other 8 it fell slower or rose. Below its own
maximum it has frequency to give: in October 2026 the owner's F750 never held its frequency flat for
half an hour below that maximum while DM fell at 50/h or faster - all six such windows were at the
maximum.

A pump that publishes no maximum is saturated when degree minutes fall at
`ANTI_WINDUP_DM_DROPPING_RATE` or faster through a window in which every reading sits at its
ceiling - the highest level the frequency has sustained in a day - or when it runs at
`COMPRESSOR_HZ_MAX` without gaining on its degree minutes. It stays saturated while the frequency
stays at that ceiling and stops climbing.

That ceiling has to be earned. It is remembered per airflow state, so a boost to 120 Hz does not
blind the guard for the day after it, and it counts only once this run or the memory has shown the
frequency span more than the band: a frozen reading, a run that began at the ceiling or the empty
memory after a restart proves nothing. A compressor within the band of its minimum is never at its
ceiling - the same F750 holds 20 Hz for up to 40 minutes after a cold start while DM falls ~110.
A level more than half the band above the highest one an hour or more old is climbing, not stuck,
and without a level that old there is no evidence yet; a slower drift is taken for the ceiling,
since an inverter with headroom adds about 7 Hz in a single cycle. Hot water - the pump's own
hot-water priority, not the temporary-lux switch, which stays on for hours after the tank is
charged - pauses the run without ending it: a verdict stands until the readings after the pause
confirm or release it, and the degree minutes the tank costs never count. A stop, a dropout, an
airflow change, the clock going back or another gap longer than half a window ends the run.

Saturated, the decision engine caps the offset at 0, the pump's own curve, for a room at the top of
its comfort band or above: a raise is declined, and one already made is withdrawn - at once, for the
coordinator's volatility gate, which holds a reversal for 45 minutes, lets that withdrawal through
(and only that: a cut it holds stays held). Lower in the band a share of the raise passes - half at
the target, all of it at the band's lower edge, where immersion heat is the heat the house is short
of. (Starting the share at the target left a house the guard had kept from banking immersion heat
colder than its band through a savings-mode peak cut or a ventilation boost.) A cut stands, and
below `MIN_TEMP_LIMIT` the house still gets everything. Withdrawing a raise drops S1 under what the
compressor can deliver, so the frequency backs off and the guard lets go until the next window
proves the ceiling again: the compressor does not sit at its ceiling throughout, and some of the
heat it would have made is given up with the immersion heat. `CompressorHealthMonitor.assess_risk()`
(HIGH above 100 Hz for 15 minutes, ELEVATED above 80 Hz for 2 hours) feeds the compressor health
sensor only.

## Sources

- NIBE F750 Installer Manual **IHB GB 1301-1** (231236), menu 4.9.3 — `start compressor`,
  `start addition` defaults and ranges.
- NIBE's degree-minute definition is consistent across the F-series and S-series manuals.

## Not sourced

That an inverter raises its frequency as DM falls is measured on one F750 (the owner's, October
2026), not read in a manual; the saturation signal assumes every NIBE inverter regulates that way.
So is that a pump's published maximum is the only limit on its frequency: no manual says what
43123 or 31043 takes into account, and on the owner's F750 the air leaving the evaporator (BT21)
reads the same spread whether the maximum is 120 Hz or lower.
The frequencies quoted are the owner's long-term statistics, which carry no ventilation history:
"with the airflow boost on" means EffektGuard had asked for it, not that the fan was measured.

The **−1500** figure itself. It is attributed in the code to "Swedish forums" and to a `stevedvo`
F2040 case study, neither of which is in this repository. It functions as an absolute backstop and
is far below the pump's own aux-start, so it is harmless as a floor — but it should not be
described as research-validated until someone can produce the research.
