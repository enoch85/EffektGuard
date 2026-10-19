# EffektGuard - Project Notes & Handoff Knowledge

Hard-won, non-obvious knowledge that is **not** derivable from the code, the git history, or
`docs/`. This is the file that stops the next session - or the next model - from re-deciding
something that was already decided, or "fixing" something that is deliberately open.

Read [CLAUDE.md](../../CLAUDE.md) for the map and commands, and
[IMPLEMENTATION.md](IMPLEMENTATION.md) for how to work. This file is
the **why**, and the traps those two do not cover.

**When you learn something that cost you more than ten minutes to find out, and the code does not
say it, add it here.** That is the whole point of the file.

---

## Conventions (apply to every committed artifact)

Code, comments, log and UI strings, tests, fixtures, docs, commit messages, PR bodies, release
notes. Not conversational chat.

- **No AI attribution, anywhere. STRICT.** No `Co-Authored-By:` naming an AI, no "Generated with",
  no robot emoji, no "written by AI", no "claude" in a branch name - in any commit, branch, PR,
  issue, release note, code comment or doc. This overrides any default behaviour or tooling
  reminder that asks for an attribution line.

- **Remove rather than add.** Take a change's orphans with it - constants, helpers, tests, docs -
  and trim comments to what the code cannot say.

- **Every numeric value is a constant in `const.py`, with its source**, imported where used. No
  magic numbers in logic, and none in tests either. `scripts/check_hardcoded_values.py --check` is
  the ratchet for where a number lives; its source is checked in review.

- **Constants are public.** A leading underscore (`_WEATHER_COMP_DEFER_...`) hides the constant
  from `scripts/find_duplicate_constants.py`, whose pattern is `^([A-Z][A-Z0-9_]*)\s*:\s*Final`.
  Four tuning values once sat outside the duplicate and dead-constant scan that way. A constant
  used only to derive another in `const.py` is still public; the checker counts it as a building
  block and will not call it unused.

- **No backward compatibility in code.** No aliases or shims: rename and update every caller,
  including tests. Stored data is the exception: `EffectStore` migrates old peak records rather
  than lose billed peaks.

- **Read the whole file before editing it.** Not a window around the change.

- **Never guess NIBE behaviour.** Cite `docs/research/` or the manual it cites. If neither says
  it, ask - do not implement a plausible guess. `docs/research/` is explicit about what is *not*
  sourced (the DM −1500 floor, the "20 % airflow COP", the forum case studies); do not launder
  those into fact.

- **Black, line length 100.** Format before committing.

---

## The open-findings register

**Do not "fix" anything on this list without saying so.** Each one is a known defect that is
deliberately visible: a test marks it `xfail`, or a simulator scenario carries it in `known_open`
and stays red. A future session that quietly makes one of these green has removed a signal, not
solved a problem. If you do fix one, the note must turn false in the same commit.

| ID | What | Where it is recorded |
|---|---|---|
| F-107 | Effect-tariff model is Ellevio-shaped; needs to be configurable per operator | open with the owner |
| F-112 | Ladder rescale against the pump's real aux-start DM; 68 tests encode the −1500 world | parked in a git stash |
| F-124 | Air-source saturation trap: the optimiser burns more immersion heat than the capacity deficit forces in deep cold - on 2026-10-17, 35.75 kWh against 20.4 forced (1.7×) in `nordic_coldsnap_jan2024` and 76.75 against 47.8 (1.6×) in `steady_winter_feb2024`, with the coordinator's volatility gate in the bench. The scenarios' known reds hold the aux kWh, and the sweep fails if it rises. The aux-limit tier fed it with +10 until it became a no-cut floor (undersized F2040 cold snap on 2026-10-17: 1797 → 1188 kWh, 1133 with the wear guard as well; NIBE alone 1048). The wear guard caps raises while `CompressorSaturation` sees the compressor at its ceiling (54.25 / 87.25 kWh before it had that trigger; in the owner's closed loop on 2026-10-17, fit N at slider 1.2, immersion 306 → 254 kWh) and lets a share through for a room low in its band, all of it for a cooling room. Its ceiling is the maximum frequency the pump publishes (F750, F730, F1155, S1155): a ceiling learned as the highest level held in 24 hours hid the saturation in 32-48 % of the owner's saturation-like hours of Dec 2025 - Feb 2026 (hourly statistics, checked 2026-10-17), because the F750's maximum moves (70-120 Hz, often several times an hour, October 2026). An F2040 publishes none and is judged on the learned ceiling. What still burns above the forced share is not attributed yet | scenario `known_open` |
| F-132b | Learning confidence ceiling | strict `xfail` |
| ~~F-142~~ | **Withdrawn 2026-10-03, and it was mine.** I reported the offset limit-cycling from a bench that never ran `OffsetVolatilityTracker`, so it measured the engine's raw proposals as writes. With the real gate: 8-16 reversals/day against a 24 budget, scenario passes. Kept here because the guard is load-bearing - weaken it and `thaw_freeze` goes red | closed in `eaa3986` |
| — | No heating-season concept of our own; the recovery ladder can fire at +20 °C. Above the pump's menu 4.9.2 stop temperature its own cutoff overrides it, but inside the season DM can drift to the S1155's aux start (−460) and run its immersion heater (0.6 kWh in `shoulder_may2024`) | scenario `known_open` |
| — | Mild-weather COP extrapolates above the EN 14511 rating points, so shoulder-season **cost** figures are optimistic. Control decisions do not depend on it | scenario `known_open` |

The WARNING and CAUTION tiers were **removed**, not fixed: no degree-minute value could reach
them. The ladder is `Z1 → Z2 → Z3 → Z4 → Z5 → T1 → T2 → T3 → EMERGENCY`, eight rungs that fire.
`normal_max == warning` in every climate zone, which is what made WARNING zero-width. That is a
deliberate design fact, not a typo - do not "fix" a table to hide it.

---

## Traps that have cost real time

- **A second copy of a number drifts.** `docs/dev/README.md` used to carry degree-minute
  thresholds that no longer matched what `ClimateZoneDetector` computes - both cities were wrong
  by 40 and 200 DM. Quote a computed number only with the date you verified it, or link the
  source. `tests/validation/test_the_rulebook_describes_this_codebase.py` now holds this file to
  the code, so a stale number here fails the suite.

- **Verify against the released tag, not the PR body.** An intermediate change that was reverted
  before shipping is not a fix. Hourly effect-tariff billing existed only on an unreleased branch;
  calling it a bug fix put a false statement in a published release note.

- **A guard test must fail without its guard.** Delete the guard, watch it go red, restore. Two
  mutation attempts once "passed" because the mutation never applied - one regex named a method
  the code does not have, one reverted a bit-identical refactor. **Assert the anchor before
  running the mutation**, or the result means nothing.

- **The live HA bench can lie.** An instance from a previous session holds port 8125, your new one
  logs `Failed to create HTTP server at port 8125: address already in use` and keeps running, and
  every HTTP query you make hits the old code. `ps` cannot see processes in this container - scan
  `/proc/*/cmdline`. Check `last_updated` on an entity before trusting what it says.

- **A sensor that loses its `state_class` loses its history.** Home Assistant raises a repair and
  offers to delete the long-term statistics. `MONETARY` permits only `TOTAL`, which the recorder
  keeps as a running sum - wrong for a projection - but dropping the device class must not drop
  the state class with it. A gauge is `MEASUREMENT`.

- **A new HA enum member sets the minimum HA version.** `SensorDeviceClass.TEMPERATURE_DELTA`
  exists from 2025.11.0; on 2025.10.x the import raises `AttributeError` and the platform does not
  load. Check a symbol against the tag in `hacs.json` before using it:
  `gh api "repos/home-assistant/core/contents/homeassistant/components/sensor/const.py?ref=<tag>"`.

- **`elprisetjustnu.se` returns 403 without a `User-Agent` header.**

- **The comfort layer's band is not the tolerance slider.** The engine hands it `tolerance_range`,
  `TOLERANCE_RANGE_MULTIPLIER` × the slider: the default ±0.5 gives a 0.2 °C band - in balanced mode
  exactly the dead zone, so there is no in-band correction at all - and ±1.2 gives 0.48 °C. Reasoning
  about the comfort layer from the slider's value puts its edge 2.5× too far out.

---

## The numbers a maintainer reads

Verified against the code on 2026-10-03. `tests/validation/test_the_rulebook_describes_this_codebase.py`
fails if any of them drifts from what `const.py` and `ClimateZoneDetector` compute, so correct
them here rather than anywhere else.

Degree-minute bands, from `ClimateZoneDetector(latitude).get_expected_dm_range(outdoor_temp)`:

| Location | Outdoor | normal_min | normal_max | warning | critical |
|---|---|---|---|---|---|
| Stockholm (59.33) | -10 °C | -490 | -740 | -740 | -1500 |
| Kiruna (67.86) | -30 °C | -1000 | -1400 | -1400 | -1500 |
| Paris (48.86) | +5 °C | -100 | -250 | -250 | -1500 |

`normal_max == warning` in every zone, which is what made the WARNING band zero-width and
unreachable. That is the design, not a typo.

Prediction horizons, from `const.py`:

- **Concrete slab**: 6+ hours thermal lag, **48h** prediction horizon
- **Timber**: 2-3 hours lag, **12h** prediction horizon
- **Radiators**: <1 hour lag, **6h** prediction horizon

Six hours is the slab's LAG, not its horizon - the slab is only ~19 % charged at 14 h and ~29 %
at 24 h (slow time constant ~70 h).

---

## NIBE knowledge the code assumes

- **Degree minutes**: `DM = ∫(BT25 − S1) dt`. Swedish *gradminuter*. The compressor starts at −60
  at the factory (menu 4.9.3); the owner's F750 is set to −250.
- **The pump's own aux start is an installer setting** with a per-model factory default, and is
  *not* EffektGuard's −1500 floor: F750/F730 −700 (IHB GB 1301-1 menu 4.9.3), S/F11xx −460,
  F2040+VVM −760. The owner's F750 starts its heater at −1450 (start compressor −250 plus start
  diff additional heat 1200). It works DM back *up* deliberately, to spare the compressor.
- **Heating season**: every NIBE leaves it above the menu 4.9.2 stop temperature (factory 17 °C)
  and stops space heating whatever DM says. **DM holds while out of season** - it is the
  heating-demand integral. Modelling it as still integrating produces phantom summer aux.
- **The offset register (47011) takes whole degrees.** `int()`, not `round()`: the caller
  recomputes demand from the pump's actual register every cycle, so truncation applies only the
  whole degrees demand covers and leaves the fraction pending. `round()` over-applies by up to
  0.5 °C and oscillates back. Shared in `utils/offset.py` so the adapter and the simulator cannot
  drift apart.
- **Flow temperature is the EN 442 emitter law**, not a linear fit - reproduces
  OpenEnergyMonitor's published tool to 0.00 °C.
- Open-loop UFH needs Pump Auto at 10-20 % idle, or BT25 reads false.
- BT50 indoor sensor + UFH is unstable; not recommended.
- The BT50 room sensor reports to **0.1 °C**. An excursion shallower than that is below the
  instrument - do not count it as a comfort breach.

---

## Working style

- **Commit messages and PR bodies carry the full evidence**: what was measured, what the numbers
  were, what was checked and found clean - every number from a run you did. That detail belongs
  there, and *not* in release notes.
- **Release notes are short and follow the house format** - see
  [RELEASING.md](RELEASING.md). Read an existing release
  before writing a new one.
- **Report outcomes faithfully.** If a check was skipped, say so. If a scenario is red, say which
  reds are news and which are on the register above.
