# Month-long optimization simulation

Drives the **real `DecisionEngine`** (with the coordinator's volatility gate and the register's
integer conversion) at 5-minute steps against a physical house + NIBE plant model (heating curve,
degree-minute hysteresis, COP from the pump profile) with **real historical data** in `data/`:

- `weather_jan2026.json` — 744 hourly Stockholm temperatures, January 2026 (ERA5-Land via Open-Meteo)
- `prices_jan2026.json` — Nord Pool SE3 day-ahead prices from the elprisetjustnu.se API, native SEK,
  31 days × 96 quarter-hours
- one weather/price pair per scenario below, fetched with `fetch_scenario_data.py`

Five houses, all targeting 22.0 °C indoor (`TARGET_INDOOR`):

| tag | house | pump profile |
|---|---|---|
| `wooden_f750` | timber frame, radiators | NIBE F750 (exhaust air) |
| `concrete_f1155` | concrete slab UFH, large villa | NIBE F1155 (ground source) |
| `apartment_f730` | apartment, deliberately oversized pump | NIBE F730 (exhaust air) |
| `villa_s1155` | timber underfloor, large villa | NIBE S1155 (ground source) |
| `airsource_f2040` | outdoor air | NIBE F2040 (air-to-water) |

Six scenarios (`--all-scenarios`), each a real weather window paired with the real prices of the same
days, each with the behaviour it pins (`expect`) and its recorded known reds (`known_open`):
`nordic_coldsnap_jan2024`, `undersized_coldsnap_jan2024` (each pump one size too small),
`steady_winter_feb2024`, `thaw_freeze_mar2024`, `shoulder_may2024`, `autumn_volatile_oct2024`.

Summary + violations + trace land in `output/`.

## Run

```bash
# Fast 2-day synthetic self-test, and the DST fall-back weekend (both in CI)
.venv/bin/python scripts/simulation/sim_harness.py --selftest
.venv/bin/python scripts/simulation/sim_harness.py --dst

# The scenario sweep (in CI), or one scenario
.venv/bin/python scripts/simulation/sim_harness.py --all-scenarios
.venv/bin/python scripts/simulation/sim_harness.py --scenario shoulder_may2024 --mode balanced

# Full 31-day January run with real data (the nominal run)
.venv/bin/python scripts/simulation/sim_harness.py

# Matched neutral baseline (same plant, offset pinned to 0) for comparison
.venv/bin/python scripts/simulation/sim_harness.py --baseline
```

Or via the test runner:

```bash
bash scripts/run_all_tests.sh simulation
```

A run is healthy when every safety invariant holds (`PASS: all safety invariants held`). The sweep
fails only on **news**: a red no known red covers, one worse than recorded, or one that stopped
failing. A scenario whose every failure is a recorded known red prints `FAIL*` (exit code 3 for a
single `--scenario` run); a known red that gets worse must be re-recorded with its cause, never
silenced.

**Always compare against `--baseline` before drawing optimization
conclusions** — an optimized run's `cost_sek` is meaningless on its own
(see `docs/dev/MULTI_SOURCE_FINDINGS.md`, finding S1: in balanced mode the
engine settles ~0.5–0.7 °C below target in a consistently expensive month,
so part of any "saving" is under-heating, not efficiency).

Cost/comfort accounting:

- `comfort_minutes_below` counts minutes below **target − configured
  tolerance** (22.0 − 0.5), the same band the engine promises to hold —
  not a looser ad-hoc threshold.
- `tariff_top3_kw` / `tariff_cost_sek` model the Swedish effect tariff on
  **quarter-hour mean power** (mean of the top-3 daily peaks × an
  illustrative rate); `total_cost_sek` = energy + tariff. Peaks are means,
  never instantaneous samples.
- Energy-model caveat: the F750 is an exhaust-air pump whose COP depends on
  exhaust and flow temperatures; the outdoor-keyed COP curve here makes
  F750 results valid for control-flow behavior, not absolute energy truth.
