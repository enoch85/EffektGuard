# Month-long optimization simulation

Drives the **real `DecisionEngine`** at 5-minute steps against a physical
house + NIBE plant model (heating curve, degree-minute hysteresis, COP from
the pump profile) with **real historical data**:

- `data/weather_jan2026.json` — 744 hourly Stockholm temperatures, January
  2026 (ERA5-Land via Open-Meteo)
- `data/prices_jan2026.json` — real Nordpool SE3 day-ahead prices, 31 days ×
  96 quarter-hours (EUR-native × daily EUR/SEK)

Two configurations, both targeting 22.0 °C indoor:

| tag | house | pump profile |
|---|---|---|
| `wooden_f750` | timber frame, radiators, τ≈30 h | NIBE F750 (ASHP) |
| `concrete_f1155` | concrete slab UFH, τ≈80 h | NIBE F1155 (GSHP) |

Checked invariants per step: DM never below the aux limit without response,
indoor never below 18 °C, offset within ±10, no engine exceptions, no offset
oscillation. Summary + violations + 30-min trace land in `output/`.

## Run

```bash
# Fast 2-day synthetic self-test (part of the test procedure)
.venv/bin/python scripts/simulation/sim_harness.py --selftest

# Full 31-day January run with real data (~10 s)
.venv/bin/python scripts/simulation/sim_harness.py

# Matched neutral baseline (same plant, offset pinned to 0) for comparison
.venv/bin/python scripts/simulation/sim_harness.py --baseline
```

Or via the test runner:

```bash
bash scripts/run_all_tests.sh simulation
```

A run is healthy when every config reports `"violations": 0` and
`"exceptions": 0`.

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
