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
```

Or via the test runner:

```bash
bash scripts/run_all_tests.sh simulation
```

A run is healthy when every config reports `"violations": 0` and
`"exceptions": 0`. Compare `cost_sek`/`indoor_mean` against a zero-offset
baseline before drawing optimization conclusions (see
`ISSUE_18_MODBUS_FINDINGS.md`, finding S1: in balanced mode the engine
currently settles ~0.5–0.7 °C below target in a consistently expensive
month).
