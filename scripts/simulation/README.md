# Simulation harness

`sim_harness.py` drives the real `DecisionEngine` - with the coordinator's volatility gate and the
register's integer conversion - at 5-minute steps against a house and NIBE plant model (heating
curve, degree-minute hysteresis, COP from the pump profile), on real data in `data/`:

- `weather_jan2026.json`: 744 hourly Stockholm temperatures, January 2026 (Open-Meteo archive API)
- `prices_jan2026.json`: Nord Pool SE3 day-ahead prices, 31 days x 96 quarter-hours, SEK/MWh
  (elprisetjustnu.se)
- one weather/price pair per scenario, fetched with `fetch_scenario_data.py`

Five houses, all targeting 22.0 C (`TARGET_INDOOR`):

| tag | house | pump |
|---|---|---|
| `wooden_f750` | timber frame, radiators | F750 (exhaust air) |
| `concrete_f1155` | concrete slab UFH, large villa | F1155 (ground source) |
| `apartment_f730` | apartment, deliberately oversized pump | F730 (exhaust air) |
| `villa_s1155` | timber underfloor, large villa | S1155 (ground source) |
| `airsource_f2040` | outdoor air | F2040 (air to water) |

Six scenarios, each a real weather window with the real prices of the same days, each with the
behaviour it pins (`expect`) and its recorded known reds (`known_open`): `nordic_coldsnap_jan2024`,
`undersized_coldsnap_jan2024` (the same days, every pump one size too small),
`steady_winter_feb2024`, `thaw_freeze_mar2024`, `shoulder_may2024`, `autumn_volatile_oct2024`.

## Run

```bash
python scripts/simulation/sim_harness.py --selftest        # 2 synthetic days (CI)
python scripts/simulation/sim_harness.py --dst             # the fall-back weekend (CI)
python scripts/simulation/sim_harness.py --all-scenarios   # the sweep (CI)
python scripts/simulation/sim_harness.py --scenario shoulder_may2024 --mode savings
python scripts/simulation/sim_harness.py                   # 31 days of January 2026
python scripts/simulation/sim_harness.py --baseline        # same plant, offset pinned to 0
bash scripts/run_all_tests.sh simulation                   # the same as --selftest
```

`--mode` is `balanced` (default), `comfort` or `savings`. Other switches, read in `main()`:
`--undersized`, `--coldsnap`, `--arctic`, `--live-se4`, `--no-price`, `--no-weather`,
`--no-forecast`, `--tuned-baseline`, `--battery`. Any of them, or a mode other than balanced, is a
different experiment: known reds are not applied and every red is news.

Per house, `output/summary-<tag>.json` holds the stats, failures and violations, and
`output/trace-<tag>.json` the trace.

## Reading a result

- `PASS: all safety invariants held` is a healthy run. `FAIL*` means every failure is a recorded
  known red (exit code 3 for a single `--scenario` run). The sweep fails only on news: a red no
  known red covers, one worse than recorded, or one that stopped failing. A known red that gets
  worse is re-recorded with its cause, never silenced.
- Compare against `--baseline` before concluding anything about savings: part of a lower `cost_sek`
  can be under-heating rather than efficiency.
- `comfort_minutes_below` counts minutes at least `SCENARIO_COMFORT_BREACH_MIN_DEPTH_C` below
  the band the mode keeps, `comfort_band_low_c`: the target minus the configured tolerance, or
  minus the mode's `room_floor_below_target` where that is wider (savings: a degree). Shallower
  dips are `comfort_minutes_grazing`.
- `tariff_top3_kw` is the mean of the three highest daily billed peaks: 15-minute mean power
  through the same `BillingPeriodAccumulator` and `effective_tariff_power_kw` production uses
  (night periods count `NIGHT_TARIFF_WEIGHT`); `tariff_cost_sek` multiplies it by the illustrative
  `SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH`; `total_cost_sek` = energy + tariff.
- The F750's COP curve is keyed on outdoor temperature, but an exhaust-air pump's COP depends on
  exhaust and flow temperatures: F750 results are evidence of control behaviour, not of energy.
