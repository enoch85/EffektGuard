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

Compressor output respects each profile's declared modulation range. The F2040 maximum comes
from NIBE's separate peak-output grid, interpolated over outdoor and water temperatures; nominal
COP rating loads are not capacity limits. The grid covers −7 to +7 C air and 25 to 55 C water.
Outside it the nearest edge is held, and `capacity_outside_published_minutes` counts that
extrapolation. Cold-end capacity is unvalidated. Profiles without a declared minimum still have
an unknown minimum output; their cycling results require further data.

The edge hold is load-bearing. On the 21-day `nordic_coldsnap_jan2024` F2040 run at `ad88d4f`
(the envelope patch on the adaptive simulator in #136), scaling only extrapolated capacity to 70%, 100%
and 130% produced respectively 395, 0 and 0 minutes below the comfort band, and 274.25, 1.25
and 0 kWh of immersion heat. These are sensitivity probes, not uncertainty bounds. A cold-weather
pass therefore cannot establish real-world comfort or backup consumption without a cold-end map.

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

- `PASS: configured simulation checks held` means those checks passed under the plant's assumptions.
  It does not validate a real home, the entire COP surface or a savings claim. `FAIL*` means every failure is a recorded
  known red (exit code 3 for a single `--scenario` run). The sweep fails only on news: a red no
  known red covers, one worse than recorded, or one that stopped failing. A known red that gets
  worse is re-recorded with its cause, never silenced.
- Compare against `--baseline`, with comparable comfort and initial/final thermal storage, before
  interpreting a cost difference. This is a modeled scenario comparison, not measured bill savings.
- `instantaneous_capacity_deficit_kwh` integrates demand above the compressor's available output.
  It ignores heat carried between intervals and does not bound unavoidable immersion use. For
  example, two one-hour intervals with demand 10/10 kW, capacity 8/12 kW and 2 kWh initially stored
  can finish with the same stored heat and no immersion, despite a 2 kWh instantaneous deficit.
  Subtracting this diagnostic from immersion consumption cannot identify the controller's effect.
- `datasheet_cop` is a nominal-rating reference, not a universal COP ceiling. Different source,
  water and load conditions require independent validation. The Carnot check and rating-point
  reproduction/hold-out tests remain. `mean_cop` is delivered emitter heat divided by modeled
  electricity excluding standby; storage changes and immersion affect that diagnostic too.
- `layer_votes` counts proposed votes before aggregation, gating and register conversion. A
  recovery vote in May does not by itself establish applied heat or extra cost.
- `zero_heat_comfort_minutes_above` repeats the room balance with supplied heat set to zero,
  starting from the same room and using the same weather and gains. When the emitter is
  nonnegative and the discrete comparison coefficient is in [0, 1], that trajectory is a lower
  temperature bound (`zero_heat_bound_valid`). A reference already over the upper-band budget
  proves the heating-only plant cannot meet that budget; the failure says so. In the 21-day May
  apartment case, even zero supplied heat gives 7,175 minutes above the band against a 720-minute
  budget. This is a model-feasibility result, not a finding about the owner's home. The reference
  can violate the lower comfort floor, so differences from it do not establish avoidable heat or savings.
- `comfort_minutes_below` counts minutes at least `SCENARIO_COMFORT_BREACH_MIN_DEPTH_C` below
  the band the mode keeps, `comfort_band_low_c`: the target minus the configured tolerance, or
  minus the mode's `room_floor_below_target` where that is wider (savings: a degree). Shallower
  dips are `comfort_minutes_grazing`.
- `tariff_top3_kw` is the mean of the three highest daily billed peaks: 15-minute mean power
  through the same `BillingPeriodAccumulator` and `effective_tariff_power_kw` production uses
  (night periods count `NIGHT_TARIFF_WEIGHT`); `tariff_cost_sek` multiplies it by the illustrative
  `SWEDISH_EFFECT_TARIFF_SEK_PER_KW_MONTH`; `total_cost_sek` = energy + tariff.
- COP uses each profile's heat source, flow temperature and heat output. Reproducing nominal
  rating points does not validate performance throughout the operating envelope, start losses,
  defrosting or the physical model of any particular home.
