# Repo-wide review #2 — validation & fix workplan (untracked working file)

Context: PRs #19 (feature+fixes) and #20 (tooling+findings) are open, CI green.
New external repo-wide review with 8 findings. Branch strategy: create
`fix/measurement-and-unit-defects` STACKED ON feature/nibe-multi-source-modbus
(coordinator conflicts with #19 otherwise); PR targets main after #19 merges,
or re-target as needed. Verify each finding in code BEFORE fixing; add a
regression test per confirmed fix; Black 100; no AI attribution anywhere.

## Findings triage (verify → fix → test)

- [x] R1 CRITICAL kW double-division: adapter converts W→kW by unit
      (nibe_adapter.get_power_consumption); reviewer says coordinator divides
      configured external meter by 1000 AGAIN (grep coordinator for "/ 1000"
      near external power sensor reads, ~_update_peak_tracking/current_power).
      Fix: single conversion point (respect unit attr), regression test with a
      kW-unit meter = 6.0 → stays 6.0.
- [x] R2 HIGH quarter peak = instantaneous sample, not 15-min mean:
      coordinator records current reading as quarter measurement
      (effect.record_quarter_measurement). Fix: accumulate samples within the
      quarter (5-min cadence → ~3 samples) and record the MEAN at quarter
      boundary; keep spike value out of peak records. Test: 9 kW spike among
      1 kW samples → recorded quarter ≈ mean, not 9.
- [x] R3 HIGH profile flow-temp unit error: profiles use
      KUEHNE_COEFFICIENT*(HLC*ΔT)**KUEHNE_POWER with HLC in W/K (weather_layer
      divides by 1000 first — check const HEAT_LOSS_DIVISOR). Affects
      f730/f750/f2040/s1155 + inherited f1155. Fix: same W→kW conversion as
      shared formula; test: HLC=180,ΔT=31 → ≈30.7°C not clamp. Note
      heat_demand_kw param unused (leave signature, document).
- [x] R4 HIGH F730/F750 exhaust-air COP vs outdoor temp: physics limitation,
      not a code bug — document in profiles + findings ("COP curves are
      approximations keyed on outdoor temp; exhaust-air source not modeled";
      simulation results for F750 are control-flow, not energy-truth).
- [x] R5 MEDIUM adaptive_learning derives "W/K" without capacitance/energy
      term: verify in optimization/adaptive_learning.py. Fix approach:
      relabel/quarantine — treat as relative decay metric, do NOT feed as
      absolute W/K into weather compensation unless user-configured; keep
      bounds; document. Check who consumes it (heat_loss_coefficient flow).
- [x] R6 MEDIUM savings /100 assumes öre/kWh: savings_calculator uses
      ORE_TO_SEK_CONVERSION unconditionally; gespot adapter preserves user
      unit. Fix: read unit_of_measurement from the gespot entity, convert
      accordingly (öre/kWh ÷100, SEK/kWh ×1, EUR→ report in EUR or skip);
      unknown unit → label output "price units", don't fake SEK. Test per unit.
- [x] R7 MEDIUM DST days: gespot_adapter forces 96 quarters + wall-clock
      indexing; spring=92 padded w/ synthetic avg, autumn duplicate ids.
      Verify; fix = index by UTC timestamp within day / handle 92/100-length
      days honestly; time_utils.get_current_quarter ambiguity. Larger — may
      document + targeted fix (no synthetic padding; classification on actual
      periods).
- [x] R8 MEDIUM DHW next-opportunity uses earliest constraint (min) where
      mandatory constraints require max: dhw_optimizer "next safe
      opportunity". Verify; fix min→max over mandatory constraints; test.
- [x] R9 scripts/demo_dhw_next_boost_clean.py: unterminated triple-quote →
      compileall fails. Fix or delete (prefer delete if demo is dead).
- [x] R10 scripts/simulate_real_day.py unrunnable (hardcoded
      /workspaces/EffektGuard/IMPLEMENTATION_PLAN/debug.log). Delete or fix
      paths; the maintained simulator is scripts/simulation/ (PR #20).
- [x] R11 Simulation harness comfort assertion looser than configured
      tolerance (target-1.0 vs tolerance 0.5): tighten comfort_minutes_below
      threshold to configured tolerance; add neutral-baseline comparison mode
      (--baseline) to the harness so cost/comfort deltas are first-class
      output (reviewer's matched-comparison approach). Quarter-average tariff
      in harness cost model.
- [x] R12 Findings doc: add "physical assumptions not validated" section
      (GM estimate, climate-zone thresholds, phase-current basis, curve
      sensitivity) + reviewer's delta table (savings ≈ under-heating).

## State when resuming
- Repo /workspace on branch: check `git branch --show-current`.
- PRs: #19 feature (2 commits), #20 tooling (4 commits) — don't break them.
- Suite baseline: 1146 passed on #19 branch; Black clean.
- Live devbox HA on :8125 + modbus sim on :5020 still running (leave alone).
