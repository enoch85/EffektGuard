# Issue #18 — NIBE F1155 via Modbus: Investigation Findings

> Working document / development record for the multi-source (Modbus /
> nibe_heatpump) support work in issue #18 — root-cause analysis, register
> research, bug findings, and the month-long simulation results (findings
> S1/S2). Kept in docs/dev as a temporary working record; may be trimmed or
> removed once the follow-ups it lists are resolved.

## TL;DR

The issue reporter runs the **official HA core `nibe_heatpump` integration** (NibeGW/MODBUS40
transport for his F1155), NOT the generic `modbus` YAML integration. Evidence: his entity id
`sensor.bt1_outdoor_temperature_40004_2` is byte-identical to `slugify()` of the yozik04/nibe
library coil name `bt1-outdoor-temperature-40004` (+ HA collision suffix `_2`), and his degree
minutes is a `number` entity — only `nibe_heatpump` produces writable coils as numbers
(`number.degree_minutes_16_bit_43005`). The generic `modbus` integration cannot create number
entities at all.

EffektGuard's adapter discovery is already ~90% source-agnostic (pure entity_id substring scan
over sensor./number./switch. registry entries). The blockers are: (1) config-flow selectors that
reject `number` entities where Modbus/nibe_heatpump expose numbers, (2) missing patterns for
nibe_heatpump/Modbus register ids (43005/40940 DM, compressor state strings), (3) no manual
override for core sensors (BT1/BT7/BT50/flow/return) when discovery misses, (4) all
`nibe_heatpump` coil entities are **disabled by default** (likely the actual cause of "BT7 not
found" — the user never enabled BT7), (5) MyUplink-hardcoded UX strings, (6) no F1155 model
profile (user forced to pick S1155; unknown keys silently fall back to F750 = wrong physics).

## Verified environment facts (primary sources)

- `nibe_heatpump` (HA core, elupus/yozik04): F-series connects via NibeGW UDP or serial
  MODBUS40; **Modbus TCP is S-series only**. Entity id = `slugify(coil.name)` where coil.name
  embeds the register id. Writable coils → `number.*` (EntityCategory.CONFIG), boolean writable →
  `switch.*`, mapped writable → `select.*`, read-only → `sensor.*` (temperature sensors carry
  device_class temperature + °C + DIAGNOSTIC category). **All coil entities disabled by
  default**; 60 s poll of enabled coils only. Availability requires coil present in last data.
  - F1155 key entities: `sensor.bt1_outdoor_temperature_40004`, `sensor.bt2_supply_temp_s1_40008`,
    `sensor.eb100_ep14_bt3_return_temp_40012`, `sensor.bt7_hw_top_40013`, `sensor.bt6_hw_load_40014`,
    `sensor.bt50_room_temp_s1_40033`, `sensor.calc_supply_s1_43009`,
    `number.degree_minutes_16_bit_43005` (factor 10, step 0.1, unit "DM", writable),
    `number.degree_minutes_32_bit_40940`, `number.heat_offset_s1_47011` (s8, ±10, step 1, writable),
    `sensor.compressor_state_ep14_43427` — **mapped strings** {20 Stopped, 40 Starting, 60 Running,
    100 Stopping}, `sensor.int_el_add_power_43084` (kW).
- Generic `modbus` integration: platforms = binary_sensor, climate, cover, fan, light, sensor,
  switch — **no number platform**. Writes via `modbus.write_register` action. Users wrap writable
  registers in **template numbers** (template integration has a number platform with `set_value`
  actions) — exactly what the issue reporter described.
- MyUplink F-series parameter ids mostly equal Modbus register ids (40004, 47011…), but DM in
  EffektGuard's pattern list is `40941` (MyUplink) while raw Modbus/nibe_heatpump DM is
  `43005` (16-bit) / `40940` (32-bit) — pattern gap.

## Reproduced live (real Modbus TCP stack in devbox HA 2026.2.3)

pymodbus 3.11.2 TCP server (F1155 register map, NIBE x10 scaling) ← HA `modbus` YAML hub "nibe"
← sensors named in the reporter's style + template numbers wrapping `modbus.write_register`:

- `sensor.bt1_outdoor_temperature_40004` = -3.2 °C ✓, `sensor.bt7_hw_top_40013` = 48.7 °C ✓,
  `sensor.degree_minutes_43005` = -150.0 ✓, `number.degree_minutes_43005_number` ✓,
  `number.heating_offset_climate_system_1_47011` ✓, `sensor.prio_43086` = 30,
  `sensor.compressor_status_ep14_43427` = 60, `sensor.heating_offset_s1_47011` = 0.

## Bugs / defects found (with suggested fixes)

### A. Blocking Modbus users (issue #18 core)

1. **Degree-minutes selector rejects number entities** — `config_flow.py:273-276` and
   reconfigure `:607-611` use `EntitySelectorConfig(domain="sensor")`. nibe_heatpump DM is a
   `number`. Adapter read path is domain-agnostic (`_read_entity_float`), so the fix is purely
   the selector: `domain=["sensor", "number", "input_number"]`.
2. **No manual override for core sensors** (BT1 outdoor, BT50 indoor, BT7/BT6 DHW,
   supply/flow, return) — config flow never asks; discovery is the only path. Fix: optional
   "manual entity mapping" fields (reconfigure too), stored in entry.data, honored by
   NibeAdapter before pattern discovery.
3. **DM pattern list misses Modbus register ids** — `nibe_adapter.py:568` patterns are
   `["degree_minutes", "40941"]`; nibe_heatpump id `number.degree_minutes_16_bit_43005`
   happens to match "degree_minutes", but raw-register naming (e.g. `sensor.nibe_43005`) and
   S-series equivalents do not. Fix: add `43005`, `40940` patterns (constants in const.py).
4. **`CONF_NIBE_ENTITY` (user-selected offset number) is dead config** —
   `nibe_adapter.py:111` stores it as `_nibe_base_entity`, never used. The write path uses
   whatever entity substring-matched `offset`/`47011` first in registry order
   (`nibe_adapter.py:569`, `:321`). The user's explicit choice must win; pattern discovery
   should only be the fallback.
5. **Offset discovery can select a read-only sensor** — patterns `["offset","47011"]` scan
   sensor+number domains; a Modbus setup has `sensor.heating_offset_s1_47011` (read) and
   maybe a template number. If the sensor registers first, `set_curve_offset` calls
   `number.set_value` on a `sensor.` → service error, write path dead. Fix: require
   domain == number for the `offset` cache key (and prefer CONF_NIBE_ENTITY, see #4).
6. **Compressor status unreadable for both nibe_heatpump and raw Modbus** —
   `_read_entity_bool` accepts only on/true/1 (`nibe_adapter.py:761`); nibe_heatpump reports
   "Running"/"Stopped" strings (mapped coil 43427), raw Modbus reports numeric 20/40/60/100.
   `is_heating` is silently always False. Fix: extend bool parsing (running/heating/60…) or
   derive from compressor_hz > 0.
7. **F1155 missing from model list** — `config_flow.py:179-184` offers only
   F730/F750/F2040/S1155. Reporter picked S1155 (GSHP physics ≈ ok, wrong identity). Unknown
   keys silently fall back to **NibeF750Profile (exhaust-air ASHP)** at `coordinator.py:132`
   — wrong DM thresholds/COP for a GSHP. Fix: add NibeF1155Profile (+ registry), make
   fallback loud.
8. **MyUplink-hardcoded UX** — "DHW sensor (BT7) not found - check MyUplink…"
   (`coordinator.py:1240`), ConfigEntryNotReady "Ensure MyUplink…" (`__init__.py:106-108`),
   strings.json abort "install NIBE Myuplink first" (+5 translations), config_flow discovery
   platform check `entity_entry.platform == "myuplink"` (`config_flow.py:372`, `:560`).
   Fix: source-neutral wording; accept platforms myuplink|nibe_heatpump|modbus|template.
9. **manifest after_dependencies lacks `modbus`/`nibe_heatpump`** — no load-ordering
   guarantee; EffektGuard may set up before Modbus entities exist. Fix: add both.

### A2. Found by LIVE reproduction (real Modbus stack, HA 2026.2.3, 2026-07-10)

21. **Discovery scans only the entity REGISTRY — YAML entities without `unique_id` are
    invisible.** Live proof: with 11 perfectly-named modbus sensors + 2 template numbers
    running, `_discover_nibe_entities` found ONLY the 2 template numbers (they have
    unique_ids). All modbus sensors (no unique_id in YAML → never registered) were invisible;
    EffektGuard logged "No outdoor temperature sensor (BT1) found!", "No indoor temperature
    sensor (BT50) found!", and the issue-#18 "DHW sensor (BT7) not found - check MyUplink…"
    — while `hass.states` held all values. Meanwhile the *manually configured* DM sensor was
    read fine ("Using configured degree minutes sensor: sensor.degree_minutes_43005 = -150.0")
    because `_read_entity_float` uses `hass.states.get`. Fix: discover over `hass.states`
    (union with registry for disabled-entity awareness), and/or add manual overrides (#2).
22. **Discovery ignores `RegistryEntry.disabled_by`** — nibe_heatpump coil entities are
    disabled by default; a disabled entry matches patterns, has state None → temperature
    validation skipped (bug #14) → cached forever → all reads default. Most probable direct
    cause of the reporter's BT7 failure. Fix: skip entries with `disabled_by` set.
23. **Stale registry entries beat live `_2` entities** — the reporter's
    `sensor.bt1_outdoor_temperature_40004_2` means the base id is a taken/stale registry
    entry; first-match-wins iteration caches the stale base entity and never considers the
    live `_2` one. Fix: prefer candidates with a currently parseable state.
24. Live cycle after setup ran optimization on fabricated defaults:
    "NIBE data retrieved: indoor 21.0°C, outdoor 0.0°C, flow 35.0°C, DM -150" (real values
    21.3 / -3.2 / 35.8). Confirms bug #17 severity: silent defaults are indistinguishable
    from real data downstream.
25. Config flow shows raw untranslated labels live (`heat_pump_model`,
    `degree_minutes_entity`, `power_sensor_entity`, …) — confirms strings.json gaps (#19).
26. Step 1 shows "Found 0 compatible entities" for a working Modbus offset number
    (platform=template ≠ myuplink, no 'nibe' substring) — confirms #8; manual selection
    still works, message is just wrong/misleading.
27. DM picker (live): lists only Sensor-domain entities; both number entities absent —
    reporter's exact "expected a sensor while I have it as a number" complaint (#1).
28. Write-path sync works against a template number: "✓ Synced with NIBE: current offset
    0°C" — number.set_value → template set_value → modbus.write_register chain is viable
    end-to-end (full write test pending).

### B. Pre-existing latent bugs (found during investigation)

10. **`coordinator.py:1844`: `getattr(nibe_data, "compressor_frequency", 0)`** — field is
    `compressor_hz`; smart grid-import fallback never triggers. Fix: use `compressor_hz`.
11. **Dead `power` cache key** — `get_power_consumption` reads `_entity_cache.get("power")`
    (`nibe_adapter.py:835`) but no discovery pattern ever populates "power".
12. **`compressor_hz` 0 becomes None** — `int(compressor_hz) if compressor_hz else None`
    (`nibe_adapter.py:283`) — falsy check swallows a legitimate 0 Hz reading.
13. **`boost_dhw` never unregistered** — missing from `_async_unregister_services`
    (`__init__.py:157-178`).
14. **Discovery accepts stateless registry entries** — temperature validation is skipped when
    the entity has no state yet (`nibe_adapter.py:643 "if state:"`); orphaned/disabled registry
    entries (e.g. the reporter's `_2`-suffix leftovers, or nibe_heatpump's disabled-by-default
    coils) get cached and then read as defaults forever. Also relevant: first-match-wins in
    arbitrary registry order.
15. **Broad substring collisions** — "compressor" (status) matches `compressor_frequency…`;
    "hot_water"/"dhw" (bool status key) match temperature/amount sensors; "offset" matches any
    offset-ish entity. First registry hit wins → nondeterministic wiring.
16. **`prio`/BE-current pattern mismatch risk** — phase-current patterns include bare register
    ids (`43086` = **Prio** on F-series Modbus, not BE1 current!). Live repro: EffektGuard
    would cache `sensor.prio_43086` as phase1_current → 30 "A" → phantom ~6.8 kW power.
    (MyUplink id mapping for BE1-3 to be confirmed by register researcher.)
17. **get_current_state silently defaults** — outdoor 0.0 °C, indoor 21 °C, supply 35 °C on
    missing/unavailable entities (docstring claims ValueError, never raises); optimization then
    runs on fabricated values. At minimum should surface data quality to the coordinator.
18. **Dual model registries** — `HeatPumpModelRegistry` (options/switch) vs hardcoded
    `HEAT_PUMP_MODELS` dict (`coordinator.py:83-88`); adding a model in one place only =
    silent F750 fallback in the coordinator.
19. **strings.json step coverage gaps** — no translations for steps `model`,
    `optional_sensors`, `reconfigure`; `user` step description hardcodes
    `number.f750_cu_3x400v_offset`; unused abort string.
20. **hacs.json vs README HA version mismatch** — 2024.1.0 vs 2025.10+.

## Design direction (to be finalized after all research lands)

Priority: (a) selector domain fixes + manual entity overrides (config flow + reconfigure),
(b) honor CONF_NIBE_ENTITY as the authoritative offset target; require number domain for
offset writes, (c) extend patterns with nibe_heatpump/Modbus ids as const.py constants,
(d) robust status parsing (strings/numeric), (e) F1155 profile + single model registry,
(f) source-neutral UX strings + after_dependencies, (g) docs (README Modbus section:
nibe_heatpump recommended path; generic modbus + template-number recipe as fallback).

## Implementation (2026-07-10) — status

All fixes from sections A/A2 and most of B implemented and verified:

- const.py: `NIBE_DISCOVERY_PATTERNS` (moved from adapter, extended with verified
  nibe_heatpump + MyUplink + Modbus namings; BE currents corrected to 40079/40081/40083;
  wrong 43086/43122/43081 removed; DM 43005/40940 added), `NIBE_DISCOVERY_EXCLUDE`
  (+ own-entity exclusion), `NIBE_DISCOVERY_CORE_KEYS` (startup rediscovery),
  `NIBE_MANUAL_OVERRIDE_KEYS`, `NIBE_STATUS_ACTIVE_STATES`,
  `NIBE_COMPRESSOR_ACTIVE_HZ_THRESHOLD`, 5 new CONF_* override keys (+ reuse of
  dead CONF_DHW_TEMP_ENTITY for BT7).
- nibe_adapter.py: discovery rewritten — manual overrides seed the cache
  (CONF_NIBE_ENTITY now authoritative for the offset write target = fixes dead
  config bug #4), state-machine scan first (fixes #21), enabled-registry
  fallback (fixes #22), rank-based live-beats-stale (fixes #23), one-key-per-
  entity exclusivity with specific-before-broad ordering (fixes #15),
  offset restricted to number domain (fixes #5); `_read_entity_bool` accepts
  mapped strings (Running/Starting, fixes #6); is_heating derives from
  compressor_hz when status unparseable; compressor_hz 0 preserved (fixes #12);
  dead "power" cache branch removed (fixes #11); re-discovery while core keys
  missing (startup race found live).
- config_flow.py: DM/override selectors accept sensor+number+input_number
  (fixes #1/#27); 6 manual override fields in optional_sensors + reconfigure
  (fixes #2); reconfigure can change the offset entity and CLEAR optional
  fields (suggested_value pattern per HA dev docs — defaults made stored
  entities unclearable); offset discovery counter source-neutral (fixes #26);
  F1155 model option (fixes #7).
- coordinator.py: single model registry w/ loud fallback (fixes #18);
  compressor_frequency getattr bug fixed (#10); BT7 + startup + applied-offset
  messages source-neutral (#8).
- __init__.py: boost_dhw unregistered (fixes #13); ConfigEntryNotReady text
  source-neutral.
- models/nibe/f1155.py: new GSHP profile (registered).
- manifest.json: after_dependencies += modbus, nibe_heatpump.
- strings.json + 5 translations: steps model/optional_sensors/reconfigure added,
  user-step text source-neutral (fixes #19/#25); reconfigure_successful abort.
- README: multi-source requirements + "Local Modbus setups" recipe (template
  number wrapping modbus.write_register).
- Tests: 14 new discovery tests (tests/unit/adapters/test_nibe_discovery.py)
  covering all three sources + edge cases; fixture/count updates. Full suite:
  **1121 passed** + Black clean.

### E2E verification (real Modbus stack, live HA 2026.2.3)

- Discovery: ONE pass, 10 entities — all real Modbus temps (BT1 -3.2 °C,
  BT7 48.7 °C, …), template-number offset, DM, compressor Hz/status; BT7 found →
  DHW optimizer loads history; Prio-43086 NOT miscached; no self-discovery.
- Coordinator state: "indoor 21.3°C, outdoor -3.2°C, flow 35.8°C, DM -150" —
  real values (previously fabricated defaults 21.0/0.0/35.0).
- WRITE path: effektguard.force_offset(2) → number.set_value →
  template number → modbus.write_register → pymodbus server logged
  "WRITE register 47011 = [2]"; register readback = 2; EffektGuard log
  "✓ Applied offset to NIBE: 0°C → 2°C". Full chain proven.
- ORGANIC write: after the 60-min override expired, the autonomous
  optimization loop later wrote register 47011 = 1 on its own (overnight,
  unattended) — the complete decision→write cycle works end-to-end on a
  pure Modbus setup.
- Two NEW bugs found live during verification and fixed: discovery summary
  block misplaced into _consider_candidate (log spam per non-matching entity);
  EffektGuard discovering its own sensor.effektguard_dhw_status as
  hot_water_status.

## Month-long simulation results (Jan 2026, real weather + real Nordpool SE3)

Harness: real DecisionEngine at 5-min steps (17,856 invocations/house), RC house +
NIBE plant (calibrated curve, DM hysteresis −60/0, COP from profile), target 22.0 °C,
mode balanced, tolerance 0.5. Houses: wooden (F750, radiators, τ≈30 h) and
concrete UFH (F1155 profile, τ≈80 h). Engine wall-clock monkeypatched to sim time.

### Robustness (all green)

- 0 exceptions, 0 invariant violations in 35,712 engine calls with genuinely
  volatile real prices (33–3748 SEK/MWh).
- DM never left healthy range (min −117 wooden / −100 concrete; aux limit −1500
  never approached; aux heat 0 kWh).
- No offset oscillation (0 sign-flip windows); offsets bounded −3…+1;
  256/313 writes ≈ 8–10/day (rate limiting fine).
- Comfort hard-floors never hit (indoor min 21.07 °C; never below 18 °C).
- New F1155 profile ran the concrete house without issues.

### Finding S1 — systematic sub-target equilibrium in balanced mode (existing logic, not the diff)

| metric (31 days) | wooden opt | wooden baseline(0) | concrete opt | concrete baseline(0) |
|---|---|---|---|---|
| indoor mean | **21.34 °C** | 22.00 °C | **21.50 °C** | 22.00 °C |
| indoor min | 21.07 | 21.92 | 21.19 | 21.97 |
| energy | 537 kWh | 543 kWh | 529 kWh | 532 kWh |
| energy cost | 588 SEK | 596 SEK | 572 SEK | 583 SEK |
| net saving | **8 SEK (1.4%)** | — | **11 SEK (1.9%)** | — |

- Offset histogram (wooden): −1 for 68% of the month, −2 for 25%, 0 for 7%,
  never positive. The engine converges to a permanent ≈−1 offset: the price
  layer's persistent negative vote balances against the comfort layer's
  positive vote only after the house has drifted ~0.5–0.7 °C below target.
- Price tracking works directionally (offset↔price corr −0.36/−0.46; house
  cooler in expensive quarters), but pre-heating above target essentially
  never happens (offset max +1, 50 samples out of 1488 on concrete only).
- Consequence: ~90% of the cost "saving" is plain under-heating (less energy),
  not load-shifting. A user who set 22.0 gets a 21.3–21.5 °C home all January
  for ≈10 SEK/month.
- Suggested fixes: (a) steady-state comfort correction — integral-style term
  that escalates the comfort layer when mean deviation stays below target
  beyond N hours (the DM side has anti-windup/debt-recovery; indoor temp has
  no equivalent); (b) constrain price-layer votes to ~zero mean over a rolling
  24 h (every "reduce in expensive" paired with "raise in cheap") so
  optimization shifts load instead of lowering the setpoint.
- Caveats: single-zone RC plant; peak-tariff benefit not billed (energy cost
  only, current_peak pinned at 6 kW); mild January (min −7.8 °C); engine had
  no weather-forecast-driven pre-heat disabled/enabled changes tested.

### Physical assumptions NOT validated (treat quantitative claims accordingly)

The simulation validates control-flow behavior (no crashes, sane offsets, DM
handling, price correlation). The following remain **assumptions**, so
absolute energy/cost numbers are indicative, not verified:

- **Degree-minute estimate**: when no DM entity is available the engine
  estimates GM from flow/return deltas; the estimator has never been checked
  against a real pump's internal GM counter.
- **Climate-zone thresholds**: latitude-derived cold/mild boundaries and the
  pre-heat trigger temperatures are engineering guesses, not fitted to
  measured Swedish/Nordic building data.
- **Phase-current basis**: 3×16 A / 400 V assumptions in power estimation are
  a common Swedish service size, not read from the installation.
- **Heating-curve sensitivity**: the ±1 curve-offset → flow-temp → indoor
  response chain uses the Kuehne formula plus an RC house model; real
  emitter/zone dynamics differ per house and are not calibrated.
- **Exhaust-air COP (F730/F750)**: COP curves are keyed on outdoor
  temperature; an exhaust-air machine's COP depends on exhaust-air and flow
  temperatures, so F750 simulation results are control-flow evidence only.
- **Learned heat-loss "coefficient"**: derived from indoor cooling rate
  without a thermal-capacitance/energy term — dimensionally a relative decay
  index, NOT W/°C. Quarantined in code (2026-07): logged as relative index,
  must not feed absolute calculations.

## Adversarial code review of the change (8 angles, 6 agents + fixes)

All confirmed findings were fixed in-tree with regression tests; suite grew
1107 → 1129 passing.

Review-found bugs in the new code (fixed):
- R1 misplaced discovery summary block executed per non-matching entity
  (log spam ×18) → moved to end of discovery; logs only when results change.
- R2 EffektGuard discovered its own sensor.effektguard_dhw_status as
  hot_water_status → own entities excluded by registry platform (not name).
- R3 ranks were rebuilt per pass, so live-beats-stale never worked across
  passes → ranks persist on the adapter (self._entity_ranks).
- R4 re-discovery stopped once core keys were merely PRESENT, even if
  registry-only (stale ids permanently blocked live `_2` entities — the exact
  issue-#18 scenario; confirmed by runtime simulation) → core keys held at
  registry-only rank count as unresolved.
- R5 hard 12-attempt cap prevented recovery when the source integration
  appears >1 h after start → slow-retry fallback (every 6th cycle) after the
  fast attempts; change-only logging keeps it silent.
- R6 claimed-entity early-return skipped rank refresh → bindings never
  upgraded registry→live and stayed stealable → rank refresh added.
- R7 "_bt2"/"bt2_" would substring-match BT20/21/22 exhaust sensors → dropped
  (BT2 covered by supply_temp/supply_line/40008).
- R8 "room_temp" matched bedroom_temp/storeroom_temp → dropped.
- R9 is_heating derived from Hz counted DHW-only runs as heating →
  gated on `not is_hot_water`.
- R10 unregistered template lookalikes could beat real integration entities →
  new rank LIVE_UNREGISTERED between live-registered and registry-only
  (generic-Modbus-only setups still discover fine).
- R11 "Found N NIBE offset entities" counted TRV/Tado calibration offsets →
  counter uses NIBE-specific patterns only (selector unchanged).
- R12 coordinator read entity_cache["offset"] before discovery (offset sync
  dead on boot; pre-existing) → manual overrides seeded in the constructor.
- R13 switch discovery lost the registry fallback (ventilation switch missed
  if its platform loads late) → registry entries included again.
- R14 conventions: _TEMPERATURE_KEYS/_RANK_* class constants (underscore
  style used nowhere else) → moved to const.py (NIBE_TEMPERATURE_KEYS,
  NIBE_DISCOVERY_RANK_*); MANUAL_SENSOR_OVERRIDE_FIELDS deduplicated to
  derive from const.NIBE_MANUAL_OVERRIDE_KEYS; f1155 profile now subclasses
  the S1155 profile (no duplicated formula bodies); duplicated test mock
  helper moved to tests/conftest.py (make_mock_async_all); en-copies in
  da/fi/no/sv reverted (HA falls back to English per-key); AdapterConfigDict/
  EffektGuardConfigDict extended with the override keys.
- R15 pre-existing: scripts/run_all_tests.sh arg parser rejected the
  documented categories (optimization/climate/learning/models/dhw/effect) →
  parser fixed; new `simulation` category added.

Verified-harmless review candidates (no action): dropped numeric patterns
43086/43122/43081 were factually wrong registers (Prio/compr-min-freq/add-op-
time), BE currents covered by current_be1/_be1 names; reconfigure explicit-
None clearing is intended and documented; data_updates None values are
truthiness-guarded everywhere.

## Testing procedure additions (kept in repo per maintainer request)

- tests/unit/adapters/test_nibe_discovery.py — 20 discovery tests covering
  nibe_heatpump/MyUplink/generic-Modbus naming, manual overrides, disabled/
  stale registry entries, startup races, rank displacement, status parsing.
- scripts/simulation/ — month-long optimization simulation (real
  DecisionEngine + plant model + real Jan-2026 weather/price data committed
  under scripts/simulation/data/). `bash scripts/run_all_tests.sh simulation`
  runs the fast self-test; full run: `python scripts/simulation/sim_harness.py`.
  Configs: wooden_f750 (NIBE F750) and concrete_f1155 (NIBE F1155).

## Live devbox instance (left running)

HA at http://localhost:8125 (hass -c /workspace/.ha-config, log
.ha-config/ha-run2.log). Modbus simulator on 127.0.0.1:5020
(scripts/simulation/nibe_modbus_simulator.py, logs register writes). EffektGuard configured
with the Modbus-backed entities, hot-water optimization ENABLED with the real
modbus temp-lux switch (switch.temporary_lux_48132, register 48132); DHW
optimizer planning on real BT7 data.

## WORKPLAN (maintainer directives 2026-07-11) — COMPLETED except W2

Clarified scope: Modbus is ADDED alongside MyUplink (MyUplink stays a
first-class source — nothing removed), and verification must be GENERAL
(whole integration), not modbus-only.

- [x] W1 MyUplink proven unchanged: TestMyUplinkF750LegacyNaming locks the
      maintainer's own f750_cu_3x400v_* naming — all 14 keys resolve, state
      reads verified (plus the HA-core "gotham_city" MyUplink naming tests).
- [x] W2 ha-docs-study COMPLETE (6/6 areas after two session-limit retries).
      The final two areas produced 16 violations; applied immediately:
      - D1 (both agents, safety-relevant): all write-path service calls were
        fire-and-forget (blocking=False) recording phantom success — offset
        write, ventilation switch, and 3 DHW temp-lux calls now use
        blocking=True with HomeAssistantError caught, so out-of-range/
        unavailable/cloud failures surface and rate-limit state is not set
        on failed writes.
      - D2: offset bookkeeping synced from the entity only once ever; now
        re-syncs when the entity disagrees and no write happened within
        NIBE_OFFSET_RESYNC_MINUTES (external changes / failed writes heal).
      - D3: offset write now validates against the target number's own
        min/max attributes (a template number left at default 0-100 would
        reject every negative offset invisibly).
      - D4: DM discovery's 'myuplink-in-entity-id' gate was dead code (HA
        myuplink ids carry the DEVICE name, never 'myuplink') — replaced
        with a registry-platform check (myuplink/nibe_heatpump).
      - D5: README Modbus recipe hardened per template docs: unique_id,
        availability guard (has_value), float(0) default, min/max required
        note, write-lag explanation.
      - D6: 'Premium subscription required' softened to the docs' wording
        ('may need a valid myUplink subscription') in README + docstring.
      Noted as follow-ups (architectural, not applied): services should be
      registered in async_setup and never unregistered; coordinator should
      pass config_entry and use runtime_data instead of hass.data; the
      hand-rolled clock-aligned scheduler bypasses documented coordinator
      machinery; raw bus.async_listen('state_changed') should be
      async_track_state_change_event; startup-grace placeholder data defeats
      ConfigEntryNotReady retry backoff; dead try/except around
      async_config_entry_first_refresh.
- [x] W3 Egress opened: weather REPLACED with official open-meteo archive API
      data (min now -11.6 °C; S3 extraction had mean 0.97 °C deviation);
      prices REBUILT from the official Swedish API (elprisetjustnu.se, native
      SEK, 96 quarters/day; prior FX-converted dataset validated to 0.13%).
- [x] W4 Live stack now simulates TWO pumps: F1155 (modbus unit 1) + F750
      (unit 2: sensor.f750_bt1_...=-3.2 °C, DM -85, own temporary-lux switch).
      EffektGuard entry pinned to the F1155 via the new override fields —
      demonstrates multi-pump disambiguation live. ("f1150" in the directive
      read as F1155, which was already simulated.)
- [x] W5 Scenario matrix (real data, 31 days):
      | scenario | indoor_min | dm_min | cost SEK | note |
      |---|---|---|---|---|
      | balanced wooden_f750 | 21.02 | −148 | 627 | reference |
      | comfort wooden_f750 | 21.05 | −164 | 629 | ≈balanced (weak mode contrast) |
      | savings wooden_f750 | **20.03** | −252 | **639** | S2: costlier AND colder |
      | coldsnap wooden_f750 (−23.6 °C week) | 20.77 | −148 | 725 | safe, aux 0 |
      | balanced concrete_f1155 | 21.20 | −105 | 600 | reference |
      | savings concrete_f1155 | 20.43 | −235 | 601 | S2 pattern repeats |
      FINDING S2: savings mode produced HIGHER cost than balanced (639 vs
      627 SEK) while dipping 2 °C below target and thrashing the offset
      (−10..+1, 439 writes) — under-heat-then-recover wastes more energy than
      steady operation. Comfort mode barely differs from balanced (mode
      differentiation weak on the comfort side). Cold snap: no violations,
      no aux, DM healthy — safety layers hold in deep cold.
- [x] W6 -32768/-3276.8 unknown-value markers rejected in _read_entity_float
      (NIBE_UNKNOWN_VALUE_MARKERS).
- [x] W7 Temp-lux discovery accepts platform nibe_heatpump and register 48132.
- [x] W8 heat_pump_model changeable via Reconfigure (keep-old-if-empty;
      HEAT_PUMP_MODEL_OPTIONS single source for both steps).
- [x] W9 hacs.json homeassistant: 2024.1.0 → 2025.10.0 (matches README).
- [x] W10 Native sv/da/no/fi translations written for every new config-flow
      key (model/optional_sensors/reconfigure steps, user step, aborts).
- [x] W11 Docs: dead IMPLEMENTATION_PLAN references removed from
      docs/dev/README.md; nibe_adapter description and offset-application
      wording made multi-source. (DHW_OPTIMIZATION.md still carries legacy
      IMPLEMENTATION_PLAN reference lines — cosmetic, noted.)
- [x] W12 Final: Black clean, **1131 tests passed**, simulation selftest green.
- [x] W13 (added directive) Latest HA + Python 3.13: venv already runs
      HA 2026.2.3 — the newest release on PyPI — on Python 3.13.14. No newer
      2026.x exists; nothing to upgrade.

## Regression audit (prompted by maintainer 2026-07-11)

Re-verified the highest-risk behavior changes rather than trusting the green suite.

- REGRESSION FOUND + FIXED: the entity min/max clamp did `float(state.attributes["min"])`
  OUTSIDE the try block. A malformed number entity raising TypeError would have
  escaped set_curve_offset; the coordinator's two call sites catch
  (HomeAssistantError, AttributeError, OSError, ValueError) but NOT TypeError, so it
  would have propagated into the update cycle. Fixed: defensive float() with
  KeyError/TypeError/ValueError fallback. New file tests/unit/adapters/test_nibe_write_path.py
  (9 tests) covers this + the blocking write + clamp + unknown-value marker.
- VERIFIED LIVE: blocking=True offset write end-to-end (the biggest behavior change,
  which never got the 8-angle review). force_offset(4) → "✓ Applied offset to NIBE:
  0°C → 4°C" → simulator "WRITE register 47011 = [4]". The initial attempt was
  correctly refused by the startup-grace period (not a regression).
- VERIFIED: both coordinator call sites (coordinator.py:1001, :1934) wrap
  set_curve_offset in try/except; reads produce real values live (indoor 21.3,
  outdoor -3.2, DM -150), not silent defaults.
- Suite now 1140 passed (1131 + 9 write-path), Black clean.

RESIDUAL RISK (honest boundary — cannot be ruled out in this environment):
- The suite is mock-based; timing/integration regressions do not surface in it.
  The live stack is the real test but exercises only ONE config (F1155 via generic
  Modbus + template numbers). NOT live-tested: real MyUplink cloud (needs OAuth
  creds), the actual nibe_heatpump integration (not installed), and EffektGuard
  driving the F750 (unit 2 exists in the simulator but the entry is pinned to F1155).
- blocking=True makes the coordinator update WAIT for the write round-trip. Instant
  for local Modbus; for MyUplink cloud it adds the API latency to the cycle. More
  correct, but a timing change unverifiable without cloud access.

### Still open / follow-ups (not blocking issue #18)

- get_current_state silent defaults (#17): surfaced but NOT redesigned —
  needs a data-quality flag consumed by the coordinator (breaking-ish change).
- Modbus/nibe_heatpump temp-lux switch: works via manual switch selection
  (nibe_heatpump exposes switch.temporary_lux_48132); auto-discovery of it
  still MyUplink-flavored in config_flow (_discover_temp_lux_entities).
- -32768 "unknown value" guard for raw Modbus sensor glitches.
- S-series myuplink parameter-id space unverified (3 distinct ID spaces exist).
- hacs.json (2024.1.0) vs README (2025.10+) HA version mismatch.
- strings for da/fi/no/sv: new keys added in English pending translation.

## Test stack (devbox, real e2e)

- Simulator: `scripts/simulation/nibe_modbus_simulator.py` (pymodbus 3.11.2, port 5020, logs writes).
- HA 2026.2.3 at :8125, `modbus:` hub "nibe" + template numbers in `/workspace/.ha-config/configuration.yaml`.
- Debug logging: effektguard+modbus+pymodbus+template at debug (targeted, disk-aware).
- Month-long January-2026 simulation (scripts/simulation/sim_harness.py): drives the REAL
  DecisionEngine at 5-min steps against an RC house + NIBE plant model
  (curve/DM/compressor-hysteresis/COP), two configs (wooden+F750 radiator,
  concrete-UFH+F1155), target 22 °C.
  Data (scripts/simulation/data/, provenance verified, nothing fabricated):
  - weather_jan2026.json: 744 hourly Stockholm temps, ERA5-Land via Open-Meteo's
    public S3 bucket (cross-checked vs ERA5 0.25°: mean |diff| 0.42 °C);
    -7.75..+2.75 °C, mean -2.0 °C.
  - prices_jan2026.json: real Nordpool SE3 day-ahead 15-min prices, 31×96,
    EUR-native (scraper repo, 960-point cross-validation, 0 mismatches) ×
    daily Fed H.10 EUR/SEK (~0.3% of ECB); 33..3748 SEK/MWh, mean 1084.
    Daily temp↔price correlation -0.40 (cold→expensive, sanity ✓).

## External review outcome (2026-07-11)

Independent review of PR #19 confirmed the implementation and found:

- **High (fixed): DHW priority classified as space heating.** With nibe_heatpump
  naming, no entity matches the hot_water_status patterns, so a DHW-priority
  run (prio 43086 = "Hot Water") with compressor "Running" read as
  is_heating=True / is_hot_water=False. Fix: new `prio` discovery key with
  value mapping (mapped strings, raw numeric 20/30; unknown values are
  conservative no-ops for undocumented MyUplink enums); priority now overrides
  the broad status reads and a hot-water-priority run never counts as heating.
  Tests cover mapped and raw numeric priorities plus the unknown-enum case.
- **Medium (fixed, pre-existing): manual offsets were not authoritative.**
  effektguard.force_offset(0) after +4°C was deferred 45 min by the
  volatile-reversal blocker. Fix: OptimizationDecision.is_manual_override flag;
  the coordinator applies user commands immediately and re-baselines the
  volatility tracker; automatic reversals remain blocked (regression tests for
  both).
- **Medium (addressed): PR scope.** Feature PR now contains only the source-
  support change + tests; the simulation harness, this findings record, and the
  governance rule live in a separate dev-tooling PR (kept, per maintainer — no
  work lost).
- Register-map traceability: references now cite the exact upstream path
  (yozik04/nibe, nibe/data/f1155_f1255.json).

## External review #2 outcome (repo-wide, 2026-07-11)

A second, repo-wide review reported 8 findings; each was verified in code
before fixing. Fixes live on `fix/measurement-and-unit-defects` (stacked on
the multi-source branch), one commit + regression test per finding:

- **R1 (critical, fixed): kW meters double-divided.** The coordinator divided
  the external meter reading by 1000 unconditionally; a kW-unit meter went
  through W→kW twice (6.0 kW became 0.006). Now converted once, by unit
  attribute.
- **R2 (high, fixed): effect-tariff quarters recorded instantaneous samples.**
  The Swedish effektavgift bills the 15-minute MEAN; the coordinator now
  accumulates samples within the quarter and records the mean at the boundary.
- **R3 (high, fixed): pump profiles fed W/K into a kW-basis formula.** The
  Kuehne flow-temp term computed ~2200 °C and was permanently masked by the
  efficiency clamp; profiles now convert to kW like weather_layer does.
- **R4 (documented): exhaust-air COP limitation** — see the physical
  assumptions section above and the profile docstrings.
- **R5 (quarantined): learned heat-loss coefficient is not W/°C** — relabeled
  a relative index; verified no consumer feeds it into the control path.
- **R6 (medium, partially fixed): savings math assumed öre/kWh.** Price-unit
  conversion now honors the GE-Spot unit attribute (öre/cent subunits ÷100,
  main units as-is), so price ranking and a single-currency spot calculation
  no longer apply an unconditional ÷100. See the open global-currency follow-up
  below before treating the resulting amount as SEK.
- **R7 (medium, fixed): DST days.** 92/100-quarter days are normalized
  honestly: duplicates dedupe first-occurrence-wins, gaps forward-fill from
  the nearest real price instead of a fabricated day-average, and empty days
  stay empty. The dense-96 shape is kept because the coordinator indexes
  today[current_quarter] positionally.
- **R8 (medium, fixed): DHW next-opportunity constraint algebra** — mandatory
  lower bounds now use max(), cheap-window candidates are clamped inside
  them, and the cooling deadline caps the result.
- **R9 (open, global-currency/tariff reporting):** Price optimization itself is
  currency-invariant: percentiles, price ratios, and cheapest-window selection
  work with any configured currency or subunit. Monetary reporting is not yet
  global. `SavingsEstimate` and the Estimated Monthly Savings sensor are SEK,
  while spot feeds can be EUR, NOK, DKK, or another currency; mixing a spot
  amount in one currency with Swedish effect-tariff savings in SEK is invalid.
  Current safe behavior must either report spot savings in its source currency
  separately, or omit it from a SEK total. A complete design needs explicit
  currency metadata for every monetary value, a configurable local effect-tariff
  model (or disabled tariff estimate), and an explicit FX source/date before a
  combined total is shown. Deferred by maintainer request (2026-07-11).
- Also: dead demo/simulation scripts removed (compileall-breaking
  triple-quote, hardcoded workspace paths); harness comfort accounting
  tightened to the configured tolerance with a `--baseline` matched-neutral
  mode and quarter-mean tariff stats (this PR).
