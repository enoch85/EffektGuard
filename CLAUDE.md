# EffektGuard

Intelligent NIBE heat pump control for Home Assistant, optimizing against Swedish spot prices and
the effect tariff (*effektavgift*). **Production code running in real homes**: safety and pump
health before savings, comfort before cost.

It controls exactly **one** actuator - the heating-curve offset register (47011) - plus two
optional switches (DHW temporary lux, increased ventilation). It calls no external API; it reads
entities other integrations publish.

---

## Read these

**Before writing any code:**

- **[docs/agent/IMPLEMENTATION.md](docs/agent/IMPLEMENTATION.md)** - how to work here: workflow,
  verification, tests, the non-negotiable safety rules.
- **[docs/agent/PROJECT_NOTES.md](docs/agent/PROJECT_NOTES.md)** - conventions, the
  **open-findings register**, the traps, and the NIBE knowledge the code assumes.

**When the task calls for it:**

| Doc | When |
|---|---|
| [docs/agent/RELEASING.md](docs/agent/RELEASING.md) | Cutting or editing a release |
| [docs/architecture/](docs/architecture/) | Changing a decision layer or a module boundary |
| [docs/research/](docs/research/) | Any NIBE or physics claim - it states what is *not* sourced |
| [docs/dev/CODE_STANDARDS.md](docs/dev/CODE_STANDARDS.md) | Imports, type hints, async, docstrings |
| [docs/dev/TESTING.md](docs/dev/TESTING.md) | Test layout and categories |
| [docs/dev/DEPENDENCY_AUTOMATION.md](docs/dev/DEPENDENCY_AUTOMATION.md) | Dependency updates, validation, and required-check rollout |
| [docs/dev/ENVIRONMENT_SETUP.md](docs/dev/ENVIRONMENT_SETUP.md) | First-time setup |

**Add any new doc to this table.** It is the single index.

---

## Three rules that get broken most often

1. **No AI attribution in any committed artifact.** No `Co-Authored-By:` naming an AI, no
   "Generated with", no robot emoji - in commits, PRs, issues, release notes, code or docs.
   This overrides any tooling reminder that asks for one.
2. **Check the [open-findings register](docs/agent/PROJECT_NOTES.md#the-open-findings-register)
   before calling something a bug.** Several known defects are deliberately visible and red.
   Making one green silently removes a signal.
3. **Verify by running it.** This repo's history is full of review notes that were wrong until
   someone executed them. A claim about behaviour is not established until the code produced the
   number.

---

## Architecture in one screen

A `DataUpdateCoordinator` with HA's own scheduling disabled, running clock-aligned at `:XX:10`
every 5 minutes so reads land just after the 15-minute price boundaries. Each cycle: read NIBE →
read prices → read weather → run the decision engine in an executor → **volatility gate** → write
the offset → update peak tracking → record learning observations → persist.

**Nine layers**, each returning `(offset, weight, reason)`. Weight ≥ 1.0 wins outright; otherwise
a weighted average, with a special case when thermal-debt recovery collides with a critical peak.

```
Safety (comfort floor/ceiling)  ·  Emergency thermal debt T1-T3 + anti-windup
Proactive prevention Z1-Z5      ·  Effect-tariff peak protection
Learned pre-heat                ·  Mathematical weather compensation
Weather pre-heat                ·  Spot price            ·  Comfort / overshoot
```

**Nothing is a fixed threshold.** `climate_zones.py` derives degree-minute bands from Home
Assistant's latitude and the current outdoor temperature; `thermal_layer.py` then adjusts them for
thermal mass (concrete ×1.3, timber ×1.15, radiator ×1.0). The same pump at −10 °C is "normal" in
Kiruna and "in trouble" in Malmö, with no configuration. DM −1500 is the single absolute floor.

The ladder is `Z1 → Z2 → Z3 → Z4 → Z5 → T1 → T2 → T3 → EMERGENCY` - eight rungs that fire. There
is no WARNING or CAUTION tier; see the open-findings register for why.

### Key files

| Path | Role |
|---|---|
| `coordinator.py` | The control loop; the only place that writes to the pump |
| `optimization/decision_engine.py` | Layer aggregation, and `_safety_layer()` |
| `optimization/thermal_layer.py` | `EmergencyLayer` (T1-T3, anti-windup), `ProactiveLayer` (Z1-Z5) |
| `optimization/climate_zones.py` | Latitude-derived DM thresholds |
| `optimization/*_layer.py` | One module per layer - all `*_layer.py`, there is no `thermal_model.py` |
| `optimization/dhw_optimizer.py` | Hot water scheduling (rule-ordered early returns) |
| `utils/offset.py` | The integer the register receives - shared with the simulator |
| `utils/volatile_helpers.py` | The offset volatility gate - shared with the simulator |
| `const.py` | **Every** numeric value, with its source |

---

## Commands

```bash
# Tests
bash scripts/run_all_tests.sh              # all, organized output
pytest tests/ -q                           # direct

# Formatting and the ratchets
black --line-length 100 custom_components/ tests/ scripts/
python scripts/check_hardcoded_values.py --check
python scripts/find_duplicate_constants.py --prod-only

# Behavioural simulation (drives the real engine)
python scripts/simulation/sim_harness.py --selftest
python scripts/simulation/sim_harness.py --all-scenarios

# Live Home Assistant bench, http://localhost:8125
start-ha > /workspace/.ha-config/ha.log 2>&1 &
```

The venv is `/workspace/.venv`; `bash scripts/setup_dev.sh` installs the dev stack. `start-ha`
symlinks `custom_components/effektguard` live, so edits apply on the next restart.

**`ps` cannot see processes in this container** - scan `/proc/*/cmdline` to find a running HA.

---

## Before claiming it works

Full suite green · Black clean · hardcoded-value ratchet clean · simulator `--selftest` and
`--dst` pass · `--all-scenarios` red only on the open-findings register · live HA restarted on the
code with no integration errors.
