# How to work in this repository

This is production code that controls heat pumps in real homes. **Safety and correctness over
speed**; comfort and pump health over savings. When uncertain about NIBE behaviour, ask and verify
rather than implement a plausible guess.

Read [CLAUDE.md](../../CLAUDE.md) for the map and commands, and
[PROJECT_NOTES.md](PROJECT_NOTES.md) for the conventions, the
open-findings register, and the traps - **before** writing code.

---

## Workflow

1. **Reproduce before changing behaviour.** Establish what the code does today by *running* it -
   the real decision engine, the simulator, or a live Home Assistant. Record the input, the
   expected result, and the observed result. If you cannot reproduce it, say so; do not present a
   suspected defect as confirmed.

2. **Trace the whole flow.** Search repo-wide for callers, constants, tests and docs that touch
   it. Read git blame and prior PRs to tell a deliberate decision from a defect. Check the
   [open-findings register](PROJECT_NOTES.md#the-open-findings-register) before
   calling anything a bug - it may be deliberately open.

3. **Define what must not change.** Name the existing behaviour the change has to preserve, and
   how you will know it did. For a tuning constant, that usually means proving the new expression
   is bit-identical.

4. **Make the smallest complete fix.** Reuse what exists. Add constants, abstractions or
   dependencies only when the requirement needs them. Fix what the outcome requires; report
   independent findings separately rather than folding in opportunistic refactors.

5. **Verify by executing, not by reading.** A claim about behaviour is not established until the
   code has run and produced the number. This repo's history is full of review notes that were
   wrong until someone ran them - three verdicts in one audit, two of them the reviewer's own.

6. **Validate the diff.** Focused tests during development, then the full suite, Black, and the
   hardcoded-value ratchet before committing. See [CLAUDE.md](../../CLAUDE.md#commands).

7. **Finish within the authorized scope.** When commit or push is authorized, complete it without
   asking again. Honour "do not push" or "leave uncommitted". Report what you ran and what you
   skipped.

---

## Verification

Facts only. Each of these has caught a real defect:

- **Run the code.** Sweep the value one step at a time rather than reasoning about a threshold.
- **Mutate the guard.** A guard test that survives its guard's deletion is a test that cannot
  fail. Delete it, watch red, restore - and **assert the mutation's anchor first**, or a no-op
  mutation will report a false pass.
- **Use the live bench.** `start-ha` gives a real Home Assistant on `localhost:8125` with this
  integration symlinked live. Config flow, entity attributes, repairs and statistics are only
  observable there. Check `last_updated` on an entity before trusting it, and confirm no stale
  instance holds the port (see the traps in the project notes).
- **Use the simulator.** `scripts/simulation/sim_harness.py` drives the real decision engine
  against real ERA5 weather paired with the real Nord Pool prices of the same days, across five
  houses. `--all-scenarios` is the sweep. A scenario that cannot fail is a demo: every one carries
  an `expect` tuple.
- **Check external APIs against the source**, not memory. For Home Assistant, read the symbol at
  the tag you claim to support.

---

## Tests

A test records behaviour we approved. It earns its place only if changing the behaviour it pins
would break something real, and no existing test already catches that change.

**Worth a test:** a safety guard, a threshold-to-response mapping, a fixed bug (name the finding).

**Not worth a test:** constants and labels, log text and log levels, class names, argument
pass-through, call counts, a library's own behaviour. Do not export a helper just to test it -
test it through the unit that uses it.

**Keep it honest.** One test per behaviour; another input on the same branch is a parametrize row,
and only when it can fail *differently*. Test at the boundary - `evaluate_layer`, not the private
method underneath. Tests must pass in any order, use no real network and no real clock, and write
nothing outside a temp folder.

Every simulator constant must say where it came from: `SOURCED` with a citation, or `ASSUMED` with
a measured sensitivity. `tests/validation/` enforces this.

---

## Safety rules that are not negotiable

- Thermal-debt thresholds come from `ClimateZoneDetector`, never hardcoded - they are
  latitude- and temperature-derived, then adjusted for thermal mass.
- All optimization respects the thermal-debt limits and the comfort floor.
- Nothing writes to the pump outside the guarded write path. Reads never drive it, and an
  unloaded integration never writes.
- An offset that cannot be confirmed is not reported as applied.
