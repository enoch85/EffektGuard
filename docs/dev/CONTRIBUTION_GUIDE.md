# Contributing

Production code in real homes: safety and pump health before savings, comfort before cost. The
rules are in [CLAUDE.md](../../CLAUDE.md), [IMPLEMENTATION.md](../agent/IMPLEMENTATION.md) and
[PROJECT_NOTES.md](../agent/PROJECT_NOTES.md). This page is the sequence.

## 1. Set up

[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md). Then read `docs/architecture/00_overview.md`, and
check the [open-findings register](../agent/PROJECT_NOTES.md#the-open-findings-register) before
calling anything a bug: some reds are deliberate.

## 2. Branch

```bash
git checkout -b fix/short-description      # or feature/...
```

One change per branch, and one pull request in progress at a time.

## 3. Change

- Reproduce first: run the engine, the simulator or the live bench, and record what the code does
  today.
- Read the whole file you edit. Trace every caller, constant, test and doc the change touches.
- New numbers go in `const.py` with their source. A claim about NIBE behaviour cites
  `docs/research/`; if nothing there says it, ask in an issue instead of guessing.
- Tests: what earns one is in IMPLEMENTATION.md (Tests). A fixed bug's test fails on `main` and
  passes on the branch. Name the file for the behaviour, e.g.
  `tests/unit/dhw/test_a_slow_tank_is_heated_in_time.py`.
- Docs: update what the change makes false (`README.md`, `docs/`), and add any new doc to the table
  in CLAUDE.md.

## 4. Check

```bash
black --line-length 100 custom_components/ tests/ scripts/
python scripts/check_hardcoded_values.py --check
bash scripts/run_all_tests.sh
python scripts/simulation/sim_harness.py --selftest
python scripts/simulation/sim_harness.py --dst
```

The complete gate list, including `--all-scenarios` and the live bench, is in CLAUDE.md ("Before
claiming it works").

## 5. Commit

The subject is a plain sentence saying what the change does; no `feat:`/`fix:` prefix. The body
says why and carries the evidence: what was run and what the numbers were. No AI attribution. From
the history, abridged:

```
Pre-heat on a colder day, not a colder night

The pre-heat trigger compared the forecast's coldest hour with the current outdoor temperature, so
an ordinary night fired it: a May night reads 8 C below the afternoon. At +2.0 that pre-heated mild
weather, and shoulder_may2024 went red on comfort.

A cold snap is a colder day. The trigger now compares the measured outdoor mean over the past 24 h
with every coming 24-hour window that starts within the horizon. [...]

Tests: a May day-night forecast no longer fires it, with or without a measured past day; [...]
```

## 6. Pull request

Pull the latest `main` into the branch, push, and open the PR on GitHub. Say what changed, why,
what you ran (the gates above) and any safety implication. Review looks for the same things:
constants rather than literals, tests that fail without the fix, research references for NIBE
behaviour, Black, docs updated. The maintainer merges.

## Questions

Open a [GitHub issue](https://github.com/enoch85/EffektGuard/issues) with logs and, for NIBE
behaviour, the manual page or research document you rely on.
