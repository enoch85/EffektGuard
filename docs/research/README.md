# Research

This directory holds the evidence for the numbers in `const.py`.

`.github/copilot-instructions.md` carries a binding rule: **never guess NIBE behaviour, verify it
against research.** For most of this project's life that rule could not be obeyed. The code and the
docs cited **fifteen** research documents as the authority for safety-critical thresholds —
`IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md`, `Swedish_NIBE_Forum_Findings.md`,
`COMPLETED/DHW_RESEARCH_FINDINGS.md`, and a dozen more — and **every one of them is absent from the
repository.** They are gitignored internal notes. Anyone cloning this repo inherited a set of
safety limits whose justification they could not read, check, or challenge.

So the citations here are to things you can actually obtain: published European standards, NIBE's
own manuals, and calculations reproduced in full so you can redo them.

## The rule

**A constant that governs heating behaviour needs a source in this directory, or a comment saying
honestly that it is a guess.** A number with a confident-sounding citation to a document nobody has
is worse than a number with no citation at all: it looks settled.

## Contents

| | |
|---|---|
| [01_degree_minutes.md](01_degree_minutes.md) | What DM is, NIBE menu 4.9.3, and why the auxiliary heater — not −1500 — is the number that governs a real F750 |
| [02_emitter_law.md](02_emitter_law.md) | EN 442 / EN 1264. How flow temperature is derived, validated against NIBE's own published curve |
| [03_concrete_slab_response.md](03_concrete_slab_response.md) | Why a concrete slab needs a 24-hour forecast horizon and a +2.0 °C pre-heat |
| [04_exhaust_air_recovery.md](04_exhaust_air_recovery.md) | Why "extra heat extracted" and "improved COP" are the same joules, and what that does to the airflow feature |

## What is *not* in here

Several numbers in `const.py` still rest on forum anecdote rather than on anything citable — the
`stevedvo` and `glyn.hudson` case studies, the "Swedish forums validated −1500" claim. They are
marked as such where they appear. **Do not launder them into facts by citing this directory.**
