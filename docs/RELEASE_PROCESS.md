# Release mechanics

What `scripts/release.sh` and `.github/workflows/release.yml` do. The judgement around them -
notes format, pre-release flips, re-cutting, the gates before cutting - is in
[docs/agent/RELEASING.md](agent/RELEASING.md).

## Tags

`vMAJOR.MINOR.PATCH`, optionally with a suffix: `v0.7.2`, `v0.7.2-beta.1`, `v0.7.2-rc.1`. The
script accepts `^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$`. A suffix containing `-alpha`,
`-beta` or `-rc` is a **pre-release** to both the script and the workflow.

## `scripts/release.sh`

```bash
bash scripts/release.sh v0.7.2-beta.1      # pre-release, from any branch
bash scripts/release.sh v0.7.2 --prod      # production, from main
```

Needs git push access, `jq` and GNU grep; `--prod` also needs an authenticated `gh`.
`--pr-only` is accepted and does nothing.

In order, the script:

1. Validates the tag format.
2. Requires branch `main` or `initial-implementation`, unless the tag is a pre-release.
3. Prompts on uncommitted changes (`y` continues).
4. Aborts if the tag already exists locally.
5. `git pull --rebase`.
6. Sets `"version"` in `custom_components/effektguard/manifest.json` to the tag, `v` included
   (`"v0.7.1"`), and the README badge to `version-0.7.1-blue`.
7. If either file changed: commits `Release vX.Y.Z of EffektGuard` and pushes the current branch.
   A re-run with both already set pushes no commit.
8. Creates the annotated tag (message `Release vX.Y.Z`, plus `(Pre-release)` for a pre-release)
   and pushes it. The workflow does the rest.

`--prod` refuses a pre-release suffix and, before step 1: `git checkout main`, `git pull --rebase`,
`bash scripts/run_pytest.sh` (`pytest tests/ -v --tb=short`, must pass), then deletes every
`vX.Y.Z-beta*` tag locally and on `origin` and every matching GitHub release. Alpha and rc tags are
left alone. `bash scripts/list_beta_releases.sh vX.Y.Z` shows what would be deleted.

Error messages and what they mean:

| Message | Cause |
|---|---|
| `Invalid version tag format!` | Tag does not match the pattern above |
| `You are not on the main or initial-implementation branch!` | Final tag from a feature branch |
| `Tag vX.Y.Z already exists!` | Local tag exists; the script never re-tags |
| `Manifest file is not valid JSON` | Also printed when `jq` is not installed |
| `Tests failed!` | `--prod` test run failed |
| `GitHub CLI (gh) is not installed!` | `--prod` without `gh` |
| `Production releases cannot have pre-release suffixes` | `--prod` with `-beta`, `-alpha` or `-rc` |

## `.github/workflows/release.yml`

Runs on a pushed tag matching `v*`:

1. Zips `custom_components/effektguard/`, `README.md`, `LICENSE` and `hacs.json` into
   `effektguard.zip`.
2. `gh release create <tag>` titled `EffektGuard <tag>`, with the zip attached and GitHub's
   generated notes (`--generate-notes`). A tag containing `-alpha`, `-beta` or `-rc` gets
   `--prerelease`; any other tag gets `--latest`.

Rewrite the generated notes in the format in [RELEASING.md](agent/RELEASING.md). To ship a plain
`vX.Y.Z` as a pre-release, flip it after the workflow has created it (same file).

## Before a tag: `validate.yml`

`.github/workflows/validate.yml` runs on pull requests, merge groups and pushes to `main` (and
`initial-implementation`), not on tags: Black and the hardcoded-value ratchet; `pip check`,
`compileall`, pytest, the simulator (`--selftest`, `--dst`, `--all-scenarios`); a HACS
distribution build, HACS and hassfest validation; workflow lint and the development-container
build.
