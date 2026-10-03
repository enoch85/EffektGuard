# Releasing

Task-specific: read this when cutting or editing a release. The mechanics live in
[RELEASE_PROCESS.md](../RELEASE_PROCESS.md); this file is the judgement around them.

---

## Release notes format

**Read an existing release before writing one** (`gh release view v0.5.0 --json body -q .body`).
Copy that format. Do not invent a structure, and do not guess at a length.

```markdown
## Key Changes

### Breaking
• **Short bold label**: one line

### Bug Fixes
• **Short bold label**: one line, symptom first, cause as a clause

### Enhancements
• **Short bold label**: one line

### Removed
• `CONSTANT_NAME` - reason

**PRs**: #n, #n

**Full Changelog**: https://github.com/enoch85/EffektGuard/compare/vX.Y.Z...vX.Y.W
```

- Bullets are `•`. Backticks for symbols. Omit empty sections. `### Breaking` goes first.
- **One line per item.** No second sentence, no callout blocks, no known-issues essay, no
  explanation of the release process. GitHub already shows the Pre-release badge.
- The reasoning, the measurements and the finding IDs belong in the PRs and commit messages,
  which stay fully detailed. The release note is for someone glancing at it.
- **Source of truth is the diff between tags**, not a PR body:
  `git log --oneline --no-merges vX.Y.Z..vX.Y.W` and `git diff vX.Y.Z..vX.Y.W`.
- **Describe the net change against the previous released tag.** A change that was made and then
  reverted before shipping is not a fix; claiming it is puts a false statement in a public note.
  Check the old tag, not the PR that introduced the intermediate state.
- Mark a superseded release with one line pointing at its replacement.

---

## Pre-release vs production

`release.yml` marks a release as pre-release **only** when the tag contains `-alpha` or `-beta`.

- **A `-rc` tag is published as `--latest`**, i.e. a full production release, even though
  `docs/RELEASE_PROCESS.md` lists `-rc` as a pre-release suffix and `release.sh` treats it as one.
  Known trap; do not use `-rc` until the workflow is fixed.
- To ship a **final version number as a pre-release** (so promotion is a flag flip, not a new
  version), push the plain tag and then flip it:

  ```bash
  gh release edit vX.Y.Z --prerelease --latest=false   # demote after the workflow creates it
  gh release edit vX.Y.Z --latest --prerelease=false   # promote when satisfied
  ```

  The workflow creates it as `--latest` first, so there is a short window - flip promptly.
- `release.sh vX.Y.Z --prod` **aborts on an existing tag**, so it is not the way to promote an
  already-published release. The flag flip above is.
- `release.sh` pushes a version-bump commit straight to `main` and requires a clean tree; its
  "uncommitted changes" prompt will abort on a non-interactive stdin.

---

## Before cutting

- Full suite green, Black clean, `check_hardcoded_values.py --check` clean.
- Simulator: nominal, `--selftest`, `--dst` pass; `--all-scenarios` fails only on entries in the
  [open-findings register](PROJECT_NOTES.md#the-open-findings-register).
- Live Home Assistant restarted on the code being released, with no integration errors.
- If a new Home Assistant API is used, confirm `hacs.json`'s minimum version actually contains it.
