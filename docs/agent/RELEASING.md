# Releasing

Task-specific: read this when cutting or editing a release. The mechanics live in
[RELEASE_PROCESS.md](../RELEASE_PROCESS.md); this file is the judgement around them.

---

## Release notes format

**Read v0.8.0's notes before writing one** (`gh release view v0.8.0 --json body -q .body`) and copy
them: one short line per item, glanceable in seconds. Write them in this order:

1. **Generate the PR list first.** It works before the tag exists:

   ```bash
   gh api repos/enoch85/EffektGuard/releases/generate-notes -f tag_name=vX.Y.W \
     -f previous_tag_name=vX.Y.Z -f target_commitish=main --jq .body
   ```

   Keep the "What's Changed" lines, and shorten each link to `#n`.
2. **Read every PR in that list** and its commits (`git log --oneline --no-merges vX.Y.Z..main`),
   then write the Summary and the bullets from them.
3. **Put the list in the "PRs included in this release" dropdown**, as in the template.

```markdown
## Summary

A short paragraph in normal prose: the release's theme and its largest changes.

A second paragraph only if updating changes something or needs an action.

## Breaking
• **Area**: what the user must do

## Bug fixes
• **Area**: what the user notices, in about ten words

## Enhancements
• **Area**: what the user gets, in about ten words

## Removed
• `CONSTANT_NAME` - reason

<details>
<summary>PRs included in this release</summary>

* PR title by @author in #n

</details>

**Full Changelog**: https://github.com/enoch85/EffektGuard/compare/vX.Y.Z...vX.Y.W
```

- **Summary**: one or two short paragraphs on the highlights and the largest changes. Mention the
  update itself only when it changes a setting or needs an action; never state the obvious ("your
  settings carry over"). Big picture only; the bullets carry the details.
- **Write for someone with no technical knowledge.** Everyday words, what it means for the house:
  "a very cold room is no longer ignored", not "an implausible indoor reading". No internal terms
  (entity, offset, degree minutes, boost, layer, register), no file or function names.
- Sections are Summary, Bug fixes, Enhancements; Breaking and Removed only when they apply.
- Bullets are `•`. Backticks for symbols. Omit empty sections.
- **Write the summary as normal paragraphs**: related sentences run on in one paragraph, and a
  blank line separates paragraphs. Never break a line inside a paragraph or a bullet: GitHub shows
  every line break in a release.
- **One short line per item: what the user notices.** No cause, no mechanism, no semicolons, no
  second clause. Label by area (Hot water, Peaks), not by PR.
- **About fifteen bullets at most.** Fold the small ones into one "Smaller fixes" line.
- No callout blocks, no known-issues essay, no explanation of the release process. GitHub already
  shows the Pre-release badge.
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

`release.yml` marks a release as pre-release when the tag contains `-alpha`, `-beta` or `-rc`.

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
- **Re-cutting a published version** on newer code, as v0.7.0 was on 2026-10-19: delete its
  release and tag (`gh release delete vX.Y.Z --yes --cleanup-tag`), then run the script. With the
  manifest and badge already at that version it pushes no commit and tags `main`'s head. HACS
  offers an update only to a higher version, so an install of the old cut gets none: it needs a
  Redownload.

---

## Before cutting

- Full suite green, Black clean, `check_hardcoded_values.py --check` clean.
- Simulator: nominal, `--selftest`, `--dst` pass; `--all-scenarios` fails only on entries in the
  [open-findings register](PROJECT_NOTES.md#the-open-findings-register).
- Live Home Assistant restarted on the code being released, with no integration errors.
- If a new Home Assistant API is used, confirm `hacs.json`'s minimum version actually contains it.
