# Dependency automation and rollout

Pip (including nested test requirements), GitHub Actions, and the Docker devcontainer are
checked Mondays at 06:00 Europe/Stockholm. Version updates wait three days; security updates
are exempt. Minor/patch updates are grouped within each ecosystem, while majors remain
individual PRs for manual review. No dependency is excluded.

The pinned Home Assistant test plugin fixes NumPy at 2.3.2 and pytest at 9.0.3. These are
compatible with HA 2026.10.0 / Python 3.14.2; newer standalone NumPy/pytest releases cannot be
installed with the complete test toolchain. Direct requirements use exact versions so
Dependabot can record updates. There was no lockfile in this repository.

Every PR, including documentation and workflow updates, runs `Lint EffektGuard`,
`Validate EffektGuard`, and `Validate Automation` with read-only permissions. There are no
path filters or job conditions. The production build is the HACS release ZIP; no Node build
applies. Docker is built on every PR. Existing HACS and Hassfest checks remain enabled.
Merge-group validation uses the same check names.

`Approve Dependabot` runs only after a successful bot-triggered PR validation. It runs no PR
code and downloads no validation artifacts. It verifies the bot's numeric identity, every
commit's author and signature, the current head SHA, and all three validation jobs. The
SHA-pinned official metadata action also verifies identity/signatures and classifies updates.
Only minor/patch updates reach approval and `gh pr merge --auto --squash --match-head-commit`.
GitHub's workflow token is used; repository merge rules and reviews still apply.

## Rollout

1. Merge the implementation PR only after explicit owner authorization and passing checks.
2. Confirm all three checks succeed on the default branch. Update existing PR branches to
   include the validation workflow before enforcing new check names. In particular, PR #24
   had no checks when this implementation began. Do not refresh Dependabot PRs without owner
   authorization; there were none open during inspection.
3. From an up-to-date default-branch checkout, run
   `python scripts/automation/require_checks.py`. It refuses to enforce checks until they
   succeed on the default branch and can report on every existing PR.
4. Run the same command with `--apply` using an administrator's normal GitHub CLI session.
   It creates a default-branch ruleset with no bypass actors and verifies the effective rules.
   Existing rulesets and classic branch protection are left intact. The pre-existing `main`
   ruleset had empty target includes and supplied no effective protection; it is preserved.
5. Verify a real Dependabot minor/patch PR: all validation checks, the bot review, auto-merge
   state, and eventual merge. Major and failed updates must remain unapproved.

The owner explicitly chose **manual releases**. No automatic distribution, deployment,
30-minute delay, or scheduled deployment polling is configured. Ordinary pushes keep their
validation behavior; `v*` tag pushes keep the existing GitHub/HACS release destination and
stable/prerelease selection. Validation builds a ZIP without publishing it. Neither a green
validation run nor a skipped approval job proves a deployment or an automatic merge.

Official references: [Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference),
[Dependabot automation](https://docs.github.com/en/code-security/tutorials/secure-your-dependencies/automate-dependabot-with-actions),
[workflow token triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow),
[metadata verification source](https://github.com/dependabot/fetch-metadata/blob/25dd0e34f4fe68f24cc83900b1fe3fe149efef98/src/dependabot/verified_commits.ts).
