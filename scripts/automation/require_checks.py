#!/usr/bin/env python3
"""Enable default-branch checks only after default and every open PR report them.

Run from a checkout of main after merging the automation PR:
    python scripts/automation/require_checks.py          # inspect readiness
    python scripts/automation/require_checks.py --apply  # enforce and verify
Existing rulesets, classic protection, and bypass policies are never rewritten.
"""

import argparse
import base64
import json
import subprocess
import sys

CHECKS = ("Lint EffektGuard", "Validate EffektGuard", "Validate Automation")
RULE_NAME = "EffektGuard PR validation"
ACTIONS_APP_ID = 15368


def gh(*args):
    result = subprocess.run(["gh", *args], check=True, text=True, capture_output=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def api(route):
    return gh("api", route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repository = gh("repo", "view", "--json", "nameWithOwner")["nameWithOwner"]
    prefix = f"repos/{repository}"
    repo = api(prefix)
    default = repo["default_branch"]
    workflow = api(f"{prefix}/contents/.github/workflows/validate.yml?ref={default}")
    workflow_text = base64.b64decode(workflow["content"]).decode()
    if not all(f"name: {name}" in workflow_text for name in CHECKS):
        sys.exit("Not ready: merge the implementation PR before enforcing checks.")
    main_sha = api(f"{prefix}/branches/{default}")["commit"]["sha"]
    checks = gh(
        "api", "--paginate", "--slurp", f"{prefix}/commits/{main_sha}/check-runs?per_page=100"
    )
    records = [check for page in checks for check in page["check_runs"]]
    for name in CHECKS:
        latest = next(
            (
                check
                for check in records
                if check["name"] == name and check["app"]["id"] == ACTIONS_APP_ID
            ),
            None,
        )
        if not latest or latest["conclusion"] != "success":
            sys.exit(f"Not ready: default-branch check {name!r} has not succeeded on {main_sha}.")
    # No stale PR is stranded waiting for checks absent from its branch/workflow.
    prs = gh(
        "pr",
        "list",
        "--state",
        "open",
        "--base",
        default,
        "--limit",
        "10000",
        "--json",
        "number,statusCheckRollup",
    )
    for pr in prs:
        present = {check.get("name") for check in pr["statusCheckRollup"]}
        missing = set(CHECKS) - present
        if missing:
            sys.exit(
                f"Not ready: PR #{pr['number']} needs its branch updated to report {sorted(missing)}."
            )
    if not repo["allow_auto_merge"] or not repo["allow_squash_merge"]:
        sys.exit("Not ready: repository auto-merge and squash merging must be enabled.")
    if not api(f"{prefix}/actions/permissions/workflow")["can_approve_pull_request_reviews"]:
        sys.exit("Not ready: workflow PR approvals must be enabled.")
    rulesets = gh("api", "--paginate", "--slurp", f"{prefix}/rulesets?per_page=100")
    existing = [rule for page in rulesets for rule in page if rule["name"] == RULE_NAME]
    if not existing and args.apply:
        payload = {
            "name": RULE_NAME,
            "target": "branch",
            "enforcement": "active",
            "bypass_actors": [],
            "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
            "rules": [
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "strict_required_status_checks_policy": True,
                        "do_not_enforce_on_create": False,
                        "required_status_checks": [
                            {"context": name, "integration_id": ACTIONS_APP_ID} for name in CHECKS
                        ],
                    },
                }
            ],
        }
        result = subprocess.run(
            ["gh", "api", "--method", "POST", f"{prefix}/rulesets", "--input", "-"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        print(f"Created ruleset {json.loads(result.stdout)['id']}.")
    effective = api(f"{prefix}/rules/branches/{default}")
    enforced = {
        item["context"]
        for rule in effective
        if rule["type"] == "required_status_checks"
        for item in rule["parameters"]["required_status_checks"]
        if item.get("integration_id") == ACTIONS_APP_ID
    }
    if args.apply or existing:
        if not set(CHECKS) <= enforced:
            sys.exit(
                "Effective rules do not enforce every intended check; inspect repository settings."
            )
        print(f"Verified effective required checks on {default}: {', '.join(CHECKS)}")
    else:
        print(f"Ready. Re-run with --apply to require {', '.join(CHECKS)} on {default}.")


if __name__ == "__main__":
    main()
