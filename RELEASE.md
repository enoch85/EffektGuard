# Quick Release Guide

## Beta Release (Developer Testing)
```bash
bash scripts/release.sh v1.3.4-beta.1
```

## Production Release (Automated Cleanup)
```bash
bash scripts/release.sh v1.3.4 --prod
```

## What `--prod` Does

1. ✅ Switches to `main` branch
2. ✅ Pulls latest changes
3. ✅ Runs `scripts/run_pytest.sh` (tests must pass)
4. ✅ Removes all `v1.3.4-beta.*` tags (local + remote)
5. ✅ Removes all `v1.3.4-beta.*` GitHub releases
6. ✅ Creates production release `v1.3.4`

## Requirements

- **Beta releases**: Git push access
- **Production releases**: Git push access + [GitHub CLI](https://cli.github.com/)

Install GitHub CLI:
```bash
# Check if installed
gh --version

# Authenticate
gh auth login
```

## Test Locally First
```bash
bash scripts/run_pytest.sh
```

---

See [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md) for full documentation.
