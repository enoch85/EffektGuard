# EffektGuard Release Process

This document describes the release process for EffektGuard.

## Release Types

### 1. Beta Releases (Developer Testing)

Beta releases are for internal testing by EffektGuard developers.

**Create a beta release:**
```bash
bash scripts/release.sh v1.3.4-beta.1
```

This will:
- Update manifest.json with the beta version
- Create and push a git tag
- Trigger GitHub Actions to create a pre-release
- Mark the release as "Pre-release" on GitHub

**Subsequent beta releases:**
```bash
bash scripts/release.sh v1.3.4-beta.2
bash scripts/release.sh v1.3.4-beta.3
# ... and so on
```

### 2. Production Releases (Public)

Production releases are for public consumption via HACS.

**Create a production release:**
```bash
bash scripts/release.sh v1.3.4 --prod
```

The `--prod` flag triggers the **automated production workflow**:

1. ✅ **Switch to main branch** - Ensures release from stable branch
2. ✅ **Pull latest changes** - Gets all merged updates
3. ✅ **Run test suite** - Validates code quality (must pass)
4. ✅ **Remove beta tags** - Cleans up all `v1.3.4-beta.*` tags (local + remote)
5. ✅ **Remove beta releases** - Deletes all `v1.3.4-beta.*` GitHub releases
6. ✅ **Create production release** - Creates final `v1.3.4` release

## Requirements

### For Beta Releases
- Git configured with push access
- On any branch (allowed for pre-releases)

### For Production Releases
- [GitHub CLI (gh)](https://cli.github.com/) installed and authenticated
- On `main` branch (enforced by script)
- All tests must pass
- No uncommitted changes (or confirmation to proceed)

## Example Workflow

### Scenario: Releasing v1.3.4

```bash
# Development cycle - multiple beta releases
bash scripts/release.sh v1.3.4-beta.1
# Test with developers...

bash scripts/release.sh v1.3.4-beta.2
# More testing...

bash scripts/release.sh v1.3.4-beta.3
# Final testing...

# Ready for production!
bash scripts/release.sh v1.3.4 --prod
```

The production command will:
```
INFO: Starting PRODUCTION release process for EffektGuard v1.3.4...
INFO: Switching to main branch...
INFO: Pulling latest changes...
INFO: Running pytest test suite...
✓ All tests passed!
INFO: Searching for beta tags for version v1.3.4...
Found beta tags:
v1.3.4-beta.1
v1.3.4-beta.2
v1.3.4-beta.3
INFO: Deleting local beta tags...
INFO: Deleting remote beta tags...
SUCCESS: Beta tags removed
INFO: Searching for beta releases for version v1.3.4...
Found beta releases:
v1.3.4-beta.1
v1.3.4-beta.2
v1.3.4-beta.3
INFO: Deleting beta releases...
SUCCESS: Beta releases removed
INFO: Creating production release...
SUCCESS: Production release v1.3.4 completed successfully!
```

## GitHub Actions Integration

All releases (beta and production) trigger automated GitHub Actions:

### Validate Workflow (`.github/workflows/validate.yml`)
Runs on every PR and push:
- Black formatting check
- pytest test suite
- HACS validation
- Hassfest validation

### Release Workflow (`.github/workflows/release.yml`)
Triggers when a version tag is pushed:
- Creates release ZIP file
- Generates release notes from git history
- Creates GitHub release
- Marks beta versions as "Pre-release"
- Marks production versions as "Latest release"

## Manual Testing

Before production release, test manually:

```bash
# Run tests locally
bash scripts/run_pytest.sh

# Check formatting
black custom_components/effektguard/ --check --line-length 100

# Validate HACS compatibility (requires hacs/action Docker image)
# Usually done automatically in CI
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality (backward compatible)
- **PATCH** version (0.0.X): Bug fixes (backward compatible)

Examples:
- `v0.0.1` - Initial alpha release
- `v0.1.0` - First feature-complete version
- `v1.0.0` - First stable release
- `v1.1.0` - Added new features
- `v1.1.1` - Bug fixes

Pre-release suffixes:
- `v1.3.4-alpha.1` - Alpha testing
- `v1.3.4-beta.1` - Beta testing (developers)
- `v1.3.4-rc.1` - Release candidate

## Troubleshooting

### Tests fail during production release

```bash
ERROR: Tests failed! Please fix failing tests before releasing.
```

**Solution:** Fix failing tests, commit changes, and try again.

### GitHub CLI not authenticated

```bash
ERROR: GitHub CLI (gh) is not installed!
```

**Solution:** Install and authenticate GitHub CLI:
```bash
# Install gh (if not installed)
# See: https://cli.github.com/

# Authenticate
gh auth login
```

### Tag already exists

```bash
ERROR: Tag v1.3.4 already exists!
```

**Solution:** 
- If incorrect: Delete tag with `git tag -d v1.3.4 && git push origin :refs/tags/v1.3.4`
- If correct: Use next version number

### Uncommitted changes

```bash
INFO: You have uncommitted changes.
Do you want to continue anyway? (y/n):
```

**Solution:** 
- Recommended: Commit or stash changes first
- Or: Type `y` to continue (not recommended for production)

## See Also

- [GitHub Actions Workflows](../.github/workflows/)
- [EffektGuard Implementation Plan](../IMPLEMENTATION_PLAN/)
- [Testing Documentation](../tests/)
