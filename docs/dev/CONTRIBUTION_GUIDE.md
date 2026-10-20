# Contribution Guide

Welcome! This guide will help you contribute to EffektGuard effectively.

## Before You Start

This is **production code affecting real homes** - heat pump health and heating comfort depend on it. We prioritize:
1. **Safety** over savings (thermal debt prevention)
2. **Comfort** over cost (moderate optimization)
3. **Quality** over speed (thorough testing required)

## Getting Started

### 1. Set Up Development Environment

Follow [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) to configure your development environment.

```bash
git clone https://github.com/enoch85/EffektGuard.git
cd EffektGuard
bash scripts/setup_dev.sh    # Create .venv (Python 3.15)
source docs/dev/activate.sh  # Activate it
pytest tests/ -v             # Verify setup
```

### 2. Understand the Architecture

Read the architecture documentation:
- `../architecture/00_overview.md` - System overview
- [CLAUDE.md](../../CLAUDE.md) and `docs/agent/` - core principles and rules
- `docs/CLIMATE_ZONES.md` - Climate-aware safety system

### 3. Check Existing Issues

Browse [GitHub Issues](https://github.com/enoch85/EffektGuard/issues) to find:
- Bug reports
- Feature requests
- Good first issues (labeled `good-first-issue`)

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Read Before Editing

**Critical**: Always read entire files before editing to understand:
- NIBE heat pump context
- Safety implications
- All locations that need changes

```bash
# Read full file
cat custom_components/effektguard/optimization/decision_engine.py

# Search for related code
grep -r "calculate_offset" custom_components/effektguard/
```

### 3. Follow Code Standards

See [CODE_STANDARDS.md](CODE_STANDARDS.md) for detailed guidelines:

- Use configuration constants (never hardcode)
- Add type hints to all functions
- Write comprehensive docstrings
- Reference research documents
- Format with Black (line length 100)

```python
# ✅ Good
from .const import DM_THRESHOLD_AUX_LIMIT

if degree_minutes <= DM_THRESHOLD_AUX_LIMIT:
    # Past the aux limit the compressor is flat out (docs/research/01_degree_minutes.md)
    return max(offset, 0.0)

# ❌ Bad
if degree_minutes < -1500:  # Magic number!
    return max(offset, 0.0)
```

### 4. Write Tests

See [TESTING.md](TESTING.md) for testing guidelines.

What earns a test - a safety guard, a threshold-to-response mapping, a fixed bug (name the finding) -
and what does not (constants and labels, log text and log levels, ...) is in
[`docs/agent/IMPLEMENTATION.md`](../agent/IMPLEMENTATION.md) (Tests). A fixed bug's test fails on
`main` and passes on the branch; name the file for the behaviour, e.g.
`tests/unit/dhw/test_a_slow_tank_is_heated_in_time.py`.

### 5. Format Code

Format with Black before committing:

```bash
black custom_components/effektguard/ --line-length 100
```

### 6. Run Tests

All tests must pass:

```bash
# Clear cache
find . -type d -name __pycache__ -exec rm -r {} +

# Run all tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=custom_components/effektguard --cov-report=term-missing
```

### 7. Verify Changes

Check for issues:

```bash
# Find hardcoded values
grep -r "= -500" custom_components/effektguard/
grep -r "= 12.0" custom_components/effektguard/

# Find old function names (if renaming)
grep -r "old_function_name" custom_components/effektguard/

# Test imports
python3 -c "from custom_components.effektguard.optimization.thermal_layer import ThermalModel"
```

## Commit Guidelines

### Commit Message Format

The subject is a plain sentence that says what the change does, as `git log` shows - no `feat:` or
`fix:` prefix. The body says why and carries the evidence, as "Working style" in
[PROJECT_NOTES.md](../agent/PROJECT_NOTES.md) describes. No AI attribution (rule 1 in
[CLAUDE.md](../../CLAUDE.md)).

### Example Commit

From the history (abridged):

```
Pre-heat on a colder day, not a colder night

The pre-heat trigger compared the forecast's coldest hour with the current outdoor temperature, so
an ordinary night fired it: a May night reads 8 C below the afternoon. At +2.0 that pre-heated mild
weather, and shoulder_may2024 went red on comfort.

A cold snap is a colder day. The trigger now compares the measured outdoor mean over the past 24 h
with every coming 24-hour window that starts within the horizon. [...]

Tests: a May day-night forecast no longer fires it, with or without a measured past day; [...]
```

## Pull Request Process

### 1. Update Documentation

If your change affects:
- User configuration → Update README.md or docs/
- API/behavior → Update architecture docs
- Development process → Update docs/dev/

### 2. Create Pull Request

```bash
git push origin feature/your-feature-name
```

On GitHub:
1. Click "New Pull Request"
2. Fill in the template:
   - **Description**: What does this change?
   - **Motivation**: Why is this needed?
   - **Testing**: How was it tested?
   - **Safety**: Any safety implications?
   - **References**: Link to research/issues

### 3. PR Review Process

Your PR will be reviewed for:
- ✅ Code quality and standards
- ✅ Test coverage (especially safety-critical code)
- ✅ NIBE-specific behavior correctness
- ✅ Documentation completeness
- ✅ Black formatting
- ✅ No hardcoded values
- ✅ Research references for NIBE decisions

### 4. Address Feedback

Respond to review comments:
```bash
# Make requested changes
git add .
git commit -m "Say a room above its band may still be cut past the aux limit"

git push origin feature/your-feature-name
```

### 5. Merge

Once approved:
- All tests pass ✅
- Code review approved ✅
- Documentation updated ✅
- Conflicts resolved ✅

The maintainer will merge your PR.

## Contribution Areas

### Good First Issues

Start with:
- Documentation improvements
- Test coverage improvements
- Code cleanup (remove duplicates, dead code)
- Adding research references to existing code

### Feature Development

Requires deep NIBE understanding:
- New optimization layers
- Safety threshold adjustments
- Thermal model improvements
- Climate zone enhancements

**Important**: Discuss major features in an issue first!

### Bug Fixes

1. Create issue describing bug
2. Include logs and reproduction steps
3. Reference real NIBE behavior (if applicable)
4. Write test that reproduces bug
5. Fix and verify test passes

## NIBE-Specific Contributions

### Required Knowledge

Before contributing NIBE-specific features, understand:
- **Degree Minutes (DM)**: Thermal balance tracking (Menu 4.9.3)
- **Pump modes**: Auto, Intermittent, Economy (system-dependent)
- **UFH types**: Concrete slab, timber, radiators (different lag times)
- **Flow temperature**: the EN 442 emitter law (`docs/research/02_emitter_law.md`)

### Research References

Always reference research when implementing NIBE behavior:
- `docs/research/` - the evidence for the numbers in `const.py`, and what is *not* sourced
- `docs/CLIMATE_ZONES.md` - Climate-aware safety system

### When Uncertain

**Never guess NIBE behavior!** Instead:
1. Search existing research documents
2. Ask in an issue with references to NIBE manuals
3. Test with real NIBE system (if available)
4. Document findings for future reference

## Safety-Critical Changes

Changes affecting safety require:
- ✅ 100% test coverage
- ✅ Research references justifying thresholds
- ✅ Real-world validation (if possible)
- ✅ Documentation of failure modes
- ✅ Graceful fallback on errors

### Example Safety Changes

- Thermal debt thresholds
- Emergency recovery logic
- Pump configuration validation
- Climate zone DM ranges

## Questions?

- Open a [GitHub Issue](https://github.com/enoch85/EffektGuard/issues)
- Reference relevant research documents
- Include logs/examples if applicable

## Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Help others learn
- Prioritize safety and quality
- Document your decisions

Thank you for contributing to EffektGuard! 🚀
