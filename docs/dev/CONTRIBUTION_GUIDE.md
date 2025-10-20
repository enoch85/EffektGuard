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
source docs/dev/activate.sh  # Activate Python 3.13 venv
pytest tests/ -v             # Verify setup
```

### 2. Understand the Architecture

Read the architecture documentation:
- `architecture/00_overview.md` - System overview
- `.github/copilot-instructions.md` - Core principles and rules
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
from .const import DM_THRESHOLD_CRITICAL

if degree_minutes < DM_THRESHOLD_CRITICAL:
    # Emergency recovery based on stevedvo case study
    return emergency_recovery_offset()

# ❌ Bad
if degree_minutes < -500:  # Magic number!
    return 3.0
```

### 4. Write Tests

See [TESTING.md](TESTING.md) for testing guidelines.

All changes must include tests:
- **New features**: Unit tests + integration tests
- **Bug fixes**: Test that reproduces bug + fix verification
- **Safety changes**: 100% test coverage required

```python
def test_thermal_debt_prevention():
    """Test that DM -500 triggers emergency recovery."""
    tracker = ThermalDebtTracker()
    tracker.degree_minutes = -500
    
    assert tracker.get_severity() == "CRITICAL"
    assert tracker.get_recovery_offset() == 3.0
```

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
python3 -c "from custom_components.effektguard.optimization.thermal_model import ThermalModel"
```

## Commit Guidelines

### Commit Message Format

```
Type: Brief description (50 chars max)

- Change 1 with context
- Change 2 with safety implications noted
- Change 3 with NIBE research reference

Tests: passing
Safety: validated
Format: black applied
```

### Commit Types

- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring (no behavior change)
- `test:` - Adding/updating tests
- `docs:` - Documentation changes
- `style:` - Formatting, typos (no code change)
- `safety:` - Safety-critical changes
- `perf:` - Performance improvements

### Example Commits

```bash
git commit -m "feat: Add climate-aware DM thresholds

- Integrate ClimateZoneDetector into decision engine
- Adapt warning thresholds based on location and outdoor temp
- Stockholm at -10°C: warning=-700, Kiruna at -30°C: warning=-1200
- Critical threshold remains -1500 (validated in Swedish forums)

Tests: Added test_climate_aware_thresholds()
Safety: Improves safety for cold climates
References: Swedish_NIBE_Forum_Findings.md"
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
git commit -m "refactor: Address PR feedback

- Update docstrings with research references
- Add test case for edge condition
- Remove hardcoded threshold"

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
- **Flow temperature**: OEM research (SPF 4.0+: Outdoor + 27°C)

### Research References

Always reference research when implementing NIBE behavior:
- `IMPLEMENTATION_PLAN/02_Research/Forum_Summary.md` - Real F2040 cases
- `IMPLEMENTATION_PLAN/02_Research/Swedish_NIBE_Forum_Findings.md` - F750 optimization
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
