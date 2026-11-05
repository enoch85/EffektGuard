# Code standards

The binding conventions - every number in `const.py` with its source, no backward-compatibility
shims, read the whole file before editing, never guess NIBE behaviour, no AI attribution - are in
[docs/agent/PROJECT_NOTES.md](../agent/PROJECT_NOTES.md) (Conventions). This page is the mechanics.

## Formatting

Black, line length 100, configured in `pyproject.toml` (`[tool.black]`). CI runs
`black custom_components/effektguard/ --check --line-length 100`. Format everything with:

```bash
black --line-length 100 custom_components/ tests/ scripts/
```

## Imports

Stdlib, third-party, Home Assistant, then local - relative within the package:

```python
import logging
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import EffektGuardCoordinator
```

## Type hints and docstrings

Type hints on every public function, using Home Assistant's own types (`HomeAssistant`,
`ConfigEntry`). Every public function and class has a docstring: what it does,
`Args:` and `Returns:` where they are not obvious, and for anything that touches the pump the
safety implication and the source in `docs/research/`. Comments say why, not what.

## Numbers

Every numeric value is a constant in `const.py`, imported where used, with a comment saying where
it came from - in code and in tests. Degree-minute thresholds come from `ClimateZoneDetector`,
never from a literal. The register integer is clamped to `MIN_OFFSET`..`MAX_OFFSET` in
`utils/offset.py`. Two scripts hold the line:

```bash
python scripts/check_hardcoded_values.py --check        # no new literals in logic (runs in CI)
python scripts/find_duplicate_constants.py --prod-only  # duplicate and unused constants
```

A constant with a leading underscore is invisible to the second script; constants are public.
Delete a dead constant with the code that used it.

## Async

Never block the event loop: no `time.sleep()`, no synchronous I/O in a coroutine. The coordinator
runs the decision engine through `hass.async_add_executor_job`, and the adapter's writes call
services with `blocking=True` so a failed write is seen rather than assumed.

## Errors

Catch the specific exception, log with context using lazy `%s` formatting
(`_LOGGER.error("Failed to read NIBE state: %s", err)`), and fall back to a safe decision. A read
failure never becomes a write.

## Home Assistant patterns

- `EffektGuardCoordinator(DataUpdateCoordinator)` is created with `update_interval=None`; the
  coordinator schedules its own clock-aligned cycle.
- Entities subclass `CoordinatorEntity` and set `_attr_has_entity_name = True`.
- Check a Home Assistant symbol against the tag in `hacs.json` before using it; a newer enum member
  raises `AttributeError` on an older release (see the traps in PROJECT_NOTES.md).

## Before committing

Black, the ratchet, the suite: the commands are in [CLAUDE.md](../../CLAUDE.md#commands), and the
full list of gates is under "Before claiming it works" there.
