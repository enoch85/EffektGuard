"""`hass.components` was removed from Home Assistant. The code still calls it.

    self.hass.components.persistent_notification.async_create(  # type: ignore[attr-defined]

The `type: ignore` carries the comment "HA dynamic component access (not in type stubs)". That is
not true. It is not a stubs gap - the attribute does not exist:

    HomeAssistant.components          -> AttributeError
    homeassistant.loader.Components   -> ImportError

Checked against Home Assistant 2026.2.3. `hacs.json` floors this integration at 2025.10.0, so the
call is broken across the entire supported range. A comment was asserting something false in order
to silence an error that was correct.

The branch is currently unreachable - `DecisionEngine.__init__` always assigns a ClimateZoneDetector,
so `self.engine.climate_detector` is never falsy - which is exactly why nobody noticed. If it ever
becomes reachable, the AttributeError is raised inside a try/except that reports "DHW calculation
error", and hot-water scheduling dies quietly behind a message about the wrong subsystem.

Note why the test suite could never have caught this: `hass` is a MagicMock in every coordinator
test, and a MagicMock answers `hass.components.persistent_notification.async_create(...)` cheerfully.
Mocking the framework mocks away the framework's own API removals. So this test asks Home Assistant
directly, and reads the source.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from homeassistant.core import HomeAssistant

from custom_components.effektguard.coordinator import EffektGuardCoordinator

COORDINATOR_SOURCE = Path(inspect.getfile(EffektGuardCoordinator)).read_text(encoding="utf-8")

# The source with comments stripped. Checked against the CODE, not against prose about the code -
# otherwise a comment explaining the removed API would trip the very test that forbids it.
CODE_ONLY = "\n".join(
    line.split("#", 1)[0]
    for line in COORDINATOR_SOURCE.splitlines()
    if not line.lstrip().startswith("#")
)


def test_home_assistant_really_has_no_components_attribute():
    """The premise. If this ever fails, HA put it back and the rest of this file is moot."""
    assert not hasattr(HomeAssistant, "components"), (
        "HomeAssistant.components exists again. It was removed; this integration used to rely on "
        "it, and these tests exist to stop that returning."
    )


def test_the_coordinator_does_not_call_a_removed_api():
    """The defect, read straight out of the source."""
    assert "hass.components" not in CODE_ONLY, (
        "coordinator.py calls `hass.components`, which Home Assistant has removed. It raises "
        "AttributeError, and the `# type: ignore[attr-defined]` on that line hides a real error "
        "behind a comment claiming it is a type-stubs gap. It is not."
    )


def test_persistent_notification_is_imported_at_module_top():
    """The project's own rule, and the fix: import the real API, at the top, like everything else."""
    assert "from homeassistant.components.persistent_notification import async_create" in (
        COORDINATOR_SOURCE
    ), (
        "The supported way to raise a notification is "
        "`homeassistant.components.persistent_notification.async_create(hass, ...)`, imported at "
        "module top."
    )
