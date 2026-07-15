"""`hass.components` was removed from Home Assistant; the coordinator must not call it.

The removed API raises AttributeError, and the `# type: ignore[attr-defined]` on the old call
claimed - falsely - that it was a type-stubs gap. The supported replacement is
`homeassistant.components.persistent_notification.async_create(hass, ...)`, imported at module
top, and these tests read the coordinator source to hold that fix in place. A MagicMock `hass`
answers `hass.components...` cheerfully, so the unit suite could never catch this by mocking.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from custom_components.effektguard.coordinator import EffektGuardCoordinator

COORDINATOR_SOURCE = Path(inspect.getfile(EffektGuardCoordinator)).read_text(encoding="utf-8")

# The source with comments stripped. Checked against the CODE, not against prose about the code -
# otherwise a comment explaining the removed API would trip the very test that forbids it.
CODE_ONLY = "\n".join(
    line.split("#", 1)[0]
    for line in COORDINATOR_SOURCE.splitlines()
    if not line.lstrip().startswith("#")
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
