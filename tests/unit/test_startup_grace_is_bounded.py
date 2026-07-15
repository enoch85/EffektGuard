"""A heat pump that never appears must eventually be reported as missing, not "still starting".

The coordinator tolerates a missing NIBE at startup (MyUplink is slow to publish entities) by
returning `startup_pending: True` while `_first_successful_update` is False. That grace must be
BOUNDED by STARTUP_MAX_GRACE_ATTEMPTS: past it, a missing pump becomes UpdateFailed rather than a
permanently green entry that reads nothing and controls nothing.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.effektguard.const import STARTUP_MAX_GRACE_ATTEMPTS
from custom_components.effektguard.coordinator import EffektGuardCoordinator


@pytest.fixture
def coordinator() -> EffektGuardCoordinator:
    """A coordinator whose NIBE never answers."""
    coord = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coord.nibe = MagicMock()
    coord.nibe.get_current_state = AsyncMock(side_effect=UpdateFailed("no such entity"))
    coord._first_successful_update = False
    coord._startup_grace_attempts = 0
    coord._schedule_aligned_refresh = MagicMock()
    coord.hass = MagicMock()
    coord.entry = MagicMock()
    coord.entry.data = {}
    return coord


async def test_it_waits_before_giving_up(coordinator):
    """The grace period must still exist: MyUplink is genuinely slow to start."""
    result = await coordinator._async_update_data()

    assert result["startup_pending"] is True, "the first attempt must be tolerated, not fatal"
    assert result["nibe"] is None


async def test_it_does_not_wait_forever(coordinator):
    """After the grace period, a missing heat pump is an error, not a pending state."""
    for _ in range(STARTUP_MAX_GRACE_ATTEMPTS):
        await coordinator._async_update_data()

    with pytest.raises(UpdateFailed) as err:
        await coordinator._async_update_data()

    assert "NIBE" in str(err.value)


async def test_the_entry_never_reports_itself_healthy_while_blind(coordinator):
    """`startup_pending` must not be returnable indefinitely.

    A config entry that stays loaded, green, and pending forever tells the user nothing is wrong
    while the integration reads nothing and controls nothing.
    """
    pending = 0
    for _ in range(STARTUP_MAX_GRACE_ATTEMPTS + 5):
        try:
            result = await coordinator._async_update_data()
        except UpdateFailed:
            break
        if result.get("startup_pending"):
            pending += 1
    else:  # pragma: no cover - only reached if it never gives up
        pytest.fail(
            f"The coordinator returned startup_pending {pending} times and never once "
            f"reported failure. A user with no NIBE at all gets a permanently green integration."
        )

    assert pending <= STARTUP_MAX_GRACE_ATTEMPTS
