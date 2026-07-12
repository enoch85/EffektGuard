"""A heat pump that never appears must eventually be reported as missing.

The coordinator tolerates a missing NIBE at startup, because MyUplink can take the best part of a
minute to publish its entities. It did so by returning `startup_pending: True` whenever
`_first_successful_update` was still False - and nothing ever set a limit on that.

So a user who picked the wrong entity, or who has no NIBE at all, gets a config entry that stays
loaded and green forever. The entities sit at "unavailable", no repair is raised, no error is
logged after the first informational line, and `last_update_success` stays True. The integration
reports that it is fine, indefinitely, while controlling nothing.

Waiting is right. Waiting FOREVER is a silent failure, and this integration writes to a heat pump:
"I am fine" must mean it.
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
