"""A price source installed after setup must be connectable without deleting the entry (issue #18).

The price entity was offered only in the initial flow. Neither Reconfigure nor the options flow had
the field, so a user who set EffektGuard up before installing GE-Spot could never connect it: the
"No electricity price source" repair told them to configure it "in the integration's options",
where no such field existed, and the only remedy was to delete and re-add the integration.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.config_flow import EffektGuardConfigFlow
from custom_components.effektguard.const import CONF_GESPOT_ENTITY, CONF_NIBE_ENTITY

GESPOT_ENTITY = "sensor.gespot_current_price_nl"


@pytest.mark.asyncio
async def test_reconfigure_connects_a_price_source_added_after_setup():
    entry = MagicMock()
    entry.data = {CONF_NIBE_ENTITY: "number.heat_offset_s1_47011", CONF_GESPOT_ENTITY: None}

    flow = EffektGuardConfigFlow()
    flow.hass = MagicMock()
    flow._get_reconfigure_entry = MagicMock(return_value=entry)
    flow.async_update_reload_and_abort = MagicMock(return_value={"type": "abort"})

    form = await flow.async_step_reconfigure()
    assert CONF_GESPOT_ENTITY in form["data_schema"].schema, (
        "Reconfigure does not offer the price entity. A user who installs GE-Spot after setting "
        "up EffektGuard has no way to connect it short of deleting the integration."
    )

    await flow.async_step_reconfigure({CONF_GESPOT_ENTITY: GESPOT_ENTITY})

    stored = flow.async_update_reload_and_abort.call_args.kwargs["data_updates"]
    assert stored.get(CONF_GESPOT_ENTITY) == GESPOT_ENTITY, (
        f"Reconfigure showed the price entity but stored {stored.get(CONF_GESPOT_ENTITY)!r}: "
        f"the adapter is rebuilt from entry.data on reload and would still have no price source."
    )
