"""An upgrade must not break setup: version-1 peak records are migrated, not parsed.

Main recorded 15-minute quarter peaks (``quarter_of_day``) in a version-1 store. This branch
bills the HOURLY mean and its records carry ``period_of_day``, but the store still declared
version 1 - so Home Assistant handed the old payload straight to ``PeakEvent.from_dict``,
which raised ``KeyError: 'period_of_day'`` inside ``async_setup_entry``. Every existing
installation failed setup on upgrade.

A quarter-hour mean is not convertible to an hourly mean - they are different billed
quantities - so migration DISCARDS the old records and the month's top-3 restarts from live
measurement. Losing at most one month of partial peak history is recoverable; failing setup
for every upgrading user is not.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import EFFECT_STORAGE_VERSION, STORAGE_KEY
from custom_components.effektguard.optimization.effect_layer import EffectManager, EffectStore

# A record exactly as main's PeakEvent.to_dict() wrote it: quarter_of_day, no source.
V1_QUARTER_RECORD = {
    "timestamp": "2026-06-15T08:00:00+02:00",
    "quarter_of_day": 32,
    "actual_power": 6.0,
    "effective_power": 6.0,
    "is_daytime": True,
}


@pytest.mark.asyncio
async def test_migration_discards_quarter_hour_records():
    """A v1 payload comes out of migration with its quarter-era peaks discarded, not crashed on."""
    store = EffectStore(MagicMock(), EFFECT_STORAGE_VERSION, STORAGE_KEY)

    migrated = await store._async_migrate_func(1, 1, {"peaks": [V1_QUARTER_RECORD]})

    assert migrated == {"peaks": []}


@pytest.mark.asyncio
async def test_migration_survives_a_malformed_v1_payload():
    """A corrupt or hand-edited v1 file must migrate to an empty history, not raise."""
    store = EffectStore(MagicMock(), EFFECT_STORAGE_VERSION, STORAGE_KEY)

    assert await store._async_migrate_func(1, 1, None) == {"peaks": []}
    assert await store._async_migrate_func(1, 1, {"junk": 1}) == {"peaks": []}


def test_the_manager_wires_the_migrating_store_above_the_quarter_era():
    """The migration only runs if the store is an EffectStore AND declares a version above 1.

    Home Assistant's Store calls ``_async_migrate_func`` only when the stored version is lower
    than the declared one. Declaring version 1 - what this integration did - hands v1 data to
    the parser unmigrated, which is the setup crash this file exists to prevent.
    """
    manager = EffectManager(MagicMock())

    assert isinstance(manager._store, EffectStore)
    assert manager._store.version == EFFECT_STORAGE_VERSION
    assert EFFECT_STORAGE_VERSION > 1
