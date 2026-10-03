"""An upgrade must not break setup: version-1 peak records are migrated, not parsed.

Version 1 recorded 15-minute quarter peaks (``quarter_of_day``). This branch bills the HOURLY mean
(``period_of_day``), and the two are different billed quantities - so migration DISCARDS the old
records and the month's top-3 restarts from live measurement. Parsing them instead raised
``KeyError: 'period_of_day'`` in ``PeakEvent.from_dict`` inside ``async_setup_entry``, failing setup
for every upgrading install. Losing at most a month of partial history is recoverable; that was not.
"""

from unittest.mock import MagicMock

import pytest

from custom_components.effektguard.const import (
    EFFECT_STORAGE_VERSION,
    POWER_SOURCE_NONE,
    STORAGE_KEY,
)
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
async def test_migration_converts_quarter_records_and_marks_them_unbillable():
    """A v1 quarter record IS the same billed quantity under the owner's 15-minute tariff.

    Migration renames `quarter_of_day` -> `period_of_day` and keeps the peak as a control
    threshold. What v1 never stored is PROVENANCE, and it cannot be reconstructed - so the
    record is marked POWER_SOURCE_NONE (unbillable) until live measurement replaces it.
    """
    store = EffectStore(MagicMock(), EFFECT_STORAGE_VERSION, STORAGE_KEY)

    migrated = await store._async_migrate_func(1, 1, {"peaks": [V1_QUARTER_RECORD]})

    assert len(migrated["peaks"]) == 1
    record = migrated["peaks"][0]
    assert record["period_of_day"] == V1_QUARTER_RECORD["quarter_of_day"]
    assert "quarter_of_day" not in record
    assert record["source"] == POWER_SOURCE_NONE, (
        "a migrated peak has unknown provenance and must not be presented as a billable "
        "meter measurement"
    )
    assert record["actual_power"] == V1_QUARTER_RECORD["actual_power"]


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
