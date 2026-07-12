"""The adapter must refuse to fabricate the inputs that drive heat-pump control.

Every primary reading used to have a plausible hard-coded fallback:

    outdoor_temp   -> 0.0
    supply_temp    -> NIBE_DEFAULT_SUPPLY_TEMP (35.0)
    indoor_temp    -> DEFAULT_INDOOR_TEMP (21.0)   <- exactly the usual target
    degree_minutes -> _estimate_degree_minutes(), invented from six magic numbers

Because get_current_state() never raised, a completely broken installation produced a
fully-populated NibeState and was indistinguishable from a healthy one - and the offset
write path ran on it. The coordinator's "NIBE required" guard was dead code.

Two distinct contracts are pinned here:

1. REQUIRED readings (outdoor, supply, degree minutes) -> raise UpdateFailed. The
   coordinator already has the degrade path for this: startup_pending before the first
   success, UpdateFailed after, entities unavailable, nothing written to the pump.

2. OPTIONAL indoor reading -> a NIBE with no room sensor (no BT50) is a LEGITIMATE
   configuration; it runs on degree minutes and the heating curve. So do not fail - but
   mark the reading invalid so comfort layers abstain instead of trusting a placeholder
   that happens to equal the target.

Degree minutes is never estimated. It is the primary thermal-debt safety signal and every
NIBE exposes it (register 40940 / 43005); guessing it drove the emergency layer on fiction.
"""

from unittest.mock import MagicMock

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import DEFAULT_INDOOR_TEMP

OUTDOOR = "sensor.nibe_bt1_outdoor"
SUPPLY = "sensor.nibe_bt25_supply"
INDOOR = "sensor.nibe_bt50_room"
DEGREE_MINUTES = "sensor.nibe_degree_minutes"
OFFSET = "number.nibe_heat_offset_s1_47011"

FULL_CACHE = {
    "outdoor_temp": OUTDOOR,
    "supply_temp": SUPPLY,
    "indoor_temp": INDOOR,
    "degree_minutes": DEGREE_MINUTES,
    "offset": OFFSET,
}

READINGS = {
    OUTDOOR: "-8.4",
    SUPPLY: "38.2",
    INDOOR: "20.6",
    DEGREE_MINUTES: "-420",
    OFFSET: "0",
}


def build_adapter(cache: dict[str, str], readings: dict[str, str]) -> NibeAdapter:
    """NibeAdapter wired to a fake state machine, with discovery pinned to `cache`."""
    hass = MagicMock()

    def get_state(entity_id: str):
        if entity_id not in readings:
            return None
        state = MagicMock()
        state.state = readings[entity_id]
        state.attributes = {"unit_of_measurement": "°C"}
        return state

    hass.states.get.side_effect = get_state

    adapter = NibeAdapter(hass, {"nibe_entity": OFFSET})
    adapter._entity_cache = dict(cache)
    # Pin discovery: the cache above IS the discovered set for this test.
    adapter._discover_nibe_entities = _noop
    return adapter


async def _noop() -> None:
    return None


class TestRequiredReadingsRefuseToBeFabricated:
    @pytest.mark.asyncio
    async def test_missing_degree_minutes_raises_instead_of_estimating(self):
        """DM is the primary safety signal. It must never be invented."""
        cache = {k: v for k, v in FULL_CACHE.items() if k != "degree_minutes"}
        adapter = build_adapter(cache, READINGS)

        with pytest.raises(UpdateFailed, match="degree minutes"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_missing_outdoor_temp_raises_instead_of_defaulting_to_zero(self):
        """Outdoor 0.0 drives the climate-aware DM thresholds and weather compensation.

        A Swedish user at -20 C read as 0 C gets the wrong DM band AND under-heating.
        """
        cache = {k: v for k, v in FULL_CACHE.items() if k != "outdoor_temp"}
        adapter = build_adapter(cache, READINGS)

        with pytest.raises(UpdateFailed, match="outdoor"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_missing_supply_temp_raises_instead_of_defaulting_to_35(self):
        cache = {k: v for k, v in FULL_CACHE.items() if k != "supply_temp"}
        adapter = build_adapter(cache, READINGS)

        with pytest.raises(UpdateFailed, match="supply"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_unavailable_entity_is_treated_as_missing(self):
        """A discovered entity reporting `unavailable` must not fall back to a constant."""
        readings = dict(READINGS)
        readings[DEGREE_MINUTES] = "unavailable"
        adapter = build_adapter(FULL_CACHE, readings)

        with pytest.raises(UpdateFailed, match="degree minutes"):
            await adapter.get_current_state()

    @pytest.mark.asyncio
    async def test_the_estimator_is_gone(self):
        """No back-door: the DM estimator must not exist at all (repo rule: no aliases)."""
        assert not hasattr(NibeAdapter, "_estimate_degree_minutes"), (
            "_estimate_degree_minutes still exists. Degree minutes must never be "
            "fabricated from a heating-curve guess."
        )


class TestIndoorSensorIsOptionalButMarkedInvalid:
    @pytest.mark.asyncio
    async def test_no_room_sensor_still_works_but_marks_indoor_invalid(self):
        """A NIBE without BT50 is a legitimate setup - it must not fail, but must not lie."""
        cache = {k: v for k, v in FULL_CACHE.items() if k != "indoor_temp"}
        adapter = build_adapter(cache, READINGS)

        state = await adapter.get_current_state()

        assert state.indoor_temp_valid is False, (
            "Indoor reading is a placeholder but is flagged as a measurement. Comfort "
            "layers would trust DEFAULT_INDOOR_TEMP, which equals the target and yields a "
            "deviation of exactly 0.0."
        )
        assert state.indoor_temp == pytest.approx(DEFAULT_INDOOR_TEMP)
        # The rest of the state is real and usable.
        assert state.degree_minutes == pytest.approx(-420.0)
        assert state.outdoor_temp == pytest.approx(-8.4)

    @pytest.mark.asyncio
    async def test_present_room_sensor_is_marked_valid(self):
        adapter = build_adapter(FULL_CACHE, READINGS)

        state = await adapter.get_current_state()

        assert state.indoor_temp_valid is True
        assert state.indoor_temp == pytest.approx(20.6)
