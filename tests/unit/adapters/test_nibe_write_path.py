"""Tests for the NIBE offset write path (set_curve_offset).

Covers the doc-driven hardening (blocking service call, entity min/max clamp,
malformed-attribute robustness, unknown-value markers) added while making the
integration multi-source. The write path must never raise into the coordinator
and must not record phantom success when the service call fails.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter
from custom_components.effektguard.const import (
    CONF_NIBE_ENTITY,
    NIBE_UNKNOWN_VALUE_MARKERS,
)


def make_adapter(offset_state="0", offset_attrs=None, call_side_effect=None):
    """Build an adapter whose offset entity is number.offset with given attrs."""
    hass = MagicMock(spec=HomeAssistant)
    state = MagicMock()
    state.state = offset_state
    state.attributes = offset_attrs if offset_attrs is not None else {}
    states = MagicMock()
    states.get = lambda entity_id: state
    hass.states = states
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=call_side_effect)

    adapter = NibeAdapter(hass, {CONF_NIBE_ENTITY: "number.offset"})
    return adapter, hass


class TestOffsetWriteBlocking:
    """The write must block and surface handler failures, not fire-and-forget."""

    async def test_successful_write_is_blocking(self):
        adapter, hass = make_adapter()
        # Big offset so the accumulator crosses the +-1 threshold from 0
        result = await adapter.set_curve_offset(3.0)

        assert result is True
        call = hass.services.async_call.call_args
        assert call.args[:2] == ("number", "set_value")
        assert call.kwargs["blocking"] is True
        assert call.kwargs.get("entity_id") or call.args[2]["entity_id"] == "number.offset"

    async def test_handler_failure_does_not_record_success(self):
        """A ServiceValidationError (HomeAssistantError subclass) must be
        caught and must NOT advance _last_nibe_offset."""
        adapter, hass = make_adapter(call_side_effect=HomeAssistantError("value out of range"))
        result = await adapter.set_curve_offset(3.0)

        assert result is False
        # bookkeeping not advanced to the failed value
        assert adapter._last_nibe_offset == 0
        assert adapter._last_write is None

    async def test_handler_failure_never_raises(self):
        """The coordinator relies on set_curve_offset never raising."""
        adapter, hass = make_adapter(call_side_effect=TypeError("boom"))
        # Must not raise
        result = await adapter.set_curve_offset(3.0)
        assert result is False


class TestOffsetClamp:
    """Respect the target number's own min/max."""

    async def test_clamps_to_entity_max(self):
        adapter, hass = make_adapter(offset_attrs={"min": -10.0, "max": 5.0})
        await adapter.set_curve_offset(8.0)

        written = hass.services.async_call.call_args.args[2]["value"]
        assert written == 5  # clamped to entity max, not MAX_OFFSET (10)

    async def test_clamps_to_entity_min_for_negative(self):
        adapter, hass = make_adapter(offset_state="0", offset_attrs={"min": -3.0, "max": 3.0})
        await adapter.set_curve_offset(-8.0)

        written = hass.services.async_call.call_args.args[2]["value"]
        assert written == -3

    async def test_default_template_number_range_still_allows_configured(self):
        """A template number left at 0-100 clamps negatives to 0 and warns
        rather than silently failing in a background task."""
        adapter, hass = make_adapter(offset_attrs={"min": 0.0, "max": 100.0})
        # -2 requested; clamps to 0, equals current -> no write, no crash
        result = await adapter.set_curve_offset(-2.0)
        assert result is False

    async def test_malformed_minmax_does_not_crash(self):
        """Non-numeric min/max attributes must be ignored, not raise."""
        adapter, hass = make_adapter(offset_attrs={"min": "n/a", "max": None})
        result = await adapter.set_curve_offset(3.0)

        # Falls back to MIN_OFFSET/MAX_OFFSET clamp, write still succeeds
        assert result is True
        written = hass.services.async_call.call_args.args[2]["value"]
        assert written == 3


class TestUnknownValueMarker:
    """-32768 / -3276.8 are 'no reading' markers, not real values."""

    async def test_marker_read_as_default(self):
        hass = MagicMock(spec=HomeAssistant)
        state = MagicMock()
        state.state = "-32768"
        state.attributes = {}
        hass.states = MagicMock()
        hass.states.get = lambda entity_id: state
        adapter = NibeAdapter(hass, {})

        value = await adapter._read_entity_float("sensor.x", default=99.0)
        assert value == 99.0

    def test_both_markers_defined(self):
        assert -32768.0 in NIBE_UNKNOWN_VALUE_MARKERS
        assert -3276.8 in NIBE_UNKNOWN_VALUE_MARKERS
