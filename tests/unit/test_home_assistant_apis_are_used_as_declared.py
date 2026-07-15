"""Two Home Assistant APIs must be handed the exact types they check for.

`calculate_optimal_schedule` must register with SupportsResponse.OPTIONAL, not a bare `True`: HA
compares the value by identity, so `True` passes `is not SupportsResponse.NONE` but fails
`is SupportsResponse.OPTIONAL`, advertising the service as response-REQUIRED.

`EffektGuardCoordinator` must pass `config_entry=` to DataUpdateCoordinator.__init__. Omitting it
falls back to a ContextVar HA removes in 2026.8, leaving `coordinator.config_entry` None for any
coordinator built outside async_setup_entry.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import SupportsResponse

from custom_components.effektguard import _async_register_services
from custom_components.effektguard.coordinator import EffektGuardCoordinator


async def test_the_service_declares_an_optional_response_not_a_required_one():
    """`supports_response=True` advertises calculate_optimal_schedule as response-REQUIRED."""
    hass = MagicMock()
    hass.services.has_service.return_value = False
    hass.services.async_register = MagicMock()

    await _async_register_services(hass)

    responses = {
        call.args[1] if len(call.args) > 1 else call.kwargs.get("service"): call.kwargs[
            "supports_response"
        ]
        for call in hass.services.async_register.call_args_list
        if "supports_response" in call.kwargs
    }

    assert responses, "no service registered a supports_response at all"

    for service, response in responses.items():
        assert isinstance(response, SupportsResponse), (
            f"{service} passed {response!r} as supports_response. Home Assistant expects a "
            f"SupportsResponse enum and compares it by identity: a bare True satisfies "
            f"`is not SupportsResponse.NONE` but fails `is SupportsResponse.OPTIONAL`, so the "
            f"service is advertised as response-REQUIRED."
        )
        assert response is SupportsResponse.OPTIONAL, (
            f"{service} returns a dict when it can and nothing when it cannot, so its response is "
            f"OPTIONAL. It declares {response!r}."
        )


def test_the_coordinator_hands_home_assistant_its_config_entry():
    """Omitting it falls back to a ContextVar that Home Assistant removes in 2026.8."""
    source = inspect.getsource(EffektGuardCoordinator.__init__)

    assert "config_entry=" in source, (
        "EffektGuardCoordinator does not pass `config_entry=` to DataUpdateCoordinator.__init__. "
        "Home Assistant falls back to a deprecated ContextVar for it - breaks_in_ha_version "
        '"2026.8" - and coordinator.config_entry is None for any coordinator constructed outside '
        "async_setup_entry, which several call sites read without checking."
    )


def test_the_config_entry_actually_arrives():
    """Behavioural, not just structural: build one and read it back."""
    hass = MagicMock()
    hass.data = {}
    hass.config = MagicMock(latitude=59.3, config_dir="/tmp/test")
    hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))

    entry = MagicMock()
    entry.data = MagicMock()
    entry.data.get.side_effect = lambda key, default=None: default
    entry.options = MagicMock()
    entry.options.get.side_effect = lambda key, default=None: default

    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=MagicMock(),
        gespot_adapter=MagicMock(),
        weather_adapter=MagicMock(),
        decision_engine=MagicMock(),
        effect_manager=MagicMock(),
        entry=entry,
    )

    assert coordinator.config_entry is entry, (
        "coordinator.config_entry is not the entry it was constructed with. Home Assistant sets it "
        "from the `config_entry=` argument; without it, it is whatever the deprecated ContextVar "
        "happened to hold - None, outside async_setup_entry."
    )
