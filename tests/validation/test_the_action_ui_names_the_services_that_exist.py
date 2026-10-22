"""The action UI describes the services the integration registers, as services.yaml describes them.

An action's name, description and field text are translations, under `services` in strings.json -
"every action and field needs one" (contributors.home-assistant.io, core/integrations/actions).
services.yaml carries the same English, which is what the backend publishes as the service
description. strings.json described a `boost_hot_water` service that does not exist (it is
`boost_dhw`) and nothing for force_offset, reset_peak_tracking, boost_dhw or
calculate_optimal_schedule; the one it did describe, boost_heating, said something else than
services.yaml. test_translation_key_parity.py holds every locale to strings.json; this holds
strings.json to the services.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from custom_components.effektguard import _async_register_services

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "effektguard"
STRINGS = json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8"))["services"]
SERVICES_YAML = yaml.safe_load((COMPONENT / "services.yaml").read_text(encoding="utf-8"))


async def _registered_services() -> set[str]:
    hass = MagicMock()
    hass.services.has_service = MagicMock(return_value=False)
    await _async_register_services(hass)
    return {call.args[1] for call in hass.services.async_register.call_args_list}


def _text(described: dict) -> dict:
    """Name, description and the same of each field - what the action UI shows."""
    return {
        "name": described["name"],
        "description": " ".join(described["description"].split()),
        "fields": {
            field: (spec["name"], " ".join(spec["description"].split()))
            for field, spec in (described.get("fields") or {}).items()
        },
    }


async def test_the_action_ui_describes_exactly_the_registered_services():
    registered = await _registered_services()

    assert set(SERVICES_YAML) == registered
    assert set(STRINGS) == registered, (
        f"strings.json describes {sorted(set(STRINGS) - registered)} that do not exist and "
        f"nothing for {sorted(registered - set(STRINGS))}"
    )


@pytest.mark.parametrize("service", sorted(SERVICES_YAML))
def test_each_action_reads_as_services_yaml_describes_it(service):
    assert service in STRINGS, f"strings.json has no text for {service}"
    assert _text(STRINGS[service]) == _text(SERVICES_YAML[service])
