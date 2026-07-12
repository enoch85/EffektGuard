"""This global is deliberate. Moving it onto the coordinator opens a one-click bypass.

`_service_last_called` is a module-level dict, and the audit filed that as a defect: "cooldowns
leak across reloads and config entries". Both halves of that are wrong, and acting on it would
remove a guard that protects the heat pump.

**Across config entries**: `manifest.json` sets `"single_config_entry": true`, so there is never
more than one. Nothing to leak into.

**Across reloads**: that is the point. The cooldowns rate-limit the two services that can actually
hurt the machine —

    boost_heating   commands MAX_OFFSET, +10.0 °C, for 45 minutes
    boost_dhw       fires the immersion heater through NIBE's temporary lux, for 60 minutes

Hold that state on the coordinator and it dies with the coordinator. **Reloading the integration —
two clicks in the UI — would then reset the rate limiter**, and a user could drive the pump to +10 °C
again immediately, and again after that. A cooldown you can clear by reloading is not a cooldown.

So the global survives the reload on purpose, and this test exists to stop it being helpfully
tidied away into per-entry state. If you need to reset a cooldown, do it explicitly and visibly -
not as a side effect of a reload.
"""

from __future__ import annotations


import inspect

from custom_components.effektguard import (
    _check_service_cooldown,
    _service_last_called,
    _update_service_timestamp,
)
from custom_components.effektguard.const import (
    DHW_BOOST_COOLDOWN_MINUTES,
    HEATING_BOOST_COOLDOWN_MINUTES,
    MAX_OFFSET,
)


def test_the_cooldown_actually_blocks_a_second_boost():
    """Precondition: the rate limiter rate-limits."""
    _service_last_called.clear()

    allowed, _ = _check_service_cooldown("boost_heating", HEATING_BOOST_COOLDOWN_MINUTES)
    assert allowed, "the first boost must be allowed"

    _update_service_timestamp("boost_heating")

    allowed, remaining = _check_service_cooldown("boost_heating", HEATING_BOOST_COOLDOWN_MINUTES)
    assert not allowed, (
        f"A second boost_heating was allowed immediately after the first. It commands "
        f"{MAX_OFFSET:+.0f} °C."
    )
    assert remaining > 0


def test_the_cooldown_state_is_not_held_on_the_coordinator():
    """Structural, and the whole point of the file.

    Anything the coordinator owns is destroyed when the entry is unloaded. Home Assistant's reload
    button unloads and re-sets-up the entry, so a cooldown living there is cleared by a reload -
    and the two services it guards are the two that can drive the pump to +10 °C and light the
    immersion heater.
    """
    from custom_components.effektguard.coordinator import EffektGuardCoordinator

    coordinator_source = inspect.getsource(EffektGuardCoordinator)

    assert "_service_last_called" not in coordinator_source, (
        "The service-cooldown state has been moved onto the coordinator. The coordinator is "
        "destroyed on unload, so reloading the integration now RESETS the cooldown on "
        "boost_heating (+10 °C) and boost_dhw (the immersion heater). A rate limiter that a reload "
        "clears is not a rate limiter. It belongs at module scope, and deliberately so."
    )


def test_no_reload_path_clears_the_cooldown():
    """A config-entry reload must not forget that a boost just happened.

    Home Assistant's reload does NOT re-import the module - it calls `async_unload_entry` and then
    `async_setup_entry` on the module already in `sys.modules`, so anything at module scope simply
    survives. The only way a reload could clear these cooldowns is if one of those paths went and
    cleared them, so that is what is checked.

    (`importlib.reload()` would be the wrong way to test this: it re-executes the module body and
    resets the dict, which is the opposite of what a config-entry reload does. It would fail here
    while the production behaviour was correct.)
    """
    import custom_components.effektguard as integration

    for name in ("async_unload_entry", "_async_unregister_services", "async_setup_entry"):
        source = inspect.getsource(getattr(integration, name))

        assert "_service_last_called" not in source, (
            f"{name} touches _service_last_called. Clearing the service cooldowns on unload or "
            f"setup makes Home Assistant's reload button a one-click reset for the rate limiter on "
            f"boost_heating ({MAX_OFFSET:+.0f} °C) and boost_dhw (the immersion heater)."
        )


def test_the_dhw_cooldown_is_long_enough_to_matter():
    """The two guarded services are the two that can hurt the machine."""
    assert HEATING_BOOST_COOLDOWN_MINUTES >= 30, (
        f"boost_heating commands {MAX_OFFSET:+.0f} °C. A {HEATING_BOOST_COOLDOWN_MINUTES}-minute "
        f"cooldown is not a meaningful limit on that."
    )
    assert DHW_BOOST_COOLDOWN_MINUTES >= 30, (
        f"boost_dhw fires the immersion heater. A {DHW_BOOST_COOLDOWN_MINUTES}-minute cooldown is "
        f"not a meaningful limit on that."
    )
