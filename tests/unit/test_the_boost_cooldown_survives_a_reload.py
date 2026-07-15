"""The `_service_last_called` cooldown dict is deliberately at MODULE scope, not on the coordinator.

It rate-limits the two services that can hurt the machine (boost_heating commands MAX_OFFSET;
boost_dhw fires the immersion heater via temporary lux). On the coordinator it would die with the
coordinator, so HA's reload button - which re-creates it - would reset the rate limiter: boost to
+10 °C, reload, boost again. `single_config_entry` is true, so a module global cannot leak across
entries. This file pins that the state stays at module scope and no reload path clears it.
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

    HA's reload does not re-import the module; it calls `async_unload_entry` then `async_setup_entry`
    on the module already in `sys.modules`, so module-scope state survives unless a path clears it.
    That is what is checked (rather than importlib.reload, which re-executes the body and would reset
    the dict - the opposite of a config-entry reload).
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
