"""The EffektGuard integration.

EffektGuard is a Home Assistant integration for intelligent NIBE heat pump control,
optimizing for Swedish electricity costs (spot prices and effect tariffs) while
maintaining comfort.

This integration leverages existing integrations (NIBE Myuplink, GE-Spot, weather)
to implement original optimization algorithms designed specifically for Sweden's
effect tariff system.
"""

import asyncio
import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.CLIMATE,
    Platform.SENSOR,
    Platform.NUMBER,
    Platform.SELECT,
    Platform.SWITCH,
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up EffektGuard from a config entry."""
    _LOGGER.info("Setting up EffektGuard integration")

    # Initialize domain data storage
    hass.data.setdefault(DOMAIN, {})

    # Create coordinator with dependency injection
    coordinator = await _create_coordinator(hass, entry)

    # Store coordinator
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Wait for Home Assistant to be ready
    await hass.async_block_till_done()

    # Give other integrations time to initialize (10 second startup delay)
    _LOGGER.debug("Waiting for dependencies to initialize...")
    await asyncio.sleep(10)

    # Perform first refresh
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as err:
        _LOGGER.error("Failed to initialize EffektGuard: %s", err)
        raise ConfigEntryNotReady from err

    # Forward setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    await _async_register_services(hass)

    # Listen for options updates
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    _LOGGER.info("EffektGuard setup complete")
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading EffektGuard integration")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Remove coordinator
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry when options change."""
    _LOGGER.debug("Reloading EffektGuard integration")
    await hass.config_entries.async_reload(entry.entry_id)


async def _create_coordinator(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> EffektGuardCoordinator:
    """Create coordinator with dependency injection.

    This factory function creates all dependencies and injects them into
    the coordinator following clean architecture principles.
    """
    from .adapters.gespot_adapter import GESpotAdapter
    from .adapters.nibe_adapter import NibeAdapter
    from .adapters.weather_adapter import WeatherAdapter
    from .optimization.decision_engine import DecisionEngine
    from .optimization.effect_manager import EffectManager
    from .optimization.price_analyzer import PriceAnalyzer
    from .optimization.thermal_model import ThermalModel

    # Create data adapters
    nibe_adapter = NibeAdapter(hass, entry.data)
    gespot_adapter = GESpotAdapter(hass, entry.data)
    weather_adapter = WeatherAdapter(hass, entry.data)

    # Create optimization components
    price_analyzer = PriceAnalyzer()
    effect_manager = EffectManager(hass)
    thermal_model = ThermalModel(
        thermal_mass=entry.options.get("thermal_mass", 1.0),
        insulation_quality=entry.options.get("insulation_quality", 1.0),
    )

    # Create decision engine with all dependencies
    decision_engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=entry.options,
    )

    # Load persistent state for effect manager
    await effect_manager.async_load()

    # Create coordinator
    coordinator = EffektGuardCoordinator(
        hass=hass,
        nibe_adapter=nibe_adapter,
        gespot_adapter=gespot_adapter,
        weather_adapter=weather_adapter,
        decision_engine=decision_engine,
        effect_manager=effect_manager,
        entry=entry,
    )

    return coordinator


async def _async_register_services(hass: HomeAssistant) -> None:
    """Register integration services."""
    # Services will be implemented in Phase 5
    # Placeholder for future service registration
    pass
