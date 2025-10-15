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
from datetime import datetime, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    BOOST_COOLDOWN_MINUTES,
    DHW_BOOST_COOLDOWN_MINUTES,
    SERVICE_RATE_LIMIT_MINUTES,
)
from .coordinator import EffektGuardCoordinator

_LOGGER = logging.getLogger(__name__)

# Service call cooldown tracking (per hass instance)
_service_last_called: dict[str, datetime] = {}


def _check_service_cooldown(service_name: str, cooldown_minutes: int) -> tuple[bool, int]:
    """Check if service is in cooldown period.

    Args:
        service_name: Name of the service to check
        cooldown_minutes: Cooldown duration in minutes

    Returns:
        Tuple of (is_allowed, remaining_time_seconds)
        - is_allowed: True if service can be called, False if in cooldown
        - remaining_time_seconds: Seconds remaining in cooldown (0 if allowed)
    """
    service_key = f"{DOMAIN}_{service_name}"
    now = dt_util.now()

    if service_key not in _service_last_called:
        return True, 0

    last_call = _service_last_called[service_key]
    cooldown = timedelta(minutes=cooldown_minutes)
    time_since_last = now - last_call

    if time_since_last >= cooldown:
        return True, 0

    remaining = cooldown - time_since_last
    remaining_seconds = int(remaining.total_seconds())
    return False, remaining_seconds


def _update_service_timestamp(service_name: str) -> None:
    """Update the last called timestamp for a service."""
    service_key = f"{DOMAIN}_{service_name}"
    _service_last_called[service_key] = dt_util.now()


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
    # Add latitude from Home Assistant config for climate zone detection
    config_with_latitude = dict(entry.options)
    config_with_latitude["latitude"] = hass.config.latitude

    decision_engine = DecisionEngine(
        price_analyzer=price_analyzer,
        effect_manager=effect_manager,
        thermal_model=thermal_model,
        config=config_with_latitude,
        thermal_predictor=None,  # Will be set after coordinator is created (Phase 6)
        weather_learner=None,  # Will be set after coordinator is created (Phase 6)
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

    # Initialize learning modules (Phase 6)
    await coordinator.async_initialize_learning()

    # Connect thermal predictor and weather learner to decision engine (Phase 6)
    decision_engine.predictor = coordinator.thermal_predictor
    decision_engine.weather_learner = coordinator.weather_learner

    # Connect heat pump model to decision engine
    decision_engine.heat_pump_model = coordinator.heat_pump_model

    return coordinator


async def _async_register_services(hass: HomeAssistant) -> None:
    """Register integration services.

    Services implemented in Phase 5:
    - force_offset: Manual heating curve override
    - reset_peak_tracking: Reset monthly peak data
    - boost_heating: Emergency comfort boost
    - calculate_optimal_schedule: Preview 24h optimization
    """
    import voluptuous as vol
    from homeassistant.helpers import config_validation as cv

    from .const import (
        ATTR_DURATION,
        ATTR_OFFSET,
        MAX_OFFSET,
        MIN_OFFSET,
        SERVICE_BOOST_HEATING,
        SERVICE_CALCULATE_OPTIMAL_SCHEDULE,
        SERVICE_FORCE_OFFSET,
        SERVICE_RESET_PEAK_TRACKING,
    )

    def get_coordinator(hass: HomeAssistant) -> EffektGuardCoordinator | None:
        """Get first available coordinator from domain data."""
        domain_data = hass.data.get(DOMAIN, {})
        for coordinator in domain_data.values():
            if isinstance(coordinator, EffektGuardCoordinator):
                return coordinator
        return None

    async def force_offset_handler(call) -> None:
        """Handle force_offset service call.

        Allows manual override of heating curve offset for specified duration.
        Useful for testing or manual adjustments.

        Rate limited to prevent excessive calls.
        """
        # Check cooldown
        is_allowed, remaining = _check_service_cooldown("force_offset", SERVICE_RATE_LIMIT_MINUTES)
        if not is_allowed:
            _LOGGER.warning(
                "force_offset called too soon. Cooldown: %d seconds remaining", remaining
            )
            return

        coordinator = get_coordinator(hass)
        if not coordinator:
            _LOGGER.error("No EffektGuard coordinator found")
            return

        offset = call.data[ATTR_OFFSET]
        duration = call.data.get(ATTR_DURATION, 60)

        _LOGGER.info("Force offset service called: offset=%s, duration=%s", offset, duration)

        # Validate offset range
        if not MIN_OFFSET <= offset <= MAX_OFFSET:
            _LOGGER.error("Offset %s outside valid range [%s, %s]", offset, MIN_OFFSET, MAX_OFFSET)
            return

        # Set override in decision engine
        coordinator.engine.set_manual_override(offset, duration)

        # Request immediate update
        await coordinator.async_request_refresh()

        # Update last called timestamp
        _update_service_timestamp("force_offset")

        _LOGGER.info("Manual offset override set: %s°C for %s minutes", offset, duration)

    async def reset_peak_tracking_handler(call) -> None:
        """Handle reset_peak_tracking service call.

        Resets monthly peak tracking. Use at start of new billing period
        to clear previous month's peak data.
        """
        coordinator = get_coordinator(hass)
        if not coordinator:
            _LOGGER.error("No EffektGuard coordinator found")
            return

        _LOGGER.info("Reset peak tracking service called")

        # Reset peaks in effect manager
        coordinator.effect.reset_monthly_peaks()

        # Save cleared state
        await coordinator.effect.async_save()

        # Update coordinator data
        await coordinator.async_request_refresh()

        _LOGGER.info("Monthly peak tracking reset successfully")

    async def boost_heating_handler(call) -> None:
        """Handle boost_heating service call.

        Emergency boost mode: maximize heating regardless of cost.
        Useful for quick temperature recovery or unexpected cold periods.

        Rate limited to prevent excessive calls.
        """
        # Check cooldown
        is_allowed, remaining = _check_service_cooldown("boost_heating", BOOST_COOLDOWN_MINUTES)
        if not is_allowed:
            remaining_minutes = int(remaining / 60)
            _LOGGER.warning(
                "boost_heating called too soon. Cooldown: %d minutes remaining", remaining_minutes
            )
            return

        coordinator = get_coordinator(hass)
        if not coordinator:
            _LOGGER.error("No EffektGuard coordinator found")
            return

        duration = call.data.get(ATTR_DURATION, 120)

        _LOGGER.info("Boost heating service called: duration=%s minutes", duration)

        # Set maximum positive offset for boost duration
        boost_offset = MAX_OFFSET  # +10°C
        coordinator.engine.set_manual_override(boost_offset, duration)

        # Request immediate update
        await coordinator.async_request_refresh()

        # Update last called timestamp
        _update_service_timestamp("boost_heating")

        _LOGGER.info("Heating boost activated: +%s°C for %s minutes", boost_offset, duration)

    async def calculate_optimal_schedule_handler(call):
        """Handle calculate_optimal_schedule service call.

        Returns 24-hour preview of optimization decisions based on current
        price forecast, weather, and system state.
        """
        coordinator = get_coordinator(hass)
        if not coordinator:
            _LOGGER.error("No EffektGuard coordinator found")
            return {"error": "No coordinator found"}

        _LOGGER.info("Calculate optimal schedule service called")

        try:
            # Get current data
            if not coordinator.data:
                return {"error": "No coordinator data available"}

            nibe_data = coordinator.data.get("nibe")
            price_data = coordinator.data.get("price")
            weather_data = coordinator.data.get("weather")

            if not all([nibe_data, price_data]):
                return {"error": "Missing required data (NIBE or price)"}

            # Calculate hourly schedule for next 24 hours
            schedule = []
            from datetime import datetime, timedelta

            current_time = datetime.now()

            for hour_offset in range(24):
                forecast_time = current_time + timedelta(hours=hour_offset)
                hour = forecast_time.hour
                quarter = (hour * 4) + (forecast_time.minute // 15)  # 0-95

                # Get price classification for this quarter
                if quarter < len(price_data.today):
                    period = price_data.today[quarter]
                    classification = coordinator.engine.price.get_current_classification(quarter)
                else:
                    # Use tomorrow's data if available
                    tomorrow_quarter = quarter - 96
                    if price_data.tomorrow and tomorrow_quarter < len(price_data.tomorrow):
                        period = price_data.tomorrow[tomorrow_quarter]
                        classification = coordinator.engine.price.get_current_classification(
                            tomorrow_quarter
                        )
                    else:
                        period = None
                        classification = "unknown"

                # Calculate what decision would be made for this time
                # Simplified preview - doesn't include all layers
                if period:
                    base_offset = coordinator.engine.price.get_base_offset(
                        quarter % 96, classification, period.is_daytime
                    )
                    estimated_offset = base_offset
                else:
                    estimated_offset = 0.0

                schedule.append(
                    {
                        "time": forecast_time.strftime("%Y-%m-%d %H:%M"),
                        "hour": hour,
                        "quarter": quarter % 96,
                        "classification": classification,
                        "estimated_offset": round(estimated_offset, 2),
                        "price": round(period.price, 3) if period else None,
                        "is_daytime": period.is_daytime if period else None,
                    }
                )

            _LOGGER.info("Calculated optimal schedule for next 24 hours")
            return {"schedule": schedule, "generated_at": current_time.isoformat()}

        except Exception as err:
            _LOGGER.error("Failed to calculate optimal schedule: %s", err)
            return {"error": str(err)}

    # Define service schemas
    force_offset_schema = vol.Schema(
        {
            vol.Required(ATTR_OFFSET): vol.All(
                vol.Coerce(float), vol.Range(min=MIN_OFFSET, max=MAX_OFFSET)
            ),
            vol.Optional(ATTR_DURATION, default=60): vol.All(
                vol.Coerce(int), vol.Range(min=0, max=480)
            ),
        }
    )

    reset_peak_tracking_schema = vol.Schema({})

    boost_heating_schema = vol.Schema(
        {
            vol.Optional(ATTR_DURATION, default=120): vol.All(
                vol.Coerce(int), vol.Range(min=30, max=360)
            ),
        }
    )

    calculate_optimal_schedule_schema = vol.Schema({})

    # Register services (only if not already registered)
    if not hass.services.has_service(DOMAIN, SERVICE_FORCE_OFFSET):
        hass.services.async_register(
            DOMAIN,
            SERVICE_FORCE_OFFSET,
            force_offset_handler,
            schema=force_offset_schema,
        )
        _LOGGER.debug("Registered service: %s", SERVICE_FORCE_OFFSET)

    if not hass.services.has_service(DOMAIN, SERVICE_RESET_PEAK_TRACKING):
        hass.services.async_register(
            DOMAIN,
            SERVICE_RESET_PEAK_TRACKING,
            reset_peak_tracking_handler,
            schema=reset_peak_tracking_schema,
        )
        _LOGGER.debug("Registered service: %s", SERVICE_RESET_PEAK_TRACKING)

    if not hass.services.has_service(DOMAIN, SERVICE_BOOST_HEATING):
        hass.services.async_register(
            DOMAIN,
            SERVICE_BOOST_HEATING,
            boost_heating_handler,
            schema=boost_heating_schema,
        )
        _LOGGER.debug("Registered service: %s", SERVICE_BOOST_HEATING)

    if not hass.services.has_service(DOMAIN, SERVICE_CALCULATE_OPTIMAL_SCHEDULE):
        hass.services.async_register(
            DOMAIN,
            SERVICE_CALCULATE_OPTIMAL_SCHEDULE,
            calculate_optimal_schedule_handler,
            schema=calculate_optimal_schedule_schema,
            supports_response=True,
        )
        _LOGGER.debug("Registered service: %s", SERVICE_CALCULATE_OPTIMAL_SCHEDULE)

    _LOGGER.info("EffektGuard services registered successfully")
