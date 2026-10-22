"""A hot-water boost the USER commanded is not the price optimizer's to cancel.

`boost_dhw` records HOW LONG the user asked for, and while that window is open:
- the hot-water optimiser leaves the switch alone - neither its price stops nor its own abort
  conditions end the boost,
- the safety gate (`hot_water_unsafe_reason`: DM at the aux limit, the house below its floor) stops
  it, and refuses to start it,
- expiry stops it through the same owned door the unload cleanup uses,
- and all of that on every control cycle, whether hot-water optimisation is switched on or not.

Only safety outranks the user; cost optimization does not.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from homeassistant.const import STATE_ON
from homeassistant.exceptions import HomeAssistantError

from custom_components.effektguard.const import (
    CONF_NIBE_TEMP_LUX_ENTITY,
    DM_THRESHOLD_AUX_LIMIT,
    MIN_TEMP_LIMIT,
)
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.climate_zones import ClimateZoneDetector
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler
from custom_components.effektguard.optimization.price_layer import PriceAnalyzer
from custom_components.effektguard.optimization.thermal_layer import EmergencyLayer

from tests.conftest import create_mock_hass

NOW = datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc)
LUX = "switch.temporary_lux_50004"
BOOST_MINUTES = 60
CYCLE = timedelta(minutes=5)


def _coordinator(lux_state: str = "off") -> EffektGuardCoordinator:
    coordinator = EffektGuardCoordinator.__new__(EffektGuardCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.services.async_call = AsyncMock()
    lux = MagicMock()
    lux.state = lux_state
    coordinator.hass.states.get = MagicMock(return_value=lux)
    coordinator.entry = MagicMock()
    coordinator.entry.data = {"target_indoor_temp": 21.0}
    coordinator.entry.options = {}
    coordinator.data = {}
    coordinator.last_update_success = True
    coordinator.temp_lux_entity = LUX
    coordinator._shutdown_requested = False
    coordinator._lux_boost_is_ours = False
    coordinator._service_boost_until = None
    coordinator._last_dhw_control_time = NOW - timedelta(hours=2)
    coordinator.dhw_optimizer = MagicMock()
    coordinator._raise_dhw_control_issue = MagicMock()
    coordinator._clear_dhw_control_issue = MagicMock()
    return coordinator


class _House:
    """A real coordinator running real control cycles; the pump and the lux switch are faked.

    The hot-water optimiser is the real one, so its decisions - and, on the old code, the abort
    conditions it hands back - are what production would see.
    """

    def __init__(self, *, hot_water_optimisation: bool, tank: float | None, outdoor: float = 0.0):
        self.dm = -200.0
        self.indoor = 21.0
        self.lux = "off"
        self.switch_calls: list[str] = []

        async def read():
            return SimpleNamespace(
                indoor_temp=self.indoor,
                outdoor_temp=outdoor,
                flow_temp=35.0,
                degree_minutes=self.dm,
                compressor_hz=None,
                is_hot_water=False,
                timestamp=NOW,
                dhw_top_temp=tank,
                dhw_amount_minutes=None,
                power_kw=None,
                current_offset=0.0,
            )

        async def switch(domain, service, data, blocking=False):
            self.switch_calls.append(service)
            self.lux = STATE_ON if service == "turn_on" else "off"

        hass = create_mock_hass()
        hass.services.async_call = AsyncMock(side_effect=switch)
        hass.states.get = MagicMock(
            side_effect=lambda entity_id: (
                SimpleNamespace(state=self.lux, last_changed=NOW) if entity_id == LUX else None
            )
        )
        nibe = MagicMock()
        nibe.get_current_state = AsyncMock(side_effect=read)
        nibe.set_curve_offset = AsyncMock(return_value=0)
        nibe.is_enhanced_ventilation_active = AsyncMock(return_value=False)
        nibe.entity_cache = {}
        detector = ClimateZoneDetector(latitude=59.33)
        engine = MagicMock()
        engine.climate_detector = detector
        engine.emergency_layer = EmergencyLayer(detector, heating_type="radiator")
        engine.price = PriceAnalyzer()
        engine.calculate_decision = MagicMock(
            return_value=OptimizationDecision(offset=0.0, reasoning="test", layers=[])
        )
        effect = MagicMock()
        effect.async_save = AsyncMock()
        effect.get_monthly_peak_summary = MagicMock(return_value={"count": 0, "highest": 0.0})
        data = {
            "enable_optimization": True,
            "enable_hot_water_optimization": hot_water_optimisation,
            "target_indoor_temp": 21.0,
            CONF_NIBE_TEMP_LUX_ENTITY: LUX,
        }
        self.coordinator = EffektGuardCoordinator(
            hass=hass,
            nibe_adapter=nibe,
            gespot_adapter=MagicMock(get_prices=AsyncMock(return_value=None)),
            weather_adapter=MagicMock(get_forecast=AsyncMock(return_value=None)),
            decision_engine=engine,
            effect_manager=effect,
            entry=MagicMock(data=data, options={}),
        )
        self.coordinator._update_peak_tracking = AsyncMock()
        self.coordinator._record_learning_observations = AsyncMock()
        self.coordinator._save_learned_data = AsyncMock()
        self.coordinator._schedule_aligned_refresh = MagicMock()
        # Past start-up: the optimiser's control runs, so on the old code it gets its say.
        self.coordinator._startup_grace_timeout = NOW - timedelta(hours=1)
        self.coordinator._startup_update_count = self.coordinator._startup_grace_updates

    async def cycle(self) -> None:
        """One control cycle, publishing its data as the aligned loop does."""
        self.coordinator.data = await self.coordinator._drive_the_pump()

    async def boost(self) -> None:
        await self.coordinator.async_start_dhw_boost(BOOST_MINUTES, NOW)


@pytest.mark.asyncio
async def test_the_price_stop_does_not_cancel_a_user_boost():
    coordinator = _coordinator(lux_state="off")
    await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)
    assert coordinator._lux_boost_is_ours is True

    # Next cycle: lux is on, prices are high, the optimizer wants it off.
    coordinator.hass.states.get.return_value.state = "on"
    coordinator.hass.services.async_call.reset_mock()

    stop = SimpleNamespace(should_heat=False, abort_conditions=[], priority_reason="EXPENSIVE")
    await coordinator._apply_dhw_control(stop, 45.0, NOW + timedelta(minutes=5))

    coordinator.hass.services.async_call.assert_not_awaited()


@pytest.mark.parametrize(
    "hot_water_optimisation,tank",
    [(True, 45.0), (False, 45.0), (True, None)],
    ids=["hot_water_optimisation_on", "hot_water_optimisation_off", "no_tank_reading"],
)
@pytest.mark.asyncio
async def test_a_user_boost_ends_when_its_duration_is_over(hot_water_optimisation, tank):
    """The duration is the service's promise in every configuration, not only with automatic
    hot-water control on (off by default) and a tank reading the optimiser can decide on."""
    with freeze_time(NOW) as clock:
        house = _House(hot_water_optimisation=hot_water_optimisation, tank=tank)
        await house.boost()

        clock.tick(timedelta(minutes=BOOST_MINUTES) - CYCLE)
        await house.cycle()
        assert house.switch_calls == ["turn_on"], "the boost was ended before its duration"

        clock.tick(2 * CYCLE)
        await house.cycle()

    assert house.switch_calls == ["turn_on", "turn_off"], (
        f"{BOOST_MINUTES + 5} minutes into a {BOOST_MINUTES}-minute boost the switch is still on: "
        f"nothing but NIBE's own lux timeout will end it"
    )
    assert house.coordinator._service_boost_until is None


@pytest.mark.parametrize(
    "dm,indoor,hot_water_optimisation",
    [(DM_THRESHOLD_AUX_LIMIT - 100, 21.0, True), (-200.0, MIN_TEMP_LIMIT - 0.5, False)],
    ids=["dm_past_the_aux_limit", "house_below_its_safety_floor"],
)
@pytest.mark.asyncio
async def test_the_safety_gate_stops_a_user_boost(dm, indoor, hot_water_optimisation):
    """The one thing that outranks the user. On the old code a decision to hold hot water back
    carried no abort conditions, so the boost ran on at any debt and in any cold."""
    with freeze_time(NOW) as clock:
        house = _House(hot_water_optimisation=hot_water_optimisation, tank=45.0, outdoor=-10.0)
        await house.boost()

        house.dm, house.indoor = dm, indoor
        clock.tick(CYCLE)
        await house.cycle()

    assert house.switch_calls == ["turn_on", "turn_off"], (
        f"DM {dm:.0f}, indoor {indoor:.1f} C - past the safety gate - and the user's hot-water "
        f"boost is still running"
    )
    assert house.coordinator._service_boost_until is None


@pytest.mark.asyncio
async def test_the_optimisers_own_abort_conditions_do_not_stop_a_user_boost():
    """Room 0.6 C below target at +3 C outside: no danger, and nothing the user asked to be
    stopped for. The optimiser's comfort abort (`indoor_temp < target - 0.5`) cut the boost
    five minutes in."""
    with freeze_time(NOW) as clock:
        house = _House(hot_water_optimisation=True, tank=40.0, outdoor=3.0)
        house.indoor = 20.4
        await house.boost()

        for _ in range(3):
            clock.tick(CYCLE)
            await house.cycle()

    assert house.switch_calls == ["turn_on"], (
        "the hot-water optimiser's own abort conditions ended a boost the user commanded; only "
        "the safety gate and the duration may"
    )


@pytest.mark.asyncio
async def test_a_user_boost_is_refused_where_the_safety_gate_would_stop_it():
    """One rule both ends: a boost started below the floor would be stopped five minutes later."""
    coordinator = _coordinator(lux_state="off")
    coordinator.dhw_optimizer = IntelligentDHWScheduler()
    coordinator.data = {
        "nibe": SimpleNamespace(degree_minutes=-200.0, indoor_temp=MIN_TEMP_LIMIT - 0.5)
    }

    with pytest.raises(HomeAssistantError):
        await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    coordinator.hass.services.async_call.assert_not_awaited()
    assert coordinator._service_boost_until is None


@pytest.mark.asyncio
async def test_a_boost_is_refused_while_optimization_is_off():
    """OFF hands control to the pump's own curve - it does not fire the immersion heater on request."""
    coordinator = _coordinator()
    coordinator.entry.data = {"enable_optimization": False}

    with pytest.raises(HomeAssistantError):
        await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    coordinator.hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_unload_cleanup_closes_the_window_too():
    coordinator = _coordinator(lux_state="on")
    await coordinator.async_start_dhw_boost(duration_minutes=60, now_time=NOW)

    await coordinator._cancel_our_dhw_boost()

    assert coordinator._service_boost_until is None
    assert coordinator._lux_boost_is_ours is False


def test_the_service_no_longer_advertises_a_temperature_it_cannot_set():
    """Temporary lux is a switch: the pump owns the temperature. services.yaml must not lie."""
    from pathlib import Path

    import yaml

    services = yaml.safe_load(
        Path("custom_components/effektguard/services.yaml").read_text(encoding="utf-8")
    )
    fields = services["boost_dhw"].get("fields", {})
    assert "target_temp" not in fields
    assert "duration" in fields
