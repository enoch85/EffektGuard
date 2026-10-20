"""The decision gets the power its own cycle measured - from the first cycle on.

Power was measured after the decision, so a fresh coordinator's first decision had none: peak
protection was off for that cycle, and the log advised configuring a power sensor the installation
had (production 2026-10-19 21:35:10 and 21:50:35, a meter configured; EG-03 of the 2026-10-20 log
investigation). Every later decision got the previous cycle's reading. Nor may it ever get
`peak_today`, a daily maximum: one morning spike would pin the effect layer to CRITICAL until
midnight.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from custom_components.effektguard.adapters.nibe_adapter import NibeState
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.optimization.decision_engine import OptimizationDecision
from custom_components.effektguard.optimization.effect_layer import EffectManager

STOCKHOLM = ZoneInfo("Europe/Stockholm")
METER = "sensor.grid_import"


async def test_each_decision_gets_the_power_of_its_own_cycle():
    hass = MagicMock()
    hass.config.latitude = 55.6
    hass.async_add_executor_job = AsyncMock(side_effect=lambda func, *args: func(*args))
    meter = MagicMock(attributes={"unit_of_measurement": "W"})
    hass.states.get = MagicMock(side_effect=lambda entity_id: meter if entity_id == METER else None)
    pump = NibeState(
        outdoor_temp=5.0,
        indoor_temp=21.0,
        supply_temp=35.0,
        return_temp=30.0,
        degree_minutes=-100.0,
        current_offset=0.0,
        is_heating=True,
        is_hot_water=False,
        timestamp=datetime(2026, 10, 19, 21, 35, 10, tzinfo=STOCKHOLM),
    )
    nibe = MagicMock(power_sensor_entity=METER, _power_sensor_entity=METER)
    nibe.get_current_state = AsyncMock(return_value=pump)
    nibe.is_enhanced_ventilation_active = AsyncMock(return_value=None)
    engine = MagicMock()
    engine.calculate_decision = MagicMock(
        return_value=OptimizationDecision(offset=0.0, reasoning="t", layers=[])
    )
    effect = EffectManager(hass)
    effect.record_period_measurement = AsyncMock(return_value=None)
    effect.async_save = AsyncMock()
    coordinator = EffektGuardCoordinator(
        hass,
        nibe,
        MagicMock(get_prices=AsyncMock(return_value=None)),
        MagicMock(get_forecast=AsyncMock(return_value=None)),
        engine,
        effect,
        MagicMock(data={}, options={}),
    )
    coordinator._record_learning_observations = AsyncMock()
    coordinator._schedule_aligned_refresh = MagicMock()

    for watts, minute in (("5000", 12), ("2000", 17)):  # 21:12 then 21:17, the next quarter
        meter.state = watts
        with patch(
            "homeassistant.util.dt.now",
            return_value=datetime(2026, 10, 19, 21, minute, 10, tzinfo=STOCKHOLM),
        ):
            await coordinator._async_update_data()

    powers = [call.args[4] for call in engine.calculate_decision.call_args_list]
    assert powers == [5.0, 2.0]
