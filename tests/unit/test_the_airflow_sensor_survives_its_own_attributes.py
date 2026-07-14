"""The airflow_thermal_gain sensor must render its attributes against the REAL optimizer.

The branch deleted ``AirflowOptimizer.get_enhancement_stats()`` (decision-history bookkeeping
nothing consumed) but left the sensor's attribute block calling it. Every state update of
``sensor.effektguard_airflow_thermal_gain`` on an F750/F730 then raised ``AttributeError``.

No existing test caught it because the fixtures' coordinator is a MagicMock, and a MagicMock
answers ``get_enhancement_stats()`` cheerfully - the same trap that hid the removed
``hass.components`` API (F-068). So this test builds the one object that matters for the
failure - a REAL ``AirflowOptimizer`` - and renders the attributes through the real sensor.
"""

from unittest.mock import MagicMock, Mock

from custom_components.effektguard.optimization.airflow_optimizer import AirflowOptimizer
from custom_components.effektguard.sensor import SENSORS, EffektGuardSensor


def test_attribute_render_calls_only_methods_the_real_optimizer_has():
    description = next(s for s in SENSORS if s.key == "airflow_thermal_gain")

    coordinator = MagicMock()
    coordinator.airflow_optimizer = AirflowOptimizer()  # the real thing - no auto-attributes
    coordinator.data = {"airflow_decision": None}

    entry = Mock()
    entry.entry_id = "test-entry"

    sensor = EffektGuardSensor(coordinator, entry, description)

    attrs = sensor.extra_state_attributes  # must not raise

    assert isinstance(attrs, dict)
