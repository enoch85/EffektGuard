"""The airflow_thermal_gain sensor must render its attributes against a REAL AirflowOptimizer.

``get_enhancement_stats()`` was deleted (bookkeeping nothing consumed), but the attribute block once
still called it, raising AttributeError on every update. A MagicMock coordinator answers any method
cheerfully and hides that, so this test wires the real optimizer and renders the real sensor.
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
