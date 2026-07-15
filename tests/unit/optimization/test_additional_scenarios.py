"""Config-key guards for DecisionEngine.

The engine reads the user's target from config key 'target_indoor_temp'. If the key it reads
ever diverges from the key the config carries, a mismatched target is silently ignored and the
engine falls back to the default. These pin the key it reads.
"""

from unittest.mock import MagicMock

from custom_components.effektguard.optimization.decision_engine import DecisionEngine
from custom_components.effektguard.const import DEFAULT_TARGET_TEMP


class TestConfigurationFlow:
    """The engine must read the target from the key the config actually carries."""

    def test_the_target_temperature_key_the_engine_reads_is_the_one_it_is_given(self):
        """The engine reads config key 'target_indoor_temp'; a config carrying any other key is
        silently ignored and it falls back to the default.
        """
        engine = DecisionEngine(
            price_analyzer=MagicMock(),
            effect_manager=MagicMock(),
            thermal_model=MagicMock(),
            config={"target_indoor_temp": 19.0},
        )

        assert engine.target_temp == 19.0, (
            f"The engine was configured with a 19.0 C target and read {engine.target_temp}. The "
            f"key it reads is 'target_indoor_temp'; a config carrying 'target_temperature' is "
            f"silently ignored, and the engine falls back to the default."
        )

    def test_a_config_without_a_target_falls_back_to_the_default(self):
        engine = DecisionEngine(
            price_analyzer=MagicMock(),
            effect_manager=MagicMock(),
            thermal_model=MagicMock(),
            config={},
        )

        assert engine.target_temp == DEFAULT_TARGET_TEMP
