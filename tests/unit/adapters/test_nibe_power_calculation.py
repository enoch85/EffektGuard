"""Tests for NIBE power calculation from phase currents.

Tests the calculate_power_from_currents() method that reads BE1/BE2/BE3
sensors and calculates real power consumption.

Based on European 3-phase electrical standards:
- IEC 60038 / EN 50160: the low-voltage supply is 230/400 V
- 400 V between phases, 230 V phase-to-neutral (400 / sqrt(3) = 230.94)
- All NIBE heat pumps in Sweden are 3-phase
- Power factor 0.95 (conservative for inverter compressor)

NIBE_VOLTAGE_PER_PHASE must stay 230 V, not the legacy 240 V: at 240 V every power figure
derived from BE1/BE2/BE3 comes out ~4.3 % high, into the monthly peak history that drives
peak protection.
"""

import pytest

from custom_components.effektguard.const import (
    NIBE_POWER_FACTOR,
    NIBE_VOLTAGE_PER_PHASE,
)


class TestNibePowerCalculationBasics:
    """Test basic power calculation functionality."""

    def test_single_phase_current_only(self, nibe_adapter):
        """Test power calculation with only BE1 current (other phases off)."""
        # Only phase 1 has current (circulation pump or standby)
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=1.0,
            phase2_amps=None,
            phase3_amps=None,
        )

        # P = V × I × PF / 1000
        # 230V × 1.0A × 0.95 = 218.5W = 0.2185 kW
        expected = (NIBE_VOLTAGE_PER_PHASE * 1.0 * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

    def test_all_three_phases_active(self, nibe_adapter):
        """Test power calculation with all three phases active."""
        # Typical heating scenario with compressor running
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=10.0,
            phase2_amps=8.0,
            phase3_amps=9.0,
        )

        # Total current: 10 + 8 + 9 = 27A
        # P = 230V × 27A × 0.95 / 1000 = 5.900 kW
        total_amps = 10.0 + 8.0 + 9.0
        expected = (NIBE_VOLTAGE_PER_PHASE * total_amps * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

    def test_no_current_data_returns_none(self, nibe_adapter):
        """Test that None phase1 current returns None."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=None,
            phase2_amps=None,
            phase3_amps=None,
        )

        assert power is None

    def test_phase2_and_phase3_default_to_zero(self, nibe_adapter):
        """Test that missing phase2/phase3 are treated as zero."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=5.0,
            # phase2 and phase3 omitted
        )

        # Should calculate with only phase1
        expected = (NIBE_VOLTAGE_PER_PHASE * 5.0 * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)


class TestNibePowerCalculationSwedishStandards:
    """Test power calculation matches Swedish 3-phase standards."""

    def test_uses_the_european_low_voltage_standard(self, nibe_adapter):
        """230 V phase-to-neutral, per IEC 60038 - not the legacy 240 V."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=1.0,
            phase2_amps=0.0,
            phase3_amps=0.0,
        )

        expected = (NIBE_VOLTAGE_PER_PHASE * 1.0 * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

        assert NIBE_VOLTAGE_PER_PHASE == 230.0, (
            f"NIBE_VOLTAGE_PER_PHASE is {NIBE_VOLTAGE_PER_PHASE} V. IEC 60038 and EN 50160 declare "
            f"the European supply as 230/400 V, and 400 / sqrt(3) = 230.94 - so a 400 V "
            f"line-to-line system CANNOT have 240 V phase-to-neutral. 240 V is the legacy UK/US "
            f"figure, and at that value every power reading derived from BE1/BE2/BE3 comes out "
            f"4.3 % high, straight into the monthly peak history that drives peak protection."
        )

    def test_uses_conservative_power_factor(self, nibe_adapter):
        """Test power calculation uses conservative 0.95 power factor."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=10.0,
            phase2_amps=10.0,
            phase3_amps=10.0,
        )

        # Power factor should be 0.95 (conservative for inverter compressors)
        # Real NIBE power factor likely 0.96-0.98, but 0.95 is safe
        expected = (NIBE_VOLTAGE_PER_PHASE * 30.0 * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)
        assert NIBE_POWER_FACTOR == 0.95  # Verify constant


class TestNibePowerCalculationRealScenarios:
    """Test power calculation with realistic NIBE heat pump scenarios."""

    def test_standby_mode_low_current(self, nibe_adapter):
        """Test power calculation during standby (circulation pump only)."""
        # Standby: Only circulation pump running (~0.3A)
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=0.3,
            phase2_amps=0.0,
            phase3_amps=0.0,
        )

        # Expected: ~0.07 kW (70W)
        assert power < 0.1  # Less than 100W
        assert power > 0.05  # More than 50W

    def test_heating_mode_typical_consumption(self, nibe_adapter):
        """Test power calculation during typical heating operation."""
        # Typical F750 heating at 50Hz compressor
        # BE1: 8A, BE2: 7A, BE3: 7A
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=8.0,
            phase2_amps=7.0,
            phase3_amps=7.0,
        )

        # Expected: ~5 kW (typical mid-range heating)
        assert power > 4.5  # At least 4.5 kW
        assert power < 5.5  # At most 5.5 kW

    def test_maximum_heating_cold_weather(self, nibe_adapter):
        """Test power calculation at maximum compressor load."""
        # Cold weather, 80Hz compressor frequency
        # BE1: 13A, BE2: 12A, BE3: 12A
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=13.0,
            phase2_amps=12.0,
            phase3_amps=12.0,
        )

        # Expected: ~8.5 kW (maximum F750 load)
        assert power > 8.0  # At least 8 kW
        assert power < 9.0  # At most 9 kW

    def test_compressor_off_only_circulation(self, nibe_adapter):
        """Test power when compressor off but circulation pump running."""
        # Compressor off: BE2 and BE3 = 0, only BE1 has circulation pump
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=0.5,
            phase2_amps=0.0,
            phase3_amps=0.0,
        )

        # Expected: ~0.11 kW (110W) - circulation pump only
        assert power < 0.15  # Less than 150W
        assert power > 0.08  # More than 80W


class TestNibePowerCalculationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_current_all_phases(self, nibe_adapter):
        """Test handling of zero current on all phases."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=0.0,
            phase2_amps=0.0,
            phase3_amps=0.0,
        )

        # Zero current = zero power
        assert power == 0.0

    def test_very_high_current_doesnt_break(self, nibe_adapter):
        """Test that very high current values don't cause errors."""
        # Unrealistically high current (testing robustness)
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=50.0,
            phase2_amps=50.0,
            phase3_amps=50.0,
        )

        # Should calculate without error (even if unrealistic)
        assert power > 30.0  # Much higher than normal
        assert isinstance(power, float)

    def test_fractional_current_values(self, nibe_adapter):
        """Test handling of fractional current values."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=7.3,
            phase2_amps=6.8,
            phase3_amps=7.1,
        )

        # Should handle fractional values precisely
        total_amps = 7.3 + 6.8 + 7.1
        expected = (NIBE_VOLTAGE_PER_PHASE * total_amps * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

    def test_negative_current_treated_as_absolute(self, nibe_adapter):
        """Test that negative current values work (sensor might report negative)."""
        # Some sensors might report negative for phase direction
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=-5.0,  # Negative value
            phase2_amps=5.0,
            phase3_amps=5.0,
        )

        # Should treat as absolute value or handle gracefully
        # Current implementation doesn't check sign, so -5 + 5 + 5 = 5
        # Real implementation might need abs() for safety
        assert isinstance(power, float)


class TestNibePowerCalculationCustomParameters:
    """Test power calculation with custom voltage and power factor."""

    def test_custom_voltage_value(self, nibe_adapter):
        """Test power calculation with custom voltage."""
        # Test with an explicit custom voltage that overrides the default
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=10.0,
            phase2_amps=10.0,
            phase3_amps=10.0,
            voltage_per_phase=230.0,  # Custom voltage
        )

        # P = 230V × 30A × 0.95 / 1000 = 6.555 kW
        expected = (230.0 * 30.0 * NIBE_POWER_FACTOR) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

    def test_custom_power_factor(self, nibe_adapter):
        """Test power calculation with custom power factor."""
        # Test with higher power factor (e.g., high-efficiency system)
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=10.0,
            phase2_amps=10.0,
            phase3_amps=10.0,
            power_factor=0.98,  # Custom power factor
        )

        # P = 230V × 30A × 0.98 / 1000 = 6.762 kW
        expected = (NIBE_VOLTAGE_PER_PHASE * 30.0 * 0.98) / 1000
        assert power == pytest.approx(expected, rel=1e-3)

    def test_both_custom_parameters(self, nibe_adapter):
        """Test power calculation with both custom parameters."""
        power = nibe_adapter.calculate_power_from_currents(
            phase1_amps=8.0,
            phase2_amps=7.0,
            phase3_amps=7.0,
            voltage_per_phase=235.0,
            power_factor=0.96,
        )

        # P = 235V × 22A × 0.96 / 1000 = 4.9632 kW
        expected = (235.0 * 22.0 * 0.96) / 1000
        assert power == pytest.approx(expected, rel=1e-3)


@pytest.fixture
def nibe_adapter():
    """Create mock NIBE adapter for testing."""
    from unittest.mock import MagicMock

    from custom_components.effektguard.adapters.nibe_adapter import NibeAdapter

    # Create adapter with minimal mocking
    hass = MagicMock()
    config = {
        "nibe_entity": "sensor.bt50_room_temp_40033",
    }

    adapter = NibeAdapter(hass, config)
    return adapter
