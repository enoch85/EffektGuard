"""NIBE F1155 heat pump profile.

Inverter-controlled GSHP - F-series ground source, predecessor of the S1155.
Available in 3 sizes: 1.5-6 kW, 4-12 kW, 4-16 kW.

Added for issue #18: F1155 owners connect via local Modbus (nibe_heatpump
integration / MODBUS40) and previously had to pick the S1155 profile.

Source: https://www.nibe.eu/en-eu/products/heat-pumps/ground-source-heat-pumps
Physics inherit from the S1155 (same GSHP family, stable 5-8°C ground
temperature); COP curve set slightly below the S1155 (older inverter
platform, SCOP ~5.0).
"""

from dataclasses import dataclass

from ..registry import HeatPumpModelRegistry
from .s1155 import NibeS1155Profile


@HeatPumpModelRegistry.register("nibe_f1155")
@dataclass
class NibeF1155Profile(NibeS1155Profile):
    """NIBE F1155 Ground Source Heat Pump (Inverter-controlled).

    **Available Sizes**: 1.5-6 kW, 4-12 kW, 4-16 kW
    **Target Market**: Small to large properties with ground source installation
    **Technology**: Inverter-controlled compressor, adjusts to heating demand
    **Key Advantage**: Stable ground temperature (5-8°C year-round) = consistent high COP

    Inherits flow-temperature and power-validation behavior from the S1155
    sibling profile (same hydraulic family); only identity, sizing, and the
    slightly lower COP curve differ. Baseline: mid-range 4-12 kW variant.
    """

    model_name: str = "F1155"
    model_type: str = "F-series GSHP"

    # Mid-range variant (4-12 kW)
    rated_power_kw: tuple[float, float] = (4.0, 12.0)  # Heat output
    typical_electrical_range_kw: tuple[float, float] = (0.6, 2.8)  # Estimated from SCOP ~5.0
    modulation_range: tuple[int, int] = (20, 120)

    typical_cop_range: tuple[float, float] = (3.3, 5.3)  # GSHP, slightly below S1155

    def __post_init__(self):
        """Initialize COP curve - S1155 shape shifted slightly down for the
        older F-series inverter platform."""
        self.cop_curve = {
            7: 5.3,
            5: 5.1,
            0: 4.8,
            -5: 4.6,  # Ground temp stable
            -10: 4.3,  # Ground temp stable
            -15: 4.1,
            -20: 3.8,
            -25: 3.6,
            -30: 3.3,  # Much better than ASHP at extreme temps
        }
