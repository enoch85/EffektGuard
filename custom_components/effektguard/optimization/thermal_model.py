"""Thermal model for building thermal behavior prediction.

Models heat storage and loss characteristics for predictive control.
Enables pre-heating and thermal energy banking strategies.
"""


class ThermalModel:
    """Model building thermal characteristics for predictive control.

    Provides thermal mass and insulation quality parameters used by
    DecisionEngine for heat loss calculations and comfort predictions.
    """

    def __init__(
        self,
        thermal_mass: float = 1.0,  # Relative scale 0.5-2.0
        insulation_quality: float = 1.0,  # Relative scale 0.5-2.0
    ):
        """Initialize thermal model.

        Args:
            thermal_mass: Relative thermal mass (1.0 = normal)
                - 0.5 = low mass (timber frame)
                - 1.0 = normal mass (mixed construction)
                - 2.0 = high mass (concrete/masonry)
            insulation_quality: Relative insulation (1.0 = normal)
                - 0.5 = poor insulation
                - 1.0 = standard insulation
                - 2.0 = excellent insulation
        """
        self.thermal_mass = thermal_mass
        self.insulation_quality = insulation_quality

    def get_prediction_horizon(self) -> float:
        """Get prediction horizon for weather forecasting.

        Base implementation returns default 12 hours.
        AdaptiveThermalModel overrides this with UFH-type-specific values.

        Returns:
            Prediction horizon in hours (default 12.0)
        """
        return 12.0  # Default medium horizon
