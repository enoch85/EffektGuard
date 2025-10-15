"""Heat pump model registry for brand-specific profiles.

Central registry that manages all supported heat pump models across manufacturers.
Allows lookup by model ID and provides discovery of supported models.
"""

from typing import Any

from .base import HeatPumpProfile


class HeatPumpModelRegistry:
    """Central registry for all supported heat pump models."""

    _models: dict[str, type[HeatPumpProfile]] = {}

    @classmethod
    def register(cls, model_id: str):
        """Decorator to register a heat pump model profile.

        Args:
            model_id: Unique identifier (e.g., "nibe_f750", "nibe_s1255")

        Returns:
            Decorator function
        """

        def decorator(profile_class: type[HeatPumpProfile]):
            cls._models[model_id] = profile_class
            return profile_class

        return decorator

    @classmethod
    def get_model(cls, model_id: str) -> HeatPumpProfile:
        """Get model profile instance by ID.

        Args:
            model_id: Model identifier

        Returns:
            HeatPumpProfile instance

        Raises:
            ValueError: If model not found
        """
        if model_id not in cls._models:
            raise ValueError(f"Unknown heat pump model: {model_id}")
        return cls._models[model_id]()

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get list of all supported model IDs.

        Returns:
            List of model ID strings
        """
        return list(cls._models.keys())

    @classmethod
    def get_models_by_manufacturer(cls, manufacturer: str) -> list[str]:
        """Get models for specific manufacturer.

        Args:
            manufacturer: Manufacturer name (e.g., "NIBE", "Vaillant")

        Returns:
            List of model IDs for that manufacturer
        """
        return [
            model_id
            for model_id, profile_class in cls._models.items()
            if profile_class().manufacturer.upper() == manufacturer.upper()
        ]

    @classmethod
    def get_models_grouped_by_manufacturer(cls) -> dict[str, list[dict[str, str]]]:
        """Get models grouped by manufacturer for UI display.

        Returns:
            Dict mapping manufacturer to list of {id, name, series} dicts
        """
        grouped: dict[str, list[dict[str, str]]] = {}

        for model_id in cls._models:
            profile = cls.get_model(model_id)
            mfr = profile.manufacturer

            if mfr not in grouped:
                grouped[mfr] = []

            grouped[mfr].append(
                {"id": model_id, "name": profile.model_name, "model_type": profile.model_type}
            )

        return grouped
