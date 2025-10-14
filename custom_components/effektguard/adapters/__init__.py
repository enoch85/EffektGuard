"""Adapters for external integration interfaces."""

from .gespot_adapter import GESpotAdapter
from .nibe_adapter import NibeAdapter
from .weather_adapter import WeatherAdapter

__all__ = ["GESpotAdapter", "NibeAdapter", "WeatherAdapter"]
