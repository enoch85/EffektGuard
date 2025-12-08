"""Optimization engine for EffektGuard.

Pure Python optimization logic independent of Home Assistant.
Implements multi-layer decision making for cost optimization while
maintaining comfort and heat pump health.
"""

from .decision_engine import DecisionEngine, LayerDecision, OptimizationDecision
from .effect_layer import EffectLayerDecision, EffectManager, PeakEvent
from .price_layer import (
    CheapestWindowResult,
    PriceAnalyzer,
    PriceData,
    PriceForecast,
    QuarterPeriod,
)
from .thermal_layer import ThermalModel, estimate_dm_recovery_time, get_thermal_debt_status

__all__ = [
    "CheapestWindowResult",
    "DecisionEngine",
    "EffectLayerDecision",
    "EffectManager",
    "LayerDecision",
    "OptimizationDecision",
    "PeakEvent",
    "PriceAnalyzer",
    "PriceData",
    "PriceForecast",
    "QuarterPeriod",
    "ThermalModel",
    "estimate_dm_recovery_time",
    "get_thermal_debt_status",
]
