"""Comfort Layer - Reactive adjustment to maintain comfort.

Phase 8 of layer refactoring: Comfort layer extraction.
Provides gentle steering toward target temperature even within tolerance zone.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ..const import (
    COMFORT_CORRECTION_MULT,
    COMFORT_TOO_COLD_CORRECTION_MULT,
    LAYER_WEIGHT_COMFORT_CRITICAL,
    LAYER_WEIGHT_COMFORT_MAX,
    LAYER_WEIGHT_COMFORT_MIN,
    MODE_CONFIGS,
    OPTIMIZATION_MODE_BALANCED,
    OptimizationModeConfig,
    OVERSHOOT_PROTECTION_FULL,
    OVERSHOOT_PROTECTION_OFFSET_MAX,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class ComfortLayerDecision:
    """Decision from the comfort layer.

    Encapsulates comfort-based temperature adjustments.
    """

    name: str
    offset: float
    weight: float
    reason: str


class ComfortLayer:
    """Comfort layer for reactive temperature adjustments.

    Provides gentle steering toward target temperature, and a graduated coast when the house is
    warmer than its band.

    Mode affects behavior:
    - Comfort mode: Tighter dead zone (0.1°C), stronger weight multiplier (1.3x)
    - Balanced mode: Standard dead zone (0.2°C), normal weights
    - Savings mode: Wider dead zone (0.3°C), weaker weight multiplier (0.7x)
    """

    def __init__(
        self,
        mode_config: Optional[OptimizationModeConfig] = None,
        tolerance_range: float = 0.5,
        target_temp: float = 21.0,
    ):
        """Initialize comfort layer.

        Args:
            mode_config: Mode configuration (dead_zone, comfort_weight_multiplier)
            tolerance_range: Temperature tolerance range (°C)
            target_temp: Target indoor temperature (°C)
        """
        self.mode_config = mode_config or MODE_CONFIGS[OPTIMIZATION_MODE_BALANCED]
        self.tolerance_range = tolerance_range
        self.target_temp = target_temp

    def evaluate_layer(self, nibe_state) -> ComfortLayerDecision:
        """Evaluate comfort layer decision.

        Above the band the response is one continuous ramp, from the in-band correction at the band
        edge to a full coast at OVERSHOOT_PROTECTION_FULL. It reads no prices: a vote that changes
        with the look-ahead is a cliff of its own (see const.py).

        Args:
            nibe_state: Current NIBE state

        Returns:
            ComfortLayerDecision with comfort correction
        """
        # A NIBE with no room sensor (no BT50) is a legitimate configuration: it runs on degree
        # minutes and the heating curve. The adapter substitutes DEFAULT_INDOOR_TEMP (21.0) so the
        # display has something to show, and sets `indoor_temp_valid=False` - in its own words, "so
        # comfort-reasoning layers abstain". This is the comfort-reasoning layer, and it did not.
        #
        # Every deviation computed from that placeholder is fiction. With a target BELOW 21.0 - and
        # 18.5 is now an allowed target - the layer reads a permanent overshoot that no amount of
        # heating can correct, because nothing is measuring the house:
        #
        #     target 19.0  ->  offset -10.00 at weight 1.00, "Overshoot: 2.0C above target"
        #
        # A full coast at critical weight, forever, on a house nobody is measuring. Only the
        # degree-minute emergency path stands between that and a cold house, and it would be
        # fighting this layer on every cycle for the whole winter.
        if not getattr(nibe_state, "indoor_temp_valid", True):
            return ComfortLayerDecision(
                name="Comfort",
                offset=0.0,
                weight=0.0,
                reason="No indoor sensor - abstaining (degree minutes protect this system)",
            )

        temp_deviation = nibe_state.indoor_temp - self.target_temp
        tolerance = self.tolerance_range
        dead_zone = self.mode_config.dead_zone
        in_band_weight = LAYER_WEIGHT_COMFORT_MIN * self.mode_config.comfort_weight_multiplier

        if abs(temp_deviation) < dead_zone:
            return ComfortLayerDecision(
                name="Comfort",
                offset=0.0,
                weight=0.0,
                reason="At target",
            )

        elif abs(temp_deviation) < tolerance:
            # Within comfort zone but drifting from target
            correction = -temp_deviation * COMFORT_CORRECTION_MULT

            if temp_deviation > 0:
                reason = f"Slightly warm (+{temp_deviation:.1f}°C), gentle reduce"
            else:
                reason = f"Slightly cool ({temp_deviation:.1f}°C), gentle boost"

            return ComfortLayerDecision(
                name="Comfort",
                offset=correction,
                weight=in_band_weight,
                reason=reason,
            )

        elif temp_deviation > 0:
            # Overshoot: ramp from the in-band correction at the band edge to a full coast. A dead
            # zone wider than the band (savings mode, small slider) moves the start out with it.
            edge = max(tolerance, dead_zone)
            edge_offset = -edge * COMFORT_CORRECTION_MULT
            fraction = (
                1.0
                if temp_deviation >= OVERSHOOT_PROTECTION_FULL
                else (temp_deviation - edge) / (OVERSHOOT_PROTECTION_FULL - edge)
            )
            offset = edge_offset + fraction * (OVERSHOOT_PROTECTION_OFFSET_MAX - edge_offset)
            weight = in_band_weight + fraction * (LAYER_WEIGHT_COMFORT_CRITICAL - in_band_weight)
            return ComfortLayerDecision(
                name="Comfort",
                offset=offset,
                weight=weight,
                reason=(
                    f"Overshoot: {temp_deviation:.1f}°C above target, reducing heat "
                    f"({offset:.1f}°C @ {weight:.2f})"
                ),
            )

        else:
            # Too cold, increase heating strongly
            correction = -(temp_deviation + tolerance) * COMFORT_TOO_COLD_CORRECTION_MULT
            return ComfortLayerDecision(
                name="Comfort",
                offset=correction,
                weight=LAYER_WEIGHT_COMFORT_MAX,
                reason=f"Too cold: {-temp_deviation:.1f}°C under",
            )
