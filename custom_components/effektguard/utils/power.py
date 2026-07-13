"""Reading a Home Assistant power entity as kilowatts.

One function, used by everything that reads the owner's power meter. Two readers that each decide for
themselves what an absent unit means will eventually disagree by a factor of a thousand, which is what
happened here: the NIBE adapter treated a unit-less sensor as kilowatts and the coordinator treated the
same sensor, in the same cycle, as watts.

There is no defensible default. This number decides whether the house is about to set a monthly billing
peak, and watts and kilowatts are three orders of magnitude apart. An unrecognised unit is refused, and
the caller withdraws whatever depends on it.
"""

import logging

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State

from ..const import KILOWATTS_PER_MEGAWATT, WATTS_PER_KILOWATT

_LOGGER = logging.getLogger(__name__)

# Every unit that IS a power. Anything else - no unit, kWh, Wh, a percentage - is refused.
# kWh is the one worth naming: it is one entry away in an entity dropdown, it is cumulative, and read
# as power it reports a house drawing its own lifetime consumption.
POWER_UNIT_FACTORS_KW: dict[str, float] = {
    "w": 1.0 / WATTS_PER_KILOWATT,
    "kw": 1.0,
    "mw": KILOWATTS_PER_MEGAWATT,
}


def power_kw_from_state(state: State | None) -> float | None:
    """Return the state's value in kW, or None if it cannot be trusted.

    None means "no power reading", and callers must treat it as exactly that - not as zero, and not as
    a reason to substitute a guess into a field that is documented to hold a measurement.
    """
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None

    unit = str(state.attributes.get("unit_of_measurement", "")).strip().lower()
    factor = POWER_UNIT_FACTORS_KW.get(unit)
    if factor is None:
        _LOGGER.warning(
            "Power sensor %s reports %s in units of %r, which is not a power unit (expected W, kW or "
            "MW). Refusing to guess: watts and kilowatts are a factor of %d apart, and this reading "
            "decides whether the house is about to set a monthly billing peak.",
            state.entity_id,
            state.state,
            unit or "none",
            int(WATTS_PER_KILOWATT),
        )
        return None

    try:
        return float(state.state) * factor
    except (ValueError, TypeError):
        _LOGGER.warning(
            "Power sensor %s reports %r, which is not a number", state.entity_id, state.state
        )
        return None
