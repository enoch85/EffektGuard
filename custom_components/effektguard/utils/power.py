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

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfPower
from homeassistant.core import State

from ..const import KILOWATTS_PER_MEGAWATT, MILLIWATTS_PER_KILOWATT, WATTS_PER_KILOWATT

_LOGGER = logging.getLogger(__name__)

# Every unit that IS a power, keyed on Home Assistant's OWN strings, CASE-SENSITIVELY.
#
# Case matters here, and it is not a style preference: HA ships both `UnitOfPower.MILLIWATT` ("mW")
# and `UnitOfPower.MEGA_WATT` ("MW"), and they differ ONLY in case. Case-folding the unit collapses
# them onto each other, and this table would then read a milliwatt sensor as MEGAWATTS - a factor
# of 10^9, classified billable, persisted as a monthly tariff peak, and pinning the effect layer to
# CRITICAL for the rest of the month. Anything else - no unit, kWh, Wh, a percentage - is refused.
# kWh is the one worth naming: it is one entry away in an entity dropdown, it is cumulative, and
# read as power it reports a house drawing its own lifetime consumption.
POWER_UNIT_FACTORS_KW: dict[str, float] = {
    UnitOfPower.MILLIWATT: 1.0 / MILLIWATTS_PER_KILOWATT,
    UnitOfPower.WATT: 1.0 / WATTS_PER_KILOWATT,
    UnitOfPower.KILO_WATT: 1.0,
    UnitOfPower.MEGA_WATT: KILOWATTS_PER_MEGAWATT,
}

# Hand-written template sensors do not always match HA's capitalisation, and refusing "kw" would
# break working installations for no safety gain. So a case-insensitive second pass is allowed -
# but ONLY where the fold is unambiguous. "mw" is not: it is both milliwatts and megawatts.
_UNAMBIGUOUS_FOLDED_UNITS: dict[str, float] = {
    folded: factor
    for folded, factor in ((unit.lower(), factor) for unit, factor in POWER_UNIT_FACTORS_KW.items())
    if [u.lower() for u in POWER_UNIT_FACTORS_KW].count(folded) == 1
}


def power_kw_from_state(state: State | None) -> float | None:
    """Return the state's value in kW, or None if it cannot be trusted.

    None means "no power reading", and callers must treat it as exactly that - not as zero, and not as
    a reason to substitute a guess into a field that is documented to hold a measurement.
    """
    if state is None or state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None

    unit = str(state.attributes.get("unit_of_measurement", "")).strip()
    factor = POWER_UNIT_FACTORS_KW.get(unit)
    if factor is None:
        factor = _UNAMBIGUOUS_FOLDED_UNITS.get(unit.lower())
    if factor is None:
        _LOGGER.warning(
            "Power sensor %s reports %s in units of %r, which is not a power unit this integration "
            "will act on (expected one of %s, spelled as Home Assistant spells them). Refusing to "
            "guess: 'mW' and 'MW' differ only in case and are a factor of %d apart, and this "
            "reading decides whether the house is about to set a monthly billing peak.",
            state.entity_id,
            state.state,
            unit or "none",
            ", ".join(POWER_UNIT_FACTORS_KW),
            int(MILLIWATTS_PER_KILOWATT * KILOWATTS_PER_MEGAWATT),
        )
        return None

    try:
        return float(state.state) * factor
    except (ValueError, TypeError):
        _LOGGER.warning(
            "Power sensor %s reports %r, which is not a number", state.entity_id, state.state
        )
        return None
