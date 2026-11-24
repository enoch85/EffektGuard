"""A situation reported once: at WARNING when it begins, at INFO when it ends, at debug between."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Episode:
    """One situation's report, which goes on only over consecutive control cycles that see it.

    The coordinator numbers its control cycles: each that reads the pump and decides (a read that
    fails decides nothing and is not one). A cycle that does not look at the situation - a manual
    command, optimisation switched off, hot water decided before its rules ran - breaks the run,
    and the situation is then reported as beginning again. So an end nobody saw is not reported,
    and a beginning is never lost. A read-only refresh is not a control cycle: with record False it
    reports nothing and changes nothing.
    """

    last_cycle: int | None = None

    def seen(self, cycle: int, record: bool = True) -> bool:
        """The situation holds in control cycle `cycle`. True if this begins it: report at WARNING."""
        if not record:
            return False
        begins = not self._goes_on(cycle)
        self.last_cycle = cycle
        return begins

    def gone(self, cycle: int, record: bool = True) -> bool:
        """The situation does not hold in `cycle`. True if this ends it: report the end at INFO."""
        if not record:
            return False
        ends = self._goes_on(cycle)
        self.last_cycle = None
        return ends

    def _goes_on(self, cycle: int) -> bool:
        return self.last_cycle is not None and cycle - self.last_cycle in (0, 1)
