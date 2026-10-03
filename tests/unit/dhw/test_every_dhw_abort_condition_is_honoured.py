"""An abort condition the decision advertises must be one the checker can act on.

`should_start_dhw` attaches `abort_conditions` to its decision, and the coordinator feeds
them to `check_abort_conditions` on every cycle while the lux window is open. Three call
sites emitted

    f"dhw_temp >= {self.user_target_temp}"

and `check_abort_conditions` had branches for `thermal_debt <` and `indoor_temp <` only.
The string fell through the if/elif chain and the function returned "no abort" - so the
tank's own target, published as an abort condition, did nothing at all. The DHW cycle ran
until NIBE stopped it or the hour's rate limit expired.

Only the `dhw_temp >=` shape is pinned here. The other two shapes, first-match ordering,
malformed conditions and the empty list are already pinned by
tests/unit/test_shared_layer_methods.py::TestCheckAbortConditions, and a second copy of
those could not fail for a reason the first one would not catch first.

The structural half of the fix is not tested here on purpose: an unrecognised condition now
logs an ERROR instead of being skipped in silence, and a log line is not a behaviour. It is
also not something a sweep over inputs can police - the emit sites for this very condition
sit behind RULE 4.5, which needs a DHW target of 60 C or more, the tank at exactly the 30 C
safety floor and a learned heating rate pinned to the 5 C/h minimum to reach at all.
"""

import pytest

from custom_components.effektguard.optimization.dhw_optimizer import IntelligentDHWScheduler


@pytest.fixture
def scheduler():
    return IntelligentDHWScheduler()


@pytest.mark.parametrize(
    "current_dhw_temp,expected_abort,why",
    [
        (51.2, True, "the tank passed the target the condition names"),
        (46.0, False, "the tank is still below target, so the cycle continues"),
        (None, False, "no reading is not the same as a reached target"),
    ],
    ids=["reached", "below", "no_sensor"],
)
def test_the_tanks_own_target_is_honoured(scheduler, current_dhw_temp, expected_abort, why):
    should_abort, reason = scheduler.check_abort_conditions(
        ["dhw_temp >= 50"],
        thermal_debt=-100.0,
        indoor_temp=21.0,
        target_indoor=21.0,
        current_dhw_temp=current_dhw_temp,
    )

    assert should_abort is expected_abort, (
        f"dhw_temp={current_dhw_temp} against a 50 C target: {why}, but the checker said "
        f"should_abort={should_abort}"
    )
    if expected_abort:
        assert (
            "51.2" in reason and "50" in reason
        ), "an abort must name the reading and the target it passed, so the log says why"
