"""A learner that logs "restored" must actually have the patterns.

The weather pattern database was persisted correctly and thrown away on every single
restart. `WeatherPatternLearner.from_dict` was a `@classmethod` that built and returned a
NEW learner; the coordinator called it on the live instance and discarded the result:

    self.weather_learner.from_dict(learned_data["weather_patterns"])   # return value dropped

So `self.weather_learner.pattern_db` stayed empty. The log line underneath then read
`summary.get("total_weeks", 0)` - a key `get_pattern_database_summary` has never returned -
so it printed "Restored weather patterns: 0 weeks of data" and looked like a cold start
rather than a bug. Nothing failed, nothing warned, and the multi-year database the
unusual-weather detector is built on could never accumulate past one session.

These tests pin the behaviour, not the implementation: persist, restore into a LIVE learner,
and require the patterns to be there and the climate zone to survive.
"""

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.effektguard.optimization.weather_learning import WeatherPatternLearner

TZ = timezone(timedelta(hours=1))


def _learner_with(patterns: int, climate_zone=None) -> WeatherPatternLearner:
    learner = WeatherPatternLearner(climate_zone_info=climate_zone)
    for day in range(patterns):
        learner.record_weather_pattern(
            date=datetime(2026, 1, 1, tzinfo=TZ) + timedelta(days=day * 8),
            daily_temps=[-5.0 - day, -3.0 - day, -7.0 - day, -4.0 - day],
        )
    return learner


def test_patterns_land_in_the_live_learner_not_a_discarded_copy():
    """The exact call the coordinator makes must populate the learner it is called on."""
    source = _learner_with(3)
    blob = source.to_dict()
    persisted = sum(len(v) for v in blob.values())
    assert persisted == 3, "precondition: three patterns were persisted"

    live = WeatherPatternLearner(climate_zone_info="cold")
    live.load_from_dict(blob)  # return value deliberately ignored, as the coordinator did

    restored = sum(len(v) for v in live.pattern_db.values())
    assert restored == persisted, (
        f"{persisted} patterns were persisted but {restored} are in the live learner. "
        "A restore whose result has to be re-assigned by the caller is a restore that "
        "silently did nothing."
    )


def test_restore_reports_how_many_patterns_arrived():
    """The count is returned so the log line states a fact instead of a hoped-for key."""
    blob = _learner_with(4).to_dict()
    live = WeatherPatternLearner()
    assert live.load_from_dict(blob) == 4


def test_restoring_keeps_the_climate_zone_the_learner_was_built_with():
    """The zone is the fallback every seasonal default reads before history exists.

    The classmethod built its replacement with a bare `cls()`, so even a caller that DID
    use the return value lost the zone and silently fell back to a 0 C winter baseline.
    """
    live = WeatherPatternLearner(climate_zone_info="extreme_cold")
    live.load_from_dict(_learner_with(1).to_dict())
    assert live.climate_zone == "extreme_cold"


@pytest.mark.parametrize("count", [25])
def test_round_trip_is_lossless(count):
    """to_dict -> load_from_dict preserves every pattern's values, not just the count."""
    source = _learner_with(count)
    live = WeatherPatternLearner()
    live.load_from_dict(source.to_dict())

    def flatten(learner):
        return sorted(
            (p.date.isoformat(), p.avg_temp, p.min_temp, p.max_temp, p.temp_range, p.year)
            for patterns in learner.pattern_db.values()
            for p in patterns
        )

    assert flatten(live) == flatten(source)
