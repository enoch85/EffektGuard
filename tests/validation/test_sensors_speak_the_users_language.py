"""The Swedish user reads every sensor in English.

`strings.json` translates the six switches. It carries no `entity.sensor.*` block at all, and not
one of the twenty-four sensor descriptions sets a `translation_key` - they set a hardcoded English
`name=` instead. So `Degree Minutes`, `Supply Temperature`, `Compressor Health Status` and the rest
stay English in sv, no, da and fi, whatever language Home Assistant is running in.

The primary audience for this integration is Swedish. This is the same defect as F-065, which was
fixed for the options flow, on the same reasoning: a Swedish owner was reading the DHW target
temperature and schedule fields - the settings that directly drive the heat pump - as raw English.
The sensors are the other half of that screen.

The switches show what the fix looks like: `translation_key="price_optimization"` plus an entry
under `entity.switch` in `strings.json`, mirrored in every locale. Home Assistant then resolves the
name by key, and `tests/validation/test_translation_key_parity.py` keeps the five locale files in
lockstep so nothing drifts.

Nothing here touches the heat pump. It is the label on the dial, not the dial.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from homeassistant.helpers.typing import UNDEFINED

from custom_components.effektguard.sensor import SENSORS

COMPONENT = Path(__file__).resolve().parents[2] / "custom_components" / "effektguard"
STRINGS = json.loads((COMPONENT / "strings.json").read_text(encoding="utf-8"))
LOCALES = ("en", "sv", "no", "da", "fi")


@pytest.mark.parametrize("description", SENSORS, ids=lambda d: d.key)
def test_every_sensor_has_a_translation_key(description):
    """Without one, Home Assistant has nothing to look the name up by."""
    assert description.translation_key, (
        f"Sensor {description.key!r} has no translation_key, so its name is permanently "
        f"{description.name!r} - in Swedish, Norwegian, Danish and Finnish too. The switches set "
        f"one; the sensors do not."
    )


@pytest.mark.parametrize("description", SENSORS, ids=lambda d: d.key)
def test_every_sensor_name_is_declared_in_strings_json(description):
    """A translation_key with nothing behind it renders as a raw key, or as nothing at all."""
    sensors = STRINGS.get("entity", {}).get("sensor", {})

    assert description.translation_key in sensors, (
        f"Sensor {description.key!r} declares translation_key="
        f"{description.translation_key!r}, and strings.json has no entity.sensor entry for it. "
        f"Home Assistant will fall back to the raw key."
    )
    assert sensors[description.translation_key].get(
        "name"
    ), f"entity.sensor.{description.translation_key} has no name in strings.json."


@pytest.mark.parametrize("locale", LOCALES)
def test_every_locale_carries_every_sensor_name(locale):
    """The parity test guards the file as a whole; this names the sensor that is missing."""
    path = COMPONENT / "translations" / f"{locale}.json"
    translated = json.loads(path.read_text(encoding="utf-8")).get("entity", {}).get("sensor", {})

    missing = [
        d.translation_key
        for d in SENSORS
        if d.translation_key and not translated.get(d.translation_key, {}).get("name")
    ]

    assert not missing, (
        f"{locale}.json is missing a name for {len(missing)} sensor(s): {', '.join(sorted(missing))}. "
        f"A user reading Home Assistant in this language sees the raw key, or a blank label."
    )


def test_the_hardcoded_english_name_is_gone():
    """Two sources for one string is one too many; they diverge, and the silent one wins.

    Home Assistant resolves the name from the translation when a translation_key is set, and only
    falls back to `name=` when the lookup fails. Keeping both means the English string sits there
    doing nothing until someone edits it, and then goes on doing nothing - which is exactly how the
    switch descriptions ended up carrying a dead `name=` that no longer matched their translation.
    """
    # EntityDescription.name defaults to the UNDEFINED sentinel, which is TRUTHY - a `getattr(d,
    # "name", None)` check silently passes on every sensor whether or not it has a name.
    with_both = [
        d.key for d in SENSORS if d.translation_key and d.name not in (UNDEFINED, None, "")
    ]

    assert not with_both, (
        f"{len(with_both)} sensor(s) carry BOTH a translation_key and a hardcoded name=: "
        f"{', '.join(sorted(with_both))}. The translation always wins, so the name is dead weight "
        f"that will silently diverge from what the user actually sees."
    )
