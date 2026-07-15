"""Every sensor must be translatable, so the Swedish user does not read the dial in English.

Home Assistant resolves an entity's name by `translation_key`. A sensor that sets a hardcoded
English `name=` instead - as all twenty-four once did - stays English in sv, no, da and fi whatever
language HA runs in, and the primary audience for this integration is Swedish.

The fix mirrors the six switches: `translation_key="..."` plus an `entity.sensor` entry in
strings.json, present in every locale (test_translation_key_parity.py keeps them in lockstep).
Nothing here touches the heat pump - it is the label on the dial, not the dial.
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
