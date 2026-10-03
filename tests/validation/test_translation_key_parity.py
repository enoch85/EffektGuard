"""Every locale must carry exactly the keys strings.json declares.

Home Assistant resolves a translation by key. A MISSING key falls back to the raw key or an empty
label; a STALE key is dead weight that quietly diverges. Neither is visible in a test run, in CI,
or in the UI of whoever wrote the change - only to the user in that language, and the primary
audience for this integration is Swedish.

This has drifted before: options.py renamed sections and added fields, strings.json and en.json
were updated, and sv/no/da/fi were not - leaving Swedish users raw keys for the DHW target and
schedule fields that directly drive the heat pump. An empty-string value counts as a failure too:
it renders as a blank label, indistinguishable from a missing translation.
"""

import json
from pathlib import Path

import pytest

COMPONENT = Path(__file__).resolve().parent.parent.parent / "custom_components" / "effektguard"
STRINGS = COMPONENT / "strings.json"
TRANSLATIONS = COMPONENT / "translations"

LOCALES = ["en", "sv", "no", "da", "fi"]


def _leaf_keys(data: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a translation dict to {dotted.key: value}."""
    out: dict[str, str] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_leaf_keys(value, path))
        else:
            out[path] = value
    return out


def _load(path: Path) -> dict[str, str]:
    return _leaf_keys(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def reference() -> dict[str, str]:
    return _load(STRINGS)


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_has_no_missing_keys(locale, reference):
    """A missing key renders as a raw key or a blank label in that language's UI."""
    translated = _load(TRANSLATIONS / f"{locale}.json")

    missing = sorted(set(reference) - set(translated))

    assert not missing, (
        f"{locale}.json is missing {len(missing)} key(s) declared in strings.json. "
        f"Users in this language see raw keys instead of labels.\n  " + "\n  ".join(missing)
    )


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_has_no_stale_keys(locale, reference):
    """A stale key is dead weight and a sign the file was not migrated with the code."""
    translated = _load(TRANSLATIONS / f"{locale}.json")

    stale = sorted(set(translated) - set(reference))

    assert not stale, (
        f"{locale}.json carries {len(stale)} key(s) that no longer exist in strings.json. "
        f"They are dead, and their presence means the file missed a rename.\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_has_no_empty_values(locale):
    """An empty string renders as a blank label - indistinguishable from a missing one."""
    translated = _load(TRANSLATIONS / f"{locale}.json")

    empty = sorted(key for key, value in translated.items() if not str(value).strip())

    assert not empty, (
        f"{locale}.json has {len(empty)} empty translation value(s), which render as blank "
        f"labels:\n  " + "\n  ".join(empty)
    )
