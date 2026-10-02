#!/usr/bin/env python3
"""Fetch REAL weather and REAL day-ahead prices for the simulator's scenarios.

Every scenario the harness runs is a pair of real measurements for the SAME real dates:

  weather  Open-Meteo ERA5 reanalysis archive (https://archive-api.open-meteo.com),
           hourly 2 m air temperature, no API key, CC-BY-4.0.
  prices   elprisetjustnu.se mirror of the Nord Pool day-ahead auction, hourly,
           per price area, no API key.

The pairing matters. A cold snap drives the Nordic price spike that follows it, so a
synthetic temperature shift against unrelated prices cannot test the thing the optimiser
exists to do - buy heat before the spike the cold is about to cause. The scenarios here
use the same calendar days for both, so that correlation is real rather than assumed.

Run: .venv/bin/python scripts/simulation/fetch_scenario_data.py [scenario ...]
Writes scripts/simulation/data/{weather,prices}_<slug>.json; refuses to overwrite
unless --force, so a committed scenario cannot be silently re-stamped.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PRICES = "https://www.elprisetjustnu.se/api/v1/prices"
ORE_PER_SEK = 100.0

# Each entry: a real place, a real window, and WHY that window is in the set. The comment is
# part of the scenario - a date range with no stated reason is a number nobody can check.
SCENARIOS = {
    "nordic_coldsnap_jan2024": {
        # The January 2024 Nordic cold wave. Sweden's coldest since 1999: -43.6 C at
        # Nikkaluokta on 3 January, and SE4 day-ahead hit its highest print of the winter
        # on 5 January as Nordic hydro and nuclear ran into a demand record. THE central
        # scenario: the cold and the price spike are the same real event, two days apart.
        "place": "Stockholm",
        "lat": 59.33,
        "lon": 18.07,
        "zone": "SE3",
        "start": date(2024, 1, 1),
        "end": date(2024, 1, 21),
        "why": "Coldest Swedish spell since 1999, with the price spike it caused",
    },
    "steady_winter_feb2024": {
        # An ORDINARY Swedish February around -10 C: no record, no spike. The control
        # case. Most of a heating season looks like this, and an optimiser that only
        # behaves well in a crisis is not an optimiser.
        "place": "Östersund",
        "lat": 63.18,
        "lon": 14.64,
        "zone": "SE2",
        "start": date(2024, 2, 1),
        "end": date(2024, 2, 21),
        "why": "Ordinary mid-winter near -10 C - the control case, not a crisis",
    },
    "thaw_freeze_mar2024": {
        # March in central Sweden: the diurnal cycle crosses 0 C almost every day. This is
        # where an air-source machine defrosts, where degree minutes swing hardest, and
        # where a controller that chases DM oscillates. The anti-windup path lives here.
        "place": "Uppsala",
        "lat": 59.86,
        "lon": 17.64,
        "zone": "SE3",
        "start": date(2024, 3, 10),
        "end": date(2024, 3, 30),
        "why": "Daily freeze-thaw crossings - defrost and DM oscillation territory",
    },
    "shoulder_may2024": {
        # Late May in southern Sweden, daytime +20 C and above. Space heating should be
        # OFF. This is the scenario that catches a controller heating in warm weather -
        # the failure the repo's own history records as "the emergency ladder fired in
        # July". A warm month must produce near-zero heating energy and no aux at all.
        "place": "Malmö",
        "lat": 55.60,
        "lon": 13.00,
        "zone": "SE4",
        "start": date(2024, 5, 20),
        "end": date(2024, 6, 9),
        "why": "Daytime +20 C and over - heating must switch itself off",
    },
    "autumn_volatile_oct2024": {
        # October in SE4, the most price-volatile Swedish zone in the most volatile
        # season: wind swings the day-ahead curve while the heating load is small but
        # non-zero. Tests the price layer where the spread is largest relative to demand.
        "place": "Malmö",
        "lat": 55.60,
        "lon": 13.00,
        "zone": "SE4",
        "start": date(2024, 10, 10),
        "end": date(2024, 10, 30),
        "why": "Largest price spread relative to heating load - the price layer's hardest case",
    },
}


# Both services are free and unauthenticated, and both expect a caller that identifies
# itself: elprisetjustnu.se answers a bare urllib request with 403. Identify the project,
# and throttle - a 21-day scenario is 21 requests, and there is no hurry.
HEADERS = {
    "User-Agent": "EffektGuard-simulator/1.0 (+https://github.com/enoch85/EffektGuard)",
    "Accept": "application/json",
}
THROTTLE_SECONDS = 0.4


def _get(url: str, attempts: int = 4) -> bytes:
    """One GET with backoff. A public archive refusing once is not a reason to fabricate."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                time.sleep(THROTTLE_SECONDS)
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last = err
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_weather(spec: dict) -> dict:
    """Hourly 2 m temperature from the ERA5 archive, stamped in local Swedish time."""
    url = (
        f"{ARCHIVE}?latitude={spec['lat']}&longitude={spec['lon']}"
        f"&start_date={spec['start'].isoformat()}&end_date={spec['end'].isoformat()}"
        "&hourly=temperature_2m&timezone=Europe%2FStockholm"
    )
    payload = json.loads(_get(url))
    hourly = payload["hourly"]
    temps = [t for t in hourly["temperature_2m"] if t is not None]
    if len(temps) != len(hourly["temperature_2m"]):
        raise RuntimeError(
            f"{spec['place']}: archive returned {len(hourly['temperature_2m']) - len(temps)} "
            "null hours. A gap is not a temperature; fix the window rather than interpolate."
        )
    return {
        "source": (
            f"Open-Meteo ERA5 archive, {spec['place']} "
            f"({spec['lat']}N {spec['lon']}E), hourly 2 m air temperature"
        ),
        "why": spec["why"],
        "min_c": min(temps),
        "max_c": max(temps),
        "hourly": {"time": hourly["time"], "temperature_2m": hourly["temperature_2m"]},
    }


def fetch_prices(spec: dict) -> dict:
    """Nord Pool day-ahead, one real day at a time, converted to öre/kWh."""
    days: dict[str, list[dict]] = {}
    day = spec["start"]
    while day <= spec["end"]:
        url = f"{PRICES}/{day.year}/{day.month:02d}-{day.day:02d}_{spec['zone']}.json"
        entries = json.loads(_get(url))
        days[day.isoformat()] = [
            {
                "start": entry["time_start"],
                "price": round(entry["SEK_per_kWh"] * ORE_PER_SEK, 4),
            }
            for entry in entries
        ]
        day += timedelta(days=1)
    flat = [entry["price"] for entries in days.values() for entry in entries]
    return {
        "source": (
            f"elprisetjustnu.se (Nord Pool day-ahead), zone {spec['zone']}, "
            f"fetched {date.today().isoformat()}"
        ),
        "unit": "öre/kWh",
        "why": spec["why"],
        "min_ore": min(flat),
        "max_ore": max(flat),
        "days": days,
    }


def main() -> int:
    force = "--force" in sys.argv
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SCENARIOS)
    unknown = [name for name in wanted if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {unknown}\nknown: {list(SCENARIOS)}")
        return 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        spec = SCENARIOS[name]
        for kind, fetch in (("weather", fetch_weather), ("prices", fetch_prices)):
            path = DATA_DIR / f"{kind}_{name}.json"
            if path.exists() and not force:
                print(f"  = {path.name} exists, keeping it (--force to refetch)")
                continue
            payload = fetch(spec)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if kind == "weather":
                print(
                    f"  + {path.name}: {payload['min_c']:.1f} to {payload['max_c']:.1f} C "
                    f"({len(payload['hourly']['time'])} h)"
                )
            else:
                print(
                    f"  + {path.name}: {payload['min_ore']:.1f} to {payload['max_ore']:.1f} "
                    f"öre/kWh ({len(payload['days'])} days)"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
