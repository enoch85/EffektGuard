# EffektGuard

<img src="icons/logo.png" alt="EffektGuard logo" width="160"/>

[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.8.0-blue)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.11%2B-blue)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github)](https://github.com/sponsors/enoch85)

[![Open EffektGuard in the Marketplace](https://my.home-assistant.io/badges/marketplace_repository.svg)](https://my.home-assistant.io/redirect/marketplace_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Open EffektGuard in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Add EffektGuard to Home Assistant](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=effektguard)

EffektGuard heats your house more when electricity is cheap and less when it is expensive. The house
stores the heat, so you stay warm and pay less. It also helps you avoid a new monthly power peak
(effect tariff).

Works with NIBE heat pumps today. Want to add another brand? See
[contributing](docs/dev/CONTRIBUTION_GUIDE.md).

> **Use at your own risk.** EffektGuard controls a real heat pump, and you are responsible for it.
> Watch it closely the first weeks.

<img src="docs/images/price-and-offset.png" alt="The spot price and the offset EffektGuard wrote, over one day" width="760"/>

*A real house, 22 October 2026 (SE4 prices, v0.7.1): more heat in the cheap night and midday hours,
down to -4 in the expensive evening.*

## What it changes on your pump

| | What it does | Default |
|---|---|---|
| Heating | Runs the pump warmer or cooler than its normal curve, -10 to +10 | On |
| Hot water | Boosts hot water (NIBE: temporary lux) when power is cheap and before your hot-water times | Off |
| Ventilation | Turns up the ventilation when the exhaust air's extra heat pays off (F750, F730) | Off |

Nothing else on the pump changes.

## Before you start

| You need | Without it |
|---|---|
| Home Assistant 2025.11 or newer | Not supported |
| Your heat pump in Home Assistant. NIBE: [MyUplink](https://www.home-assistant.io/integrations/myuplink/), [nibe_heatpump](https://www.home-assistant.io/integrations/nibe_heatpump/) (turn on its entities) or [Modbus](https://www.home-assistant.io/integrations/modbus/) ([how](docs/LOCAL_MODBUS.md)) | Nothing works |
| Your model: NIBE F730, F750, F1155, F2040 or S1155 | Pick the closest and [ask for yours](https://github.com/enoch85/EffektGuard/issues/new?labels=enhancement) |
| [GE-Spot](https://github.com/enoch85/ge-spot) for prices (from Nord Pool, ENTSO-E and others) | No price optimization |
| Optional: weather with an hourly forecast | No heating ahead of a colder day |
| Optional: a whole-house power meter | Peak protection uses the pump's current readings, if it has them |

**Turn off the pump's own price control first**, or the two fight. NIBE: *Smart Control* in
myUplink (`switch.<pump>_smart_control`).

## Install

1. **Marketplace** (Home Assistant 2026.11 and newer): the first button. Download, restart.
2. **HACS** (2026.10 and older): the second button. If [HACS](https://hacs.xyz/) asks to add the
   repository, say yes. Download, restart.
3. **By hand**: copy `custom_components/effektguard/` into your `custom_components/` folder and
   restart.

Then add it: the third button, or *Settings → Devices & services → Add integration → EffektGuard*.

## Set up

Five questions. It finds the rest itself.

1. **Offset entity** - shifts the pump's heating curve. NIBE F-series: register 47011, e.g.
   `number.heat_offset_s1_47011`.
2. **Price sensor** - GE-Spot's current price. Follow the price: yes or no.
3. **Pump model** - F750 by default.
4. **Weather** (optional). Peak protection: on or off.
5. **More sensors** - degree minutes, power meter, hot-water boost switch (nibe_heatpump: an
   `input_boolean` helper for its select), extra room sensors and how to combine them, and manual
   picks if it found the wrong ones.

To change something later, go to *Settings → Devices & services → EffektGuard*:

| To change | Click |
|---|---|
| Mode, tolerance, house, hot water, ventilation | The gear icon ([Settings](#settings)) |
| An entity or the pump model | ⋮ → *Reconfigure* |
| Price optimization or peak protection | Their switches on the EffektGuard device |

## Daily use

<img src="docs/images/thermostat-and-switches.png" alt="The EffektGuard thermostat and switches" width="520"/>

Use the EffektGuard thermostat, `climate.effektguard`:

| Choose | What happens |
|---|---|
| Target | Your room temperature, 18.5-25 °C |
| Heat | On |
| Off | Paused: the pump runs its normal curve and EffektGuard does nothing, not even safety |
| Preset none, comfort, eco | Balanced, Comfort or Savings [mode](#settings) |
| Preset away | Savings, 3 °C below your target (never below 18.5 °C). The thermostat keeps showing your target; hot water is unchanged |

Why did it do that? See `sensor.effektguard_optimization_reasoning`. After a restart it waits a
cycle or two before changing anything.

These charts on your own dashboard: [dashboard-charts.yaml](docs/dashboard-charts.yaml) (needs
[apexcharts-card](https://github.com/RomRider/apexcharts-card)).

## Settings

The gear icon on EffektGuard:

| Setting | Range (default) | What it does |
|---|---|---|
| Mode | Comfort, Balanced, Savings (Balanced) | Comfort: steadiest temperature (reacts to 0.1 °C), follows the price less. Savings: more swing (reacts to 0.3 °C), follows the price more, cuts the heat fully at the most expensive times |
| Tolerance | 0.5-3.0 (0.5) | How far the room may drift to save money: ±0.4 × this value. Higher also scales the normal price step, from 20 % (0.5) to 100 % (3.0) |
| Thermal mass, insulation | 0.5-2.0 (1.0) | Your house. Thermal mass 1.5 or more: concrete slab. 1.2 or more: timber floor heating |
| Hot water | Target 45-60 °C (50), amount 1-30 min (5), ready at 07:00 and 18:00 | See [Hot water](#hot-water) |
| Ventilation | Normal 100-300, boosted 150-400 m³/h (150, 252) | NIBE F750 and F730 only |

Switches on by default: Enable Optimization, Price Optimization, Peak Protection, Weather
Prediction. Off: Hot Water Optimization, Airflow Optimization.

## How it decides

Every 5 minutes it sets the heat, in whole degrees, from:

- **Price** - cheap: more heat. Expensive: less. Extra before a price jump, less before a drop.
  Least at the most expensive times.
- **Colder day ahead** - heats in advance, earlier in a heavy house. Mostly at cheap times, never at
  the most expensive.
- **Monthly peak** - less heat near this month's highest power use. Needs peak protection, and a
  power meter or the pump's current readings.
- **Falling behind** - when the pump can't keep up (NIBE degree minutes), more heat: a little at
  first, more if it gets worse and the room is below target or power is cheap. When it starts
  depends on your location, the weather and your house.
- **Comfort** - pulls the room back to your target.

Always wins: safety, the most expensive times, a new monthly peak, a room far too warm. Except while
catching up after falling far behind: then price and power peaks only limit the extra heat to +0.5
to +1 °C. When the room drifts below your tolerance, the cuts ease off. It holds back a reversal of
2 °C or more within 45 minutes. [Safety](#safety) comes first.
[Details](docs/architecture/00_overview.md).

## Hot water

<img src="docs/images/hot-water.png" alt="The spot price, the tank temperature and the hot-water boosts, from 03:00 to midnight" width="760"/>

*Same day from 03:00: two boosts at cheap midday prices (69-83 öre/kWh) took the tank from 38 to
50 °C. No heating in the expensive evening (up to 294 öre/kWh), and still 46 °C at the end of the
day.*

Off by default. Turn on `switch.effektguard_hot_water_optimization` and pick the boost switch in
setup. Then EffektGuard:

- heats the tank when power is cheap
- has hot water ready at your hot-water times
- runs each boost at least 15 minutes, unless the house needs the heat or the tank can't be read
- leaves your own boosts alone

The pump still heats a cold tank by itself. [More](docs/DHW_OPTIMIZATION.md).

## Safety

- **Room below 18 °C**: full heat (+10), a quarter less if the room is already warming faster than
  0.3 °C/h. Needs a room sensor.
- **Pump very far behind** (past -1500 degree minutes): never below normal (0) unless the room is
  too warm. No extra heat for being behind; other reasons to add heat still count, cut back if the
  compressor is already at full speed.
- **In both cases** it stops its own hot-water boosts and starts none.
- Whole degrees only, -10 to +10, within your offset entity's range. On its own, at most one change
  every 5 minutes.
- Can't read the pump? It changes nothing. `force_offset` can't override any of this.

## Troubleshooting

| You see | Do this |
|---|---|
| Heating changes you didn't expect | Turn off the pump's own price control ([Before you start](#before-you-start)) |
| Repair: "No electricity price source" | Install GE-Spot, then pick its sensor: ⋮ → *Reconfigure* |
| Repair: "No hot-water control entity" | Pick the boost switch: ⋮ → *Reconfigure*. nibe_heatpump needs an `input_boolean` helper, and has none for the S-series |
| EffektGuard entities unavailable | It can't read the pump (MyUplink: the cloud may be slow or down). It changes nothing until it can |
| The heat setting doesn't move | Is the thermostat on Heat? Read the reasoning sensor. Changes under a whole degree are not sent |
| Peaks look wrong | Your power meter must go unavailable when it stops updating (MQTT `expire_after`, a template's `availability`) |
| Anything else | ⋮ → *Download diagnostics*, then open an [issue](https://github.com/enoch85/EffektGuard/issues) |

## Services

| Service | Does | Limits |
|---|---|---|
| `effektguard.force_offset` | Holds the heat setting you choose | -10 to +10 °C for 0-480 min (0 = one cycle); once per 5 min |
| `effektguard.boost_heating` | Full heat (+10) | 30-360 min; once per 45 min |
| `effektguard.boost_dhw` | Hot-water boost | 30-180 min; once per 60 min; not during the [safety limits](#safety) |
| `effektguard.reset_peak_tracking` | Clears the monthly peaks, e.g. at a new billing period | |
| `effektguard.calculate_optimal_schedule` | 24-hour price preview | |

The first three need EffektGuard on, and act at once.

More: [architecture](docs/architecture/00_overview.md), [climate zones](docs/CLIMATE_ZONES.md),
[hot water](docs/DHW_OPTIMIZATION.md) and its [setup](docs/DHW_SETUP.md),
[local Modbus](docs/LOCAL_MODBUS.md), [development](docs/dev/README.md),
[contributor rulebook](CLAUDE.md).

## Support and license

[GitHub issues](https://github.com/enoch85/EffektGuard/issues) · MIT, see [LICENSE](LICENSE) ·
[@enoch85](https://github.com/enoch85)
