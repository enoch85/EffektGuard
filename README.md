# EffektGuard

<img src="icons/logo.png" alt="EffektGuard logo" width="160"/>

[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.9.0-blue)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.11%2B-blue)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github)](https://github.com/sponsors/enoch85)

Heat your house when electricity is cheap, and save when it is expensive.

EffektGuard turns your house into a heat battery. It stores heat in the cheap hours, lets the house
coast through the expensive ones, and helps you avoid a new monthly power peak. Safety and the
pump's health come first, and the mode you pick decides how warm the house stays.

Set it up once in Home Assistant, and it runs on its own.

It works with NIBE heat pumps today. Other brands are welcome, see
[contributing](docs/dev/CONTRIBUTION_GUIDE.md).

<img src="docs/images/price-and-offset.png" alt="One day on a real house: EffektGuard's offset goes up when the price is low and down when it is high" width="760"/>

<sub><i>One day on a real house: more heat in the cheap hours, less in the expensive evening.</i></sub>

> [!WARNING]
> Use at your own risk. EffektGuard controls a real heat pump. Watch it closely the first weeks.

## What it does

| Feature | What it does |
|---|---|
| Heating | More heat when power is cheap, less when it is expensive |
| Hot water | Heats the tank when power is cheap, and has it ready when you need it |
| Weather | Heats ahead of a colder day, mostly while power is cheap |
| Power peaks | Holds back to avoid a new monthly peak |
| Ventilation | Turns it up when the extra heat pays off (NIBE F750, F730) |

<img src="docs/images/hot-water.png" alt="Hot water heated in the cheap midday hours, and the tank coasting through the expensive evening" width="760"/>

<sub><i>Hot water heated at cheap midday prices, and not in the expensive evening.</i></sub>

## What you need

| You need | Details |
|---|---|
| Home Assistant | 2025.11 or newer |
| Heat pump | NIBE F730, F750, F1155, F2040 or S1155, through MyUplink, nibe_heatpump or [Modbus](docs/LOCAL_MODBUS.md). Another model? [Ask for it](https://github.com/enoch85/EffektGuard/issues/new?labels=enhancement) |
| Electricity price | [GE-Spot](https://github.com/enoch85/ge-spot), with prices from Nord Pool, ENTSO-E, Energy-Charts, Energi Data Service, Strømligning, OMIE, AEMO, ComEd or Amber |
| Weather forecast | Met.no, which comes with Home Assistant. SMHI, OpenWeatherMap, AccuWeather and others with an hourly forecast work too |
| Power meter | Optional, for peak protection (otherwise the pump's own current readings) |

> [!IMPORTANT]
> Turn off the pump's own price control first, or the two will fight. On a NIBE it is Smart
> Control in myUplink.

## Install

[![Open EffektGuard in the Marketplace](https://my.home-assistant.io/badges/marketplace_repository.svg)](https://my.home-assistant.io/redirect/marketplace_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Open EffektGuard in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Add EffektGuard to Home Assistant](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=effektguard)

## Daily use

<img src="docs/images/thermostat-and-switches.png" alt="The EffektGuard thermostat and its switches" width="520"/>

<sub><i>The EffektGuard thermostat and its switches.</i></sub>

| Thermostat | What happens |
|---|---|
| Target | Your room temperature, 18.5-25 °C |
| Heat or Off | On, or paused with the pump on its normal curve |
| Preset | Comfort, Balanced (none) or Savings (eco) |
| Away | Savings, 3 °C below your target, never below 18.5 °C |

> [!NOTE]
> Wondering why the heat changed? `sensor.effektguard_optimization_reasoning` shows the reason.

## Safety

- Room below 18 °C: it calls for full heat.
- Pump far behind on heat: it never runs cooler than normal, unless the room is too warm.
- If it can't read the pump, it changes nothing.

## Documentation

- [User guide](docs/USER_GUIDE.md): setup, settings, how it decides, safety, troubleshooting
- [Hot water](docs/DHW_OPTIMIZATION.md)
- [Dashboard charts](docs/dashboard-charts.yaml) like the ones above
- [Architecture](docs/architecture/00_overview.md) and [development](docs/dev/README.md)

## Support and license

[GitHub issues](https://github.com/enoch85/EffektGuard/issues) · MIT, see [LICENSE](LICENSE) ·
[@enoch85](https://github.com/enoch85)
