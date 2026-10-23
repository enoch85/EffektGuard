# EffektGuard

<img src="icons/logo.png" alt="EffektGuard logo" width="160"/>

[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.8.0-blue)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.11%2B-blue)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github)](https://github.com/sponsors/enoch85)

[![Open EffektGuard in the Marketplace](https://my.home-assistant.io/badges/marketplace_repository.svg)](https://my.home-assistant.io/redirect/marketplace_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Open EffektGuard in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Add EffektGuard to Home Assistant](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=effektguard)

EffektGuard runs your heat pump on the electricity price: your house stores heat when power is cheap,
and you pay less.

It works with NIBE heat pumps today. Other brands are welcome, see
[contributing](docs/dev/CONTRIBUTION_GUIDE.md).

EffektGuard controls a real heat pump: use it at your own risk, and watch it the first weeks.

<img src="docs/images/price-and-offset.png" alt="One day on a real house: EffektGuard's offset goes up when the price is low and down when it is high" width="760"/>

## What it does

| Feature | What it does |
|---|---|
| Heating | More heat when power is cheap, less when it is expensive |
| Hot water | Heats the tank when power is cheap, and has it ready when you need it |
| Weather | Heats ahead of a colder day |
| Power peaks | Holds back to avoid a new monthly peak |
| Ventilation | Turns it up when the extra heat pays off (NIBE F750, F730) |

<img src="docs/images/hot-water.png" alt="Hot water heated in the cheap midday hours, and the tank coasting through the expensive evening" width="760"/>

## What you need

| You need | Details |
|---|---|
| Home Assistant | 2025.11 or newer |
| Heat pump | NIBE F730, F750, F1155, F2040 or S1155, through MyUplink, nibe_heatpump or [Modbus](docs/LOCAL_MODBUS.md). Another model? [Ask for it](https://github.com/enoch85/EffektGuard/issues/new?labels=enhancement) |
| Electricity price | [GE-Spot](https://github.com/enoch85/ge-spot) |
| Weather forecast | Optional, to heat ahead of a colder day |
| Power meter | Optional, for peak protection (otherwise the pump's own current readings) |

Turn off the pump's own price control first, or the two will fight. On a NIBE it is Smart Control
in myUplink.

## Install

1. Click the Marketplace button (Home Assistant 2026.11 and newer) or the HACS button (older).
   Download and restart.
2. Click the third button to add EffektGuard, and answer the setup questions.

Then use the EffektGuard thermostat to set your temperature and mode.

<img src="docs/images/thermostat-and-switches.png" alt="The EffektGuard thermostat and its switches" width="520"/>

## Documentation

- [User guide](docs/USER_GUIDE.md): setup, settings, how it decides, safety, troubleshooting
- [Hot water](docs/DHW_OPTIMIZATION.md)
- [Dashboard charts](docs/dashboard-charts.yaml) like the ones above
- [Architecture](docs/architecture/00_overview.md) and [development](docs/dev/README.md)

## Support and license

[GitHub issues](https://github.com/enoch85/EffektGuard/issues) · MIT, see [LICENSE](LICENSE) ·
[@enoch85](https://github.com/enoch85)
