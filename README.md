# EffektGuard

<img src="icons/logo.png" alt="EffektGuard logo" width="160"/>

[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
![Version](https://img.shields.io/badge/version-0.8.0-blue)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2025.11%2B-blue)
[![Sponsor](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-1f425f?logo=github)](https://github.com/sponsors/enoch85)

[![Open EffektGuard in the Marketplace](https://my.home-assistant.io/badges/marketplace_repository.svg)](https://my.home-assistant.io/redirect/marketplace_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Open EffektGuard in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=enoch85&repository=EffektGuard&category=integration)
[![Add EffektGuard to Home Assistant](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=effektguard)

EffektGuard runs your heat pump on the electricity price. It heats more when power is cheap and less
when it is expensive, so your house stores heat and you pay less. It also helps you avoid a new
monthly power peak.

It works with NIBE heat pumps today. Other brands are welcome, see
[contributing](docs/dev/CONTRIBUTION_GUIDE.md).

EffektGuard controls a real heat pump: use it at your own risk, and watch it the first weeks.

<img src="docs/images/price-and-offset.png" alt="One day on a real house: EffektGuard's offset goes up when the price is low and down when it is high" width="760"/>

## What it does

- Heating: runs the pump warmer or cooler than its normal curve. On by default.
- Hot water: heats the tank when power is cheap and before your hot-water times. Off by default.
- Ventilation: turns up the ventilation when the extra heat pays off, on a NIBE F750 or F730. Off
  by default.

## What you need

- Home Assistant 2025.11 or newer
- A NIBE F730, F750, F1155, F2040 or S1155 in Home Assistant, through MyUplink, nibe_heatpump or
  [Modbus](docs/LOCAL_MODBUS.md). Another model? Pick the closest and
  [ask for yours](https://github.com/enoch85/EffektGuard/issues/new?labels=enhancement).
- [GE-Spot](https://github.com/enoch85/ge-spot) for the electricity price
- Optional: a weather forecast and a whole-house power meter

Turn off the pump's own price control first, or the two will fight. On a NIBE it is Smart Control
in myUplink.

## Install

1. Click the Marketplace button (Home Assistant 2026.11 and newer) or the HACS button (older).
   Download and restart.
2. Click the third button to add EffektGuard, and answer the setup questions.

Then use the EffektGuard thermostat to set your temperature and mode.

## Documentation

- [User guide](docs/USER_GUIDE.md): setup, settings, how it decides, safety, troubleshooting
- [Hot water](docs/DHW_OPTIMIZATION.md)
- [Dashboard charts](docs/dashboard-charts.yaml) like the one above
- [Architecture](docs/architecture/00_overview.md) and [development](docs/dev/README.md)

## Support and license

[GitHub issues](https://github.com/enoch85/EffektGuard/issues) · MIT, see [LICENSE](LICENSE) ·
[@enoch85](https://github.com/enoch85)
