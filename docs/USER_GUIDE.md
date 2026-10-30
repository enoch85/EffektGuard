# User guide

The details behind the [README](../README.md).

## What you need, and why

| You need | Without it |
|---|---|
| Home Assistant 2025.11 or newer | Not supported |
| Your heat pump in Home Assistant: [MyUplink](https://www.home-assistant.io/integrations/myuplink/), [nibe_heatpump](https://www.home-assistant.io/integrations/nibe_heatpump/) (turn on its entities) or [Modbus](https://www.home-assistant.io/integrations/modbus/) ([how](LOCAL_MODBUS.md)) | Nothing works |
| A model profile: NIBE F730, F750, F1155, F2040 or S1155 | Pick the closest |
| [GE-Spot](https://github.com/enoch85/ge-spot), which gets prices from Nord Pool, ENTSO-E and others | No price optimization |
| A weather entity with an hourly forecast: Met.no (comes with Home Assistant), SMHI, OpenWeatherMap, AccuWeather or others (optional) | No heating ahead of a colder day |
| A whole-house power meter (optional) | Peak protection uses the pump's current readings, if it has them |

Turn off the pump's own price control, or the two fight. On a NIBE it is Smart Control in myUplink
(`switch.<pump>_smart_control`).

## Install

- Marketplace (Home Assistant 2026.11 and newer): the first button in the README. Download, restart.
- HACS (2026.10 and older): the second button. If HACS asks to add the repository, say yes.
  Download, restart.
- By hand: copy `custom_components/effektguard/` into your `custom_components/` folder and restart.

Then add it with the third button, or Settings → Devices & services → Add integration →
EffektGuard.

## Setup

The setup asks five things and finds the rest itself:

1. Offset entity: the setting that shifts the pump's heating curve. On a NIBE F-series it is
   register 47011, for example `number.heat_offset_s1_47011`.
2. Price sensor: GE-Spot's current price, and whether to follow the price.
3. Pump model: F750 by default.
4. Weather entity (optional), and whether peak protection is on.
5. More sensors: degree minutes, the power meter, the hot-water boost switch (on nibe_heatpump an
   `input_boolean` helper for its select), extra room sensors and how to combine them, and manual
   picks if it found the wrong ones.

To change something later, go to Settings → Devices & services → EffektGuard:

| To change | Click |
|---|---|
| Mode, tolerance, house, hot water, ventilation | The gear icon (see [Settings](#settings)) |
| An entity or the pump model | ⋮ → Reconfigure |
| Price optimization or peak protection | Their switches on the EffektGuard device |

## The thermostat

<img src="images/thermostat-and-switches.png" alt="The EffektGuard thermostat and switches" width="520"/>

`climate.effektguard` is the day-to-day control:

| Choose | What happens |
|---|---|
| Target | Your room temperature, 18.5-25 °C |
| Heat | On |
| Off | Paused: the pump runs its normal curve and EffektGuard does nothing, not even safety |
| Preset none, comfort, eco | Balanced, Comfort or Savings [mode](#settings) |
| Preset away | Savings, 3 °C below your target (never below 18.5 °C). The thermostat keeps showing your target, and hot water is unchanged |

To see why it did something, look at `sensor.effektguard_optimization_reasoning`. After a restart
it waits a cycle or two before changing anything.

## Settings

Under the gear icon:

| Setting | Range (default) | What it does |
|---|---|---|
| Mode | Comfort, Balanced, Savings (Balanced) | Comfort keeps the room at your target or above, and follows the price least. Balanced may let it dip about 0.2 °C below your target to save more. Savings may let it dip about 1 °C below, and follows the price most |
| Tolerance | 0.5-3.0 (0.5) | How far the room may drift to save money: ±0.4 × this value. Higher also makes the price steps bigger, from 20 % (0.5) to 100 % (3.0), but a price never takes more than 2 °C off |
| Thermal mass, insulation | 0.5-2.0 (1.0) | Your house. Thermal mass 1.5 or more is a concrete slab, 1.2 or more timber floor heating |
| Hot water | Target 45-60 °C (50), amount 1-30 min (5), ready at 07:00 and 18:00 | See [Hot water](#hot-water) |
| Ventilation | Normal 100-300, boosted 150-400 m³/h (150, 252) | NIBE F750 and F730 only |

Switches on by default: Enable Optimization, Price Optimization, Peak Protection, Weather
Prediction. Off by default: Hot Water Optimization, Airflow Optimization.

## How it decides

Every 5 minutes it sets the heat, in whole degrees, from:

- Price: more heat when power is cheap, less when it is expensive. Extra before a price jump, less
  before a drop, least at the most expensive times.
- Colder weather: when a colder day is coming, it heats ahead, earlier in a heavy house. Mostly at
  cheap times, never at the most expensive.
- Monthly peak: less heat close to this month's highest power use. Needs peak protection, and a
  power meter or the pump's current readings.
- Falling behind: when the pump can't keep up (NIBE's degree minutes drop) and the room is below
  the level your mode keeps (your target in Comfort, about 0.2 °C below in Balanced, about 1 °C
  below in Savings), more heat. A little at first, more if it gets worse. Where this starts depends
  on your location, the weather and your house.
- The pump's electric heater: while the room is at or above that level, it keeps the pump from
  starting its electric heater, which gives one kWh of heat for each kWh of electricity where the
  compressor gives more. Below that level the heater may help. It learns where your pump starts
  the heater by watching the heater's power (on nibe_heatpump, turn on "Int. el.add. Power");
  until it has seen that, it uses your model's factory setting.
- Cold your pump can't keep up with: once EffektGuard has seen your pump need its electric heater
  at some outdoor temperature, it never turns the heat below the pump's own heating curve when it
  is that cold, or will be within 12 hours. Heat saved then could only come back from the heater.
- Comfort: it pulls the room back to your target.

Safety, the most expensive times, a new monthly peak and a room far too warm always win. Two
exceptions: while the pump catches up after falling far behind, a price or power peak only limits
the extra heat to +0.5 to +1 °C; and in a cold your pump can't keep up with, none of them turns the
heat below the pump's own heating curve.

When the room drifts below your tolerance, the cuts for price and peaks ease off. To avoid
flip-flopping, it holds back a reversal of 2 °C or more within 45 minutes. Safety comes first.
[Architecture](architecture/00_overview.md) has the full picture.

## Hot water

<img src="images/hot-water.png" alt="The spot price, the tank temperature and the hot-water boosts, from 03:00 to midnight" width="760"/>

Off by default. Turn on Hot Water Optimization and pick the boost switch in setup. Then EffektGuard
heats the tank when power is cheap and has hot water ready at your hot-water times. Each boost runs
at least 15 minutes, unless the house needs the heat or the tank can't be read. A boost you start
yourself is left alone, and the pump still heats a cold tank by itself.
[Hot water in detail](DHW_OPTIMIZATION.md).

## Safety

- Room below 18 °C: full heat (+10). If the room is already warming up quickly (over 0.3 °C an
  hour), it eases to +7.5 so it doesn't overshoot. This needs a room sensor.
- Pump very far behind (past -1500 degree minutes): it never runs cooler than normal, unless the
  room is at or above the level your mode keeps. It adds no heat just to catch up: the compressor
  is already flat out, so the extra would come from the electric heater.
- In both cases it stops its own hot-water boosts and starts none.
- It only sets whole degrees from -10 to +10, within your offset entity's range, and on its own at
  most once every 5 minutes.
- If it can't read the pump, it changes nothing. `force_offset` can't override any of this.

## Services

| Service | Does | Limits |
|---|---|---|
| `effektguard.force_offset` | Holds the heat setting you choose | -10 to +10 °C for 0-480 min (0 = one cycle), once per 5 min |
| `effektguard.boost_heating` | Full heat (+10) | 30-360 min, once per 45 min |
| `effektguard.boost_dhw` | Hot-water boost | 30-180 min, once per 60 min, not during the [safety limits](#safety) |
| `effektguard.reset_peak_tracking` | Clears the monthly peaks, for example at a new billing period | |
| `effektguard.calculate_optimal_schedule` | A 24-hour price preview | |

The first three need EffektGuard on, and act at once.

## Dashboard charts

The README's chart on your own dashboard, plus the hot-water chart:
[dashboard-charts.yaml](dashboard-charts.yaml). It needs
[apexcharts-card](https://github.com/RomRider/apexcharts-card).

## Troubleshooting

| You see | Do this |
|---|---|
| Heating changes you didn't expect | Turn off the pump's own price control (see [What you need](#what-you-need-and-why)) |
| Repair: "No electricity price source" | Install GE-Spot, then pick its sensor: ⋮ → Reconfigure |
| Repair: "No hot-water control entity" | Pick the boost switch: ⋮ → Reconfigure. nibe_heatpump needs an `input_boolean` helper, and has none for the S-series |
| EffektGuard entities unavailable | It can't read the pump (with MyUplink, the cloud may be slow or down). It changes nothing until it can |
| The heat setting doesn't move | Check that the thermostat is on Heat, and read the reasoning sensor. Changes under a whole degree are not sent |
| Peaks look wrong | Your power meter must go unavailable when it stops updating (MQTT `expire_after`, a template's `availability`) |
| Anything else | ⋮ → Download diagnostics, then open an [issue](https://github.com/enoch85/EffektGuard/issues) |
