# Airflow optimisation (exhaust-air pumps)

An exhaust-air pump draws its heat from the air it ventilates out of the house. Running the fan
faster brings more air past the evaporator, and also brings in more cold outdoor air for the house
to reheat. EffektGuard weighs the two and switches NIBE's "increased ventilation" on when the
balance is positive. Code: `optimization/airflow_optimizer.py`; control: `coordinator.py`
`_apply_airflow_decision`.

## When it exists

Only for models with `supports_exhaust_airflow` (`models/nibe/f750.py`, `f730.py`). Then, and only
then, the integration creates:

| Entity | What it shows |
|---|---|
| `switch.effektguard_airflow_optimization` | Master switch, off by default. Off, the decision is still computed and shown; nothing is written. |
| `sensor.effektguard_airflow_enhancement` | `standard` or `enhanced`; attributes `reason`, `duration_minutes`, `decision_time`, `flow_standard_m3h`, `flow_enhanced_m3h` |
| `sensor.effektguard_airflow_thermal_gain` | the net gain in kW; attribute `mode` |

Options (Configure, section "Airflow Optimization (Exhaust Air)"): the standard rate
(100-300 m³/h, default 150) and the enhanced rate (150-400 m³/h, default 252). The pump's switch
is found by name: `switch.*_increased_ventilation`.

## The decision

Every cycle `AirflowOptimizer.evaluate_from_nibe` reads outdoor and indoor temperature, the held
target, the compressor as a share of 20-120 Hz, and the indoor trend. `evaluate_airflow` returns
`standard` at the first rule that applies:

1. outdoor below -15 °C (`AIRFLOW_OUTDOOR_TEMP_MIN`)
2. indoor less than 0.2 °C below target (`AIRFLOW_INDOOR_DEFICIT_MIN`)
3. indoor warming faster than +0.1 °C/h, or cooling faster than -0.1 °C/h
4. compressor below max(61 %, 61 − 2.5 × outdoor °C) (`AIRFLOW_COMPRESSOR_*`; 61 % is 81 Hz)
5. net gain at or below 0 kW

Otherwise `enhanced` for 15 / 20 / 45 / 60 minutes by deficit (below 0.3 / 0.5 / 1.0 °C / above),
capped at 30 minutes below -5 °C outdoor and 20 minutes below -10 °C.

### The net gain

```
net gain = extra evaporator extraction − extra ventilation loss
         = ṁ·cp·12 K                   − ṁ·cp·(indoor − outdoor)
```

with ṁ the extra mass flow (150 → 252 m³/h, 1.2 kg/m³, cp 1.005 kJ/kg·K). There is no
separate "better COP" term: it is the same heat counted twice
([research 04](../research/04_exhaust_air_recovery.md)). Computed by `calculate_net_thermal_gain`
at 21 °C indoor:

| Outdoor | Net gain | Compressor needed |
|---|---|---|
| +10 °C | +0.03 kW | 61 % |
| +5 °C | -0.14 kW | 61 % |
| 0 °C | -0.31 kW | 61 % |
| -5 °C | -0.48 kW | 73.5 % |
| -10 °C | -0.65 kW | 86 % |
| -15 °C | -0.82 kW | 98.5 % |

Break-even is indoor minus 12 °C, about +9 °C outdoor. In the heating season the feature therefore
does not enhance.

## Writing to the fan

`_apply_airflow_decision` runs only when the switch is on, EffektGuard is on, and the startup
grace period is over:

- Once enhanced, the fan stays enhanced for at least the decision's duration, never under 15 min
  (`NIBE_VENTILATION_MIN_ENHANCED_DURATION`), and after that until the decision returns `standard`.
  There is no maximum run time.
- Once back at normal, it is not enhanced again for 15 min (`NIBE_VENTILATION_MIN_REST_DURATION`).
- `NibeAdapter.set_enhanced_ventilation` writes at most every 5 min and skips a write when the
  switch already shows the wanted state.
- Switching `switch.effektguard_enable_optimization` off returns the fan to normal; so does turning
  the airflow switch off.
