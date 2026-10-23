# Local Modbus

With Home Assistant's generic [Modbus](https://www.home-assistant.io/integrations/modbus/)
integration, define sensors for the registers EffektGuard reads and wrap the offset register in a
template number so EffektGuard can write it.

- **Needed**: outdoor BT1 (40004), supply (40008 or 40071) and degree minutes (43005). Without them
  EffektGuard writes nothing.
- **Optional**: room BT50 (40033), tank BT7 (40013), compressor frequency (43136) and the phase
  currents BE1-BE3 (40083, 40081, 40079).
- **Scale**: temperatures, degree minutes, compressor frequency and the phase currents are raw x10,
  so use `scale: 0.1`. The offset register is not scaled.
- **Found by name**: EffektGuard finds an entity by the register number in its entity id (for
  example `sensor.bt1_outdoor_40004`), and a temperature only with `unit_of_measurement: °C`. Or
  choose them under the overrides in setup's sensor step.

These are the F-series registers. An S-series numbers them differently (offset 40031, degree minutes
40019, and its 40033 is not the room), so choose its entities under the overrides.

The offset register, 47011, as a template number:

```yaml
template:
  - number:
      - name: "Heat Offset S1 47011"
        unique_id: nibe_heat_offset_s1_47011
        state: "{{ states('sensor.heat_offset_s1_47011') | float(0) }}"
        availability: "{{ has_value('sensor.heat_offset_s1_47011') }}"
        min: -10
        max: 10
        step: 1
        set_value:
          - action: modbus.write_register
            data:
              hub: nibe
              slave: 1
              address: 47011
              value: "{{ (value | int) % 65536 }}"
```

`min` and `max` are required: a template number defaults to 0-100, and EffektGuard clamps to the
entity's range, so without them a negative offset would never reach the pump. The displayed value
catches up on the sensor's next poll (`scan_interval`, 15 s by default). Select this number as the
offset entity in the first setup step.
