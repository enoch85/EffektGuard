# Local Modbus

With Home Assistant's generic [Modbus](https://www.home-assistant.io/integrations/modbus/)
integration, define sensors for the registers EffektGuard needs (temperatures and degree minutes are
raw x10, use `scale: 0.1`; the offset register is not scaled) and wrap the offset register in a
template number so EffektGuard can write it:

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
