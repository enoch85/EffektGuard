# Services

Five services, declared in `services.yaml` and handled in `__init__.py`
(`_async_register_services`). The three that command the pump refuse to run while
`switch.effektguard_enable_optimization` is off.

| Service | Fields | What it does | Cooldown |
|---|---|---|---|
| `effektguard.force_offset` | `offset` -10 to 10 (required); `duration` 0-480 min, default 60 | Overrides the heating-curve offset and writes it at once. `duration: 0` means one control cycle. | 5 min |
| `effektguard.boost_heating` | `duration` 30-360 min, default 120 | `force_offset` with offset +10 (`MAX_OFFSET`). | 45 min |
| `effektguard.boost_dhw` | `duration` 30-180 min, default 90 | Turns on the configured temporary-lux entity; EffektGuard turns it off when the duration ends. Price optimization will not cancel it, the thermal-debt safety stop will. See [DHW_SETUP.md](../DHW_SETUP.md). | 60 min |
| `effektguard.reset_peak_tracking` | none | Clears the month's peak list (`EffectManager.reset_monthly_peaks`), saves, refreshes. | none |
| `effektguard.calculate_optimal_schedule` | none; returns a response | 24 hourly rows from the spot-price layer alone. | none |

Cooldowns, a missing coordinator, EffektGuard switched off, or `boost_dhw` without a temporary-lux
entity all raise a `ServiceValidationError`, which Home Assistant shows to the caller. The cooldown
clocks live at module level, so reloading the integration does not reset them.

## Calling them

```yaml
action: effektguard.force_offset
data:
  offset: 2.5
  duration: 60
```

```yaml
action: effektguard.boost_heating
data:
  duration: 120
```

```yaml
action: effektguard.boost_dhw
data:
  duration: 90
```

```yaml
action: effektguard.reset_peak_tracking
```

```yaml
action: effektguard.calculate_optimal_schedule
response_variable: schedule
```

The response is `{"schedule": [...], "generated_at": "..."}`, or `{"error": "..."}` when price or
NIBE data is missing. Each row has `time`, `hour` (clock hour), `quarter` (0-95),
`classification` (`very_cheap` … `peak`), `estimated_offset`, `price` and `is_daytime`. It is the
spot-price layer's base offset for that quarter, not a full decision.

## How an override is applied

`force_offset` and `boost_heating` go through `coordinator.async_apply_manual_override`:

1. `DecisionEngine.set_manual_override(offset, duration)` stores the offset and its expiry.
2. The coordinator runs a control cycle immediately. `calculate_decision` returns the override with
   weight 1.0 before any layer is evaluated.
3. The write bypasses the offset volatility gate and the 5-minute write cooldown
   (`force_write=True`). The register is still integer-only: `offset: 2.5` from a register at 0
   writes +2 (`utils/offset.py` truncates). If nothing reaches the register the service raises and
   the override is cleared.
4. The override expires by time (`_check_manual_override`), or after that one write when the
   duration is 0 (`consume_manual_override`). Then the layers resume.

The absolute safety floor (`DecisionEngine._absolute_safety_floor`) still applies: indoor below
18 °C means +10, and degree minutes at or past -1500 mean not below 0. An override below the floor
is raised to it, with a warning in the log; one above it is applied as asked.
