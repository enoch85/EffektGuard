#!/bin/bash
# Bring up (or repair) the week-long live observation. Idempotent - safe to run any number of times.
#
# RUN THIS AFTER EVERY REBOOT. This box has no init: pid 1 is `docker-init -- sleep infinity`, there
# is no systemd and no cron, so nothing starts Home Assistant or the watcher when the machine comes
# back. That is the box, not the script.
#
#     bash scripts/start_week.sh
#
# What it does:
#   - starts Home Assistant if it is not answering on :8125
#   - starts the watcher if it is not already running (pid file, checked against /proc)
#   - prints what it found and what it did

set -u
cd /workspace || exit 1

ha_is_up() { curl -s -o /dev/null -m 8 "http://localhost:8125/" 2>/dev/null; }
# Alive AND actually a week_watch - a pid file alone would happily point at a recycled pid.
watcher_is_up() {
  local pidfile=/workspace/.ha-config/week_watch.pid pid
  [ -f "$pidfile" ] || return 1
  pid=$(tr -dc '0-9' <"$pidfile")
  [ -n "$pid" ] && [ -d "/proc/$pid" ] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q week_watch.sh
}
# The heat pump itself. Without it HA has no BT1/BT25/degree-minutes, EffektGuard correctly refuses
# to control on incomplete data, and the week records nothing but that refusal. The first version of
# this script started Home Assistant and the watcher and forgot the pump they were meant to watch.
pump_is_up() { python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(3)
try:
    s.connect(('127.0.0.1', 5020))
except OSError:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; }

if pump_is_up; then
  echo "NIBE simulator : already up (modbus :5020)"
else
  echo "NIBE simulator : down - starting"
  setsid nohup /workspace/.venv/bin/python scripts/simulation/nibe_modbus_simulator.py \
    >>/workspace/.ha-config/nibe_sim.log 2>&1 </dev/null &
  for _ in $(seq 1 15); do
    sleep 1
    pump_is_up && break
  done
  pump_is_up && echo "NIBE simulator : up" || echo "NIBE simulator : FAILED - see .ha-config/nibe_sim.log"
fi

if ha_is_up; then
  echo "Home Assistant : already up"
else
  echo "Home Assistant : not answering - starting"
  nohup start-ha >>/workspace/.ha-config/ha.log 2>&1 &
  for _ in $(seq 1 36); do
    sleep 10
    ha_is_up && break
  done
  ha_is_up && echo "Home Assistant : up" || echo "Home Assistant : STILL DOWN - check .ha-config/ha.log"
fi

if watcher_is_up; then
  echo "watcher        : already running"
else
  echo "watcher        : starting"
  setsid nohup bash /workspace/scripts/week_watch.sh >/dev/null 2>&1 </dev/null &
  sleep 3
  watcher_is_up && echo "watcher        : up" || echo "watcher        : FAILED - see .ha-config/week_watch.log"
fi

echo "records        : $(($(wc -l </workspace/.ha-config/week_watch.csv 2>/dev/null || echo 1) - 1)) samples in .ha-config/week_watch.csv"
