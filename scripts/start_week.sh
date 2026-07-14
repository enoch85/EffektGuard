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
#   - starts the watcher if it is not already holding its lock (flock makes a second copy exit)
#   - prints what it found and what it did

set -u
cd /workspace || exit 1

ha_is_up() { curl -s -o /dev/null -m 8 "http://localhost:8125/" 2>/dev/null; }
watcher_is_up() { flock -n /workspace/.ha-config/week_watch.lock true 2>/dev/null && return 1 || return 0; }

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
