#!/bin/bash
# Week-long live observation of EffektGuard against real SE4 spot prices.
#
# Do not run this directly - run scripts/start_week.sh, which is idempotent and also brings
# Home Assistant up. This script assumes it is the only copy of itself.
#
# Two jobs:
#   1. keep Home Assistant up (restart it if it dies)
#   2. snapshot what the integration actually DID, every 15 minutes, to a CSV
#
# SINGLE INSTANCE, ENFORCED. The first version was started twice - `pkill -f` does not reliably
# reach a process in its own `setsid` session - and both copies appended to the same CSV. One had
# a formatting bug, so the file ended up half-good and half-corrupt, which is worse than either.
# flock makes a second copy exit immediately.
#
# THIS BOX HAS NO INIT. pid 1 is `docker-init -- sleep infinity`: no systemd, no cron, nothing that
# runs on boot. A reboot kills Home Assistant and this watcher, and NOTHING brings them back.
# After a reboot somebody has to run scripts/start_week.sh. That is a property of the box, not
# something the script can fix.
#
# Output: /workspace/.ha-config/week_watch.csv   (git-excluded, like the rest of .ha-config)

set -u

LOG=/workspace/.ha-config/ha.log
CSV=/workspace/.ha-config/week_watch.csv
WATCHLOG=/workspace/.ha-config/week_watch.log
LOCK=/workspace/.ha-config/week_watch.lock
INTERVAL=900 # 15 minutes

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -u +%FT%TZ) another week_watch is already running - exiting" >>"$WATCHLOG"
  exit 0
fi

ha_is_up() {
  curl -s -o /dev/null -m 8 "http://localhost:8125/" 2>/dev/null
}

token() {
  local cid="http://localhost:8125/" fid code
  fid=$(curl -s -m 10 -X POST http://localhost:8125/auth/login_flow \
    -H 'Content-Type: application/json' \
    -d "{\"client_id\":\"$cid\",\"handler\":[\"homeassistant\",null],\"redirect_uri\":\"$cid\"}" |
    python3 -c "import sys,json;print(json.load(sys.stdin).get('flow_id',''))" 2>/dev/null) || return 1
  [ -z "$fid" ] && return 1
  code=$(curl -s -m 10 -X POST "http://localhost:8125/auth/login_flow/$fid" \
    -H 'Content-Type: application/json' \
    -d "{\"client_id\":\"$cid\",\"username\":\"dev\",\"password\":\"dev\"}" |
    python3 -c "import sys,json;print(json.load(sys.stdin).get('result',''))" 2>/dev/null) || return 1
  [ -z "$code" ] && return 1
  curl -s -m 10 -X POST http://localhost:8125/auth/token \
    -d "grant_type=authorization_code&code=$code&client_id=$cid" |
    python3 -c "import sys,json;print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null
}

[ -f "$CSV" ] || echo "utc,offset,degree_minutes,indoor,supply,outdoor,price_ore,peak_today_kw,peak_month_kw,hvac,errors,restarts" >"$CSV"

echo "$(date -u +%FT%TZ) week_watch started (pid $$)" >>"$WATCHLOG"
RESTARTS=0

while true; do
  if ! ha_is_up; then
    RESTARTS=$((RESTARTS + 1))
    echo "$(date -u +%FT%TZ) HA not answering - starting it (#$RESTARTS)" >>"$WATCHLOG"
    nohup start-ha >>"$LOG" 2>&1 &
    for _ in $(seq 1 30); do
      sleep 10
      ha_is_up && break
    done
  fi

  TOK=$(token) || TOK=""
  if [ -n "$TOK" ]; then
    # `grep -c` prints 0 AND exits 1 when it matches nothing, so a `|| echo 0` fallback appends a
    # SECOND zero and splits the CSV row in half. Force it to one integer, always.
    ERRS=$(grep -c "ERROR.*effektguard" "$LOG" 2>/dev/null | head -1 | tr -dc '0-9')
    ERRS=${ERRS:-0}

    ROW=$(curl -s -m 15 -H "Authorization: Bearer $TOK" http://localhost:8125/api/states |
      RESTARTS="$RESTARTS" ERRS="$ERRS" python3 -c "
import sys, json, os, datetime
try:
    states = {e['entity_id']: e for e in json.load(sys.stdin)}
except Exception:
    sys.exit(1)
def s(eid, attr=None):
    e = states.get(eid)
    if not e:
        return ''
    v = e['attributes'].get(attr) if attr else e['state']
    return '' if v in (None, 'unknown', 'unavailable') else v
row = [
    datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    s('sensor.effektguard_current_offset'),
    s('sensor.effektguard_degree_minutes'),
    s('climate.effektguard', 'current_temperature'),
    s('sensor.effektguard_supply_temperature'),
    s('climate.effektguard', 'outdoor_temp'),
    s('climate.effektguard', 'current_price'),
    s('sensor.effektguard_peak_today'),
    s('sensor.effektguard_monthly_peak'),
    s('climate.effektguard'),
    ''.join(c for c in os.environ.get('ERRS', '0') if c.isdigit()) or '0',
    ''.join(c for c in os.environ.get('RESTARTS', '0') if c.isdigit()) or '0',
]
line = ','.join(str(x).replace(',', ' ').replace(chr(10), ' ') for x in row)
if line.count(',') != 11:
    sys.exit(1)
print(line)
")
    # Only a row with exactly 12 fields is written. A malformed record over seven days is worse
    # than a missing one: it looks fine until the day somebody tries to read it.
    if [ -n "$ROW" ]; then
      echo "$ROW" >>"$CSV"
    else
      echo "$(date -u +%FT%TZ) skipped a malformed/empty sample" >>"$WATCHLOG"
    fi
  else
    echo "$(date -u +%FT%TZ) could not authenticate to HA - skipping this sample" >>"$WATCHLOG"
  fi

  sleep "$INTERVAL"
done
