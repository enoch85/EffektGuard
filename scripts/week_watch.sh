#!/bin/bash
# Week-long live observation of EffektGuard against real SE4 spot prices.
#
# Runs detached, independent of any agent session. Two jobs:
#   1. keep Home Assistant up (restart it if it dies)
#   2. snapshot what the integration actually DID, every 15 minutes, to a CSV
#
# The point is a record that exists whether or not anyone is watching. Everything the
# integration decided is in ha.log, but the log rotates and is 500 MB of DEBUG; this is the
# few columns that answer "how did it go".
#
# Output: /workspace/.ha-config/week_watch.csv   (git-excluded, like the rest of .ha-config)

LOG=/workspace/.ha-config/ha.log
CSV=/workspace/.ha-config/week_watch.csv
WATCHLOG=/workspace/.ha-config/week_watch.log

ha_is_up() {
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -q "bin/hass .*/workspace/.ha-config" && return 0
  done
  return 1
}

token() {
  local cid="http://localhost:8125/"
  local fid code
  fid=$(curl -s -m 10 -X POST http://localhost:8125/auth/login_flow \
        -H 'Content-Type: application/json' \
        -d "{\"client_id\":\"$cid\",\"handler\":[\"homeassistant\",null],\"redirect_uri\":\"$cid\"}" \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('flow_id',''))" 2>/dev/null)
  [ -z "$fid" ] && return 1
  code=$(curl -s -m 10 -X POST "http://localhost:8125/auth/login_flow/$fid" \
        -H 'Content-Type: application/json' \
        -d "{\"client_id\":\"$cid\",\"username\":\"dev\",\"password\":\"dev\"}" \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',''))" 2>/dev/null)
  [ -z "$code" ] && return 1
  curl -s -m 10 -X POST http://localhost:8125/auth/token \
       -d "grant_type=authorization_code&code=$code&client_id=$cid" \
       | python3 -c "import sys,json;print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null
}

[ -f "$CSV" ] || echo "utc,offset,degree_minutes,indoor,supply,outdoor,price_ore,peak_today_kw,peak_month_kw,hvac,errors,restarts" > "$CSV"

RESTARTS=0
while true; do
  if ! ha_is_up; then
    RESTARTS=$((RESTARTS+1))
    echo "$(date -u +%FT%TZ) HA down - restarting (#$RESTARTS)" >> "$WATCHLOG"
    nohup start-ha >> "$LOG" 2>&1 &
    sleep 120
  fi

  TOK=$(token)
  if [ -n "$TOK" ]; then
    # grep -c prints 0 AND exits 1 when there are no matches, so `|| echo 0` used to append a
    # SECOND zero and split the CSV row in half. Take the first line and nothing else.
    ERRS=$(grep -c "ERROR.*effektguard" "$LOG" 2>/dev/null | head -1 | tr -dc '0-9')
    ERRS=${ERRS:-0}
    curl -s -m 15 -H "Authorization: Bearer $TOK" http://localhost:8125/api/states \
      | RESTARTS="$RESTARTS" ERRS="$ERRS" python3 -c "
import sys, json, os, datetime
try:
    states = {e['entity_id']: e for e in json.load(sys.stdin)}
except Exception:
    sys.exit(0)
def s(eid, attr=None):
    e = states.get(eid)
    if not e: return ''
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
    os.environ.get('ERRS','0').strip(),
    os.environ.get('RESTARTS','0'),
]
print(','.join(str(x) for x in row))
" >> "$CSV"
  fi
  sleep 900   # 15 minutes
done
