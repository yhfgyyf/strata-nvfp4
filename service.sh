#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
config="${STRATA_CONFIG:-$root/strata.json}"
config="$(realpath "$config")"
python="$root/.venv/bin/python"
mkdir -p "$root/logs"
exec 9>"$root/logs/service.lock"
flock -n 9 || { echo "Another start/stop operation is running."; exit 1; }
port="$($python -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$config")"
pid=""
for p in $(pgrep -u "$(id -u)" -f 'serve.server' || true); do
  command="$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)" || continue
  if [[ "$command" == "$python -m serve.server --engine strata --config $config " ]]; then
    [[ -z "$pid" ]] || { echo "Multiple matching servers; refusing."; exit 1; }
    pid="$p"
  fi
done
case "${1:-status}" in
  start)
    [[ -z "$pid" ]] || { echo "Already running: PID $pid"; exit 0; }
    if ss -H -ltn "sport = :$port" | grep -q .; then
      echo "Port $port is already occupied."; exit 1
    fi
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
      echo "GPU is occupied. Stop its current workload first."; exit 1
    fi
    cd "$root"
    PYTHONUNBUFFERED=1 nohup setsid "$python" -m serve.server --engine strata --config "$config" \
      >> "$root/logs/server.log" 2>&1 < /dev/null 9>&- &
    echo "$!" > "$root/logs/service.pid"
    echo "Starting PID $!; log: $root/logs/server.log"
    ;;
  stop)
    [[ -n "$pid" ]] || { echo "Not running."; exit 0; }
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    [[ "$pgid" == "$pid" ]] || { echo "Shared process group; use the foreground terminal to stop it."; exit 1; }
    "$python" - "$port" "$config" <<'PY'
import json,sys,urllib.request
key=json.load(open(sys.argv[2])).get('api_key')
headers={'Authorization':'Bearer '+key} if key else {}
try:
    request=urllib.request.Request('http://127.0.0.1:'+sys.argv[1]+'/v1/status',headers=headers)
    with urllib.request.urlopen(request,timeout=5) as f:
        state=json.load(f)
except OSError:
    state={}
if state.get('activity',{}).get('in_flight',0):
    raise SystemExit('Requests are active; wait before stopping.')
PY
    kill -TERM "$pid"
    for ((i=0;i<60;i++)); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$pid" 2>/dev/null || true
    rm -f "$root/logs/service.pid"
    echo "Stopped."
    ;;
  status)
    if [[ -n "$pid" ]]; then
      echo "PID $pid; http://127.0.0.1:$port"
      "$python" - "$port" "$config" <<'PY'
import json,sys,urllib.request
key=json.load(open(sys.argv[2])).get('api_key')
headers={'Authorization':'Bearer '+key} if key else {}
request=urllib.request.Request('http://127.0.0.1:'+sys.argv[1]+'/v1/status',headers=headers)
with urllib.request.urlopen(request,timeout=5) as response:
    print(response.read().decode())
PY
    else
      echo "Not running."
    fi
    ;;
  *) echo "Usage: $0 {start|stop|status}"; exit 2 ;;
esac
