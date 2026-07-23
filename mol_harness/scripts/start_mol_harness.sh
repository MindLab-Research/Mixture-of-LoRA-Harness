#!/usr/bin/env bash
# Start one model-selected MoL proxy after validating its native engine.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000}"
export PROXY_PORT="${PROXY_PORT:-8200}"
export ENTRY_ROUTE="${ENTRY_ROUTE:-L0}"
export MOL_USE_MODEL_ROUTER="${MOL_USE_MODEL_ROUTER:-1}"
export MOL_PURE_MODEL_ROUTE="${MOL_PURE_MODEL_ROUTE:-1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Macaron-V1-Venti}"

if [[ -z "${MOL_API_KEY:-}" && -n "${MOL_API_KEY_FILE:-}" ]]; then
  if [[ ! -f "$MOL_API_KEY_FILE" ]]; then
    printf 'MOL_API_KEY_FILE does not exist: %s\n' "$MOL_API_KEY_FILE" >&2
    exit 1
  fi
  IFS= read -r MOL_API_KEY < "$MOL_API_KEY_FILE"
  export MOL_API_KEY
fi

if [[ -z "${MOL_API_KEY:-}" && "${MOL_ALLOW_UNAUTHENTICATED:-0}" != "1" ]]; then
  printf 'set MOL_API_KEY or explicitly set MOL_ALLOW_UNAUTHENTICATED=1\n' >&2
  exit 1
fi

"$PYTHON_BIN" - "$UPSTREAM" "$PROXY_PORT" <<'PY'
import json
import socket
import sys
import urllib.request

from mol_harness.model_config import ACTIVE_MODEL_CONFIG

upstream, proxy_port = sys.argv[1], int(sys.argv[2])
required = {
    task.adapter_name for task in ACTIVE_MODEL_CONFIG.tasks.values()
    if task.adapter_name
}
with urllib.request.urlopen(upstream + "/health", timeout=5) as response:
    if response.status != 200:
        raise SystemExit(f"upstream health returned HTTP {response.status}")
with urllib.request.urlopen(upstream + "/v1/models", timeout=10) as response:
    payload = json.load(response)
available = {
    item.get("id") for item in payload.get("data", [])
    if isinstance(item, dict) and item.get("id")
}
missing = sorted(required - available)
if missing:
    raise SystemExit("upstream is missing profile adapters: " + ", ".join(missing))
with socket.socket() as sock:
    sock.settimeout(0.2)
    if sock.connect_ex(("127.0.0.1", proxy_port)) == 0:
        raise SystemExit(f"proxy port {proxy_port} is already in use")
print(
    f"validated model={ACTIVE_MODEL_CONFIG.model_name}, upstream={upstream}, "
    f"adapters={sorted(required)}, proxy_port={proxy_port}"
)
PY

LOG_DIR="${MOL_HARNESS_LOG_DIR:-$PROJECT_ROOT/logs}"
PID_FILE="${MOL_HARNESS_PID_FILE:-$PROJECT_ROOT/mol_harness.pid}"
LOG_PREFIX="${MOL_HARNESS_LOG_PREFIX:-mol_harness}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_ROOT"
nohup setsid "$PYTHON_BIN" -m mol_harness.proxy >"$LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
printf 'mol_harness launched model=%s pid=%s port=%s log=%s\n' \
  "$SERVED_MODEL_NAME" "$pid" "$PROXY_PORT" "$LOG"
