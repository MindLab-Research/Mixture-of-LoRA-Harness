#!/usr/bin/env bash
# Internal helper used by the Venti and Tall restart entrypoints.
set -euo pipefail

if (( $# != 3 )); then
  printf 'usage: %s <slug> <served-model-name> <engine-launcher>\n' "$0" >&2
  exit 2
fi

SLUG="$1"
PUBLIC_MODEL="$2"
ENGINE_LAUNCHER="$3"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_CONNECT_HOST="${VLLM_CONNECT_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8200}"
START_TIMEOUT_S="${ENGINE_START_TIMEOUT_S:-3600}"
LOG_DIR="${MODEL_STACK_LOG_DIR:-$PROJECT_ROOT/logs}"
ENGINE_PID_FILE="$PROJECT_ROOT/vllm_${SLUG}.pid"
HARNESS_PID_FILE="$PROJECT_ROOT/mol_harness_${SLUG}.pid"

mkdir -p "$LOG_DIR"

stop_pid_file() {
  local pid_file="$1" label="$2" pid deadline
  [[ -f "$pid_file" ]] || return 0
  pid="$(<"$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf 'stopping %s pid=%s\n' "$label" "$pid"
    kill -TERM "$pid"
    deadline=$((SECONDS + 120))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      printf '%s pid=%s did not stop after 120 seconds\n' "$label" "$pid" >&2
      exit 1
    fi
  fi
  rm -f "$pid_file"
}

stop_pid_file "$HARNESS_PID_FILE" "$PUBLIC_MODEL harness"
stop_pid_file "$ENGINE_PID_FILE" "$PUBLIC_MODEL engine"

port_is_open() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

for port in "$VLLM_PORT" "$PROXY_PORT"; do
  if port_is_open "$port"; then
    printf 'port %s is occupied by a process not owned by %s PID files\n' \
      "$port" "$SLUG" >&2
    exit 1
  fi
done

ENGINE_LOG="$LOG_DIR/vllm_${SLUG}_$(date +%Y%m%d_%H%M%S).log"
setsid env \
  PYTHON_BIN="$PYTHON_BIN" \
  VLLM_HOST="$VLLM_HOST" \
  VLLM_PORT="$VLLM_PORT" \
  "$ENGINE_LAUNCHER" </dev/null >"$ENGINE_LOG" 2>&1 &
engine_pid=$!
printf '%s\n' "$engine_pid" >"$ENGINE_PID_FILE"

deadline=$((SECONDS + START_TIMEOUT_S))
until curl -fsS --max-time 5 "http://$VLLM_CONNECT_HOST:$VLLM_PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$engine_pid" 2>/dev/null; then
    printf '%s engine exited during startup; log=%s\n' "$PUBLIC_MODEL" "$ENGINE_LOG" >&2
    tail -n 80 "$ENGINE_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    printf '%s engine did not become ready within %ss; log=%s\n' \
      "$PUBLIC_MODEL" "$START_TIMEOUT_S" "$ENGINE_LOG" >&2
    kill -TERM "$engine_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

if ! env \
    PYTHON_BIN="$PYTHON_BIN" \
    SERVED_MODEL_NAME="$PUBLIC_MODEL" \
    UPSTREAM="http://$VLLM_CONNECT_HOST:$VLLM_PORT" \
    PROXY_PORT="$PROXY_PORT" \
    MOL_HARNESS_PID_FILE="$HARNESS_PID_FILE" \
    MOL_HARNESS_LOG_PREFIX="mol_harness_${SLUG}" \
    "$HERE/start_mol_harness.sh"; then
  kill -TERM "$engine_pid" 2>/dev/null || true
  exit 1
fi

printf '%s stack ready: engine=http://%s:%s proxy=http://127.0.0.1:%s\n' \
  "$PUBLIC_MODEL" "$VLLM_CONNECT_HOST" "$VLLM_PORT" "$PROXY_PORT"
