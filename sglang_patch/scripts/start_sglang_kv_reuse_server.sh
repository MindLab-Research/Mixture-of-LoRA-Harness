#!/usr/bin/env bash
set -euo pipefail

PATCH_ROOT="${PATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SGLANG_BASE_PYTHON="${SGLANG_BASE_PYTHON:?Set SGLANG_BASE_PYTHON to the upstream SGLang python package directory}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base model path}"
MODEL_NAME="${MODEL_NAME:-zai-org/glm-5.1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
LORA_USE_VIRTUAL_EXPERTS="${LORA_USE_VIRTUAL_EXPERTS:-1}"
LORA_PATHS_ARGS="${LORA_PATHS_ARGS:-}"

if [[ -z "${LORA_PATHS_ARGS}" ]]; then
  cat >&2 <<'EOF'
LORA_PATHS_ARGS is required, for example:
  export LORA_PATHS_ARGS='shared-lora=/path/to/l0 coding-lora=/path/to/l2'
EOF
  exit 2
fi

export SGLANG_BASE_PYTHON
export PYTHONPATH="${PATCH_ROOT}/overlay/python:${SGLANG_BASE_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a LORA_PATHS_ARRAY <<< "${LORA_PATHS_ARGS}"
LORA_PATHS_COUNT="${#LORA_PATHS_ARRAY[@]}"
MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-${LORA_PATHS_COUNT}}"
if (( MAX_LORAS_PER_BATCH < LORA_PATHS_COUNT )); then
  echo "MAX_LORAS_PER_BATCH=${MAX_LORAS_PER_BATCH} is smaller than loaded LoRAs=${LORA_PATHS_COUNT}; using ${LORA_PATHS_COUNT}."
  MAX_LORAS_PER_BATCH="${LORA_PATHS_COUNT}"
fi

cmd=(
  /usr/bin/python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp "${TP}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --enable-metrics
  --enable-mfu-metrics
  --disable-custom-all-reduce
  --enable-lora
  --lora-paths "${LORA_PATHS_ARRAY[@]}"
  --max-loras-per-batch "${MAX_LORAS_PER_BATCH}"
  --max-lora-rank "${MAX_LORA_RANK}"
  --disable-overlap-schedule
)

if [[ "${LORA_USE_VIRTUAL_EXPERTS}" == "1" || "${LORA_USE_VIRTUAL_EXPERTS}" == "true" ]]; then
  cmd+=(--lora-use-virtual-experts)
fi

echo "PATCH_ROOT=${PATCH_ROOT}"
echo "SGLANG_BASE_PYTHON=${SGLANG_BASE_PYTHON}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "HOST=${HOST} PORT=${PORT} TP=${TP}"
echo "LORA_PATHS_ARGS=${LORA_PATHS_ARGS}"
echo "LORA_USE_VIRTUAL_EXPERTS=${LORA_USE_VIRTUAL_EXPERTS}"
echo "Command: ${cmd[*]}"
exec "${cmd[@]}"
