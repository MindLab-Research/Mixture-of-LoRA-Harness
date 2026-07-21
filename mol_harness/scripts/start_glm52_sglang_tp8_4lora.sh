#!/usr/bin/env bash
# Native SGLang + GLM-5.2-FP8 + four Macaron-V1-Venti LoRA adapters.
# Additional command-line arguments are forwarded verbatim to SGLang.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-/root/glm52_local}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/base}"
LORA_ROOT="${LORA_ROOT:-${MODEL_ROOT}/loras}"
MODEL_NAME="${MODEL_NAME:-glm52-fp8-official}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_LOADED_LORAS="${MAX_LOADED_LORAS:-4}"
MAX_LORAS_PER_BATCH="${MAX_LORAS_PER_BATCH:-2}"

for path in "${MODEL_PATH}" \
  "${LORA_ROOT}/L0" "${LORA_ROOT}/L1" \
  "${LORA_ROOT}/L2" "${LORA_ROOT}/L3"; do
  if [[ ! -d "${path}" ]]; then
    printf 'missing model or adapter directory: %s\n' "${path}" >&2
    exit 1
  fi
done

unset MOL_PATCH_WORKER MOL_MINI_PATCH

cmd=(
  "${PYTHON_BIN}" -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp "${TP}"
  --context-length "${CONTEXT_LENGTH}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --enable-metrics
  --enable-mfu-metrics
  --disable-custom-all-reduce
  --enable-lora
  --lora-paths
    "L0=${LORA_ROOT}/L0"
    "L1=${LORA_ROOT}/L1"
    "L2=${LORA_ROOT}/L2"
    "L3=${LORA_ROOT}/L3"
  --max-loaded-loras "${MAX_LOADED_LORAS}"
  --max-loras-per-batch "${MAX_LORAS_PER_BATCH}"
  --max-lora-rank "${MAX_LORA_RANK}"
  --lora-use-virtual-experts
)

# These correctness-first defaults avoid the concurrent multi-LoRA corruption
# and scheduling hazards observed in the validated SGLang 0.5.13 environment.
if [[ "${DISABLE_OVERLAP:-1}" == "1" ]]; then
  cmd+=(--disable-overlap-schedule)
fi
if [[ "${DISABLE_CUDA_GRAPH:-1}" == "1" ]]; then
  cmd+=(--disable-cuda-graph)
fi
cmd+=("$@")

printf 'Starting native SGLang:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
