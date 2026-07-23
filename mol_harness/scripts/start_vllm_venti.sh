#!/usr/bin/env bash
# Macaron-V1-Venti clean engine: native GLM-5.2 TP8/PP1 without DCP.
set -euo pipefail

MODEL_ROOT="${VENTI_MODEL_ROOT:-/root/glm52_local}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
EXPECTED_VLLM_VERSION="0.24.0"
EXPECTED_BASE_SHARDS=141

umask 077
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH MOL_PATCH_WORKER MOL_MINI_PATCH VLLM_LORA_ENABLE_DUAL_STREAM
unset VLLM_DISTRIBUTED_USE_SPLIT_GROUP VLLM_SCHEDULER_STEP_TRACE_DIR
unset VLLM_SCHEDULER_STEP_TRACE_MAX_STEPS

actual_vllm_version="$($PYTHON_BIN -c 'import vllm; print(vllm.__version__)')"
if [[ "${actual_vllm_version%%+*}" != "$EXPECTED_VLLM_VERSION" ]]; then
  printf 'expected vLLM %s, found %s\n' \
    "$EXPECTED_VLLM_VERSION" "$actual_vllm_version" >&2
  exit 1
fi

"$PYTHON_BIN" - "$MODEL_ROOT" "$EXPECTED_BASE_SHARDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_shards = int(sys.argv[2])
if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
    raise SystemExit(f"model root must be a real local directory: {root}")

base = root / "base"
index_path = base / "model.safetensors.index.json"
for required in (base / "config.json", index_path):
    if required.is_symlink() or not required.is_file() or required.stat().st_size == 0:
        raise SystemExit(f"missing real local model file: {required}")
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if len(shards) != expected_shards:
    raise SystemExit(f"expected {expected_shards} base shards, found {len(shards)}")
for shard in shards:
    path = base / shard
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing real local model shard: {path}")
for name in ("L0", "L1", "L2", "L3"):
    adapter = root / "loras" / name
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("r", 0)) > 16:
        raise SystemExit(f"{name} rank exceeds Venti max rank 16")
    if weights_path.is_symlink() or not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise SystemExit(f"missing real local adapter weights: {weights_path}")
print(f"validated Venti: {len(shards)} base shards and four adapters")
PY

gpu_count="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < 8 )); then
  printf 'expected at least 8 visible GPUs, found %s\n' "$gpu_count" >&2
  exit 1
fi
if [[ "${VLLM_CONFIG_CHECK_ONLY:-0}" == "1" ]]; then
  printf 'Venti configuration validated with vLLM %s and %s GPUs\n' \
    "$actual_vllm_version" "$gpu_count"
  exit 0
fi

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ROOT/base" \
  --served-model-name glm52-fp8-official \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --attention-backend FLASHMLA_SPARSE \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.915 \
  --max-num-seqs 8 \
  --max-cudagraph-capture-size 8 \
  --enable-lora \
  --lora-modules \
    L0="$MODEL_ROOT/loras/L0" \
    L1="$MODEL_ROOT/loras/L1" \
    L2="$MODEL_ROOT/loras/L2" \
    L3="$MODEL_ROOT/loras/L3" \
  --max-lora-rank 16 \
  --max-loras 4 \
  --max-cpu-loras 4 \
  --enable-prefix-caching \
  --max-model-len 262144 \
  --no-async-scheduling \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice
