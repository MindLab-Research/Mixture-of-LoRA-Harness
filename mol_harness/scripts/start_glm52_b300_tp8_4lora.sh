#!/usr/bin/env bash
# GLM-5.2-FP8 + four LoRAs on one 8 x B300 (reported as NVIDIA L20D) Engine.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
MODEL_ROOT="${MODEL_ROOT:-/root/glm52_local}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.24.0}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA L20D}"

unset PYTHONPATH MOL_PATCH_WORKER MOL_MINI_PATCH
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"

"$PYTHON_BIN" - "$EXPECTED_VLLM_VERSION" "$EXPECTED_GPU_COUNT" \
  "$EXPECTED_GPU_NAME" "$MODEL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import torch
import vllm

expected_version, expected_count, expected_name, configured_root = sys.argv[1:]
if vllm.__version__.split("+", 1)[0] != expected_version:
    raise SystemExit(
        f"expected vLLM {expected_version}, found {vllm.__version__}"
    )

gpu_count = torch.cuda.device_count()
if gpu_count != int(expected_count):
    raise SystemExit(f"expected {expected_count} visible GPUs, found {gpu_count}")
for index in range(gpu_count):
    actual_name = torch.cuda.get_device_name(index)
    if actual_name != expected_name:
        raise SystemExit(
            f"expected GPU {index} to be {expected_name!r}, found {actual_name!r}"
        )

root = Path(configured_root)
if root.is_symlink() or not root.is_dir():
    raise SystemExit(f"model root must be a real directory: {root}")
base = root / "base"
index_path = base / "model.safetensors.index.json"
for required in (base / "config.json", index_path):
    if required.is_symlink() or not required.is_file() or required.stat().st_size == 0:
        raise SystemExit(f"missing local model file: {required}")
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if len(shards) != 141:
    raise SystemExit(f"expected 141 base shards, found {len(shards)}")
for shard in shards:
    path = base / shard
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing local model shard: {path}")
for name in ("L0", "L1", "L2", "L3"):
    adapter = root / "loras" / name
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing local adapter file: {path}")
    json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
print(
    f"validated vLLM {vllm.__version__}, {gpu_count} x {expected_name}, "
    "141 base shards, and L0/L1/L2/L3"
)
PY

if [[ "${VLLM_CONFIG_CHECK_ONLY:-0}" == "1" ]]; then
  exit 0
fi

kv_cache_args=()
if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  kv_cache_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ROOT/base" \
  --served-model-name glm52-fp8-official \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend FLASHMLA_SPARSE \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
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
  --no-async-scheduling \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  "${kv_cache_args[@]}"
