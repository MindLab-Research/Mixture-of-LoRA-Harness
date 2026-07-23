#!/usr/bin/env bash
# Restart the Macaron-V1-Tall TP8 engine and its selected harness.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
exec "$HERE/restart_model_stack.sh" \
  tall Macaron-V1-Tall "$HERE/start_vllm_tall.sh"
