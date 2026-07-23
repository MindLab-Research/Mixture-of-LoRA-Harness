#!/usr/bin/env bash
# Restart the native Macaron-V1-Venti TP8 engine and selected harness.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
exec "$HERE/restart_model_stack.sh" \
  venti Macaron-V1-Venti "$HERE/start_vllm_venti.sh"
