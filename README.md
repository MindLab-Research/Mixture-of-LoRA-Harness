# Mixture-of-LoRA Harness

Mixture-of-LoRA Harness is a lightweight router and SGLang overlay for serving
multiple LoRA adapters behind one OpenAI-compatible endpoint.

The repository is intentionally source-only. It does not include model weights,
shadow LoRA directories, logs, trace data, or benchmark results.

## Components

- `mol_harness/`: LoRA Library parser, router prompt builder, and deterministic
  metadata guardrails.
- `examples/lora_library/`: small placeholder task descriptions for L0-L4.
- `lora_library/glm51_current/`: sanitized L0-L4 task descriptions used in our
  GLM5.1 experiments. It contains metadata only, with placeholder checkpoint
  and dataset paths.
- `examples/route_decode_client.py`: one OpenAI-compatible completion request
  that uses `route_decode_v2` custom params.
- `oai_api_connectivity_test/`: external-client examples for `lora_adapter_id`
  selection, including `auto` routing and direct `L0`-`L4` adapter inference.
- `examples/scripts/create_shadow_loras.py`: prepares SGLang-loadable shadow
  LoRA directories by patching PEFT config and symlinking weights.
- `sglang_patch/`: non-invasive SGLang overlay patch. It is activated with
  `PYTHONPATH`, not by editing the installed SGLang source tree.

## Routing Design

L0 is both the entry router LoRA and the general chat LoRA. Specialist LoRAs are
selected only when the current user request clearly matches a task family.

The router is hybrid:

- The prompt route is produced by the model from task descriptions, rules, and
  examples.
- The metadata guardrail can override noisy model output when strong LoRA
  Library signals clearly match a specialist.
- Invalid, missing, or ambiguous routes fall back to L0.

The intended multi-turn policy is:

- L0 routing sees only the current user query.
- L0 chat can see full conversation history.
- Specialist LoRAs see the current query plus same-task history.
- Cross-task specialist history is masked at the harness policy level.
- After a specialist finishes a turn, the next new task starts from L0 again.
- If `enable_kv_reuse=true`, the patched SGLang path can reuse the continuous
  current-query prefix after removing router-only prompt/decode tokens.

## Install

```bash
git clone https://github.com/MindLab-Research/Mixture-of-LoRA-Harness-alpha.git
cd Mixture-of-LoRA-Harness-alpha
python3 -m pip install -r requirements.txt
```

## Configure a LoRA Library

Copy `examples/lora_library/` and edit each `source_path` and `adapter_name`:

```bash
cp -r examples/lora_library my_lora_library
$EDITOR my_lora_library/L2.md
```

Each markdown file defines one route:

- `id`: route id returned by the router, for example `L2`.
- `adapter_name`: the server-visible LoRA model id, for example `l2_coding`.
- `source_path`: the local adapter checkpoint directory. This is where the LoRA
  adapter path is declared in the library.

Prepare shadow LoRA directories:

```bash
python3 examples/scripts/create_shadow_loras.py \
  --library-dir my_lora_library \
  --output-dir shadow_loras
```

`shadow_loras/` is ignored by git because it points to model weights.

This repository also includes `lora_library/glm51_current/`, which preserves
the current L0-L4 descriptions, routing rules, signal lists, and dataset
reference structure. Checkpoint and dataset paths are sanitized placeholders;
no LoRA weights or private storage paths are committed.

To prepare the GLM5.1 library:

```bash
python3 examples/scripts/create_shadow_loras.py \
  --library-dir lora_library/glm51_current \
  --output-dir shadow_loras
```

## Start Patched SGLang

```bash
export PATCH_ROOT=$PWD/sglang_patch
export SGLANG_BASE_PYTHON=/path/to/upstream/sglang/python
export MODEL_PATH=/path/to/base/model
export MODEL_NAME=your-base-model-name
export LORA_PATHS_ARGS='l0_chat=shadow_loras/L0 l2_coding=shadow_loras/L2'

bash sglang_patch/scripts/start_sglang_kv_reuse_server.sh
```

`LORA_PATHS_ARGS` is where the actual adapter paths are passed to SGLang. After
startup, requests use `adapter_name`; they do not send filesystem paths.

For an L0-L4 setup:

```bash
export LORA_PATHS_ARGS='l0_chat=shadow_loras/L0 l1_living_vita_tau3=shadow_loras/L1 l2_swe_tb2=shadow_loras/L2 l3_a2ui=shadow_loras/L3 l4_openclaw_pinch=shadow_loras/L4'
```

Configure route decode:

```bash
curl -s http://127.0.0.1:30000/v1/configure_lora_router \
  -H 'Content-Type: application/json' \
  -d '{"lora_pool":["l0_chat","l2_coding"],"switch_every_n_tokens":0,"mode":"route_decode_v2"}'
```

## Send a Test Request

You can first test metadata routing without a model server:

```bash
python3 examples/offline_router_demo.py \
  --library-dir examples/lora_library \
  "Fix a failing pytest in this repository and run verification."
```

To test the patched SGLang path:

```bash
python3 examples/route_decode_client.py \
  --base-url http://127.0.0.1:30000 \
  --library-dir my_lora_library \
  --model l0_chat \
  "Fix a failing pytest in this repository and run verification."
```

The response includes normal completion text plus `lora_router_*` metadata when
the patched SGLang overlay is active.

## Validation Boundary

There are two different validation scopes:

- Route/KV mechanism tests use short completion decodes to verify
  `route_decode_v2`, same-request LoRA switching, router-token trimming, and
  the `enable_kv_reuse=true/false` paths.
- Full task tests with tools are end-to-end task execution tests. They route
  with L0, switch to the selected adapter, let the adapter issue tool calls, and
  replay fake tool results until final answer decode. Teacher-forcing runs are
  diagnostics only and should not be counted as natural model performance.

Do not compare 16-token route/KV smoke tests with full tool-loop task
completion scores as if they measured the same thing.

## Notes

- `route_decode_v2` reuses a continuous KV prefix after router-only tokens are
  trimmed. It does not implement arbitrary non-contiguous GPU KV cache splicing.
- The overlay is tied to the upstream SGLang version it was derived from. Check
  `sglang_patch/changed_files.txt` before applying it to a different version.
- LoRA Library checkpoint and dataset paths are placeholders. No adapter weights
  or private storage paths are committed; users should replace paths with
  locations they can access.
