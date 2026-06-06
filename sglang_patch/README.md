# SGLang Route-Decode LoRA KV Reuse Patch

This directory contains a non-invasive SGLang overlay patch for same-request
LoRA routing and KV-prefix reuse. It is applied by putting
`sglang_patch/overlay/python` before the upstream SGLang package on
`PYTHONPATH`; the upstream installation is not edited in place.

## What It Adds

- `/v1/configure_lora_router` to configure the loaded LoRA pool.
- `route_decode` and `route_decode_v2` router modes.
- Same-request flow: prefill/decode with entry LoRA, parse a route id, trim
  router-only tokens, switch to the selected LoRA, and continue decode.
- Metadata fields in OpenAI-compatible completion output, prefixed with
  `lora_router_`.
- `route_decode_v2` metadata for multi-turn harness policies, including current
  query prefix length, same-task reused length, cross-task masked length, and
  whether current-query KV was reused.
- `/v1/lora_router_library` to expose a markdown LoRA Library from the server
  environment when configured.

## Boundary

SGLang stores a request KV cache as a continuous sequence. This patch trims the
router prompt/decode suffix and reuses a continuous prefix for the selected
adapter. The higher-level harness can implement policy-level masking of
cross-task history by choosing which text is included in each request. Arbitrary
non-contiguous GPU KV splicing is not claimed here.

## Launch

```bash
export PATCH_ROOT=/path/to/Mixture-of-LoRA-Harness/sglang_patch
export SGLANG_BASE_PYTHON=/path/to/upstream/sglang/python
export MODEL_PATH=/path/to/base/model
export MODEL_NAME=your-base-model-name
export LORA_PATHS_ARGS='l0_chat=/path/to/shadow_loras/L0 l2_coding=/path/to/shadow_loras/L2'

bash "$PATCH_ROOT/scripts/start_sglang_kv_reuse_server.sh"
```

`LORA_PATHS_ARGS` is the adapter-path handoff point. Each item maps a
server-visible adapter name to a local shadow LoRA directory. Route-decode
requests refer to these already-loaded adapter names; they do not send
filesystem paths.

Then configure router mode:

```bash
curl -s http://127.0.0.1:30000/v1/configure_lora_router \
  -H 'Content-Type: application/json' \
  -d '{"lora_pool":["l0_chat","l2_coding"],"switch_every_n_tokens":0,"mode":"route_decode_v2"}'
```

`MAX_LORAS_PER_BATCH` defaults to the number of LoRAs in `LORA_PATHS_ARGS`.
Override it only when you understand the LoRA memory-pool tradeoff.

## Compatibility

The overlay is file-level and must match the upstream SGLang version it was
developed against. See `changed_files.txt` and `patches/` for the touched
modules and a compact patch summary.
