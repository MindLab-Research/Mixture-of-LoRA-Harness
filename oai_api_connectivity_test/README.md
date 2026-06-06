# OpenAI-Compatible LoRA Adapter Test

This directory contains a small external-client test for a patched SGLang
OpenAI-compatible service.

The test is intentionally a pure client. It does not import the harness package
and does not need local access to LoRA weights. By default it fetches task
metadata from the server via `GET /v1/lora_router_library`; `--library-dir` is
only a fallback for offline/local development.

The client exposes one selector:

- `--lora_adapter_id auto`: default. Start from L0 and use the automatic
  `route_decode_v2` LoRA router.
- `--lora_adapter_id L0`: directly decode with the L0 chat adapter.
- `--lora_adapter_id L1`: directly decode with the L1 living/vita/tau3 adapter.
- `--lora_adapter_id L2`: directly decode with the L2 swe/tb2 adapter.
- `--lora_adapter_id L3`: directly decode with the L3 a2ui adapter.
- `--lora_adapter_id L4`: directly decode with the L4 openclaw/pinch adapter.

By default, the script sends one request using the current
`--lora_adapter_id` value. Since the default adapter selector is `auto`, running
the script with no arguments returns one `auto` result. The built-in prompt
samples are not all executed unless you pass `--sample all`.

For explicit `L0`-`L4` requests, the client sends a normal `/v1/completions`
request whose `model` is the corresponding server-visible `adapter_name`
returned by the service LoRA Library. The whole completion uses that single
adapter.

For `auto`, the client sends a `/v1/completions` request to L0 with
`custom_params.lora_router.mode=route_decode_v2`. The patched server decodes the
route, trims router-only context, switches to the selected adapter, and then
continues decoding. The client also forwards `route_signals` loaded from the
server LoRA Library so the patched server can apply its own fallback override if
L0 does not emit a parseable `model_id=...` line.

Runtime requests do not send adapter checkpoint paths. The LoRA checkpoint paths
are declared in `lora_library/glm51_current/*.md` and must already be loaded by
the SGLang service through `LORA_PATHS_ARGS` or `--lora-paths`.

The default base URL is `SGLANG_BASE_URL` if that environment variable is set;
otherwise it falls back to `http://127.0.0.1:30000`.

## Minimal Examples

Automatic routing:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id auto \
  --prompt "Fix a failing pytest in this repository and run verification."
```

Direct L0 chat:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id L0 \
  --prompt "Explain KV cache reuse in one short paragraph."
```

Direct L1 personal-agent task:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id L1 \
  --prompt "你好，帮我安排明晚和朋友聚餐，预算五百以内。"
```

Direct L2 coding task:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id L2 \
  --prompt "Fix a failing pytest in this repository and run the relevant verification command."
```

Direct L3 UI task:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id L3 \
  --prompt "Create a dashboard UI screen with filters, a data table, loading state, and empty state."
```

Direct L4 workspace task:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --lora_adapter_id L4 \
  --prompt "Read files under input/, produce the required JSON artifact under output/, and summarize what changed."
```

Run the built-in samples for `auto` and all explicit adapters:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://YOUR_SGLANG_HOST:30000 \
  --sample all
```

When running directly on the server, override the public URL with the local
SGLang address:

```bash
python3 oai_api_connectivity_test/test_lora_adapter_oai_api.py \
  --base-url http://127.0.0.1:30000 \
  --lora_adapter_id auto
```

## Notes

- `auto` requires the patched SGLang overlay and the
  `/v1/configure_lora_router` endpoint.
- Explicit `L0`-`L4` requests only require that the chosen adapter is already
  loaded by the server.
- Use `--disable-kv-reuse` to ask `route_decode_v2` not to reuse the L0
  current-query prefix after routing.
- Use `--library-dir` if your adapter names differ from
  `lora_library/glm51_current`.
