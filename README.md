# Macaron-V1-Venti Mixture-of-LoRA Harness

This repository serves **Macaron-V1-Venti**, a 748B-parameter model composed
of a 744B GLM-5.2 base model and four 1B LoRA specialists. It exposes one
OpenAI-compatible model while routing each request through the appropriate
adapter.

The release uses native multi-LoRA support from either vLLM or SGLang. No
engine monkey patch is required.

## Architecture

```text
OpenAI client
    |
    v
MoL Proxy :30000
    |
    v
SGLang Model Gateway :30001
    |
    v
vLLM or SGLang Engine :8000
    |
    +-- L0  general chat and model identity
    +-- L1  personal agent and service workflows
    +-- L2  coding and terminal workflows
    `-- L3  UI and A2UI generation
```

For every new user request, the Proxy performs:

1. **Route**: L0 classifies the current request into one canonical adapter.
2. **Answer**: the selected adapter answers using its own conversation view.
3. **Summary**: the selected adapter creates a short cross-adapter summary
   that is stored by the Proxy and is not returned to the client.

Tool-result continuations remain on the adapter that issued the tool call and
do not run another route hop. Stable per-adapter prompts allow the native
engine prefix cache to reuse each adapter's existing KV prefix.

## Release Contents

- `mol_harness/`: Proxy, router, state management, Chat Completions, and
  Responses API implementation.
- `lora_library/mol_glm52/`: the four evaluated Macaron-V1-Venti adapter
  descriptions and routing boundaries.
- `sgl-model-gateway/src/`: Gateway control plane and HTTP routing source.
- `mol_harness/scripts/`: native vLLM and SGLang GLM-5.2 launchers.

Model weights, logs, benchmarks, experimental patches, and test suites are not
included.

## Requirements

- Linux and a CUDA environment that supports GLM-5.2.
- Python 3.10 or newer.
- Either vLLM 0.24.0 or SGLang 0.5.13.post1 with native multi-LoRA support.
- Rust, Cargo, `protoc`, a C/C++ toolchain, and `pkg-config` to build the
  Gateway.
- `curl` and `jq` for registration and health checks.

The checked-in engine profiles use tensor parallelism across eight GPUs.
Adjust parallelism, memory utilization, context length, and LoRA residency for
different hardware before serving traffic.

## Model Layout

Both engine launchers use this layout by default:

```text
/root/glm52_local/
|-- base/
|   |-- config.json
|   `-- model-*.safetensors
`-- loras/
    |-- L0/
    |   |-- adapter_config.json
    |   `-- adapter_model.safetensors
    |-- L1/
    |-- L2/
    `-- L3/
```

Override `MODEL_ROOT`, or set `MODEL_PATH` and `LORA_ROOT` separately for
the SGLang launcher. Keep the engine-visible adapter names exactly
`L0`, `L1`, `L2`, and `L3`.

## Install

```bash
git clone https://github.com/MindLab-Research/Mixture-of-LoRA-Harness-alpha.git
cd Mixture-of-LoRA-Harness-alpha

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install vLLM or SGLang in the CUDA environment used by the Engine. Engine
packages are intentionally not pinned in `requirements.txt` because their
wheels are CUDA- and host-specific.

Build the Gateway:

```bash
cd sgl-model-gateway
command -v protoc
cargo build --release --bin smg --features vendored-openssl
cd ..
```

The binary is written to `sgl-model-gateway/target/release/smg`.

## Start an Engine

Choose exactly one of the following engines. Both serve the base model as
`glm52-fp8-official` and the adapters as `L0` through `L3` on port 8000.

### vLLM

The vLLM launcher is the validated GLM-5.2 FP8 TP8 profile. It checks vLLM
0.24.0, the GPU inventory, 141 base-model shards, and all four adapter files
before starting:

```bash
MODEL_ROOT=/root/glm52_local \
VLLM_CONFIG_CHECK_ONLY=1 \
  ./mol_harness/scripts/start_glm52_b300_tp8_4lora.sh

MODEL_ROOT=/root/glm52_local \
  ./mol_harness/scripts/start_glm52_b300_tp8_4lora.sh
```

The launcher executes `python -m vllm.entrypoints.openai.api_server` with:

- tensor parallel size 8;
- `FLASHMLA_SPARSE` attention;
- native multi-LoRA and prefix caching;
- GLM reasoning and tool-call parsers;
- `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` for FP8 multi-LoRA correctness.

The default hardware check expects eight GPUs named `NVIDIA L20D` in the
validated B300 environment. Set `EXPECTED_GPU_COUNT` and
`EXPECTED_GPU_NAME` to the exact values reported by PyTorch on another
validated host. Other useful overrides are `PYTHON_BIN`, `HOST`, `PORT`,
and `KV_CACHE_DTYPE`.

### SGLang

The SGLang launcher uses native multi-LoRA serving and does not load an MoL
overlay:

```bash
MODEL_ROOT=/root/glm52_local \
  ./mol_harness/scripts/start_glm52_sglang_tp8_4lora.sh
```

It executes `python -m sglang.launch_server` with TP8, a 262144-token context,
the four adapters, GLM reasoning/tool parsers, and virtual LoRA experts.
`--disable-cuda-graph` and `--disable-overlap-schedule` are enabled by
default because they are the validated correctness-first settings for
concurrent multi-LoRA requests.

Common overrides:

| Variable | Default |
|---|---|
| `MODEL_PATH` | `$MODEL_ROOT/base` |
| `LORA_ROOT` | `$MODEL_ROOT/loras` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` |
| `TP` | `8` |
| `CONTEXT_LENGTH` | `262144` |
| `MEM_FRACTION_STATIC` | `0.93` |
| `MAX_LORAS_PER_BATCH` | `2` |

Append additional native SGLang arguments after the launcher path. Argument
boundaries and quoted values are preserved by the shell:

```bash
MODEL_ROOT=/root/glm52_local \
  ./mol_harness/scripts/start_glm52_sglang_tp8_4lora.sh \
  --max-running-requests 64
```

For either engine, wait until both endpoints succeed:

```bash
curl -sf http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models | jq .
```

Confirm that `L0`, `L1`, `L2`, and `L3` appear in the model list.

## Start the Gateway

Start the Gateway with an empty worker registry:

```bash
./sgl-model-gateway/target/release/smg launch \
  --host 127.0.0.1 \
  --port 30001 \
  --policy manual \
  --assignment-mode min_load_then_group \
  --max-idle-secs 1800
```

Register the healthy Engine. Set `ENGINE_RUNTIME` to exactly `vllm` or
`sglang`; the Gateway uses this value to preserve the correct request shape
for each runtime.

```bash
ENGINE_RUNTIME=vllm

WORKER_JSON="$(jq -n \
  --arg runtime "$ENGINE_RUNTIME" \
  '{
    url: "http://127.0.0.1:8000",
    model_id: "glm52-fp8-official",
    worker_type: "regular",
    runtime: $runtime,
    priority: 50,
    cost: 1.0,
    labels: {deployment: "macaron-v1-venti"}
  }')"

REGISTERED="$(curl -sS -X POST http://127.0.0.1:30001/workers \
  -H 'Content-Type: application/json' \
  -d "$WORKER_JSON")"
printf '%s\n' "$REGISTERED" | jq .
WORKER_ID="$(printf '%s' "$REGISTERED" | jq -r '.worker_id')"

until curl -sf "http://127.0.0.1:30001/workers/$WORKER_ID" \
  | jq -e '.is_healthy == true' >/dev/null; do
  sleep 5
done

curl -sf http://127.0.0.1:30001/readiness | jq .
```

For SGLang, use `ENGINE_RUNTIME=sglang`; no other registration field changes.

## Start the Proxy

```bash
UPSTREAM=http://127.0.0.1:30001 \
PROXY_PORT=30000 \
LIBRARY_DIR=lora_library/mol_glm52 \
MOL_USE_MODEL_ROUTER=1 \
MOL_PURE_MODEL_ROUTE=1 \
SERVED_MODEL_NAME=Macaron-V1-Venti \
  python3 -m mol_harness.proxy
```

Set `MOL_API_KEY` to require `Authorization: Bearer <key>` on model
endpoints. `GET /health` remains unauthenticated.

Key Proxy settings:

| Variable | Default | Purpose |
|---|---:|---|
| `UPSTREAM` | `http://127.0.0.1:30001` | Gateway URL |
| `PROXY_PORT` | `30000` | Public listener |
| `LIBRARY_DIR` | `lora_library/mol_glm52` | Macaron routing metadata |
| `ROUTER_MAX_TOKENS` | `24` | L0 route decode budget |
| `SUMMARY_MAX_OUT` | `192` | Hidden summary budget |
| `MOL_MAX_INFLIGHT_REQUESTS` | `256` | Active orchestration limit |
| `MOL_MAX_QUEUED_REQUESTS` | `256` | Admission queue limit |
| `CONVO_TTL_S` | `1800` | Idle state lifetime |
| `MOL_API_KEY` | unset | Optional API authentication |

Use a process supervisor for production. Do not use broad `pkill` commands on
shared inference hosts.

## Verify

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:30001/readiness | jq .
curl -sS http://127.0.0.1:30000/health
curl -sS http://127.0.0.1:30000/v1/models | jq .
```

The Proxy exposes one public model, `Macaron-V1-Venti`. Internal base and
adapter model names are not returned to clients.

Chat Completions:

```bash
curl -sS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Macaron-V1-Venti",
    "messages": [{"role": "user", "content": "Who are you?"}],
    "max_tokens": 256
  }' | jq .
```

Responses API:

```bash
curl -sS http://127.0.0.1:30000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Macaron-V1-Venti",
    "input": "Write a Python function that validates an email address.",
    "max_output_tokens": 512
  }' | jq .
```

Supported public endpoints:

| Method | Path |
|---|---|
| `GET` | `/health` |
| `GET` | `/v1/models` |
| `POST` | `/v1/chat/completions` |
| `POST` | `/v1/responses` |

## Shutdown

Stop components in this order:

1. Stop accepting new traffic at the Proxy and allow in-flight requests to
   drain.
2. Delete the Engine worker from the Gateway with
   `DELETE /workers/$WORKER_ID`.
3. Stop the Gateway.
4. Stop the vLLM or SGLang Engine.

This prevents new requests from being routed to an Engine that is shutting
down.
