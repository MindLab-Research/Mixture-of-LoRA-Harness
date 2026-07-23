# Macaron Mixture-of-LoRA Harness

Mixture-of-LoRA Harness serves two model profiles:

- **Macaron-V1-Venti**, a 748B-parameter model composed of a 744B GLM-5.2
  base model and four 1B LoRA specialists.
- **Macaron-V1-Tall**, a 50B-parameter model composed of a 35B
  Qwen3.6-35B-A3B base model and four 3.7B Rank-64 LoRA specialists.

One process exposes one OpenAI-compatible model. For each new user request,
the harness routes inference to `L0`, `L1`, `L2`, or `L3` while keeping the
adapter names internal.

The release uses native multi-LoRA support from vLLM or SGLang. Routing,
per-adapter context, summaries, and tool-call stickiness live in the Proxy.

## Hosted Service

Visit [mint.macaron.im](https://mint.macaron.im) to register and try Macaron.

## Architecture

The checked-in restart entrypoints use a direct single-engine topology:

```text
OpenAI client
    |
    v
MoL Proxy :8200
    |  route -> answer -> summary
    v
vLLM or SGLang :8000
    |
    +-- L0  general chat, model identity, and routing
    +-- L1  personal-agent and service workflows
    +-- L2  coding and terminal workflows
    `-- L3  UI and A2UI generation
```

Every new user request runs three internal steps:

1. **Route**: L0 selects one canonical adapter.
2. **Answer**: the selected adapter answers from its own conversation view.
3. **Summary**: the selected adapter creates a short cross-adapter summary.

The summary is stored by the Proxy and is not returned to the client. A tool
result continues on the adapter that issued the tool call and does not run
another route hop. Stable per-adapter prompts allow the native engine prefix
cache to reuse each adapter's existing KV prefix.

## Model Profiles

`SERVED_MODEL_NAME` selects one profile directory when the Proxy starts:

| Served model | Base model | Profile directory |
|---|---|---|
| `Macaron-V1-Venti` | GLM-5.2 | `mol_harness/Macaron-V1-Venti/` |
| `Macaron-V1-Tall` | Qwen3.6-35B-A3B | `mol_harness/Macaron-V1-Tall/` |

Each profile contains:

- `lora.md`: aggregated L0-L3 descriptions and adapter mappings.
- `route.md`: the L0 route prompt.
- `intro.md`: model-specific identity instructions.
- `summary.md`: the hidden task-summary prompt.

The Engine must expose adapter names `L0`, `L1`, `L2`, and `L3`, matching the
selected profile's `lora.md`.

## Release Contents

- `mol_harness/`: Proxy, router, state management, Chat Completions, and
  Responses API implementation.
- `mol_harness/Macaron-V1-Venti/`: Venti profile prompts and descriptions.
- `mol_harness/Macaron-V1-Tall/`: Tall profile prompts and descriptions.
- `mol_harness/scripts/`: validated production launchers.

Model weights, logs, benchmarks, experimental patches, and test suites are not
included.

## Requirements

- Linux with a CUDA environment supported by the selected base model.
- Python 3.10 or newer.
- vLLM 0.24.0 or SGLang 0.5.13.post1 with native multi-LoRA support.
- Eight visible GPUs for the checked-in vLLM production profiles.
- `curl` and `jq` for API and health checks.

Review parallelism, memory utilization, context length, and LoRA residency
before using a different host.

## Model Layout

The vLLM launchers use `/root/glm52_local` for Venti and
`/root/qwen36_local` for Tall. Both directories follow this layout:

```text
/root/<model-root>/
|-- base/
|   |-- config.json
|   |-- model.safetensors.index.json
|   `-- model-*.safetensors
`-- loras/
    |-- L0/
    |   |-- adapter_config.json
    |   `-- adapter_model.safetensors
    |-- L1/
    |-- L2/
    `-- L3/
```

The launchers require real local files rather than symlinked model shards.

## Install

```bash
git clone https://github.com/MindLab-Research/Mixture-of-LoRA-Harness.git
cd Mixture-of-LoRA-Harness

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install vLLM or SGLang in the CUDA environment used by the Engine. Engine
packages are not pinned in `requirements.txt` because their wheels depend on
the host CUDA stack.

## Start vLLM

Run the matching configuration check before starting a stack:

```bash
VLLM_CONFIG_CHECK_ONLY=1 \
  ./mol_harness/scripts/start_vllm_venti.sh

VLLM_CONFIG_CHECK_ONLY=1 \
  ./mol_harness/scripts/start_vllm_tall.sh
```

Start exactly one complete Engine and Proxy stack:

```bash
MOL_API_KEY_FILE=/path/to/api-key \
  ./mol_harness/scripts/restart_vllm_tp8_venti.sh

# Or start Tall instead:
MOL_API_KEY_FILE=/path/to/api-key \
  ./mol_harness/scripts/restart_vllm_tp8_tall.sh
```

For an isolated development host, replace `MOL_API_KEY_FILE` with
`MOL_ALLOW_UNAUTHENTICATED=1`.

The restart helper:

1. Stops the selected profile's Proxy and Engine from their PID files.
2. Refuses startup while ports 8000 or 8200 are occupied.
3. Validates vLLM 0.24.0, eight GPUs, model shards, and all four adapters.
4. Starts vLLM on `127.0.0.1:8000` and waits up to one hour for `/health`.
5. Verifies that `/v1/models` contains L0-L3.
6. Selects the matching profile and starts the Proxy on port 8200.

Production profile differences:

| Setting | Macaron-V1-Venti | Macaron-V1-Tall |
|---|---|---|
| Base model | GLM-5.2 | Qwen3.6-35B-A3B |
| Model root | `/root/glm52_local` | `/root/qwen36_local` |
| Base shards | 141 | 26 |
| Parallelism | TP8 / PP1 / DCP disabled | TP8 / PP1 / DCP disabled |
| Attention and KV | `FLASHMLA_SPARSE`, automatic KV dtype | hybrid attention, BF16, automatic KV dtype |
| LoRA residency | four adapters, maximum rank 16 | four adapters, rank 64 |
| Parsers | reasoning `glm45`, tools `glm47` | reasoning `qwen3`, tools `qwen3_coder` |
| Prompt token details | disabled | `--enable-prompt-tokens-details` |

Both profiles use a 262,144-token context limit, `max_num_seqs=8`,
`gpu_memory_utilization=0.915`, CUDA Graph capture up to 8, prefix caching, and
disabled async scheduling.

The profiles default to the same eight GPUs and ports, so they cannot run
together on one eight-GPU host. The helper stops only the profile named by its
entrypoint; stop the active stack before switching between Venti and Tall.

## Start SGLang

The checked-in SGLang launcher currently targets the Venti GLM-5.2 profile:

```bash
MODEL_ROOT=/root/glm52_local \
  ./mol_harness/scripts/start_glm52_sglang_tp8_4lora.sh
```

It uses TP8, a 262,144-token context, native multi-LoRA, virtual LoRA experts,
and GLM reasoning/tool parsers. CUDA Graph and overlap scheduling are disabled
by default as correctness-first settings for concurrent multi-LoRA requests.

After the Engine is healthy, start a direct Venti Proxy:

```bash
MOL_API_KEY_FILE=/path/to/api-key \
SERVED_MODEL_NAME=Macaron-V1-Venti \
UPSTREAM=http://127.0.0.1:8000 \
PROXY_PORT=8200 \
  ./mol_harness/scripts/start_mol_harness.sh
```

## API

The Proxy listens on port 8200 and exposes only the selected public model:

```bash
BASE_URL=http://127.0.0.1:8200
API_KEY=replace-me

curl -sS "$BASE_URL/health"
curl -sS "$BASE_URL/v1/models" \
  -H "Authorization: Bearer $API_KEY" | jq .
```

Chat Completions:

```bash
curl -sS "$BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Macaron-V1-Venti",
    "messages": [{"role": "user", "content": "Who are you?"}],
    "max_tokens": 256
  }' | jq .
```

Responses API:

```bash
curl -sS "$BASE_URL/v1/responses" \
  -H "Authorization: Bearer $API_KEY" \
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

Chat Completions is stateless: clients resend conversation history. Responses
stores state in the Proxy and continues from `previous_response_id`.

## Shutdown

Stop accepting new traffic and drain in-flight requests, then stop the Proxy
before the vLLM or SGLang Engine.

This prevents new requests from reaching an Engine that is shutting down.

## Acknowledgements

We thank the [SGLang](https://github.com/sgl-project/sglang) and
[vLLM](https://github.com/vllm-project/vllm) teams for their support and for
the open-source inference systems that make this work possible.

## License

Released under the [MIT License](LICENSE).

A [Mind Lab](https://macaron.im/mindlab) Contribution - A Lab for Experiential Intelligence.
