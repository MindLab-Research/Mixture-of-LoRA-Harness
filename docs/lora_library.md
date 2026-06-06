# LoRA Library Format

The harness reads one markdown file per route. Each file has YAML-style front
matter plus markdown sections.

Required front matter:

```yaml
---
id: L2
level: L2
task: coding
adapter_name: l2_coding
source_path: /path/to/lora
priority: 100
---
```

Front matter fields:

- `id`: route id emitted by the router, such as `L0`, `L1`, or `L2`.
- `adapter_name`: the LoRA model id served by SGLang. Requests and router
  metadata use this name after the adapter is loaded.
- `source_path`: the local filesystem path to the LoRA adapter checkpoint. The
  helper script reads this path to create `shadow_loras/<id>/`.
- `priority`: deterministic routing tie-breaker used by the metadata guardrail.

The OpenAI-compatible request does not send `source_path`. Adapter paths are
passed to SGLang at server launch through `--lora-paths` or `LORA_PATHS_ARGS`.
At request time the harness sends only route ids and already-loaded
`adapter_name` values.

Supported sections:

- `## Description`: compact task summary used in the router prompt.
- `## Routing Rules`: bullet rules for the language-model router.
- `## Strong Signals`: comma-separated high-confidence trigger phrases.
- `## Positive Signals`: comma-separated weaker trigger phrases.
- `## Negative Signals`: comma-separated phrases that should suppress the route.
- `## Examples`: route examples in the form `prompt => L2`.
- `## Datasets`: optional local dataset paths for evaluation scripts.

The routing harness combines two signals:

- A router prompt rendered from descriptions, rules, and examples.
- A deterministic metadata guardrail from strong/positive/negative signals.

The guardrail is specialist-first when metadata has a clear match: a specialist
metadata match overrides noisy model output. Invalid, missing, or ambiguous
routes fall back to L0. In hybrid deployments, a valid model-produced route can
still be accepted when metadata does not identify a stronger specialist match.
