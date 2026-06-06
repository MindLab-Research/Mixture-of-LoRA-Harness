#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request
import uuid


DEFAULT_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")
ENTRY_ROUTE_ID = "L0"
LORA_CHOICES = ("auto", "L0", "L1", "L2", "L3", "L4")

SAMPLE_PROMPTS = {
    "auto": "Fix a failing pytest in this repository and run the relevant verification command.",
    "L0": "Explain KV cache reuse in one short paragraph.",
    "L1": "你好，帮我安排明晚和朋友聚餐，预算五百以内，最好顺便推荐附近能散步的地方。",
    "L2": "Fix a failing pytest in this repository and run the relevant verification command.",
    "L3": "Create a dashboard UI screen with filters, a data table, loading state, and empty state.",
    "L4": "Read files under input/, produce the required JSON artifact under output/, and summarize what changed.",
}


def default_library_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    current = root / "lora_library" / "glm51_current"
    if current.exists():
        return current
    return root / "lora_library"


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict[str, Any], float]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw else {}
            return resp.status, body, time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw}
        return exc.code, body, time.perf_counter() - start
    except urllib.error.URLError as exc:
        return 599, {"error": "request_failed", "message": str(exc)}, time.perf_counter() - start


def parse_lora_markdown(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        _, raw_meta, body = text.split("---", 2)
        for line in raw_meta.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip().lower().replace(" ", "_")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    def section_text(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    def bullets(name: str) -> list[str]:
        out: list[str] = []
        for line in sections.get(name, []):
            stripped = line.strip()
            if stripped.startswith("- "):
                out.append(stripped[2:].strip())
        return out

    def comma_list(name: str) -> list[str]:
        raw = section_text(name)
        return [item.strip() for item in raw.split(",") if item.strip()]

    def section_items(name: str) -> list[str]:
        out: list[str] = []
        for line in sections.get(name, []):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            out.append(stripped)
        return out

    task_id = meta.get("id") or path.stem
    dataset = meta.get("dataset", "")
    datasets = section_items("datasets")
    if dataset and dataset not in datasets:
        datasets.insert(0, dataset)
    return {
        "id": task_id,
        "task": meta.get("task", task_id),
        "level": meta.get("level", ""),
        "adapter_name": meta.get("adapter_name", ""),
        "source_path": meta.get("source_path", ""),
        "priority": int(meta.get("priority", "0") or 0),
        "description": section_text("description"),
        "routing_rules": bullets("routing_rules"),
        "strong_signals": comma_list("strong_signals"),
        "positive_signals": comma_list("positive_signals"),
        "negative_signals": comma_list("negative_signals"),
        "examples": section_items("examples"),
        "datasets": datasets,
        "library_path": str(path),
    }


def load_local_lora_library(
    library_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str], str]:
    if not library_dir.exists():
        raise FileNotFoundError(f"Missing LoRA Library directory: {library_dir}")

    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(library_dir.glob("*.md")):
        item = parse_lora_markdown(path)
        tasks[item["id"]] = item
    if not tasks:
        raise FileNotFoundError(f"No .md task files found in LoRA Library: {library_dir}")
    if ENTRY_ROUTE_ID not in tasks:
        raise FileNotFoundError(f"LoRA Library must include {ENTRY_ROUTE_ID}.md: {library_dir}")

    route_to_adapter = {
        task_id: str(task.get("adapter_name") or task_id)
        for task_id, task in tasks.items()
    }
    return tasks, tasks[ENTRY_ROUTE_ID], route_to_adapter, f"local:{library_dir}"


def fetch_remote_lora_library(
    base_url: str,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str], str]:
    status, data, wall = request_json(
        "GET",
        f"{base_url.rstrip('/')}/v1/lora_router_library",
        timeout=timeout,
    )
    if status >= 400 or not data.get("success"):
        message = data.get("message") or data.get("raw") or data.get("error") or "unknown error"
        raise RuntimeError(f"Remote LoRA Library unavailable: HTTP {status}: {message}")

    tasks = data.get("tasks") or {}
    entry_metadata = data.get("entry_metadata") or data.get("base_model_metadata") or {}
    route_to_adapter = data.get("route_to_adapter") or {}
    if not tasks or not entry_metadata or not route_to_adapter:
        raise RuntimeError("Remote LoRA Library response is missing tasks/entry metadata/route_to_adapter.")
    return tasks, entry_metadata, route_to_adapter, f"remote:{data.get('library_dir') or ''}:wall_s={wall:.4f}"


def load_client_lora_library(
    base_url: str,
    library_dir: Path,
    timeout: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str], str]:
    try:
        return fetch_remote_lora_library(base_url, timeout)
    except Exception as remote_exc:
        if library_dir.exists():
            tasks, entry_metadata, route_to_adapter, source = load_local_lora_library(library_dir)
            return tasks, entry_metadata, route_to_adapter, f"{source}; remote_error={remote_exc}"
        raise RuntimeError(
            "Could not load LoRA Library from the remote service, and no local "
            f"fallback exists at {library_dir}. Make sure the SGLang service has "
            "the /v1/lora_router_library endpoint enabled, or pass --library-dir "
            "to a local copy of the md library."
        ) from remote_exc


def sorted_tasks(tasks: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(
        tasks.items(),
        key=lambda item: (item[0] != ENTRY_ROUTE_ID, -int(item[1].get("priority", 0) or 0), item[0]),
    )


def build_model_id_list(tasks: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"- {task_id}: {task.get('description', '')}"
        for task_id, task in sorted_tasks(tasks)
    )


def build_routing_rules(tasks: dict[str, dict[str, Any]]) -> str:
    rules: list[str] = []
    for task_id, task in sorted_tasks(tasks):
        for rule in task.get("routing_rules", []):
            rules.append(f"{task_id}: {rule}")
    return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))


def build_routing_examples(tasks: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for task_id, task in sorted_tasks(tasks):
        for example in task.get("examples", []):
            if "=>" in example:
                user, route = example.split("=>", 1)
                route = route.strip() or task_id
            else:
                user, route = example, task_id
            lines.append(f"User: {user.strip()}\nmodel_id={route}")
    return "\n".join(lines)


def build_router_instruction(tasks: dict[str, dict[str, Any]]) -> str:
    examples = build_routing_examples(tasks)
    examples_block = f"Examples:\n{examples}\n\n" if examples else ""
    return (
        "Router instruction:\n"
        "You are running inside L0, the entry chat LoRA for a Mixture-of-LoRA service. "
        "Choose exactly one model_id for the next decoding phase. "
        "Classify by the user's goal and execution environment, not by keyword overlap. "
        "Return L0 for ordinary chat, general reasoning, language work, or ambiguous requests. "
        "Route to specialist LoRAs only when one task family clearly matches. "
        "Slash-separated labels mean one adapter covers all listed task families.\n\n"
        "Available model ids:\n"
        f"{build_model_id_list(tasks)}\n\n"
        "Routing rules:\n"
        f"{build_routing_rules(tasks)}\n\n"
        f"{examples_block}"
        "Return exactly one line in this format and nothing else:\n"
        f"model_id=<{'|'.join(tasks)}>\n\n"
        "model_id="
    )


def build_route_signals(tasks: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    route_signals: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        if task_id == ENTRY_ROUTE_ID:
            continue
        signals: list[str] = []
        for field in ("strong_signals", "positive_signals"):
            for signal in task.get(field, []):
                if isinstance(signal, str) and signal.strip():
                    signals.append(signal.strip())
        route_signals[task_id] = list(dict.fromkeys(signals))
    return route_signals


def answer_prompt(user_text: str) -> str:
    return f"User request:\n{user_text.strip()}\n\nAnswer:"


def tokenize_count(base_url: str, text: str, timeout: float) -> int:
    status, data, _ = request_json(
        "POST",
        f"{base_url.rstrip('/')}/tokenize",
        {"prompt": text, "add_special_tokens": False},
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"tokenize failed HTTP {status}: {json.dumps(data, ensure_ascii=False)}")
    return int(data["count"])


def loaded_model_ids(base_url: str, timeout: float) -> set[str] | None:
    status, data, _ = request_json(
        "GET",
        f"{base_url.rstrip('/')}/v1/models",
        timeout=timeout,
    )
    if status >= 400:
        return None
    return {
        str(item.get("id"))
        for item in data.get("data", [])
        if item.get("id")
    }


def configure_lora_router(
    base_url: str,
    route_to_adapter: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    requested_adapter_names = list(dict.fromkeys(route_to_adapter.values()))
    loaded = loaded_model_ids(base_url, timeout)
    configured_adapter_names = requested_adapter_names
    if loaded is not None:
        configured_adapter_names = [
            name for name in requested_adapter_names if name in loaded
        ]
    status, data, wall = request_json(
        "POST",
        f"{base_url.rstrip('/')}/v1/configure_lora_router",
        {
            "lora_pool": configured_adapter_names,
            "switch_every_n_tokens": 0,
            "mode": "route_decode_v2",
        },
        timeout=timeout,
    )
    data["http_status"] = status
    data["wall_s"] = round(wall, 4)
    data["requested_lora_pool"] = requested_adapter_names
    data["configured_lora_pool"] = configured_adapter_names
    if loaded is not None:
        data["unloaded_lora_pool"] = [
            name for name in requested_adapter_names if name not in loaded
        ]
    return data


def build_auto_payload(
    *,
    base_url: str,
    tasks: dict[str, dict[str, Any]],
    route_to_adapter: dict[str, str],
    user_text: str,
    router_max_tokens: int,
    max_tokens: int,
    temperature: float,
    enable_kv_reuse: bool,
    timeout: float,
) -> dict[str, Any]:
    entry_model = route_to_adapter[ENTRY_ROUTE_ID]
    prefix = answer_prompt(user_text)
    prefix_tokens = tokenize_count(base_url, prefix, timeout)
    keep_prefix_tokens = prefix_tokens if enable_kv_reuse else 0
    return {
        "model": entry_model,
        "prompt": prefix + "\n\n" + build_router_instruction(tasks),
        "max_tokens": router_max_tokens + max_tokens,
        "temperature": temperature,
        "ignore_eos": True,
        "rid": f"oai-lora-auto-{uuid.uuid4().hex[:10]}",
        "custom_params": {
            "lora_router": {
                "mode": "route_decode_v2",
                "entry_route_id": ENTRY_ROUTE_ID,
                "base_route_id": ENTRY_ROUTE_ID,
                "base_route_adapter": entry_model,
                "route_to_adapter": route_to_adapter,
                "router_max_tokens": router_max_tokens,
                "decode_tokens": max_tokens,
                "keep_prefix_token_count": keep_prefix_tokens,
                "query_prefix_token_count": prefix_tokens,
                "specialist_context_token_count": prefix_tokens,
                "query_cache_reused_token_count": prefix_tokens if enable_kv_reuse else 0,
                "task_reprefill_token_count": 0 if enable_kv_reuse else prefix_tokens,
                "task_reprefill_required": not enable_kv_reuse,
                "enable_kv_reuse": enable_kv_reuse,
                "user_text": user_text,
                "route_signals": build_route_signals(tasks),
            }
        },
    }


def build_direct_payload(
    *,
    route_to_adapter: dict[str, str],
    lora_adapter_id: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": route_to_adapter[lora_adapter_id],
        "prompt": answer_prompt(user_text),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": False,
        "rid": f"oai-lora-{lora_adapter_id.lower()}-{uuid.uuid4().hex[:10]}",
    }


def response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("text") or "")


def run_one(
    *,
    base_url: str,
    tasks: dict[str, dict[str, Any]],
    route_to_adapter: dict[str, str],
    lora_adapter_id: str,
    prompt: str,
    max_tokens: int,
    router_max_tokens: int,
    temperature: float,
    enable_kv_reuse: bool,
    configure_router: bool,
    timeout: float,
) -> dict[str, Any]:
    if lora_adapter_id == "auto":
        config_result = (
            configure_lora_router(base_url, route_to_adapter, timeout)
            if configure_router
            else None
        )
        payload = build_auto_payload(
            base_url=base_url,
            tasks=tasks,
            route_to_adapter=route_to_adapter,
            user_text=prompt,
            router_max_tokens=router_max_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_kv_reuse=enable_kv_reuse,
            timeout=timeout,
        )
    else:
        config_result = None
        payload = build_direct_payload(
            route_to_adapter=route_to_adapter,
            lora_adapter_id=lora_adapter_id,
            user_text=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    status, data, wall = request_json(
        "POST",
        f"{base_url.rstrip('/')}/v1/completions",
        payload,
        timeout=timeout,
    )
    metadata = data.get("metadata") or {}
    return {
        "lora_adapter_id": lora_adapter_id,
        "prompt": prompt,
        "http_status": status,
        "wall_s": round(wall, 4),
        "request_model": payload["model"],
        "request_id": payload["rid"],
        "selected_route": metadata.get("lora_router_selected_route") or lora_adapter_id,
        "selected_adapter": metadata.get("lora_router_selected_adapter") or payload["model"],
        "router_metadata": metadata,
        "router_config": config_result,
        "text": response_text(data),
        "raw_response": data if status >= 400 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI-compatible LoRA adapter test. Use lora_adapter_id=auto for "
            "L0 route_decode_v2 routing, or L0-L4 for direct adapter inference."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--library-dir", type=Path, default=default_library_dir())
    parser.add_argument(
        "--lora_adapter_id",
        "--lora-adapter-id",
        dest="lora_adapter_id",
        choices=LORA_CHOICES,
        default="auto",
        help="Adapter selector for one request. Default: auto.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, the built-in prompt for the selected sample is used.",
    )
    parser.add_argument(
        "--sample",
        choices=(*LORA_CHOICES, "all"),
        default=None,
        help=(
            "Select built-in sample prompt(s). Default is the current --lora_adapter_id only. "
            "Use --sample all to run auto and L0-L4."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--router-max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-kv-reuse", action="store_true")
    parser.add_argument("--no-configure-router", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    tasks, entry_metadata, route_to_adapter, library_source = load_client_lora_library(
        base_url,
        args.library_dir,
        args.timeout,
    )
    _ = entry_metadata
    missing = [item for item in LORA_CHOICES if item != "auto" and item not in route_to_adapter]
    if missing:
        raise ValueError(f"LoRA Library is missing route ids: {missing}")

    if args.sample == "all":
        requested = list(LORA_CHOICES)
    else:
        requested = [args.sample or args.lora_adapter_id]

    results = []
    for adapter_id in requested:
        prompt = args.prompt or SAMPLE_PROMPTS[adapter_id]
        results.append(
            run_one(
                base_url=base_url,
                tasks=tasks,
                route_to_adapter=route_to_adapter,
                lora_adapter_id=adapter_id,
                prompt=prompt,
                max_tokens=args.max_tokens,
                router_max_tokens=args.router_max_tokens,
                temperature=args.temperature,
                enable_kv_reuse=not args.disable_kv_reuse,
                configure_router=not args.no_configure_router,
                timeout=args.timeout,
            )
        )

    print(
        json.dumps(
            {
                "library_source": library_source,
                "requested_samples": requested,
                "available_samples": list(SAMPLE_PROMPTS),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(item["http_status"] < 400 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
