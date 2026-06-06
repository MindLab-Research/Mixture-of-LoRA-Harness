#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mol_harness import RouterHarness, load_lora_library  # noqa: E402


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 120.0) -> tuple[int, dict]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw}
        return exc.code, body


def tokenize_count(base_url: str, text: str) -> int:
    status, data = request_json(
        "POST",
        f"{base_url.rstrip('/')}/tokenize",
        {"prompt": text, "add_special_tokens": False},
        timeout=60,
    )
    if status >= 400:
        raise RuntimeError(f"tokenize failed HTTP {status}: {json.dumps(data, ensure_ascii=False)}")
    return int(data["count"])


def answer_prefix(user_text: str) -> str:
    return f"User request:\n{user_text.strip()}\n\nAnswer:"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one route_decode_v2 request to a patched SGLang server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/lora_library"))
    parser.add_argument("--model", default="l0_chat", help="Entry LoRA model name served by SGLang.")
    parser.add_argument("--router-max-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument(
        "--disable-kv-reuse",
        action="store_true",
        help=(
            "Do not ask route_decode_v2 to keep the current-query KV prefix. "
            "Full no-reuse production tests should use a route-then-reprefill harness."
        ),
    )
    parser.add_argument("prompt", nargs="?", default="Fix a failing pytest in this repository and run verification.")
    args = parser.parse_args()

    tasks = load_lora_library(args.library_dir)
    harness = RouterHarness(tasks)
    user_text = args.prompt
    prefix = answer_prefix(user_text)
    keep_prefix_tokens = tokenize_count(args.base_url, prefix)
    enable_kv_reuse = not args.disable_kv_reuse
    request_keep_prefix_tokens = keep_prefix_tokens if enable_kv_reuse else 0
    deterministic = harness.route_by_library(user_text)

    payload = {
        "model": args.model,
        "prompt": prefix + "\n\n" + harness.router_instruction(),
        "max_tokens": args.router_max_tokens + args.decode_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "rid": f"mol-route-decode-{uuid.uuid4().hex[:10]}",
        "custom_params": {
            "lora_router": {
                "mode": "route_decode_v2",
                "entry_route_id": "L0",
                "base_route_id": "L0",
                "route_to_adapter": harness.route_to_adapter(),
                "router_max_tokens": args.router_max_tokens,
                "decode_tokens": args.decode_tokens,
                "keep_prefix_token_count": request_keep_prefix_tokens,
                "query_prefix_token_count": keep_prefix_tokens,
                "specialist_context_token_count": keep_prefix_tokens,
                "query_cache_reused_token_count": keep_prefix_tokens if enable_kv_reuse else 0,
                "task_reprefill_token_count": 0 if enable_kv_reuse else keep_prefix_tokens,
                "task_reprefill_required": not enable_kv_reuse,
                "enable_kv_reuse": enable_kv_reuse,
                "deterministic_route": deterministic.route_id,
                "deterministic_route_decision": deterministic.decision,
                "user_text": user_text,
            }
        },
    }

    start = time.perf_counter()
    status, data = request_json(
        "POST",
        f"{args.base_url.rstrip('/')}/v1/completions",
        payload,
        timeout=180,
    )
    wall = time.perf_counter() - start
    print(json.dumps({"http_status": status, "wall_s": round(wall, 3)}, indent=2))
    print(json.dumps(data, ensure_ascii=False, indent=2)[:8000])
    return 0 if status < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
