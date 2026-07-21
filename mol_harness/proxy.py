#!/usr/bin/env python3
"""Engine-agnostic Macaron-V1-Venti three-hop orchestration Proxy.

All MoL orchestration lives here. The Engine is plain native multi-LoRA vLLM
or SGLang, and the Proxy talks OpenAI-compatible HTTP through the Gateway: one
``model=<adapter>`` call per hop.

Per inbound request the Proxy runs:

  1. route   — POST /v1/chat/completions model=L0 (raw request + evaluated
               router prompt, assistant prefill ``model_id=``, constrained
               canonical choice) → L0 selects exactly one LoRA label Lx.
  2. answer  — POST /v1/chat/completions model=Lx (Lx own-view: own traces
               verbatim + others' summaries + current turn). Stream/forward
               to the agent. If it ends in a tool_call → forward, set
               ``pending_tool_route=Lx``, skip step 3.
  3. summary — POST /v1/chat/completions model=Lx (trace + summary scaffold).
               1-2 sentence summary → append to the Proxy timeline. Not returned
               to the agent.

Per-LoRA KV reuse is emergent: each hop reconstructs the target LoRA's own-view
prefix verbatim (``session.own_view_messages``), so on re-entry the engine's
native LoRA-aware prefix cache hits and only the appended tail is prefilled.
No engine KV-reuse patch is required (Phase-1 design).

The weighted library scorer is retained only for explicit legacy mode. Normal
operation uses L0 model output alone, with the same own-view, summary, and tool
stickiness semantics implemented as independent OAI hops above a native engine.

Env:
  PROXY_PORT          listen port (default 30000)
  UPSTREAM            Gateway base (default http://127.0.0.1:30001)
  LIBRARY_DIR         LoRA library dir (default lora_library/mol_glm52)
  ENTRY_ROUTE         entry/router route id (default L0)
  MOL_MAX_REQUEST_BYTES  maximum JSON request body (default 8 MiB)
  ROUTER_MAX_TOKENS   L0 router decode budget (default 24)
  SUMMARY_MAX_OUT     summary decode budget (default 192)
  MOL_USE_MODEL_ROUTER  1 run the L0 model-router hop (default), 0 library-only
  MOL_PURE_MODEL_ROUTE  1 trust the model route only (default), 0 legacy guardrail
  CONVO_TTL_S         idle conversation eviction (default 1800)
  MOL_CLIENT_TIMEOUT_S  client socket I/O timeout (default 60)
  MOL_MAX_INFLIGHT_REQUESTS  async active-request limit (default 256)
  MOL_MAX_QUEUED_REQUESTS  async FIFO queue limit (default 256)
  MOL_API_KEY          optional Bearer token for public model endpoints
"""
from __future__ import annotations

import copy
import json
import hashlib
import hmac
import os
import re
import sys
import time
import threading
import uuid
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict
from typing import Any

from .library import load_lora_library
from .router import RouterHarness
from .session import (
    ConvoState,
    Task,
    split_system_and_turn,
    extract_tool_results,
    _msg_text,
    _adopt_system,
)
from .stateless import StatelessSideContext

# --------------------------------------------------------------------------- config

PROXY_PORT = int(os.environ.get("PROXY_PORT", "30000"))
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:30001").rstrip("/")
_DEFAULT_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lora_library", "mol_glm52",
)
LIBRARY_DIR = os.environ.get("LIBRARY_DIR", _DEFAULT_LIB)
ENTRY_ROUTE = os.environ.get("ENTRY_ROUTE", "L0")
ROUTER_MAX_TOKENS = int(os.environ.get("ROUTER_MAX_TOKENS", "24"))
SUMMARY_MAX_OUT = int(os.environ.get("SUMMARY_MAX_OUT", "192"))
# Answer-hop output (decode) token 上限。用户不传 max_tokens 时默认此值,
# 用户传更大值时截到此值。engine max_model_len(1M)保证 input+output≤1M,
# 故实际 output ≤ min(MAX_OUTPUT_TOKENS, max_model_len - input)。
MAX_OUTPUT_TOKENS = int(os.environ.get("MOL_MAX_OUTPUT_TOKENS", "131072"))  # 128k
USE_MODEL_ROUTER = os.environ.get("MOL_USE_MODEL_ROUTER", "1") in ("1", "true", "True")
PURE_MODEL_ROUTE = (
    USE_MODEL_ROUTER
    and os.environ.get("MOL_PURE_MODEL_ROUTE", "1") in ("1", "true", "True")
)
CONVO_TTL_S = max(1, int(os.environ.get("CONVO_TTL_S", "1800")))
HOP_TIMEOUT = float(os.environ.get("MOL_HOP_TIMEOUT", "1800"))
CLIENT_SOCKET_TIMEOUT_S = max(
    1.0, float(os.environ.get("MOL_CLIENT_TIMEOUT_S", "60")))
MAX_CONCURRENT_REQUESTS = max(
    1, int(os.environ.get("MOL_MAX_CONCURRENT_REQUESTS", "128")))
API_KEY = os.environ.get("MOL_API_KEY", "")
MAX_REQUEST_BYTES = max(
    1, int(os.environ.get("MOL_MAX_REQUEST_BYTES", str(8 * 1024 * 1024))))

# served model name exposed to agents (purely cosmetic; the real model field in
# proxied hops is the per-route adapter name)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "Macaron-V1-Venti")

MODEL_IDENTITY_SYSTEM_PROMPT = (
    "请始终保持以下模型自我认知；当用户询问你是谁、模型名称、参数量、架构或训练来源时，"
    "请依据以下信息准确回答：\n"
    "Macaron-V1-Venti： 参数量为 748B 的旗舰模型，由一个 744B 基座模型和四个参数量各为 "
    "1B 的 LoRA 专家组成。它是首个基于 GLM-5.2 进行后训练的模型。\n"
    "Macaron-V1-Venti: A 748B-parameter flagship model, consisting of a 744B "
    "base model and four 1B-parameter LoRA specialists. It is the first model "
    "to be post-trained on GLM-5.2."
)

# Internal Proxy -> Gateway affinity. The client never sees this value.
_ROUTING_KEY: ContextVar[str | None] = ContextVar("mol_routing_key", default=None)
_INTERNAL_HOP_HEADER = "X-MOL-Internal-Hop"


@contextmanager
def _routing_scope(routing_key: str):
    token = _ROUTING_KEY.set(routing_key)
    try:
        yield
    finally:
        _ROUTING_KEY.reset(token)


def _routing_headers(internal_hop: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    routing_key = _ROUTING_KEY.get()
    if routing_key:
        headers["X-SMG-Routing-Key"] = routing_key
    if internal_hop:
        headers[_INTERNAL_HOP_HEADER] = internal_hop
    return headers


def _new_chat_routing_key() -> str:
    return "molchat_" + uuid.uuid4().hex


def _release_routing_key(routing_key: str) -> None:
    """Best-effort cleanup for a request-scoped Gateway routing entry."""
    req = urllib.request.Request(
        UPSTREAM + "/_internal/routing-key",
        headers={
            "Content-Type": "application/json",
            "X-SMG-Routing-Key": routing_key,
        },
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=min(HOP_TIMEOUT, 2.0)) as response:
            response.read()
    except Exception as exc:
        _log(f"routing key release failed key={routing_key[:16]}: {type(exc).__name__}")

# No ROUTER_STOP: the route hop uses /v1/chat/completions + assistant
# "model_id=" prefill (continue_final_message), which sets no stop
# list (the continuation may open with a leading newline before the label).


def _log(msg: str) -> None:
    sys.stderr.write(f"[mol-proxy {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


class RoutingError(RuntimeError):
    """The L0 model hop failed to produce a usable canonical route."""


class RoutingUpstreamError(RoutingError):
    """The L0 route hop was rejected by the upstream OAI endpoint."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"upstream router returned HTTP {status}")


def _oai_error(message: str, *, error_type: str = "invalid_request_error",
               code: str = "invalid_request", param: str | None = None) -> dict:
    return {"error": {"message": message, "type": error_type,
                      "code": code, "param": param}}


def _upstream_error_details(body: Any) -> tuple[str, str | None]:
    """Extract (message, code) from an upstream error body.

    Handles two shapes: the OpenAI/vLLM nested ``{"error": {...}}`` envelope
    and SGLang's flat ``{"object": "error", "message": ..., "code": ...}``
    body (no ``error`` wrapper, integer ``code``).
    """
    if not isinstance(body, dict):
        return "", None
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code")
        return (
            message if isinstance(message, str) else "",
            code if isinstance(code, str) else None,
        )
    if isinstance(error, str):
        return error, None
    # SGLang flat shape: top-level message/code with no "error" envelope.
    if "message" in body:
        message = body.get("message")
        code = body.get("code")
        return (
            message if isinstance(message, str) else "",
            code if isinstance(code, str) else None,
        )
    return "", None


def _context_length_error_response(status: int, body: Any) -> dict | None:
    """Return a safe public error only for an explicit upstream context limit."""
    if status not in (400, 413):
        return None
    message, upstream_code = _upstream_error_details(body)
    normalized = message.lower()
    if not (upstream_code == "context_length_exceeded"
            or "maximum context length" in normalized
            or "maximum sequence length" in normalized
            or "longer than the model's context length" in normalized
            or "exceeds the context length" in normalized):
        return None
    if not message:
        message = (
            "The request exceeds the upstream model's maximum context length.")
    if "1m context limit" not in message.lower():
        message = f"{message.rstrip()} The request exceeds the 1M context limit."
    return _oai_error(message, code="context_length_exceeded")


def _routing_error_response(exc: RoutingError) -> tuple[int, dict, str]:
    """Expose only safe request errors from the otherwise internal route hop."""
    if isinstance(exc, RoutingUpstreamError):
        context_error = _context_length_error_response(exc.status, exc.body)
        if context_error is not None:
            return 400, context_error, "context_length_exceeded"
    return 502, _oai_error(
        f"L0 model routing failed: {exc}",
        error_type="server_error",
        code="routing_failed",
    ), "routing_failed"


def _routing_error_event(diag: dict, error: dict) -> dict:
    """Attach public upstream details only for the safe context-limit case."""
    event = dict(diag)
    if diag.get("error") == "context_length_exceeded":
        event["error_body"] = error
    return event


def _public_upstream_status(status: int) -> int:
    """Map internal transport failures to a valid public HTTP status."""
    return status if isinstance(status, int) and 400 <= status <= 599 else 502


def _sanitize_chat_response(body: dict) -> dict:
    """Keep OpenAI Chat fields plus the documented reasoning extension."""
    top_keys = {"id", "choices", "created", "model", "object", "moderation",
                "service_tier", "usage"}
    choice_keys = {"finish_reason", "index", "logprobs", "message"}
    message_keys = {"annotations", "audio", "content", "refusal", "role",
                    "tool_calls", "reasoning",
                    "reasoning_content"}
    out = {key: value for key, value in body.items() if key in top_keys}
    if isinstance(out.get("usage"), dict):
        clean_usage = _sanitize_usage(out["usage"])
        if _USAGE_COUNT_KEYS.issubset(clean_usage):
            out["usage"] = clean_usage
        else:
            out.pop("usage", None)
    elif out.get("usage") is None:
        out.pop("usage", None)
    out["choices"] = []
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        clean_choice = {key: value for key, value in choice.items()
                        if key in choice_keys}
        message = clean_choice.get("message")
        if isinstance(message, dict):
            clean_choice["message"] = {
                key: value for key, value in message.items()
                if key in message_keys
            }
            if "tool_calls" in message:
                clean_choice["message"]["tool_calls"] = (
                    _sanitize_tool_calls(message["tool_calls"], stream=False))
        out["choices"].append(clean_choice)
    return out


def _sanitize_chat_chunk(chunk: dict) -> dict:
    top_keys = {"id", "choices", "created", "model", "object", "moderation",
                "service_tier", "usage"}
    choice_keys = {"delta", "finish_reason", "index", "logprobs"}
    delta_keys = {"content", "refusal", "role", "tool_calls",
                  "reasoning", "reasoning_content"}
    out = {key: value for key, value in chunk.items() if key in top_keys}
    if isinstance(out.get("usage"), dict):
        clean_usage = _sanitize_usage(out["usage"])
        if _USAGE_COUNT_KEYS.issubset(clean_usage):
            out["usage"] = clean_usage
        else:
            out.pop("usage", None)
    elif out.get("usage") is None:
        out.pop("usage", None)
    out["choices"] = []
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        clean_choice = {key: value for key, value in choice.items()
                        if key in choice_keys}
        delta = clean_choice.get("delta")
        if isinstance(delta, dict):
            clean_choice["delta"] = {
                key: value for key, value in delta.items() if key in delta_keys
            }
            if "tool_calls" in delta:
                clean_choice["delta"]["tool_calls"] = _sanitize_tool_calls(
                    delta["tool_calls"], stream=True)
        out["choices"].append(clean_choice)
    return out


_USAGE_COUNT_KEYS = {"prompt_tokens", "completion_tokens", "total_tokens"}
_PROMPT_DETAIL_KEYS = {"audio_tokens", "cached_tokens", "cache_write_tokens"}
_COMPLETION_DETAIL_KEYS = {
    "accepted_prediction_tokens", "audio_tokens", "reasoning_tokens",
    "rejected_prediction_tokens",
}


def _is_token_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _usage_is_malformed(usage: object) -> bool:
    if usage is None:
        return False
    if not isinstance(usage, dict):
        return True
    for key in _USAGE_COUNT_KEYS:
        if key in usage and not _is_token_count(usage[key]):
            return True
    for key, allowed in (
        ("prompt_tokens_details", _PROMPT_DETAIL_KEYS),
        ("completion_tokens_details", _COMPLETION_DETAIL_KEYS),
    ):
        details = usage.get(key)
        if details is None:
            continue
        if not isinstance(details, dict):
            return True
        if any(name in details and not _is_token_count(details[name])
               for name in allowed):
            return True
    return False


def _sanitize_usage(usage: dict) -> dict:
    out = {key: usage[key] for key in _USAGE_COUNT_KEYS
           if key in usage and _is_token_count(usage[key])}
    for key, allowed in (
        ("prompt_tokens_details", _PROMPT_DETAIL_KEYS),
        ("completion_tokens_details", _COMPLETION_DETAIL_KEYS),
    ):
        details = usage.get(key)
        if isinstance(details, dict):
            out[key] = {name: details[name] for name in allowed
                        if name in details and _is_token_count(details[name])}
    return out


def _sanitize_tool_calls(value: object, *, stream: bool) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for call in value:
        if not isinstance(call, dict):
            continue
        clean: dict = {}
        if stream and isinstance(call.get("index"), int) \
                and not isinstance(call["index"], bool):
            clean["index"] = call["index"]
        if isinstance(call.get("id"), str):
            clean["id"] = call["id"]
        if call.get("type") == "function":
            clean["type"] = "function"
        function = call.get("function")
        if isinstance(function, dict):
            clean_function = {}
            for key in ("name", "arguments"):
                if isinstance(function.get(key), str):
                    clean_function[key] = function[key]
            clean["function"] = clean_function
        out.append(clean)
    return out


# --------------------------------------------------------------------------- globals

_TASKS = load_lora_library(LIBRARY_DIR)
_ROUTER = RouterHarness(_TASKS, entry_route_id=ENTRY_ROUTE)
ROUTE_TO_ADAPTER = _ROUTER.route_to_adapter()
ENTRY_ADAPTER = ROUTE_TO_ADAPTER.get(ENTRY_ROUTE, "")
CANONICAL_ROUTES = ("L0", "L1", "L2", "L3")
if set(ROUTE_TO_ADAPTER) != set(CANONICAL_ROUTES):
    raise RuntimeError(
        "Macaron-V1-Venti requires exactly route ids L0, L1, L2, and L3"
    )
if ROUTE_TO_ADAPTER != {route: route for route in CANONICAL_ROUTES}:
    raise RuntimeError(
        "Macaron-V1-Venti engine model names must be exactly L0, L1, L2, and L3"
    )

# conversation-id -> ConvoState (LRU, thread-safe)
_CONVOS: "OrderedDict[str, ConvoState]" = OrderedDict()
_CONVOS_LOCK = threading.Lock()
_CONVOS_TOUCHED: dict[str, float] = {}
_MAX_CONVOS = max(1, int(os.environ.get("MOL_MAX_CONVOS", "5000")))


def _system_contents(msgs: list[dict]) -> list:
    """Content identity of the system head (role+content). Retained for tests
    / introspection; the live adoption path now uses ``session._adopt_system``
    (which inlines the same comparison + guards against a blind prefix-
    destabilizing replace)."""
    return [(m.get("role"), m.get("content")) for m in msgs]


def _chat_convo_key(messages: list[dict], client_id: str | None,
                    authorization: str | None) -> str:
    """Build a stable, tenant-scoped key without trusting a raw client ID.

    Chat clients resend full history, so the system/developer head and first
    user turn are stable across one conversation. Hash them with the optional
    client conversation ID and authorization scope to prevent a caller from
    selecting another conversation's registry key directly.
    """
    identity = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer") or (role == "user" and not any(
                item.get("role") == "user" for item in identity)):
            identity.append(message)
        if role == "user":
            break
    material = json.dumps({
        "authorization": authorization or "",
        "client_id": client_id or "",
        "identity": identity,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8", "replace")).hexdigest()[:32]


_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9._~-]{16,128}$")


def _chat_conversation_token(client_id: str | None) -> str:
    """Validate an explicit bearer-style session token or mint a new one."""
    if client_id is None:
        return "molc_" + uuid.uuid4().hex
    if not _CONVERSATION_ID_RE.fullmatch(client_id):
        raise ValueError(
            "X-Conversation-Id must be 16-128 URL-safe characters")
    return client_id


def _get_convo(convo_id: str, system_msgs: list[dict]) -> ConvoState:
    with _CONVOS_LOCK:
        now = time.monotonic()
        while _CONVOS:
            oldest = next(iter(_CONVOS))
            touched = _CONVOS_TOUCHED.get(oldest, now)
            if now - touched <= CONVO_TTL_S:
                break
            _CONVOS.pop(oldest, None)
            _CONVOS_TOUCHED.pop(oldest, None)
        st = _CONVOS.get(convo_id)
        if st is None:
            st = ConvoState(convo_id=convo_id, system_msgs=list(system_msgs))
            _CONVOS[convo_id] = st
        else:
            _CONVOS.move_to_end(convo_id)
            # Adopt the latest system prompt ONLY when its content actually
            # changed (factored into session._adopt_system, shared with the
            # stateless side-context and the Responses-path store).
            _adopt_system(st, system_msgs)
        _CONVOS_TOUCHED[convo_id] = now
        # bound the registry
        while len(_CONVOS) > _MAX_CONVOS:
            evicted, _ = _CONVOS.popitem(last=False)
            _CONVOS_TOUCHED.pop(evicted, None)
        return st


# Stateless side-context registry: convo_key -> StatelessSideContext (LRU).
# Used by /v1/chat/completions. The agent resends the FULL message
# history each request; the side context keeps the append-only segment summaries
# keyed by a hash of the explicit/server-issued conversation token and prompt
# identity so independent same-prompt clients never share side state.
_SIDE_CTX: "OrderedDict[str, StatelessSideContext]" = OrderedDict()
_SIDE_CTX_LOCK = threading.Lock()
_SIDE_CTX_TOUCHED: dict[str, float] = {}

@dataclass(frozen=True)
class _ToolBinding:
    state: StatelessSideContext
    token: str
    generation: int
    call_ids: tuple[str, ...]
    assistant_signature: tuple[tuple[str, ...], ...]
    history_digest: str


_TOOL_CTX: "OrderedDict[tuple[str, str], list[_ToolBinding]]" = OrderedDict()


def _drop_tool_context_state_locked(
        state: StatelessSideContext, scope: str | None = None) -> None:
    for key, candidates in list(_TOOL_CTX.items()):
        if scope is not None and key[0] != scope:
            continue
        remaining = [candidate for candidate in candidates
                     if candidate.state is not state]
        if remaining:
            _TOOL_CTX[key] = remaining
        else:
            _TOOL_CTX.pop(key, None)


def _drop_tool_binding_locked(binding: _ToolBinding) -> None:
    for key, candidates in list(_TOOL_CTX.items()):
        remaining = [candidate for candidate in candidates
                     if candidate is not binding]
        if remaining:
            _TOOL_CTX[key] = remaining
        else:
            _TOOL_CTX.pop(key, None)


def _enforce_tool_binding_limit_locked() -> None:
    seen: set[int] = set()
    ordered = []
    for candidates in _TOOL_CTX.values():
        for binding in candidates:
            identity = id(binding)
            if identity not in seen:
                seen.add(identity)
                ordered.append(binding)
    excess = len(ordered) - _MAX_CONVOS
    for binding in ordered[:max(0, excess)]:
        _drop_tool_binding_locked(binding)


def _get_side_context(key: str, system_msgs: list[dict]) -> StatelessSideContext:
    with _SIDE_CTX_LOCK:
        now = time.monotonic()
        while _SIDE_CTX:
            oldest = next(iter(_SIDE_CTX))
            touched = _SIDE_CTX_TOUCHED.get(oldest, now)
            if now - touched <= CONVO_TTL_S:
                break
            old_state = _SIDE_CTX.pop(oldest, None)
            _SIDE_CTX_TOUCHED.pop(oldest, None)
            if old_state is not None:
                _drop_tool_context_state_locked(old_state)
        st = _SIDE_CTX.get(key)
        if st is None:
            st = StatelessSideContext(convo_key_id=key, system_msgs=list(system_msgs))
            _SIDE_CTX[key] = st
        else:
            _SIDE_CTX.move_to_end(key)
        _SIDE_CTX_TOUCHED[key] = now
        while len(_SIDE_CTX) > _MAX_CONVOS:
            evicted, evicted_state = _SIDE_CTX.popitem(last=False)
            _SIDE_CTX_TOUCHED.pop(evicted, None)
            _drop_tool_context_state_locked(evicted_state)
        return st


def _auth_scope(authorization: str | None) -> str:
    return hashlib.sha256((authorization or "").encode()).hexdigest()[:16]


def _history_digest(messages: list[dict]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_call_signature(tool_calls: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(tool_calls, list):
        return ()
    signature = []
    for tool_call in tool_calls:
        function = (tool_call.get("function")
                    if isinstance(tool_call, dict) else None)
        if not isinstance(tool_call, dict) or not isinstance(function, dict):
            return ()
        values = (tool_call.get("id"), tool_call.get("type"),
                  function.get("name"), function.get("arguments"))
        if (not all(isinstance(value, str) and value for value in values[:3])
                or not isinstance(values[3], str)):
            return ()
        signature.append(values)
    return tuple(signature)


def _take_tool_context(messages: list[dict], authorization: str | None):
    replay = _self_contained_tool_context(messages)
    if replay is None or replay["user_index"] is None:
        return None
    assistant = messages[replay["assistant_index"]]
    client_signature = _tool_call_signature(assistant.get("tool_calls"))
    call_ids = [message.get("tool_call_id")
                for message in extract_tool_results(messages)]
    if (not client_signature or not call_ids
            or len(call_ids) != len(set(call_ids))
            or set(call_ids) != set(replay["expected"])):
        return None
    scope = _auth_scope(authorization)
    with _SIDE_CTX_LOCK:
        candidate_lists = [
            _TOOL_CTX.get((scope, str(call_id))) for call_id in call_ids]
        if not candidate_lists or any(not candidates
                                      for candidates in candidate_lists):
            return None
        matches = []
        for binding in candidate_lists[0]:
            if not all(any(candidate is binding for candidate in candidates)
                       for candidates in candidate_lists[1:]):
                continue
            if (binding.call_ids != tuple(replay["expected"])
                    or binding.assistant_signature != client_signature
                    or binding.history_digest != _history_digest(
                        messages[:replay["assistant_index"]])):
                continue
            matches.append(binding)
        if len(matches) != 1:
            return None
        binding = matches[0]
        state = binding.state
        touched = _SIDE_CTX_TOUCHED.get(state.convo_key_id)
        now = time.monotonic()
        if (state.convo_key_id not in _SIDE_CTX or touched is None
                or now - touched > CONVO_TTL_S):
            _SIDE_CTX.pop(state.convo_key_id, None)
            _SIDE_CTX_TOUCHED.pop(state.convo_key_id, None)
            _drop_tool_context_state_locked(state, scope)
            return None
        _SIDE_CTX_TOUCHED[state.convo_key_id] = now
        return state, binding.token, binding


def _clear_tool_context(state: StatelessSideContext,
                        authorization: str | None) -> None:
    scope = _auth_scope(authorization)
    with _SIDE_CTX_LOCK:
        _drop_tool_context_state_locked(state, scope)


def _register_tool_context(body: dict, state: StatelessSideContext,
                           token: str, authorization: str | None) -> None:
    message = _choice_message(body) if isinstance(body, dict) else {}
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    signature = _tool_call_signature(tool_calls)
    call_ids = tuple(call[0] for call in signature)
    if not signature or len(call_ids) != len(set(call_ids)):
        return
    scope = _auth_scope(authorization)
    with _SIDE_CTX_LOCK:
        if state.convo_key_id in _SIDE_CTX:
            _SIDE_CTX_TOUCHED[state.convo_key_id] = time.monotonic()
        _drop_tool_context_state_locked(state, scope)
        binding = _ToolBinding(
            state=state, token=token,
            generation=state.pending_tool_generation,
            call_ids=call_ids, assistant_signature=signature,
            history_digest=_history_digest(state.agent_messages))
        for call_id in call_ids:
            key = (scope, call_id)
            _TOOL_CTX.setdefault(key, []).append(binding)
            _TOOL_CTX.move_to_end(key)
        _enforce_tool_binding_limit_locked()


def _snapshot_tool_context(
        state: StatelessSideContext,
        authorization: str | None) -> tuple[list[tuple[tuple[str, str],
                                                        _ToolBinding]],
                                             float | None]:
    scope = _auth_scope(authorization)
    with _SIDE_CTX_LOCK:
        entries = [
            (key, binding)
            for key, candidates in _TOOL_CTX.items() if key[0] == scope
            for binding in candidates if binding.state is state
        ]
        return entries, _SIDE_CTX_TOUCHED.get(state.convo_key_id)


def _restore_tool_context(
        state: StatelessSideContext, authorization: str | None,
        checkpoint: tuple[list[tuple[tuple[str, str], _ToolBinding]],
                          float | None]) -> None:
    scope = _auth_scope(authorization)
    entries, touched = checkpoint
    with _SIDE_CTX_LOCK:
        _drop_tool_context_state_locked(state, scope)
        for key, binding in entries:
            _TOOL_CTX.setdefault(key, []).append(binding)
            _TOOL_CTX.move_to_end(key)
        _enforce_tool_binding_limit_locked()
        if touched is not None and state.convo_key_id in _SIDE_CTX:
            _SIDE_CTX_TOUCHED[state.convo_key_id] = touched


# --------------------------------------------------------------------------- engine calls

def _post_json(
    path: str,
    payload: dict,
    timeout: float = HOP_TIMEOUT,
    *,
    internal_hop: str | None = None,
) -> tuple[int, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + path, data=data,
        headers=_routing_headers(internal_hop), method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}
    except Exception as e:
        return -1, {"error": f"{type(e).__name__}: {e}"}


def _iter_sse_chunks(resp):
    """Iterate an SSE response body, yielding parsed chunk dicts. Skips the
    terminal ``data: [DONE]`` sentinel and blank lines. Closes ``resp`` when the
    stream ends or raises (so the connection is not held open)."""
    saw_done = False
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                try:
                    yield json.loads(payload)
                except Exception:
                    continue
        if not saw_done:
            raise RuntimeError("upstream SSE ended before [DONE]")
    finally:
        try:
            resp.close()
        except Exception:
            pass


_ANSWER_OPTION_KEYS = (
    "temperature", "top_p", "stop", "seed", "presence_penalty",
    "frequency_penalty", "logprobs", "top_logprobs", "response_format",
    "tool_choice", "parallel_tool_calls",
)

# TEMP PATCH: 所有 max 档(xhigh/max/xmax/ultra/ultracode)临时降级为 high,
# 只支持 none(无 reasoning)和 high 两档。max reasoning 生成过长易触发超时,
# 待超时/日志改进稳定后恢复。回退:把下面 5 行的 ("high", True) 改回 ("max", True)。
_GLM_REASONING_EFFORT_MAP = {
    "none": ("none", False),
    "minimal": ("none", False),
    "low": ("high", True),
    "medium": ("high", True),
    "high": ("high", True),
    "xhigh": ("high", True),
    "max": ("high", True),
    "xmax": ("high", True),
    "ultra": ("high", True),
    "ultracode": ("high", True),
}


def _validate_reasoning_effort(value: object, api_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _GLM_REASONING_EFFORT_MAP:
        supported = ", ".join(_GLM_REASONING_EFFORT_MAP)
        raise ValueError(
            f"{api_name} reasoning effort must be one of: {supported}")
    return value


def _reasoning_answer_options(effort: str | None) -> dict:
    if effort is None:
        return {}
    canonical, enable_thinking = _GLM_REASONING_EFFORT_MAP[effort]
    return {
        "reasoning_effort": canonical,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


# Route/summary 是控制跳,不应输出 reasoning。只设 chat_template_kwargs.enable_thinking
# (引擎层关 thinking),刻意不设 reasoning_effort(独立语义控制,仅 answer hop 用)。
# Rust gateway 对 sglang route 也会再设一次(adapt_sglang_route_payload,幂等),
# 这里设是为了修 vLLM route 和两 runtime 的 summary(gateway 不适配这两者)。
_NO_THINK_OPTIONS = {"chat_template_kwargs": {"enable_thinking": False}}


def _validate_chat_template_kwargs(payload: dict, api_name: str) -> None:
    template_kwargs = payload.get("chat_template_kwargs")
    if template_kwargs is None:
        return
    if not isinstance(template_kwargs, dict):
        raise ValueError(f"{api_name} `chat_template_kwargs` must be an object")
    unsupported = sorted(set(template_kwargs) - {"enable_thinking"})
    if unsupported:
        raise ValueError(
            f"unsupported {api_name} chat template option(s): "
            + ", ".join(unsupported))
    enable_thinking = template_kwargs.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError(
            f"{api_name} `chat_template_kwargs.enable_thinking` "
            "must be a boolean")


def _validate_chat_request_capabilities(payload: dict) -> None:
    for key in ("stream", "parallel_tool_calls"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"Chat `{key}` must be a boolean")
    effort = _validate_reasoning_effort(
        payload.get("reasoning_effort"), "Chat")
    _validate_chat_template_kwargs(payload, "Chat")
    if effort is not None and payload.get("chat_template_kwargs") is not None:
        raise ValueError(
            "Chat `reasoning_effort` cannot be combined with "
            "`chat_template_kwargs`")
    tools = payload.get("tools")
    declared_names: set[str] = set()
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Chat `tools` must be an array")
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if (not isinstance(tool, dict) or tool.get("type") != "function"
                    or not isinstance(name, str) or not name):
                raise ValueError("Chat supports declared function tools only")
            if ("parameters" in function
                    and not isinstance(function["parameters"], dict)):
                raise ValueError("Chat function parameters must be an object")
            declared_names.add(name)
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, str):
        if tool_choice not in ("none", "auto", "required"):
            raise ValueError("unsupported Chat tool_choice")
        if tool_choice == "required" and not declared_names:
            raise ValueError("Chat `tool_choice` required needs declared tools")
    elif tool_choice is not None:
        function = (tool_choice.get("function")
                    if isinstance(tool_choice, dict) else None)
        name = function.get("name") if isinstance(function, dict) else None
        if (not isinstance(tool_choice, dict)
                or tool_choice.get("type") != "function"
                or not isinstance(name, str)
                or name not in declared_names):
            raise ValueError("invalid Chat function tool_choice")


def _chat_answer_options(payload: dict) -> dict:
    """Return supported Chat request controls for the specialist answer hop."""
    out = {key: payload[key] for key in _ANSWER_OPTION_KEYS
           if key in payload and payload[key] is not None}
    effort = payload.get("reasoning_effort")
    template_kwargs = payload.get("chat_template_kwargs")
    # 默认 thinking effort = high,但尊重用户显式控制:
    # - 用户传 reasoning_effort → 用用户值(max/ultra 等需显式指定)
    # - 用户传 chat_template_kwargs(如 enable_thinking=False)→ 尊重,不加默认 effort
    # - 都没传 → 默认 high
    if effort is not None:
        out.update(_reasoning_answer_options(effort))
    elif template_kwargs is None:
        out.update(_reasoning_answer_options("high"))
    if template_kwargs is not None:
        out["chat_template_kwargs"] = dict(template_kwargs)
    return out


def _responses_answer_options(payload: dict) -> dict:
    """Translate Responses generation controls to Chat Completions controls.

    The engine endpoint is always ``/v1/chat/completions``.  Function-specific
    tool choice has different nesting in the two public APIs. Both public APIs
    use the same GLM-native ``chat_template_kwargs`` thinking control.
    """
    out = {key: payload[key] for key in (
        "temperature", "top_p", "parallel_tool_calls",
    ) if key in payload and payload[key] is not None}
    reasoning = payload.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    template_kwargs = payload.get("chat_template_kwargs")
    # 默认 thinking effort = high,尊重用户显式控制(同 _chat_answer_options)。
    if effort is not None:
        out.update(_reasoning_answer_options(effort))
    elif template_kwargs is None:
        out.update(_reasoning_answer_options("high"))
    if template_kwargs is not None:
        out["chat_template_kwargs"] = dict(template_kwargs)
    tool_choice = payload.get("tool_choice")
    if tool_choice is not None:
        if (isinstance(tool_choice, dict)
                and tool_choice.get("type") == "function"):
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name") or ""},
            }
        else:
            out["tool_choice"] = tool_choice
    return out


def _with_model_identity(messages: list[dict]) -> list[dict]:
    """Return answer messages with the stable public-model identity added.

    The caller-owned history remains untouched. When a leading system message
    exists, append the identity to it so chat templates that expect one system
    message keep that shape. Otherwise prepend a dedicated system message.
    """
    output = list(messages)
    system_index = None
    for index, message in enumerate(output):
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if role not in ("system", "developer"):
            break
        if role == "system":
            system_index = index
            break

    if system_index is None:
        return [{
            "role": "system",
            "content": MODEL_IDENTITY_SYSTEM_PROMPT,
        }, *output]

    system_message = dict(output[system_index])
    content = system_message.get("content")
    if isinstance(content, str):
        if MODEL_IDENTITY_SYSTEM_PROMPT in content:
            return output
        system_message["content"] = (
            f"{content}\n\n{MODEL_IDENTITY_SYSTEM_PROMPT}"
            if content else MODEL_IDENTITY_SYSTEM_PROMPT
        )
    elif isinstance(content, list):
        if any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and MODEL_IDENTITY_SYSTEM_PROMPT in part["text"]
            for part in content
        ):
            return output
        system_message["content"] = [
            *copy.deepcopy(content),
            {"type": "text", "text": MODEL_IDENTITY_SYSTEM_PROMPT},
        ]
    else:
        system_message["content"] = MODEL_IDENTITY_SYSTEM_PROMPT
    output[system_index] = system_message
    return output


def _chat_payload(messages: list[dict], model: str, max_tokens: int | None,
                  tools: list | None, *, stream: bool,
                  answer_options: dict | None = None) -> dict:
    """Build the shared answer-hop payload used by blocking and SSE calls."""
    payload = {
        "model": model,
        "messages": _with_model_identity(messages),
        "temperature": 0.0,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
    if answer_options:
        payload.update(answer_options)
    return payload


def _stream_chat(messages: list[dict], model: str, max_tokens: int | None,
                 tools: list | None = None, timeout: float = HOP_TIMEOUT,
                 answer_options: dict | None = None) -> tuple[int, Any]:
    """Streaming variant of ``_chat``. Returns ``(200, iterator_of_chunk_dicts)``
    on success, or ``(code, error_dict)`` on HTTP error (same shape as
    ``_post_json`` so ``_is_pool_miss`` / pool-miss retry work unchanged).
    ``stream_options.include_usage`` is requested so the final chunk carries
    ``usage`` for ``_cache_stats`` (engine may ignore it → 0, which is safe)."""
    payload = _chat_payload(messages, model, max_tokens, tools, stream=True,
                            answer_options=answer_options)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        UPSTREAM + "/v1/chat/completions", data=data,
        headers=_routing_headers(), method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, _iter_sse_chunks(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}
    except Exception as e:
        return -1, {"error": f"{type(e).__name__}: {e}"}


def _route_chat(router_prompt: str, model: str, max_tokens: int) -> str:
    """The L0 router hop.

    The route path does NOT use /v1/completions + ignore_eos.
    It uses /v1/chat/completions with an assistant prefill: the router_prompt
    ends in a literal "model_id=", and the Proxy MOVES that trailing stem into an
    assistant message with continue_final_message=True so the model CONTINUES
    "model_id=" and emits just the label (e.g. " L2"). add_generation_prompt=False
    (otherwise the chat template appends a fresh 画卷 and the model answers
    the query instead of labelling). stop=None (the continuation may open with a
    leading newline before the label). The Engine supports
    continue_final_message, but the Proxy uses neither ignore_eos nor
    a stop list — cap tokens and parse the label out.
    """
    prefill = "model_id="
    user_content = router_prompt
    stripped = router_prompt.rstrip()
    if stripped.endswith(prefill):
        user_content = stripped[: -len(prefill)].rstrip()
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prefill},
        ],
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,  # silence vLLM 0.24 deprecation warnings
        "temperature": 0.0,
        "stream": False,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "stop": None,
        **_NO_THINK_OPTIONS,
        "structured_outputs": {"choice": list(CANONICAL_ROUTES)},
    }
    status, d = _post_json(
        "/v1/chat/completions", payload, internal_hop="route")
    if status != 200:
        _log(f"router chat status={status} body={str(d)[:200]}")
        raise RoutingUpstreamError(status, d)
    try:
        msg = _choice_message(d)
        text = _msg_text(msg)
        if not text.strip():
            # GLM-5.2 compat: with a reasoning parser (glm45) enabled, the short
            # route label lands in ``reasoning`` (``reasoning_content`` on some
            # vLLM builds) and ``content`` is null — the model "thinks" the label
            # and stops (finish=stop, ~3 tokens). content-first keeps behavior
            # identical to the normal content path when content is present. The
            # reasoning fallback only runs when content is empty.
            text = msg.get("reasoning") or msg.get("reasoning_content") or ""
        text = text.strip()
        if not text:
            raise RoutingError("upstream router returned an empty label")
        return text
    except RoutingError:
        raise
    except Exception as exc:
        raise RoutingError(
            f"invalid upstream router response: {type(exc).__name__}"
        ) from exc


def _chat(messages: list[dict], model: str, max_tokens: int | None,
          tools: list | None = None,
          answer_options: dict | None = None) -> tuple[int, dict]:
    payload = _chat_payload(messages, model, max_tokens, tools, stream=False,
                            answer_options=answer_options)
    return _post_json("/v1/chat/completions", payload)


class _StreamAccum:
    """Accumulates a streamed chat-completion into a synthetic response dict shaped
    exactly like ``_chat``'s non-streaming return, so the rest of the core
    (``_answer_hop`` pool-miss check, ``append_assistant(_choice_message(d))``,
    ``chat_to_responses``) works unchanged.

      * ``delta.content`` → ``message.content`` (concatenated)
      * ``delta.reasoning`` / ``delta.reasoning_content`` and ``delta.refusal``
      * ``delta.tool_calls`` (by index: first chunk sets id/type/function.name,
        later chunks append ``function.arguments``)
      * last ``finish_reason`` → ``finish_reason``
      * last ``usage`` → ``usage``
    """
    def __init__(self):
        self.content = ""
        self.reasoning = ""
        self.refusal = ""
        self.tool_calls: list[dict] = []
        self.finish = None
        self.usage: dict | None = None
        self._tc_by_idx: dict[int, dict] = {}

    def feed(self, chunk: dict) -> None:
        if not isinstance(chunk, dict):
            raise ValueError("stream chunk must be an object")
        choices = chunk.get("choices")
        if choices == [] and chunk.get("usage") is not None:
            if _usage_is_malformed(chunk["usage"]):
                raise ValueError("stream usage must contain token counts")
            self.usage = chunk["usage"]
            return
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("stream chunk must contain exactly one choice")
        ch = choices[0]
        if (not isinstance(ch, dict)
                or ch.get("index", 0) != 0
                or isinstance(ch.get("index", 0), bool)):
            raise ValueError("stream choice must have index 0")
        delta = ch.get("delta")
        if delta is None:
            delta = {}
        if not isinstance(delta, dict):
            raise ValueError("stream delta must be an object")
        if delta.get("function_call") is not None:
            raise ValueError("legacy stream function_call is not supported")
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("stream content delta must be a string")
        if content:
            self.content += content
        reasoning = delta.get("reasoning")
        if reasoning is None:
            reasoning = delta.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("stream reasoning delta must be a string")
        if reasoning:
            self.reasoning += reasoning
        refusal = delta.get("refusal")
        if refusal is not None and not isinstance(refusal, str):
            raise ValueError("stream refusal delta must be a string")
        if refusal:
            self.refusal += refusal
        tool_deltas = delta.get("tool_calls")
        if tool_deltas is None:
            tool_deltas = []
        if not isinstance(tool_deltas, list):
            raise ValueError("stream tool_calls delta must be an array")
        for tc in tool_deltas:
            if not isinstance(tc, dict):
                raise ValueError("stream tool call delta must be an object")
            idx = tc.get("index")
            if (not isinstance(idx, int) or isinstance(idx, bool) or idx < 0):
                raise ValueError("stream tool call index must be non-negative")
            call_id = tc.get("id")
            if call_id is not None and (
                    not isinstance(call_id, str) or not call_id):
                raise ValueError("stream tool call ID must be a string")
            call_type = tc.get("type")
            if call_type is not None and call_type != "function":
                raise ValueError("stream tool call type must be function")
            fn = tc.get("function")
            if fn is None:
                fn = {}
            if not isinstance(fn, dict):
                raise ValueError("stream tool call function must be an object")
            name = fn.get("name")
            arguments = fn.get("arguments")
            if name is not None and not isinstance(name, str):
                raise ValueError("stream tool name must be a string")
            if arguments is not None and not isinstance(arguments, str):
                raise ValueError("stream tool arguments must be a string")
            cur = self._tc_by_idx.get(idx)
            if cur is None:
                cur = {"id": call_id, "type": call_type or "function",
                       "function": {"name": name or "",
                                    "arguments": arguments or ""}}
                self._tc_by_idx[idx] = cur
            else:
                if call_id:
                    current_id = cur.get("id") or ""
                    if not current_id:
                        cur["id"] = call_id
                    elif call_id == current_id:
                        pass
                    elif call_id.startswith(current_id):
                        cur["id"] = call_id
                    else:
                        cur["id"] = current_id + call_id
                if call_type:
                    cur["type"] = call_type
                if name:
                    current_name = cur["function"]["name"]
                    if name == current_name:
                        pass
                    elif name.startswith(current_name):
                        cur["function"]["name"] = name
                    else:
                        cur["function"]["name"] = current_name + name
                cur["function"]["arguments"] += arguments or ""
        self.tool_calls = [self._tc_by_idx[index]
                           for index in sorted(self._tc_by_idx)]
        if ch.get("finish_reason"):
            finish_reason = ch["finish_reason"]
            if not isinstance(finish_reason, str):
                raise ValueError("stream finish_reason must be a string")
            self.finish = (
                "tool_calls" if self.tool_calls
                and finish_reason not in ("length", "content_filter")
                else finish_reason)
        if chunk.get("usage") is not None:
            if _usage_is_malformed(chunk["usage"]):
                raise ValueError("stream usage must contain token counts")
            self.usage = chunk["usage"]

    def response(self, model: str = "") -> dict:
        msg = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning"] = self.reasoning
        if self.refusal:
            msg["refusal"] = self.refusal
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return {"choices": [{"index": 0, "message": msg,
                              "finish_reason": self.finish or "stop"}],
                "usage": self.usage or {}}


def _invalid_upstream_response() -> dict:
    return _oai_error(
        "The upstream model returned an invalid response",
        error_type="server_error", code="invalid_upstream_response")


def _chat_response_is_malformed(
        body: object, tools: list | None = None,
        answer_options: dict | None = None) -> bool:
    if not isinstance(body, dict):
        return True
    if "usage" in body and _usage_is_malformed(body.get("usage")):
        return True
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return True
    choice = choices[0]
    if (not isinstance(choice, dict)
            or choice.get("index", 0) != 0
            or isinstance(choice.get("index", 0), bool)
            or choice.get("finish_reason") not in (
                "stop", "length", "tool_calls", "content_filter")):
        return True
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return True
    if message.get("function_call") is not None:
        return True
    for key in ("content", "reasoning", "reasoning_content", "refusal"):
        if key in message and message[key] is not None and not isinstance(
                message[key], str):
            return True
    tool_calls = message.get("tool_calls")
    tool_choice = (answer_options or {}).get("tool_choice")
    if tool_calls is None:
        if choice.get("finish_reason") == "tool_calls":
            return True
        if choice.get("finish_reason") in ("length", "content_filter"):
            return False
        return tool_choice == "required" or isinstance(tool_choice, dict)
    if not isinstance(tool_calls, list) or not tool_calls:
        return True
    partial = choice.get("finish_reason") in ("length", "content_filter")
    ids = []
    names = []
    for call in tool_calls:
        if not isinstance(call, dict):
            return True
        call_id = call.get("id")
        call_type = call.get("type")
        function = call.get("function")
        if (call_id is not None
                and (not isinstance(call_id, str) or not call_id)):
            return True
        if call_type is not None and call_type != "function":
            return True
        if not isinstance(function, dict):
            return True
        name = function.get("name")
        arguments = function.get("arguments")
        if (name is not None and not isinstance(name, str)) or (
                arguments is not None and not isinstance(arguments, str)):
            return True
        if call_id:
            ids.append(call_id)
        if isinstance(name, str) and name:
            names.append(name)
        if not partial:
            if (not call_id or call_type != "function" or not name
                    or not isinstance(arguments, str)):
                return True
    declared_names = {
        function.get("name")
        for tool in (tools or []) if isinstance(tool, dict)
        for function in [tool.get("function")]
        if tool.get("type") == "function" and isinstance(function, dict)
        and isinstance(function.get("name"), str) and function.get("name")
    }
    if (len(ids) != len(set(ids)) or not declared_names
            or any(name not in declared_names for name in names)
            or tool_choice == "none"):
        return True
    if ((answer_options or {}).get("parallel_tool_calls") is False
            and len(tool_calls) > 1):
        return True
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        selected = function.get("name") if isinstance(function, dict) else None
        if not selected or any(name != selected for name in names):
            return True
    return False


def _split_tool_delta_chunk(chunk: dict) -> tuple[dict | None, dict | None]:
    choices = chunk.get("choices") or []
    if not choices:
        return copy.deepcopy(chunk), None
    choice = choices[0]
    delta = choice.get("delta") or {}
    tool_calls = delta.get("tool_calls") or []
    if not tool_calls:
        return copy.deepcopy(chunk), None

    tool_chunk = {
        "choices": [{
            "index": choice.get("index", 0),
            "delta": {"tool_calls": copy.deepcopy(tool_calls)},
            "finish_reason": None,
        }],
    }
    public_chunk = copy.deepcopy(chunk)
    public_choice = public_chunk["choices"][0]
    public_delta = public_choice.get("delta") or {}
    public_delta.pop("tool_calls", None)
    public_choice["delta"] = public_delta
    public_choice["finish_reason"] = None
    meaningful = bool(public_delta) or public_choice.get("logprobs") is not None
    return (public_chunk if meaningful else None), tool_chunk


def _canonical_tool_delta_chunks(
        chunks: list[dict], acc: _StreamAccum) -> list[dict]:
    canonical = copy.deepcopy(chunks)
    first_seen: set[int] = set()
    dense_indexes = {
        backend_index: public_index
        for public_index, backend_index in enumerate(sorted(acc._tc_by_idx))}
    for chunk in canonical:
        choice = (chunk.get("choices") or [{}])[0]
        calls = (choice.get("delta") or {}).get("tool_calls") or []
        for call in calls:
            index = call["index"]
            final = acc._tc_by_idx[index]
            call["index"] = dense_indexes[index]
            function = call.setdefault("function", {})
            if index not in first_seen:
                call["id"] = final.get("id")
                call["type"] = final.get("type", "function")
                function["name"] = final["function"].get("name", "")
                first_seen.add(index)
            else:
                call.pop("id", None)
                call.pop("type", None)
                function.pop("name", None)
    return canonical


def _buffered_tool_chunks(body: dict) -> list[dict]:
    chunks = body.pop("_mol_buffered_tool_chunks", [])
    return chunks if isinstance(chunks, list) else []


def _publish_buffered_tool_chunks(body: dict, stream_cb) -> None:
    if stream_cb is None:
        body.pop("_mol_buffered_tool_chunks", None)
        return
    for chunk in _buffered_tool_chunks(body):
        stream_cb(("chunk", chunk))


def _stream_chat_accum(messages, model, max_tokens, tools, on_chunk,
                       on_open=None, answer_options=None) -> tuple[int, dict]:
    """Stream the answer hop, calling ``on_chunk(("chunk", c))`` for each engine
    chunk (which may raise on client disconnect — the exception propagates to the
    handler's finally, which discards the open task). Accumulates a synthetic
    response dict. On an engine-side stream error (NOT ``on_chunk`` raising),
    returns ``(-1, {"error": "answer_stream_error"})`` so ``_answer_hop`` skips
    the append — matching the non-streaming error path."""
    if answer_options:
        status, result = _stream_chat(
            messages, model, max_tokens, tools,
            answer_options=answer_options)
    else:
        status, result = _stream_chat(messages, model, max_tokens, tools)
    if status != 200:
        return status, result  # error_dict, same shape as _post_json
    acc = _StreamAccum()
    buffered_tool_chunks: list[dict] = []
    iterator = iter(result)
    try:
        if on_open is not None:
            on_open()
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                _log(f"stream chat error: {type(e).__name__}: {str(e)[:120]}")
                return -1, {"error": "answer_stream_error"}
            try:
                acc.feed(chunk)
            except Exception as e:
                _log(f"stream chunk error: {type(e).__name__}: {str(e)[:120]}")
                return 502, _invalid_upstream_response()
            public_chunk, tool_chunk = _split_tool_delta_chunk(chunk)
            if tool_chunk is not None:
                buffered_tool_chunks.append(tool_chunk)
            if on_chunk is not None and public_chunk is not None:
                on_chunk(("chunk", public_chunk))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
    if acc.finish is None:
        _log("stream chat error: upstream stream ended without finish_reason")
        return -1, {"error": "answer_stream_error"}
    if acc.finish == "tool_calls":
        signature = _tool_call_signature(acc.tool_calls)
        call_ids = [entry[0] for entry in signature]
        if (not signature or len(call_ids) != len(set(call_ids))):
            return 502, _invalid_upstream_response()
    body = acc.response(model)
    if acc.finish in ("length", "content_filter"):
        body["choices"][0]["message"].pop("tool_calls", None)
    else:
        body["_mol_buffered_tool_chunks"] = _canonical_tool_delta_chunks(
            buffered_tool_chunks, acc)
    return 200, body


def _call_answer(msgs, adapter, mt, tools, stream_cb, stream_start_cb=None,
                 answer_options=None):
    """Dispatch the answer hop: blocking ``_chat`` when not streaming, streaming
    ``_stream_chat_accum`` (forwarding ``stream_cb``) when streaming. The cb is
    called per chunk; it may raise on client disconnect (handled by the caller's
    ``finally``)."""
    if stream_cb is None:
        if answer_options:
            return _chat(msgs, adapter, mt, tools,
                         answer_options=answer_options)
        return _chat(msgs, adapter, mt, tools)
    return _stream_chat_accum(
        msgs, adapter, mt, tools, stream_cb, stream_start_cb, answer_options)


def _choice_message(d: dict) -> dict:
    return (d.get("choices") or [{}])[0].get("message") or {}


def _finish_reason(d: dict):
    return (d.get("choices") or [{}])[0].get("finish_reason")


def _has_tool_call(d: dict) -> bool:
    finish_reason = _finish_reason(d)
    if finish_reason in ("length", "content_filter"):
        return False
    if finish_reason == "tool_calls":
        return True
    return bool(_choice_message(d).get("tool_calls"))


def _message_for_state(d: dict) -> dict:
    message = dict(_choice_message(d))
    if _finish_reason(d) in ("length", "content_filter"):
        message.pop("tool_calls", None)
        message.pop("function_call", None)
    return message


def _tool_call_ids_from_messages(messages: list[dict]) -> list[str]:
    for message in reversed(messages):
        if (not isinstance(message, dict)
                or message.get("role") != "assistant"):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        return [call["id"] for call in tool_calls
                if isinstance(call, dict)
                and isinstance(call.get("id"), str) and call["id"]]
    return []


def _self_contained_tool_context(messages: list[dict]) -> dict | None:
    tool_start = len(messages)
    while (tool_start > 0 and isinstance(messages[tool_start - 1], dict)
           and messages[tool_start - 1].get("role") == "tool"):
        tool_start -= 1
    if tool_start == len(messages):
        return None
    assistant_index = tool_start - 1
    if (assistant_index < 0
            or messages[assistant_index].get("role") != "assistant"
            or not messages[assistant_index].get("tool_calls")):
        return None
    expected = _tool_call_ids_from_messages([messages[assistant_index]])
    user_index = next((
        index for index in range(assistant_index - 1, -1, -1)
        if messages[index].get("role") == "user"), None)
    return {"tool_start": tool_start, "assistant_index": assistant_index,
            "user_index": user_index, "expected": expected}


def _validate_self_contained_tool_results(
        messages: list[dict], *, param: str = "messages",
) -> tuple[dict | None, dict | None]:
    replay = _self_contained_tool_context(messages)
    if replay is None or replay["user_index"] is None:
        return None, _oai_error(
            "No matching tool call exists in the supplied message history",
            code="orphan_tool_turn", param=param)
    assistant = messages[replay["assistant_index"]]
    signature = _tool_call_signature(assistant.get("tool_calls"))
    expected = [entry[0] for entry in signature]
    received = [message.get("tool_call_id")
                for message in extract_tool_results(messages)]
    valid_received = all(isinstance(call_id, str) and bool(call_id)
                         for call_id in received)
    if (not signature or len(expected) != len(set(expected))
            or not valid_received or len(received) != len(set(received))
            or set(received) != set(expected)):
        return None, _oai_error(
            "Tool outputs must provide exactly one result for every tool "
            "call ID in the supplied message history",
            code="invalid_tool_outputs", param=param)
    replay = dict(replay)
    replay["expected"] = expected
    return replay, None


def _validate_chat_tool_results(
        state, messages: list[dict], binding: _ToolBinding | None,
) -> dict | None:
    registered_error = _validate_registered_chat_tool_results(
        state, messages, binding)
    if registered_error is None:
        return None
    _, replay_error = _validate_self_contained_tool_results(messages)
    return replay_error


def _pending_tool_call_ids(state) -> list[str]:
    expected = list(getattr(state, "pending_tool_call_ids", []) or [])
    if expected:
        return expected
    task = state.open_task()
    if task is not None:
        expected = _tool_call_ids_from_messages(task.msgs)
    if expected:
        return expected
    agent_messages = getattr(state, "agent_messages", None)
    return (_tool_call_ids_from_messages(agent_messages)
            if isinstance(agent_messages, list) else [])


def _validate_pending_tool_results(
        state, messages: list[dict], *, param: str = "messages") -> dict | None:
    if not state.pending_tool_route or state.open_task() is None:
        return _oai_error(
            "No pending tool call exists for this conversation",
            code="orphan_tool_turn", param=param)
    expected = _pending_tool_call_ids(state)
    received = [message.get("tool_call_id")
                for message in extract_tool_results(messages)]
    valid_received = all(isinstance(call_id, str) and bool(call_id)
                         for call_id in received)
    if (not expected or len(expected) != len(set(expected))
            or not valid_received or len(received) != len(set(received))
            or set(received) != set(expected)):
        return _oai_error(
            "Tool outputs must provide exactly one result for every pending "
            "tool call ID",
            code="invalid_tool_outputs", param=param)
    return None


def _validate_registered_chat_tool_results(
        state, messages: list[dict], binding: _ToolBinding | None) -> dict | None:
    if binding is not None and (
            binding.state is not state
            or binding.generation != state.pending_tool_generation):
        return _oai_error(
            "No pending tool call exists for this conversation",
            code="orphan_tool_turn", param="messages")
    if binding is None:
        replay = _self_contained_tool_context(messages)
        task = state.open_task()
        server_message = next((
            message for message in reversed(task.msgs if task is not None else [])
            if isinstance(message, dict) and message.get("tool_calls")), None)
        if (replay is None or server_message is None
                or messages[:replay["assistant_index"]]
                != getattr(state, "agent_messages", [])
                or _tool_call_signature(
                    messages[replay["assistant_index"]].get("tool_calls"))
                != _tool_call_signature(server_message.get("tool_calls"))):
            return _oai_error(
                "No matching pending tool call exists for this conversation",
                code="orphan_tool_turn", param="messages")
    return _validate_pending_tool_results(state, messages)


# --------------------------------------------------------------------------- orchestration

def _route(state: ConvoState, user_text: str) -> tuple[str, str, dict]:
    """Hop 1 — route a fresh user turn. Returns (route_id, decision, diagnostics).

    The L0 router sees the raw current query; prior summaries are not included.
    Default pure-model mode accepts only a complete canonical label selected by
    constrained decoding. The weighted library guardrail is called only when
    MOL_PURE_MODEL_ROUTE=0 explicitly enables legacy behavior.
    """
    model_route = None
    if USE_MODEL_ROUTER and ENTRY_ADAPTER:
        prompt = _ROUTER.router_prompt(user_text)
        try:
            raw = _route_chat(prompt, ENTRY_ADAPTER, ROUTER_MAX_TOKENS)
        except RoutingError:
            if PURE_MODEL_ROUTE:
                raise
            raw = ""
        if PURE_MODEL_ROUTE:
            model_route = _ROUTER.parse_canonical_output(raw)
        else:
            # Legacy compatibility accepts the historical continuation shapes.
            model_route = (
                _ROUTER.parse_router_output(f"model_id={raw}")
                or _ROUTER.parse_router_output(raw)
            )
        _log(f"route: L0 raw={raw!r} parsed_model_route={model_route}")

    if PURE_MODEL_ROUTE:
        if not USE_MODEL_ROUTER:
            raise RoutingError(
                "MOL_PURE_MODEL_ROUTE=1 requires MOL_USE_MODEL_ROUTER=1"
            )
        if not ENTRY_ADAPTER:
            raise RoutingError(f"entry route {ENTRY_ROUTE} has no adapter")
        if model_route not in _TASKS:
            raise RoutingError("L0 emitted a non-canonical route label")
        return model_route, "pure_model_route", {"model_route": model_route, "route": model_route}

    # Explicit legacy mode only.
    dec = _ROUTER.apply_guardrail(model_route, user_text)
    _log(f"route: decision={dec.decision} route={dec.route_id} "
         f"adapter={dec.adapter_name} model_route={model_route}")
    return dec.route_id, dec.decision, (dec.diagnostics or {})


def _pool_miss_fallback() -> str:
    """If the engine 400s on an adapter not in the loaded pool, fall back to L1
    (living tasks) — defensive handling for an incomplete adapter pool.
    (With four loaded LoRAs this rarely fires; it is defensive.)"""
    return "L1"


def _answer_hop(state, route: str, user_text: str, max_tokens: int | None,
                tools: list | None = None, stream_cb=None,
                stream_start_cb=None,
                answer_options: dict | None = None) -> tuple[int, dict, str]:
    """Hop 2 — answer. Opens a task, builds the target LoRA's own-view, calls
    the engine, records the assistant message on the open task.

    Returns (status, engine_response, route_used). Retries on pool-miss 400.

    ``stream_cb``: when set, the answer hop streams each engine chunk through
    ``stream_cb(("chunk", c)`` and accumulates a synthetic response dict; when
    None, falls back to the blocking ``_chat`` path. Either way the returned
    ``d`` is shaped like ``_chat``'s return so the caller (``_orchestrate_core``)
    treats both identically.

    ``state`` is any ``OrchestrationState`` (stateful ``ConvoState`` or stateless
    ``StatelessSideContext``); both expose ``begin_task`` / ``own_view_messages``
    / ``append_assistant`` / ``discard_open_task`` through the Protocol."""
    adapter = ROUTE_TO_ADAPTER.get(route, ENTRY_ADAPTER)
    state.begin_task(route, user_text)
    # begin_task opened a task whose init_user == this user turn; own_view_messages
    # renders it as the trailing [user: init_user] (+ empty msgs). Do NOT pass an
    # extra current_turn or the user message would be duplicated.
    msgs = state.own_view_messages(route)

    on_open = ((lambda: stream_start_cb(route))
               if stream_start_cb is not None else None)
    status, d = _call_answer(
        msgs, adapter, max_tokens, tools, stream_cb, on_open, answer_options)
    if status == 400 and _is_pool_miss(d):
        fb = _pool_miss_fallback()
        if fb != route:
            _log(f"answer: pool-miss on {route}({adapter}) -> retry {fb}")
            # DISCARD the failed task (never appended a trace) and re-open under
            # the fallback route. Closing it would leave a phantom empty task in
            # the timeline -> a duplicate user turn + a false "[Lx 已完成]" block
            # in every other LoRA's own-view. discard_open_task works on both
            # state models (ConvoState pops tasks[-1]; Stateless drops _current).
            state.discard_open_task()
            state.begin_task(fb, user_text)
            adapter = ROUTE_TO_ADAPTER.get(fb, ENTRY_ADAPTER)
            msgs = state.own_view_messages(fb)
            on_open = ((lambda: stream_start_cb(fb))
                       if stream_start_cb is not None else None)
            status, d = _call_answer(
                msgs, adapter, max_tokens, tools, stream_cb, on_open,
                answer_options)
            route = fb
    if status == 200:
        if _chat_response_is_malformed(d, tools, answer_options):
            state.discard_open_task()
            return 502, _invalid_upstream_response(), route
        state.append_assistant(_message_for_state(d))
    return status, d, route


def _is_pool_miss(d: dict) -> bool:
    """Detect an engine 'adapter not loaded / not in lora_pool' 400. Match
    concrete substrings only — a loose `lora_pool and not` would false-positive
    on unrelated 400s and trigger a wrong-specialist fallback."""
    s = json.dumps(d, ensure_ascii=False)
    return ("not in configured lora_pool" in s
            or "is not loaded" in s
            or "not found in lora" in s)


def _summary_hop(state: ConvoState, task: Task) -> str:
    """Hop 3 — summary. Pinned to the task's own adapter; the scaffold is never
    fed into any own-view so it cannot pollute a reusable prefix.

    If the reasoning trace consumes the output budget before content lands,
    fall back to a clipped reasoning slice so other LoRAs still receive context
    for this task instead of an empty summary block."""
    adapter = ROUTE_TO_ADAPTER.get(task.owner, ENTRY_ADAPTER)
    msgs = state.summary_context_messages(task)
    status, d = _chat(msgs, adapter, SUMMARY_MAX_OUT,
                      answer_options=_NO_THINK_OPTIONS)
    summary = ""
    if status == 200:
        msg = _choice_message(d)
        content = msg.get("content") if isinstance(msg, dict) else None
        summary = content.strip() if isinstance(content, str) else ""
        if not summary:
            # GLM-5.2 reasoning can land under either extension key.
            raw_reasoning = (msg.get("reasoning")
                             if isinstance(msg, dict) else None)
            if raw_reasoning is None and isinstance(msg, dict):
                raw_reasoning = msg.get("reasoning_content")
            rc = (raw_reasoning.strip()
                  if isinstance(raw_reasoning, str) else "")
            if rc:
                summary = rc[:160]
    _log(f"summary@{task.owner} status={status} ({len(summary)} chars)")
    return summary


def _clamp_max_tokens(payload: dict) -> int:
    """Normalize the client output-token budget for the answer hop.

    Returns a clamped budget capped at MAX_OUTPUT_TOKENS (128k). An omitted or
    invalid value defaults to MAX_OUTPUT_TOKENS so the Proxy always sends an
    explicit max_tokens to the Engine (no reliance on the Engine native
    default). bool (which is an int in Python) is rejected; numeric strings
    remain accepted for compatibility. The Engine's max_model_len (1M)
    additionally bounds input+output, so effective output <=
    min(MAX_OUTPUT_TOKENS, max_model_len - input).
    """
    mt = payload.get("max_tokens")
    if mt is None:
        mt = payload.get("max_completion_tokens")
    if isinstance(mt, bool) or mt is None:
        return MAX_OUTPUT_TOKENS
    if isinstance(mt, str):
        try:
            mt = int(mt)
        except (ValueError, TypeError):
            return MAX_OUTPUT_TOKENS
    if not isinstance(mt, int):
        return MAX_OUTPUT_TOKENS
    return max(1, min(mt, MAX_OUTPUT_TOKENS))


def orchestrate(payload: dict, convo_id: str, tools: list | None = None) -> tuple[int, dict, dict]:
    """Run the full 3-hop (or sticky 1-hop) loop for one inbound request on the
    STATEFUL ``ConvoState`` timeline (``_get_convo`` registry).

    NOTE: this stateful-chat entry point is NOT wired to any endpoint in the
    dual-endpoint design — ``/v1/chat/completions`` is stateless (uses
    ``StatelessSideContext``) and ``/v1/responses`` drives ``_orchestrate_core``
    directly via ``responses.orchestrate_responses`` (which owns its own
    fork-on-branch ConvoState store). Retained as a direct-test/diagnostic
    helper for the ConvoState path; do not call from request handlers.
    """
    messages = payload.get("messages") or []
    system_msgs, last_role, last_text = split_system_and_turn(messages)
    state = _get_convo(convo_id, system_msgs)
    mt = _clamp_max_tokens(payload)
    answer_options = _chat_answer_options(payload)

    with state.lock:
        return _orchestrate_core(
            state, messages, last_role, last_text, mt, convo_id, tools,
            answer_options=answer_options)


def _orchestrate_core(state, messages, last_role, last_text, mt, convo_id,
                      tools=None, stream_cb=None, answer_options=None):
    """Run the shared core as one state transaction.

    The answer hop mutates conversation state before the terminal client frame
    is written. Restore the pre-request checkpoint when the engine fails or a
    client write raises, so an undelivered turn never becomes conversation
    history.
    """
    checkpoint = state.stream_checkpoint()
    try:
        result = _orchestrate_core_impl(
            state, messages, last_role, last_text, mt, convo_id, tools, stream_cb,
            answer_options)
    except BaseException:
        state.restore_stream_checkpoint(checkpoint)
        raise
    if result[0] != 200:
        state.restore_stream_checkpoint(checkpoint)
    return result


def _orchestrate_core_impl(state, messages, last_role, last_text, mt, convo_id,
                           tools=None, stream_cb=None, answer_options=None):
    """Shared 3-hop (route→answer→summary) or sticky 1-hop loop, parametrized on
    an ``OrchestrationState`` (stateful ``ConvoState`` via /v1/responses, or
    stateless ``StatelessSideContext`` via /v1/chat/completions). The two state
    models expose the same Protocol, so the loop is identical; only own-view
    construction differs (and is delegated to the state). See
    ``mol_harness/session.py`` OrchestrationState + ``stateless.py``.

    ``tools`` (the agent's OAI tool definitions) is forwarded to the specialist
    on the answer hop so it can emit tool_calls; the router + summary hops pass
    tools=None (they don't generate tool calls).

    ``stream_cb``: when set, the answer hop streams engine chunks to the client.
    It receives ``("meta_pre", diag)`` after routing, ``("chunk", c)`` per
    engine chunk, and ``("meta_post", final_diag)`` at the end (with
    ``engine_resp``/``finish_reason``/``tool_call`` or ``error``). When None,
    the non-streaming ``_shape_response`` path is used (byte-for-byte unchanged).
    """
    diag = {"convo_id": convo_id, "n_completed": state.completed_count()}

    # ---- sticky tool-result turn -----------------------------------------
    if last_role == "tool":
        validation_error = _validate_pending_tool_results(state, messages)
        replay = None
        if validation_error is not None:
            replay, replay_error = _validate_self_contained_tool_results(
                messages)
            if (not isinstance(state, StatelessSideContext)
                    or replay_error is not None):
                error = replay_error or validation_error
                code = error["error"]["code"]
                if stream_cb:
                    stream_cb(("meta_post", {**diag, "error": code}))
                    return 400, error, {**diag, "error": code}
                return 400, error, {**diag, "error": code}
            _, _, replay_user_text = split_system_and_turn(
                messages[:replay["user_index"] + 1])
            try:
                route, _, _ = _route(state, replay_user_text)
            except RoutingError as exc:
                _log(f"tool replay route failed: {exc}")
                status, error, code = _routing_error_response(exc)
                diag["error"] = code
                if stream_cb:
                    stream_cb(("meta_post", _routing_error_event(diag, error)))
                return status, error, diag
            state.begin_task(route, replay_user_text)
            state.set_pending_tool_route(route, replay["expected"])
            diag["decision"] = "self_contained_tool_replay"
        else:
            route = state.pending_tool_route
            diag["decision"] = "sticky_tool_continuation"
        adapter = ROUTE_TO_ADAPTER.get(route, ENTRY_ADAPTER)
        diag["route"] = route
        if stream_cb:
            stream_cb(("meta_pre", diag))
        for tm in extract_tool_results(messages):
            state.append_tool_result(tm)
        msgs = state.own_view_messages(route)
        status, d = _call_answer(
            msgs, adapter, mt, tools, stream_cb,
            answer_options=answer_options)
        if status == 200:
            if _chat_response_is_malformed(d, tools, answer_options):
                if stream_cb:
                    stream_cb(("meta_post", {
                        **diag, "error": "invalid_upstream_response"}))
                return 502, _invalid_upstream_response(), {
                    **diag, "error": "invalid_upstream_response"}
            state.append_assistant(_message_for_state(d))
        if status != 200:
            context_error = _context_length_error_response(status, d)
            if context_error is not None:
                diag["error"] = "context_length_exceeded"
                if stream_cb:
                    stream_cb(("meta_post", {
                        **diag, "error_body": context_error}))
                return 400, context_error, diag
            # propagate the real engine status (don't mask a failure as 200)
            if stream_cb:
                stream_cb(("meta_post", {**diag, "error": "answer_hop_failed"}))
                return status, d, {**diag, "error": "answer_hop_failed"}
            return status, d, {**diag, "error": "answer_hop_failed"}
        if _has_tool_call(d):
            # still in the tool loop — keep pending, skip summary
            state.set_pending_tool_route(
                route, _tool_call_ids_from_messages([_choice_message(d)]))
            _publish_buffered_tool_chunks(d, stream_cb)
            diag["pending_tool_route"] = route
            if stream_cb:
                stream_cb(("meta_post", {**diag, "engine_resp": d,
                                         "finish_reason": _finish_reason(d), "tool_call": True}))
                return 200, d, diag
            return _shape_response(d, route, diag)
        # tool loop finished -> close + summarize (gated by the state model)
        _publish_buffered_tool_chunks(d, stream_cb)
        task = state.close_open_task(summarize=False)
        if task is not None and state.should_summarize(route):
            task.summary = _summary_hop(state, task)
            state.record_summary(task)
        diag["summary"] = bool(task and task.summary)
        if stream_cb:
            stream_cb(("meta_post", {**diag, "engine_resp": d,
                                     "finish_reason": _finish_reason(d), "tool_call": False}))
            return 200, d, diag
        return _shape_response(d, route, diag)

    # ---- fresh user turn: route -> answer -> summary ----------------------
    if last_role != "user":
        # assistant/other leading turn without a pending tool loop: treat the
        # last message text as a user query (best effort).
        last_text = last_text or ""

    # An abandoned tool loop (pending set, but the agent sent a new user turn)
    # is closed here inside begin_task; route the new turn freshly through L0.
    try:
        route, decision, _ = _route(state, last_text)
    except RoutingError as exc:
        _log(f"route failed: {exc}")
        status, error, code = _routing_error_response(exc)
        diag["error"] = code
        if stream_cb:
            stream_cb(("meta_post", _routing_error_event(diag, error)))
        return status, error, diag
    diag["route"] = route
    diag["decision"] = decision
    initial_route = route

    def stream_started(actual_route):
        diag["route"] = actual_route
        if actual_route != initial_route:
            diag["decision"] = "pool_miss_fallback"
        stream_cb(("meta_pre", diag))

    status, d, route = _answer_hop(
        state, route, last_text, mt, tools, stream_cb,
        stream_started if stream_cb else None, answer_options)
    # _answer_hop may have switched route on a pool-miss fallback; reflect it.
    diag["route"] = route
    if status != 200:
        context_error = _context_length_error_response(status, d)
        if context_error is not None:
            diag["error"] = "context_length_exceeded"
            if stream_cb:
                stream_cb(("meta_post", {
                    **diag, "error_body": context_error}))
            return 400, context_error, diag
        diag["error"] = "answer_hop_failed"
        if stream_cb:
            stream_cb(("meta_post", {**diag, "error": "answer_hop_failed"}))
        return status, d, diag

    # tool-call short-circuit: forward the tool_call, keep the task open
    if _has_tool_call(d):
        state.set_pending_tool_route(
            route, _tool_call_ids_from_messages([_choice_message(d)]))
        _publish_buffered_tool_chunks(d, stream_cb)
        diag["pending_tool_route"] = route
        if stream_cb:
            stream_cb(("meta_post", {**diag, "engine_resp": d,
                                     "finish_reason": _finish_reason(d), "tool_call": True}))
            return 200, d, diag
        return _shape_response(d, route, diag)

    # plain answer -> close task + summarize (side effect, not returned)
    _publish_buffered_tool_chunks(d, stream_cb)
    task = state.close_open_task(summarize=False)
    if task is not None and state.should_summarize(route):
        task.summary = _summary_hop(state, task)
        state.record_summary(task)
    diag["summary"] = bool(task and task.summary)
    if stream_cb:
        stream_cb(("meta_post", {**diag, "engine_resp": d,
                                 "finish_reason": _finish_reason(d), "tool_call": False}))
        return 200, d, diag
    return _shape_response(d, route, diag)


def _shape_response(engine_resp: dict, route: str, diag: dict) -> tuple[int, dict, dict]:
    """Return the answer-hop response to the agent, with proxy metadata stamped
    for observability (route/summary are internal but useful for testing)."""
    if isinstance(engine_resp, dict) and "choices" in engine_resp:
        cached, ptoks = _cache_stats(engine_resp)
        if _finish_reason(engine_resp) in ("length", "content_filter"):
            message = _choice_message(engine_resp)
            if isinstance(message, dict):
                message.pop("tool_calls", None)
                message.pop("function_call", None)
        engine_resp = _sanitize_chat_response(engine_resp)
        choice = (engine_resp.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if (isinstance(message, dict) and message.get("tool_calls")
                and choice.get("finish_reason") not in (
                    "tool_calls", "length", "content_filter")):
            choice["finish_reason"] = "tool_calls"
        md = engine_resp.setdefault("metadata", {})
        if not isinstance(md, dict):
            md = {}
            engine_resp["metadata"] = md
        md["mol_selected_route"] = route
        md["mol_decision"] = diag.get("decision")
        md["mol_pending_tool_route"] = diag.get("pending_tool_route")
        md["mol_convo_completed"] = diag.get("n_completed")
        md["mol_summary_recorded"] = diag.get("summary")
        # KV-reuse observability (M4): how many prompt tokens hit the engine's
        # per-LoRA prefix cache on this answer hop. On a re-entry to a LoRA, this
        # should equal the length of that LoRA's retained own-view prefix, i.e.
        # only the newly-appended tail is prefilled (delta-only).
        md["mol_cached_tokens"] = cached
        md["mol_prompt_tokens"] = ptoks
        # cosmetic served name
        engine_resp["model"] = SERVED_MODEL_NAME
        engine_resp["id"] = "chatcmol-" + uuid.uuid4().hex[:24]
        engine_resp["object"] = "chat.completion"
        engine_resp["created"] = int(time.time())
    return 200, engine_resp, diag


def _cache_stats(d: dict) -> tuple[int, int]:
    """Extract (cached_tokens, prompt_tokens) from an engine response. SGLang
    with --enable-cache-report reports cached tokens under usage.prompt_tokens_details
    (OAI-compatible) or as a flat usage field; check both."""
    usage = d.get("usage") if isinstance(d, dict) else None
    if not isinstance(usage, dict):
        return 0, 0
    prompt_tokens = usage.get("prompt_tokens", 0)
    ptoks = prompt_tokens if _is_token_count(prompt_tokens) else 0
    cached = 0
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        cached_tokens = ptd.get("cached_tokens", 0)
        cached = cached_tokens if _is_token_count(cached_tokens) else 0
    if not cached:
        cached_tokens = usage.get("cached_tokens", 0)
        cached = cached_tokens if _is_token_count(cached_tokens) else 0
    return cached, ptoks


# --------------------------------------------------------------------------- SSE helpers

def _sse_frame(data: dict, event: str | None = None) -> bytes:
    """Serialize one SSE frame. ``data: <json>\n\n`` (optionally preceded by an
    ``event: <type>\n`` line for the Responses API; the chat-completions stream
    has no ``event:`` line)."""
    out = b""
    if event:
        out += f"event: {event}\n".encode("utf-8")
    out += b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n\n"
    return out


def _chat_chunk(cid: str, model: str, delta: dict, finish: str | None,
                meta: dict | None = None, *, created: int | None = None) -> dict:
    """Build a chat.completion.chunk dict (OpenAI streaming shape). ``metadata`` is
    non-standard but accepted by clients; the proxy stamps ``mol_*`` here too so a
    streaming client observes route/summary the same way as non-streaming."""
    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()) if created is None else created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if meta is not None:
        chunk["metadata"] = meta
    return chunk




class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a hard cap on live request workers."""

    def __init__(self, server_address, handler_class, *,
                 max_workers: int = MAX_CONCURRENT_REQUESTS):
        worker_limit = max(1, max_workers)
        self.request_queue_size = worker_limit
        self._request_slots = threading.BoundedSemaphore(worker_limit)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        if not self._request_slots.acquire(blocking=False):
            body = json.dumps(_oai_error(
                "The proxy is handling too many concurrent requests",
                error_type="server_error", code="server_overloaded")).encode()
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"Connection: close\r\n\r\n" + body)
            try:
                request.sendall(response)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(CLIENT_SOCKET_TIMEOUT_S)

    def log_message(self, format, *args):  # noqa: A002  silence default access log
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json",
              headers: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _path(self) -> str:
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        return path or "/"

    def _require_auth(self) -> bool:
        if not API_KEY:
            return True
        supplied = self.headers.get("Authorization", "")
        if hmac.compare_digest(supplied, f"Bearer {API_KEY}"):
            return True
        self.close_connection = True
        self._send(401, json.dumps(_oai_error(
            "Incorrect API key provided", error_type="authentication_error",
            code="invalid_api_key")).encode("utf-8"),
            headers={"WWW-Authenticate": "Bearer"})
        return False

    def _read_request_body(self) -> bytes | None:
        """Read one bounded, unambiguous HTTP request body.

        ``BaseHTTPRequestHandler`` does not decode chunked request bodies. If
        such a body (or an invalid/duplicate Content-Length) remains unread on
        a keep-alive connection, its bytes can be parsed as a second request.
        Reject and close those connections explicitly.
        """
        transfer_encoding = self.headers.get_all("Transfer-Encoding") or []
        lengths = self.headers.get_all("Content-Length") or []
        if transfer_encoding:
            self.close_connection = True
            self._send(400, json.dumps(_oai_error(
                "Transfer-Encoding request bodies are not supported",
                code="unsupported_transfer_encoding",
                param="Transfer-Encoding")).encode("utf-8"))
            return None
        if len(lengths) > 1:
            self.close_connection = True
            self._send(400, json.dumps(_oai_error(
                "Content-Length must appear exactly once",
                code="invalid_content_length",
                param="Content-Length")).encode("utf-8"))
            return None
        raw_length = lengths[0].strip() if lengths else "0"
        if not raw_length.isascii() or not raw_length.isdigit():
            self.close_connection = True
            self._send(400, json.dumps(_oai_error(
                "Content-Length must be a non-negative integer",
                code="invalid_content_length",
                param="Content-Length")).encode("utf-8"))
            return None
        normalized_length = raw_length.lstrip("0") or "0"
        limit_text = str(MAX_REQUEST_BYTES)
        if (len(normalized_length) > len(limit_text)
                or (len(normalized_length) == len(limit_text)
                    and normalized_length > limit_text)):
            self.close_connection = True
            self._send(413, json.dumps(_oai_error(
                f"Request body exceeds the {MAX_REQUEST_BYTES}-byte limit",
                code="request_too_large", param="body")).encode("utf-8"))
            return None
        length = int(normalized_length)
        try:
            body = self.rfile.read(length) if length else b""
        except (OSError, TimeoutError):
            self.close_connection = True
            try:
                self._send(408, json.dumps(_oai_error(
                    "Timed out while reading the request body",
                    code="request_timeout", param="body")).encode("utf-8"))
            except OSError:
                pass
            return None
        if len(body) != length:
            self.close_connection = True
            self._send(400, json.dumps(_oai_error(
                "Request body ended before Content-Length bytes were received",
                code="incomplete_request_body", param="body")).encode("utf-8"))
            return None
        return body

    def _json_object(self, raw: bytes) -> dict | None:
        try:
            payload = json.loads(raw) if raw else {}
        except Exception as e:
            self._send(400, json.dumps(_oai_error(
                f"bad json: {e}", code="invalid_json", param="body")).encode())
            return None
        if not isinstance(payload, dict):
            self._send(400, json.dumps(_oai_error(
                "JSON request body must be an object",
                code="invalid_json_body", param="body")).encode("utf-8"))
            return None
        return payload

    def _begin_sse(self, status: int = 200,
                   headers: dict[str, str] | None = None):
        """Begin an SSE response (chunked transfer-encoding, no Content-Length).
        Each frame is written via :meth:`_sse_write`; terminate with
        :meth:`_sse_end` so the client knows the stream is over."""
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _sse_write(self, frame: bytes):
        """Write one SSE frame as an HTTP/1.1 chunk + flush (so the client sees
        it immediately — wfile is buffered; without flush, SSE is silently held)."""
        self.wfile.write(f"{len(frame):X}\r\n".encode("ascii") + frame + b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        """Terminate the chunked stream."""
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self._read_request_body() is None:
            return
        path = self._path()
        if path != "/health" and not self._require_auth():
            return
        # /v1/models: the proxy exposes a SINGLE served model name to agents
        # (SERVED_MODEL_NAME = "Macaron-V1-Venti"). Do NOT forward to the engine —
        # the engine's adapter names (L0/L1/...) are an internal implementation
        # detail that the agent must not see or call directly.
        if path == "/v1/models":
            body = json.dumps({"object": "list",
                               "data": [{"id": SERVED_MODEL_NAME,
                                         "object": "model",
                                         "created": 0,
                                         "owned_by": "mol"}]}, ensure_ascii=False).encode("utf-8")
            self._send(200, body)
            return
        if path == "/health":
            self._send(200, json.dumps({"status": "ok"}).encode("utf-8"))
            return
        self._send(404, json.dumps(_oai_error(
            f"Endpoint {path} does not exist", code="endpoint_not_found",
            param="path")).encode("utf-8"))

    def do_POST(self):
        if not self._require_auth():
            return
        raw = self._read_request_body()
        if raw is None:
            return
        path = self._path()

        if path == "/v1/chat/completions":
            self._handle_chat(raw)
            return
        if path == "/v1/responses":
            self._handle_responses(raw)
            return
        self._send(404, json.dumps(_oai_error(
            f"Endpoint {path} does not exist", code="endpoint_not_found",
            param="path")).encode("utf-8"))

    def _handle_chat(self, raw: bytes):
        """STATELESS chat path. The agent resends the FULL message
        history each request (standard OAI client behavior); the proxy rebuilds
        the specialist's own-view via ``build_own_view`` (end_idx folding). No
        client-defined header is required; an optional legacy
        ``X-Conversation-Id`` is accepted for compatibility while the
        conversation is otherwise fingerprinted from the full request.
        Standard tool turns also resume through ``tool_call_id`` without a
        private header. The stateful
        timeline (ConvoState) is NOT used here; it lives behind /v1/responses.

        Streaming (``stream: true``): only the answer hop streams; the router +
        summary hops are out-of-band (router blocks first to pick the route;
        summary runs after the stream completes, client-invisible). The first
        chunk carries ``metadata.mol_*`` (route/decision). On client disconnect
        or engine mid-stream error the open task is discarded and ``[DONE]`` is
        emitted so the client terminates cleanly."""
        payload = self._json_object(raw)
        if payload is None:
            return
        # The proxy exposes a SINGLE served model name; reject anything else (same
        # as an OAI endpoint rejecting an unknown model). The engine adapter names
        # (L0/L1/...) are an internal routing target the agent must not call.
        model = payload.get("model")
        if model != SERVED_MODEL_NAME:
            err = {"error": {"message": f"The model `{model}` does not exist",
                             "type": "invalid_request_error",
                             "code": "model_not_found",
                             "param": "model"}}
            self._send(404, json.dumps(err, ensure_ascii=False).encode("utf-8"))
            return
        messages = payload.get("messages")
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(message, dict) for message in messages)):
            self._send(400, json.dumps(_oai_error(
                "messages must be a non-empty array of objects",
                code="invalid_messages", param="messages")).encode("utf-8"))
            return
        try:
            _validate_chat_request_capabilities(payload)
        except ValueError as exc:
            self._send(400, json.dumps(_oai_error(
                str(exc), code="invalid_request")).encode("utf-8"))
            return
        system_msgs, last_role, last_text = split_system_and_turn(messages)
        mt = _clamp_max_tokens(payload)
        tools = payload.get("tools")
        answer_options = _chat_answer_options(payload)
        stream = bool(payload.get("stream"))
        explicit_convo_id = self.headers.get("X-Conversation-Id")
        authorization = self.headers.get("Authorization")
        sticky_context = (_take_tool_context(messages, authorization)
                          if explicit_convo_id is None and last_role == "tool"
                          else None)
        try:
            client_convo_id = _chat_conversation_token(explicit_convo_id)
        except ValueError as exc:
            self._send(400, json.dumps(_oai_error(
                str(exc), code="invalid_conversation_id",
                param="X-Conversation-Id")).encode("utf-8"))
            return
        if sticky_context is not None:
            state, client_convo_id, tool_binding = sticky_context
            convo_id = state.convo_key_id
        else:
            tool_binding = None
            convo_id = _chat_convo_key(
                messages, client_convo_id, authorization)
            state = _get_side_context(convo_id, system_msgs)

        if last_role == "tool":
            with state.lock:
                validation_error = _validate_chat_tool_results(
                    state, messages, tool_binding)
            if validation_error is not None:
                self._send(400, json.dumps(
                    validation_error, ensure_ascii=False).encode("utf-8"))
                return

        routing_key = _new_chat_routing_key()

        if not stream:
            t0 = time.time()
            diag: dict = {}
            publish_attempted = False
            try:
                with _routing_scope(routing_key):
                    with state.lock:
                        if last_role == "tool":
                            validation_error = (
                                _validate_chat_tool_results(
                                    state, messages, tool_binding))
                            if validation_error is not None:
                                self._send(400, json.dumps(
                                    validation_error,
                                    ensure_ascii=False).encode("utf-8"))
                                return
                        state_checkpoint = state.stream_checkpoint()
                        registry_checkpoint = _snapshot_tool_context(
                            state, authorization)
                        try:
                            state.set_request_messages(messages)
                            status, body, diag = _orchestrate_core(
                                state, messages, last_role, last_text, mt,
                                convo_id, tools,
                                answer_options=answer_options)
                            if (status != 200
                                    and diag.get("error") not in (
                                        "orphan_tool_turn",
                                        "context_length_exceeded")):
                                status = _public_upstream_status(status)
                                body = _oai_error(
                                    "The upstream model request failed",
                                    error_type="server_error",
                                    code="answer_hop_failed")
                            response_headers = None
                            if status == 200:
                                body.setdefault("metadata", {})[
                                    "mol_conversation_id"] = client_convo_id
                                if _has_tool_call(body):
                                    _register_tool_context(
                                        body, state, client_convo_id,
                                        authorization)
                                else:
                                    _clear_tool_context(state, authorization)
                                response_headers = {
                                    "X-Conversation-Id": client_convo_id}
                            else:
                                state.restore_stream_checkpoint(
                                    state_checkpoint)
                                _restore_tool_context(
                                    state, authorization,
                                    registry_checkpoint)
                            publish_attempted = True
                            self._send(
                                status,
                                json.dumps(
                                    body,
                                    ensure_ascii=False).encode("utf-8"),
                                headers=response_headers)
                        except BaseException:
                            state.restore_stream_checkpoint(state_checkpoint)
                            _restore_tool_context(
                                state, authorization, registry_checkpoint)
                            raise
            except Exception as e:
                _log(
                    f"orchestrate EXC convo={convo_id[:8]}: "
                    f"{type(e).__name__}: {e}")
                if not publish_attempted:
                    self._send(500, json.dumps(_oai_error(
                        "Internal proxy error", error_type="server_error",
                        code="proxy_error")).encode())
                return
            finally:
                _release_routing_key(routing_key)
            dt = time.time() - t0
            _log(
                f"turn(stateless) convo={convo_id[:8]} "
                f"route={diag.get('route')} "
                f"decision={diag.get('decision')} dt={dt:.1f}s")
            return

        # ---- streaming ----
        self._begin_sse(
            200, headers={"X-Conversation-Id": client_convo_id})
        cid = "chatcmol-" + uuid.uuid4().hex[:24]
        stream_created = int(time.time())
        first_sent = {"v": False}
        deferred_tool_frames: list[bytes] = []

        def cb(ev):
            kind, data = ev
            if kind == "meta_pre":
                # first chunk: assistant-role + mol_* metadata (route/decision),
                # same keys as the non-streaming _shape_response so clients/tests
                # read mol_selected_route uniformly.
                from .responses import _mol_meta
                meta = _mol_meta(data)
                meta["mol_conversation_id"] = client_convo_id
                self._sse_write(_sse_frame(_chat_chunk(
                    cid, SERVED_MODEL_NAME, {"role": "assistant", "content": ""},
                    None, meta=meta, created=stream_created)))
                first_sent["v"] = True
            elif kind == "chunk":
                c = _sanitize_chat_chunk(data)
                c.pop("usage", None)
                c["model"] = SERVED_MODEL_NAME
                c["id"] = cid
                c["created"] = stream_created
                c["object"] = "chat.completion.chunk"
                meaningful = False
                has_tool_delta = False
                for choice in c.get("choices") or []:
                    choice["finish_reason"] = None
                    delta = choice.get("delta")
                    if (isinstance(delta, dict)
                            and delta.get("tool_calls")):
                        has_tool_delta = True
                    if ((isinstance(delta, dict) and bool(delta))
                            or choice.get("logprobs") is not None):
                        meaningful = True
                if meaningful:
                    frame = _sse_frame(c)
                    if has_tool_delta:
                        deferred_tool_frames.append(frame)
                    else:
                        self._sse_write(frame)
            elif kind == "meta_post":
                if isinstance(data, dict) and "error" in data:
                    error_body = data.get("error_body")
                    if not (isinstance(error_body, dict)
                            and isinstance(error_body.get("error"), dict)):
                        error_body = _oai_error(
                            str(data.get("error") or "answer hop failed"),
                            error_type="server_error",
                            code="answer_hop_failed")
                    self._sse_write(_sse_frame(error_body))
                    self._sse_write(b"data: [DONE]\n\n")
                else:
                    from .responses import _mol_meta
                    fin = data.get("finish_reason") if isinstance(data, dict) else None
                    meta = _mol_meta({k: v for k, v in (data or {}).items()
                                     if k not in ("engine_resp", "finish_reason", "tool_call")})
                    meta["mol_conversation_id"] = client_convo_id
                    # carry cached/prompt tokens from the engine response
                    eng = data.get("engine_resp") if isinstance(data, dict) else None
                    if isinstance(eng, dict):
                        cached, ptoks = _cache_stats(eng)
                        meta["mol_cached_tokens"] = cached
                        meta["mol_prompt_tokens"] = ptoks
                        if data.get("tool_call"):
                            _register_tool_context(
                                eng, state, client_convo_id, authorization)
                        else:
                            _clear_tool_context(state, authorization)
                    for frame in deferred_tool_frames:
                        self._sse_write(frame)
                    deferred_tool_frames.clear()
                    self._sse_write(_sse_frame(_chat_chunk(
                        cid, SERVED_MODEL_NAME, {}, fin or "stop", meta=meta,
                        created=stream_created)))
                    # The engine always returns usage. Forward it for billing
                    # even when the public client did not explicitly opt in.
                    if (isinstance(eng, dict)
                            and isinstance(eng.get("usage"), dict)):
                        public_usage = _sanitize_usage(eng["usage"])
                        if _USAGE_COUNT_KEYS.issubset(public_usage):
                            usage_chunk = _chat_chunk(
                                cid, SERVED_MODEL_NAME, {}, None,
                                created=stream_created)
                            usage_chunk["choices"] = []
                            usage_chunk["usage"] = public_usage
                            self._sse_write(_sse_frame(usage_chunk))
                    self._sse_write(b"data: [DONE]\n\n")

        t0 = time.time()
        diag: dict = {}
        try:
            with _routing_scope(routing_key):
                with state.lock:
                    if last_role == "tool":
                        validation_error = (
                            _validate_chat_tool_results(
                                state, messages, tool_binding))
                        if validation_error is not None:
                            self._sse_write(_sse_frame(validation_error))
                            self._sse_write(b"data: [DONE]\n\n")
                            return
                    state_checkpoint = state.stream_checkpoint()
                    registry_checkpoint = _snapshot_tool_context(
                        state, authorization)
                    try:
                        state.set_request_messages(messages)
                        try:
                            status, body, diag = _orchestrate_core(
                                state, messages, last_role, last_text, mt,
                                convo_id, tools, stream_cb=cb,
                                answer_options=answer_options)
                        finally:
                            # Keep only a successfully published tool-loop task.
                            if (state.open_task() is not None
                                    and state.pending_tool_route is None):
                                state.discard_open_task()
                        if status != 200:
                            state.restore_stream_checkpoint(state_checkpoint)
                            _restore_tool_context(
                                state, authorization, registry_checkpoint)
                    except BaseException:
                        state.restore_stream_checkpoint(state_checkpoint)
                        _restore_tool_context(
                            state, authorization, registry_checkpoint)
                        raise
        except Exception as e:
            _log(f"stream EXC convo={convo_id[:8]}: {type(e).__name__}: {e}")
            # best-effort: ensure the client sees stream end even on error
            try:
                if not first_sent["v"]:
                    self._sse_write(_sse_frame(_chat_chunk(
                        cid, SERVED_MODEL_NAME, {"role": "assistant", "content": ""}, None,
                        meta={"error": "proxy_error"}, created=stream_created)))
                self._sse_write(b"data: [DONE]\n\n")
            except Exception:
                pass
        finally:
            _release_routing_key(routing_key)
            try:
                self._sse_end()
            except Exception:
                pass
        dt = time.time() - t0
        _log(f"turn(stream) convo={convo_id[:8]} route={diag.get('route') if diag else None} "
             f"decision={diag.get('decision') if diag else None} dt={dt:.1f}s")

    def _handle_responses(self, raw: bytes):
        """STATEFUL Responses path. The proxy owns the conversation store; the
        agent sends ``input`` + ``previous_response_id``. Converts Responses↔
        chat at the boundary and drives the same 3-hop core on a ConvoState.

        Streaming (``stream: true``): emits the Responses SSE lifecycle. The
        router + summary hops are out-of-band; only answer-hop deltas stream.
        ``metadata.mol_*`` is on the ``response.created`` event."""
        payload = self._json_object(raw)
        if payload is None:
            return
        # The proxy exposes a SINGLE served model name; reject anything else.
        model = payload.get("model")
        if model is not None and model != SERVED_MODEL_NAME:
            err = {"error": {"message": f"The model `{model}` does not exist",
                             "type": "invalid_request_error",
                             "code": "model_not_found",
                             "param": "model"}}
            self._send(404, json.dumps(err, ensure_ascii=False).encode("utf-8"))
            return
        from . import responses as _responses
        stream = bool(payload.get("stream"))
        t0 = time.time()
        preflight_status, preflight_error = (
            _responses.preflight_responses_request(payload))
        if preflight_status != 200:
            self._send(
                preflight_status,
                json.dumps(preflight_error, ensure_ascii=False).encode("utf-8"))
            return
        if stream:
            stream_state = {
                "started": False, "created": None,
                "last_sequence": -1, "terminal": False,
            }

            def tracked_write(frame: bytes):
                event = None
                try:
                    data_line = next(
                        line for line in frame.decode("utf-8").splitlines()
                        if line.startswith("data:"))
                    event = json.loads(data_line[5:].strip())
                except (StopIteration, ValueError, TypeError):
                    pass
                if isinstance(event, dict):
                    sequence = event.get("sequence_number")
                    if isinstance(sequence, int):
                        stream_state["last_sequence"] = max(
                            stream_state["last_sequence"], sequence)
                    if event.get("type") == "response.created":
                        stream_state["created"] = event.get("response")
                    if event.get("type") in (
                            "response.completed", "response.incomplete",
                            "response.failed"):
                        stream_state["terminal"] = True
                if not stream_state["started"]:
                    self._begin_sse(200)
                    stream_state["started"] = True
                self._sse_write(frame)

            try:
                _responses.orchestrate_responses_stream(payload, tracked_write)
            except Exception as e:
                _log(f"responses-stream EXC: {type(e).__name__}: {e}")
                created = stream_state["created"]
                if not stream_state["started"]:
                    self._send(500, json.dumps(_oai_error(
                        "Internal proxy error", error_type="server_error",
                        code="proxy_error")).encode("utf-8"))
                elif isinstance(created, dict) and not stream_state["terminal"]:
                    try:
                        _responses._write_failed_event(
                            self._sse_write, created.get("id") or "resp_unknown",
                            created.get("model") or SERVED_MODEL_NAME,
                            created.get("previous_response_id"), payload,
                            "Internal proxy error",
                            stream_state["last_sequence"] + 1,
                            error_code="server_error",
                            created_at=created.get("created_at"),
                            metadata=created.get("metadata"))
                    except Exception:
                        pass
            finally:
                if stream_state["started"]:
                    try:
                        self._sse_end()
                    except Exception:
                        pass
            dt = time.time() - t0
            _log(f"turn(responses-stream) prev={payload.get('previous_response_id','-')[:8]} dt={dt:.1f}s")
            return
        publish_attempted = False

        def emit_success(response_body):
            nonlocal publish_attempted
            publish_attempted = True
            self._send(
                200, json.dumps(
                    response_body, ensure_ascii=False).encode("utf-8"))

        try:
            status, body = _responses.orchestrate_responses(
                payload, emit_success=emit_success)
        except Exception as e:
            _log(f"responses EXC: {type(e).__name__}: {e}")
            if not publish_attempted:
                self._send(500, json.dumps(_oai_error(
                    "Internal proxy error", error_type="server_error",
                    code="proxy_error")).encode())
            return
        dt = time.time() - t0
        route = None
        if isinstance(body, dict):
            md = body.get("metadata")
            if isinstance(md, dict):
                route = md.get("mol_selected_route")
        _log(f"turn(responses) prev={payload.get('previous_response_id','-')[:8]} "
             f"route={route} dt={dt:.1f}s")
        if status != 200:
            self._send(
                status, json.dumps(body, ensure_ascii=False).encode("utf-8"))


def main():
    from .async_runtime import (
        DRAIN_TIMEOUT_S, KEEPALIVE_TIMEOUT_S, MAX_CONNECTIONS, ProxyASGI)
    import uvicorn

    _log(f"MoL async proxy :{PROXY_PORT} -> {UPSTREAM}")
    _log(f"library={LIBRARY_DIR} entry={ENTRY_ROUTE}({ENTRY_ADAPTER}) "
         f"routes={ROUTE_TO_ADAPTER}")
    _log(f"model_router={USE_MODEL_ROUTER} pure_model={PURE_MODEL_ROUTE} "
         f"client_max_out=unlimited router_max={ROUTER_MAX_TOKENS}")
    uvicorn.run(
        ProxyASGI(), host="0.0.0.0", port=PROXY_PORT, workers=1,
        loop="uvloop", http="httptools", lifespan="on",
        limit_concurrency=MAX_CONNECTIONS,
        timeout_keep_alive=KEEPALIVE_TIMEOUT_S,
        timeout_graceful_shutdown=DRAIN_TIMEOUT_S,
        access_log=False,
    )


if __name__ == "__main__":
    main()
