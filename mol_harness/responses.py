"""Stateful Responses API adapter and store for the MoL Proxy.

The dual-endpoint Proxy exposes two client-facing
endpoints; this module implements the **stateful** one, ``/v1/responses``:

  * the agent sends ``input`` (string | array) + ``previous_response_id``;
  * the PROXY owns the conversation store keyed by ``resp_id``;
  * internally it reuses the stateful ``ConvoState`` timeline and the shared
    3-hop core (``proxy._orchestrate_core``) by converting Responses ↔ chat at
    the boundary — the engine always sees ``/v1/chat/completions``, never
    ``/v1/responses``.

This is the *stateful* counterpart to ``stateless.StatelessSideContext``
(``/v1/chat/completions``). The two never share state: independent
registries (``_RESPONSES`` here vs ``_SIDE_CTX`` in proxy.py).

Response chaining (``previous_response_id``):
  * Each ``StoredResponse`` owns a **snapshot** of the ``ConvoState`` as it was
    at the moment that response was generated (an immutable point in the
    timeline). To continue from ``previous_response_id=R``, the proxy forks
    (deep-copies) R's snapshot and appends the new turn to the copy.
  * Linear chain (R2 from R1) and branch (R3 also from R1, not R2) BOTH fork
    R1's immutable snapshot — so a branch never sees a sibling's turn. This
    matches OpenAI Responses semantics ("continue after R1 was generated"),
    not "continue after whatever the shared state currently looks like".
  * Per-LoRA KV reuse is preserved: the own-view is rebuilt from ``tasks`` each
    hop, and the forked snapshot's tasks are identical to the parent's, so the
    engine sees the same stable prefix on re-entry (only the appended tail is
    prefilled) regardless of whether the ConvoState is shared or copied.

Tool format: Responses output ``function_call`` items (flat
``{name, arguments, call_id}``) ↔ chat ``message.tool_calls``. A specialist's
tool_call response becomes a ``function_call`` output item; the agent's next
``function_call_output`` input becomes a ``tool`` message that drives the
shared core's sticky-tool branch (``ConvoState.pending_tool_route`` carries
across the response chain via the forked snapshot).

Both non-streaming and streaming Responses are supported. Streaming emits the
Responses SSE lifecycle while the engine remains on ``/v1/chat/completions``.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .session import ConvoState, split_system_and_turn
from . import proxy as _proxy


# resp_id -> StoredResponse. LRU-bounded + TTL-evicted like the convo registry.
_RESPONSES: "OrderedDict[str, StoredResponse]" = OrderedDict()
_RESPONSES_LOCK = threading.Lock()
_PENDING_RESPONSES: dict[str, threading.Event] = {}
_PENDING_RESPONSE_SIZES: dict[str, int] = {}
_MAX_RESPONSES = max(
    1, int(__import__("os").environ.get("MOL_MAX_RESPONSES", "5000")))
_MAX_RESPONSE_STATE_BYTES = max(1, int(__import__("os").environ.get(
    "MOL_MAX_RESPONSE_STATE_BYTES", str(512 * 1024 * 1024))))
_MAX_PENDING_RESPONSES = max(1, int(__import__("os").environ.get(
    "MOL_MAX_PENDING_RESPONSES", "128")))
_MAX_PENDING_RESPONSE_STATE_BYTES = max(
    _MAX_RESPONSE_STATE_BYTES,
    int(__import__("os").environ.get(
        "MOL_MAX_PENDING_RESPONSE_STATE_BYTES",
        str(_MAX_RESPONSE_STATE_BYTES))))
_PENDING_WAIT_S = max(
    1.0, float(__import__("os").environ.get("MOL_PENDING_WAIT_S", "60")))
_RESPONSES_CONDITION = threading.Condition(_RESPONSES_LOCK)


class ResponseStoreCapacityError(RuntimeError):
    pass


@dataclass
class StoredResponse:
    """One stored Responses-API object + the ``ConvoState`` snapshot it owns.

    ``state`` is the timeline AS IT WAS when this response was generated — an
    immutable point in the conversation. A continuation (``previous_response_id``
    = this ``resp_id``) forks this snapshot (deep-copy) and appends its turn to
    the copy, so siblings (linear or branch) never mutate each other's history.
    """
    resp_id: str
    state: ConvoState
    parent_id: str | None = None
    routing_root_id: str | None = None
    touched_at: float = field(default_factory=time.monotonic)
    approx_bytes: int = 0


# --------------------------------------------------------------------------- Responses -> chat

def responses_input_to_oai_messages(input_items: Any, instructions: str | None) -> list[dict]:
    """Convert a Responses-API ``input`` (string | array) to OAI chat messages.

      * ``instructions`` (top-level system prompt) -> leading system message.
      * string ``input`` -> one user message.
      * array ``input`` -> per item:
          - ``message``          -> OAI message (role preserved;
            content as str or flattened from content parts).
          - ``function_call``    -> assistant message with ``tool_calls``
            (the specialist's tool_call, resent by the agent to continue).
          - ``function_call_output`` -> ``tool`` message (the tool result).
          - ``reasoning``        -> assistant ``reasoning_content`` extension.

    Consecutive reasoning/function-call items belong to one assistant turn.  In
    particular, parallel function calls must become one Chat assistant message
    with multiple ``tool_calls`` rather than several invalid assistant turns.
    """
    msgs: list[dict] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})

    if isinstance(input_items, str):
        msgs.append({"role": "user", "content": input_items})
        return msgs
    if not isinstance(input_items, list) or not input_items:
        raise ValueError("Responses `input` must be a string or non-empty array")

    pending_assistant: dict | None = None

    def assistant() -> dict:
        nonlocal pending_assistant
        if pending_assistant is None:
            pending_assistant = {"role": "assistant", "content": ""}
        return pending_assistant

    def flush_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None:
            msgs.append(pending_assistant)
            pending_assistant = None

    for item in input_items:
        if not isinstance(item, dict):
            raise ValueError("Responses input items must be objects")
        itype = item.get("type")
        if itype is None and "role" in item and "content" in item:
            itype = "message"
        if itype == "message":
            role = item.get("role") or "user"
            if role not in ("user", "assistant", "system", "developer"):
                raise ValueError(f"unsupported Responses message role: {role!r}")
            content = item.get("content")
            if role == "assistant":
                text, refusal = _assistant_input_content(content)
                turn = assistant()
                turn["content"] = str(turn.get("content") or "") + text
                if refusal:
                    turn["refusal"] = str(turn.get("refusal") or "") + refusal
            else:
                flush_assistant()
                msgs.append({"role": role,
                             "content": _flatten_input_content(content)})
        elif itype == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "Responses function_call requires a non-empty call_id")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Responses function_call requires a non-empty name")
            if not isinstance(arguments, str):
                raise ValueError(
                    "Responses function_call arguments must be a string")
            turn = assistant()
            turn.setdefault("tool_calls", []).append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            })
        elif itype == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "Responses function_call_output requires a non-empty call_id")
            flush_assistant()
            # tool result fed back for a sticky continuation
            msgs.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _flatten_input_content(item.get("output")),
            })
        elif itype == "reasoning":
            if item.get("encrypted_content") is not None:
                raise ValueError(
                    "Responses encrypted reasoning input is not supported")
            text = _reasoning_item_text(item)
            if text:
                turn = assistant()
                turn["reasoning_content"] = (
                    str(turn.get("reasoning_content") or "") + text)
        elif itype == "output_text":
            flush_assistant()
            # sometimes appears as a top-level input item (echoed text)
            msgs.append({"role": _role_or_user(item), "content": _flatten_content(item.get("text"))})
        else:
            raise ValueError(f"unsupported Responses input item type: {itype!r}")
    flush_assistant()
    return msgs


def responses_tools_to_oai_tools(tools: list | None) -> list[dict] | None:
    """Convert Responses-API tools (flat ``{type:"function", name, ...}``) to
    OAI chat tools (``{type:"function", function:{name,...}}``).

    The current engine integration supports function tools only.  Reject other
    tool kinds explicitly instead of silently removing requested capabilities.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("Responses tools must be an array")
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise ValueError("Responses tools must be objects")
        if t.get("type") != "function":
            raise ValueError(
                f"unsupported Responses tool type: {t.get('type')!r}")
        name = t.get("name")
        description = t.get("description", "")
        parameters = t.get(
            "parameters", {"type": "object", "properties": {}})
        if not isinstance(name, str) or not name:
            raise ValueError("Responses function tool requires a name")
        if not isinstance(description, str):
            raise ValueError("Responses function description must be a string")
        if not isinstance(parameters, dict):
            raise ValueError("Responses function parameters must be an object")
        fn = {"name": name, "description": description,
              "parameters": parameters}
        if "strict" in t:
            if not isinstance(t["strict"], bool):
                raise ValueError("Responses function strict must be a boolean")
            fn["strict"] = t["strict"]
        out.append({"type": "function", "function": fn})
    return out or None


# --------------------------------------------------------------------------- chat -> Responses

def chat_to_responses(chat_body: dict, resp_id: str, prev_id: str | None,
                     model: str, diag: dict | None = None,
                     request_payload: dict | None = None,
                     item_layout: list[dict] | None = None,
                     created_at: int | None = None) -> dict:
    """Convert an OAI chat response body to a Responses-API object.

      * assistant content -> an ``output`` item ``{type:"message", role:"assistant",
        content:[{type:"output_text", text}]}``.
      * each ``tool_calls[*]`` -> an ``output`` item ``{type:"function_call",
        name, arguments, call_id}``.
      * ``finish_reason:length`` -> ``status:"incomplete"``; else ``"completed"``.
      * ``usage`` mapped (input/output tokens).
      * proxy diagnostics (mol_*) carried in ``metadata``.
    """
    choices = chat_body.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")

    request_payload = request_payload or {}
    status = ("incomplete" if finish in ("length", "content_filter")
              else "completed")
    item_status = status

    reasoning_text = _flatten_content(
        message.get("reasoning") or message.get("reasoning_content"))
    content_text = _flatten_content(message.get("content"))
    refusal_text = _flatten_content(message.get("refusal"))
    tool_calls = message.get("tool_calls") or []

    reasoning_item = None
    if reasoning_text:
        reasoning_item = {
            "id": _response_item_id(resp_id, "rs", 0),
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": reasoning_text}],
            "status": item_status,
        }

    message_item = None
    message_content = []
    if content_text:
        message_content.append({"type": "output_text", "text": content_text,
                                "annotations": []})
    if refusal_text:
        message_content.append({"type": "refusal", "refusal": refusal_text})
    if message_content or (not reasoning_item and not tool_calls):
        if not message_content:
            message_content.append({"type": "output_text", "text": "",
                                    "annotations": []})
        message_item = {
            "id": _response_item_id(resp_id, "msg", 0),
            "type": "message",
            "role": "assistant",
            "status": item_status,
            "content": message_content,
        }

    tool_items = []
    for ordinal, tc in enumerate(tool_calls):
        fn = (tc or {}).get("function") or {}
        tool_items.append({
            "id": _response_item_id(resp_id, "fc", ordinal),
            "type": "function_call",
            "name": fn.get("name") or "",
            "arguments": fn.get("arguments") or "",
            "call_id": (tc or {}).get("id") or _gen_tool_id(),
            "status": item_status,
        })

    output_by_kind = {
        "reasoning": reasoning_item,
        "message": message_item,
    }
    if item_layout:
        output = []
        for entry in item_layout:
            kind = entry.get("kind")
            if kind == "tool":
                ordinal = int(entry.get(
                    "chat_index", entry.get("tool_ordinal", -1)))
                item = tool_items[ordinal] if 0 <= ordinal < len(tool_items) else None
            else:
                item = output_by_kind.get(kind)
            if item is not None:
                item["id"] = entry["id"]
                if kind == "message" and entry.get("content_order"):
                    parts = {
                        ("text" if part.get("type") == "output_text"
                         else "refusal"): part
                        for part in item.get("content") or []
                    }
                    item["content"] = [parts[part_kind]
                                       for part_kind in entry["content_order"]
                                       if part_kind in parts]
                output.append(item)
    else:
        output = ([reasoning_item] if reasoning_item else [])
        if message_item:
            output.append(message_item)
        output.extend(tool_items)

    usage = chat_body.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at if created_at is not None else _created_at(),
        "model": model,
        "status": status,
        "previous_response_id": prev_id,
        "output": output,
        "parallel_tool_calls": bool(request_payload.get(
            "parallel_tool_calls", True)),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": _response_tools(request_payload.get("tools")),
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {
                "cache_write_tokens": int(
                    prompt_details.get("cache_write_tokens", 0) or 0),
                "cached_tokens": int(
                    prompt_details.get("cached_tokens", 0) or 0),
            },
            "output_tokens": output_tokens,
            "output_tokens_details": {
                "reasoning_tokens": int(
                    completion_details.get("reasoning_tokens", 0) or 0),
            },
            "total_tokens": int(
                usage.get("total_tokens", input_tokens + output_tokens) or 0),
        },
    }
    if status == "incomplete":
        resp["incomplete_details"] = {
            "reason": ("content_filter" if finish == "content_filter"
                       else "max_output_tokens")
        }
    # carry proxy diagnostics into metadata, using the SAME key names as the
    # chat path (``proxy._shape_response``): mol_selected_route /
    # mol_decision / mol_pending_tool_route / mol_convo_completed /
    # mol_summary_recorded / mol_cached_tokens / mol_prompt_tokens. Downstream
    # tooling (e2e tests, m4_kv_reuse) checks these uniformly across endpoints.
    md = _mol_meta(diag or {}, stringify=True)
    cached, ptoks = _proxy._cache_stats(chat_body)
    md["mol_cached_tokens"] = str(cached)
    md["mol_prompt_tokens"] = str(ptoks)
    resp["metadata"] = md
    return resp


# --------------------------------------------------------------------------- store + orchestration

def _fork_state(state: ConvoState) -> ConvoState:
    """Deep-copy a ConvoState for a response-chain branch. Tasks are plain
    dataclasses (deepcopy-able); the threading.Lock is NOT copied (a fresh one
    is created) — copying a lock would either fail or wrongly share it across
    two divergent timelines."""
    new = ConvoState(convo_id=state.convo_id + "_branch", system_msgs=copy.deepcopy(state.system_msgs))
    new.tasks = copy.deepcopy(state.tasks)
    new.pending_tool_route = state.pending_tool_route
    new.pending_tool_call_ids = list(state.pending_tool_call_ids)
    new.pending_tool_generation = state.pending_tool_generation
    new.pending_context_msgs = copy.deepcopy(state.pending_context_msgs)
    return new


def _resolve_state_with_root(
    prev_id: str | None,
) -> tuple[ConvoState | None, str | None, int, str | None]:
    """Resolve the ConvoState for a request.

    Returns ``(state, parent_id, http_status, routing_root_id)``.

      * no ``prev_id`` -> a brand-new ConvoState (start of a new conversation).
      * ``prev_id`` found -> FORK a deep-copy of that response's snapshot (both
        linear and branch continuations fork; the snapshot is immutable so a
        branch never sees a sibling's turn).
      * ``prev_id`` not found -> 404 (None state, status 404).
    """
    if not prev_id:
        return ConvoState(convo_id="resp_root"), None, 200, None
    while True:
        state_to_fork = None
        routing_root_id = None
        with _RESPONSES_LOCK:
            _evict_expired_responses_locked()
            ready = _PENDING_RESPONSES.get(prev_id)
            if ready is None:
                stored = _RESPONSES.get(prev_id)
                if stored is None:
                    return None, None, 404, None
                stored.touched_at = time.monotonic()
                _RESPONSES.move_to_end(prev_id)
                state_to_fork = stored.state
                routing_root_id = stored.routing_root_id
        if state_to_fork is not None:
            return (_fork_state(state_to_fork), prev_id, 200, routing_root_id)
        # A terminal event is currently publishing this response. Wait outside
        # the global mutex, then retry after commit or rollback.
        if not ready.wait(timeout=_PENDING_WAIT_S):
            return None, prev_id, 409, None


def _resolve_state(prev_id: str | None) -> tuple[ConvoState | None, str | None, int]:
    """Backward-compatible state resolver without exposing routing metadata."""
    state, parent_id, status, _ = _resolve_state_with_root(prev_id)
    return state, parent_id, status


def _response_availability(resp_id: str) -> int:
    while True:
        with _RESPONSES_LOCK:
            _evict_expired_responses_locked()
            ready = _PENDING_RESPONSES.get(resp_id)
            if ready is None:
                return 200 if resp_id in _RESPONSES else 404
        if not ready.wait(timeout=_PENDING_WAIT_S):
            return 409


def _preflight_tool_output_error(
        prev_id: str | None, messages: list[dict]) -> dict | None:
    _, last_role, _ = split_system_and_turn(messages)
    if last_role != "tool":
        return None
    state = None
    if prev_id:
        with _RESPONSES_LOCK:
            stored = _RESPONSES.get(prev_id)
            if stored is not None:
                state = stored.state
    if state is None:
        return _proxy._oai_error(
            "No pending tool call exists for this response chain",
            code="orphan_tool_turn", param="input")
    with state.lock:
        return _proxy._validate_pending_tool_results(
            state, messages, param="input")


def preflight_responses_request(payload: dict) -> tuple[int, dict | None]:
    """Validate all errors that must be returned before opening an SSE stream."""
    try:
        _validate_request_capabilities(payload)
        messages = responses_input_to_oai_messages(
            payload.get("input"), payload.get("instructions"))
        responses_tools_to_oai_tools(payload.get("tools"))
    except ValueError as exc:
        return 400, _proxy._oai_error(
            str(exc), code="unsupported_parameter")
    if not messages:
        return 400, _proxy._oai_error(
            "Responses `input` is empty", code="invalid_input", param="input")
    prev_id = payload.get("previous_response_id")
    if prev_id:
        availability = _response_availability(prev_id)
    else:
        availability = 200
    if availability != 200:
        if availability == 409:
            return 409, _proxy._oai_error(
                f"previous_response_id {prev_id!r} is still being committed",
                error_type="conflict", code="response_not_ready",
                param="previous_response_id")
        return 404, _proxy._oai_error(
            f"previous_response_id {prev_id!r} not found",
            error_type="not_found", code="previous_response_not_found",
            param="previous_response_id")
    tool_error = _preflight_tool_output_error(prev_id, messages)
    if tool_error is not None:
        return 400, tool_error
    return 200, None


def _enforce_response_limit_locked() -> None:
    """Evict committed LRU entries to both count and byte budgets."""
    total_bytes = sum(
        stored.approx_bytes for stored in _RESPONSES.values())
    while (_RESPONSES and (
            len(_RESPONSES) > _MAX_RESPONSES
            or total_bytes > _MAX_RESPONSE_STATE_BYTES)):
        _, victim = _RESPONSES.popitem(last=False)
        total_bytes -= victim.approx_bytes


def _evict_expired_responses_locked() -> None:
    now = time.monotonic()
    for key, stored in list(_RESPONSES.items()):
        if now - stored.touched_at > _proxy.CONVO_TTL_S:
            _RESPONSES.pop(key, None)


def _state_approx_bytes(state: ConvoState) -> int:
    payload = {
        "convo_id": state.convo_id,
        "system_msgs": state.system_msgs,
        "tasks": [{
            "owner": task.owner,
            "init_user": task.init_user,
            "msgs": task.msgs,
            "context_msgs": task.context_msgs,
            "summary": task.summary,
            "closed": task.closed,
        } for task in state.tasks],
        "pending_tool_route": state.pending_tool_route,
        "pending_tool_call_ids": state.pending_tool_call_ids,
        "pending_tool_generation": state.pending_tool_generation,
        "pending_context_msgs": state.pending_context_msgs,
    }
    return len(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8"))


def _stored_response(resp_id: str, state: ConvoState,
                     parent_id: str | None, approx_bytes: int,
                     routing_root_id: str | None = None) -> StoredResponse:
    return StoredResponse(
        resp_id=resp_id, state=state, parent_id=parent_id,
        routing_root_id=routing_root_id, approx_bytes=approx_bytes)


def _store(
    resp_id: str,
    state: ConvoState,
    response_obj: dict,
    parent_id: str | None,
    routing_root_id: str | None = None,
) -> None:
    approx_bytes = _state_approx_bytes(state)
    if approx_bytes > _MAX_RESPONSE_STATE_BYTES:
        raise ResponseStoreCapacityError(
            "response state exceeds the configured byte budget")
    with _RESPONSES_LOCK:
        _evict_expired_responses_locked()
        _RESPONSES[resp_id] = _stored_response(
            resp_id, state, parent_id, approx_bytes, routing_root_id)
        _RESPONSES.move_to_end(resp_id)
        _enforce_response_limit_locked()


def _reserve_pending_response(
        resp_id: str, approx_bytes: int, *,
        enforce_state_limit: bool = True) -> threading.Event:
    if enforce_state_limit and approx_bytes > _MAX_RESPONSE_STATE_BYTES:
        raise ResponseStoreCapacityError(
            "response state exceeds the configured byte budget")
    deadline = time.monotonic() + _PENDING_WAIT_S
    with _RESPONSES_CONDITION:
        while True:
            pending_bytes = sum(
                _PENDING_RESPONSE_SIZES.get(key, 0)
                for key in _PENDING_RESPONSES)
            if (len(_PENDING_RESPONSES) < _MAX_PENDING_RESPONSES
                    and pending_bytes + approx_bytes
                    <= _MAX_PENDING_RESPONSE_STATE_BYTES):
                ready = threading.Event()
                _PENDING_RESPONSES[resp_id] = ready
                _PENDING_RESPONSE_SIZES[resp_id] = approx_bytes
                return ready
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResponseStoreCapacityError(
                    "response publication capacity is temporarily exhausted")
            _RESPONSES_CONDITION.wait(timeout=remaining)


def _resize_pending_response(
        resp_id: str, ready: threading.Event, approx_bytes: int, *,
        enforce_state_limit: bool = True) -> None:
    if enforce_state_limit and approx_bytes > _MAX_RESPONSE_STATE_BYTES:
        raise ResponseStoreCapacityError(
            "response state exceeds the configured byte budget")
    deadline = time.monotonic() + _PENDING_WAIT_S
    with _RESPONSES_CONDITION:
        if _PENDING_RESPONSES.get(resp_id) is not ready:
            raise RuntimeError("response publication reservation was lost")
        while True:
            other_bytes = sum(
                _PENDING_RESPONSE_SIZES.get(key, 0)
                for key in _PENDING_RESPONSES if key != resp_id)
            if other_bytes + approx_bytes <= _MAX_PENDING_RESPONSE_STATE_BYTES:
                _PENDING_RESPONSE_SIZES[resp_id] = approx_bytes
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResponseStoreCapacityError(
                    "response publication capacity is temporarily exhausted")
            _RESPONSES_CONDITION.wait(timeout=remaining)


def _release_pending_response(
        resp_id: str, ready: threading.Event) -> None:
    with _RESPONSES_CONDITION:
        current = _PENDING_RESPONSES.get(resp_id)
        if current is ready:
            _PENDING_RESPONSES.pop(resp_id, None)
            _PENDING_RESPONSE_SIZES.pop(resp_id, None)
        ready.set()
        _RESPONSES_CONDITION.notify_all()


def _store_then_emit(resp_id: str, state: ConvoState, response_obj: dict,
                     parent_id: str | None, emit,
                     routing_root_id: str | None = None,
                     pending_ready: threading.Event | None = None) -> None:
    """Publish a stored response before its completed event.

    An in-flight marker lets unrelated conversations continue without holding
    the global store mutex across network I/O. A continuation for this exact
    response waits on the marker, so it can never fork state that is later
    removed after a failed terminal write.
    """
    approx_bytes = _state_approx_bytes(state)
    ready = (pending_ready if pending_ready is not None
             else _reserve_pending_response(resp_id, approx_bytes))
    try:
        if pending_ready is not None:
            _resize_pending_response(resp_id, ready, approx_bytes)
        emit()
    except BaseException:
        _release_pending_response(resp_id, ready)
        raise
    else:
        with _RESPONSES_CONDITION:
            _evict_expired_responses_locked()
            _RESPONSES[resp_id] = _stored_response(
                resp_id, state, parent_id, approx_bytes, routing_root_id)
            _RESPONSES.move_to_end(resp_id)
            _enforce_response_limit_locked()
            _PENDING_RESPONSES.pop(resp_id, None)
            _PENDING_RESPONSE_SIZES.pop(resp_id, None)
            ready.set()
            _RESPONSES_CONDITION.notify_all()


def orchestrate_responses(payload: dict, emit_success=None) -> tuple[int, dict]:
    """Run the 3-hop core on the stateful Responses path.

    Returns (http_status, responses_object). Resolves ``previous_response_id``
    to a ConvoState (shared on a linear chain, forked on a branch), converts the
    Responses input→chat, drives ``_orchestrate_core``, converts the chat
    response→a Responses object, and stores it (if ``store`` != False).

    A tool_call answer leaves the ``ConvoState`` with ``pending_tool_route`` set
    (carried in the shared/forked state); the agent's next
    ``function_call_output`` input becomes a ``tool`` message that the core's
    sticky branch continues — same as the chat path, just across response turns.
    """
    instructions = payload.get("instructions")
    try:
        _validate_request_capabilities(payload)
        oai_messages = responses_input_to_oai_messages(
            payload.get("input"), instructions)
        tools = responses_tools_to_oai_tools(payload.get("tools"))
    except ValueError as exc:
        return 400, _proxy._oai_error(
            str(exc), code="unsupported_parameter")
    if not oai_messages:
        return 400, _proxy._oai_error(
            "Responses `input` is empty", code="invalid_input", param="input")
    prev_id = payload.get("previous_response_id")
    state, parent_id, rstatus, parent_root = _resolve_state_with_root(prev_id)
    if rstatus != 200:
        if rstatus == 409:
            return 409, _proxy._oai_error(
                f"previous_response_id {prev_id!r} is still being committed",
                error_type="conflict", code="response_not_ready",
                param="previous_response_id")
        return 404, _proxy._oai_error(
            f"previous_response_id {prev_id!r} not found",
            error_type="not_found", code="previous_response_not_found",
            param="previous_response_id")
    # adopt the (possibly updated) system head onto the resolved state so the
    # own-view prefix stays byte-stable (only replaces on genuine content change).
    system_msgs, last_role, last_text = split_system_and_turn(oai_messages)
    _proxy._adopt_system(state, system_msgs, clear_empty=True)
    if last_role == "tool":
        validation_error = _proxy._validate_pending_tool_results(
            state, oai_messages, param="input")
        if validation_error is not None:
            return 400, validation_error

    # max_output_tokens (Responses field) -> optional Engine max_tokens
    mt = _proxy._clamp_max_tokens({"max_tokens": payload.get("max_output_tokens")})
    answer_options = _proxy._responses_answer_options(payload)

    resp_id = "resp_" + uuid.uuid4().hex[:24]
    model = payload.get("model") or _proxy.SERVED_MODEL_NAME
    routing_root = parent_root or resp_id
    stored = False

    try:
        with _proxy._routing_scope(routing_root):
            with state.lock:
                if last_role != "tool":
                    state.stage_external_history(
                        oai_messages[len(system_msgs):-1])
                if answer_options:
                    status, chat_body, diag = _proxy._orchestrate_core(
                        state, oai_messages, last_role, last_text, mt, resp_id, tools,
                        answer_options=answer_options)
                else:
                    status, chat_body, diag = _proxy._orchestrate_core(
                        state, oai_messages, last_role, last_text, mt, resp_id, tools)

        if status != 200:
            if diag.get("error") == "context_length_exceeded":
                return _proxy._public_upstream_status(status), chat_body
            # propagate engine error (e.g. pool-miss fallback exhausted) as a
            # Responses-shaped error object
            return _proxy._public_upstream_status(status), _proxy._oai_error(
                "answer hop failed", error_type="server_error",
                code="answer_hop_failed")

        responses_obj = chat_to_responses(
            chat_body, resp_id, prev_id, model, diag, request_payload=payload)

        store = payload.get("store", True)
        if store is not False:
            try:
                if emit_success is None:
                    _store(resp_id, state, responses_obj, parent_id,
                           routing_root)
                else:
                    _store_then_emit(
                        resp_id, state, responses_obj, parent_id,
                        lambda: emit_success(responses_obj),
                        routing_root_id=routing_root)
                stored = True
            except ResponseStoreCapacityError:
                return 503, _proxy._oai_error(
                    "The response state store is at capacity",
                    error_type="server_error", code="response_store_capacity")
        elif emit_success is not None:
            emit_success(responses_obj)
        return 200, responses_obj
    finally:
        if parent_root is None and not stored:
            _proxy._release_routing_key(routing_root)


def _mol_meta(diag: dict, *, stringify: bool = False) -> dict:
    """Map proxy diag → mol_* metadata keys. Shared by the non-streaming
    ``chat_to_responses`` and the streaming ``response.created``/``completed``
    events so the two stay in sync. Mirrors ``proxy._shape_response`` key names."""
    md = {}
    if isinstance(diag, dict):
        if "route" in diag:
            md["mol_selected_route"] = diag["route"]
        if "decision" in diag:
            md["mol_decision"] = diag["decision"]
        if "pending_tool_route" in diag:
            md["mol_pending_tool_route"] = diag["pending_tool_route"]
        if "n_completed" in diag:
            md["mol_convo_completed"] = diag["n_completed"]
        if "summary" in diag:
            md["mol_summary_recorded"] = diag["summary"]
    if stringify:
        return {key: _metadata_value(value) for key, value in md.items()
                if value is not None}
    return md


def orchestrate_responses_stream(payload: dict, write) -> int:
    """Streaming Responses path. Mirrors ``orchestrate_responses`` but streams the
    answer hop as a Responses SSE event lifecycle (response.created → in_progress
    → output_item.added → content_part.added → output_text.delta ×N → …done →
    output_item.done → response.completed). The router + summary hops are
    out-of-band (client-invisible). On client disconnect / mid-stream engine
    error the open task is discarded and ``response.failed`` is emitted.

    ``write`` is a callable(bytes_frame) (the handler's ``_sse_write``). Returns
    the final http status (always 200 for SSE; pre-errors become failed events)."""
    prev_id = payload.get("previous_response_id")
    model = payload.get("model") or _proxy.SERVED_MODEL_NAME
    state, parent_id, rstatus, parent_root = _resolve_state_with_root(prev_id)
    if rstatus != 200:
        _write_failed_event(
            write, "resp_unknown", model, prev_id, payload,
            f"previous_response_id {prev_id!r} not found", 0,
            error_code="invalid_prompt")
        return 200

    instructions = payload.get("instructions")
    try:
        _validate_request_capabilities(payload)
        oai_messages = responses_input_to_oai_messages(
            payload.get("input"), instructions)
        tools = responses_tools_to_oai_tools(payload.get("tools"))
    except ValueError as exc:
        _write_failed_event(
            write, "resp_unknown", model, prev_id, payload, str(exc), 0,
            error_code="invalid_prompt")
        return 200
    if not oai_messages:
        _write_failed_event(
            write, "resp_unknown", model, prev_id, payload, "input is empty", 0,
            error_code="invalid_prompt")
        return 200
    system_msgs, last_role, last_text = split_system_and_turn(oai_messages)
    _proxy._adopt_system(state, system_msgs, clear_empty=True)
    mt = _proxy._clamp_max_tokens({"max_tokens": payload.get("max_output_tokens")})
    answer_options = _proxy._responses_answer_options(payload)

    resp_id = "resp_" + uuid.uuid4().hex[:24]
    routing_root = parent_root or resp_id

    diag = {}
    stream_cb = _make_responses_cb(
        write, resp_id, model, prev_id, payload)
    should_store = payload.get("store", True) is not False
    pending_ready = None
    stored = False
    if should_store:
        try:
            # Publish the ID as in-flight before response.created and account
            # for the forked state immediately. The per-response hard limit is
            # enforced at terminal publication after generation finishes.
            pending_ready = _reserve_pending_response(
                resp_id, _state_approx_bytes(state),
                enforce_state_limit=False)
        except ResponseStoreCapacityError:
            stream_cb(("meta_post", {"error": "response_store_capacity"}))
            return 200
    try:
        with _proxy._routing_scope(routing_root):
            with state.lock:
                if last_role != "tool":
                    state.stage_external_history(
                        oai_messages[len(system_msgs):-1])
                if pending_ready is not None:
                    try:
                        _resize_pending_response(
                            resp_id, pending_ready, _state_approx_bytes(state),
                            enforce_state_limit=False)
                    except ResponseStoreCapacityError:
                        stream_cb(("meta_post", {
                            "error": "response_store_capacity"}))
                        return 200
                try:
                    if answer_options:
                        status, chat_body, diag = _proxy._orchestrate_core(
                            state, oai_messages, last_role, last_text, mt, resp_id,
                            tools, stream_cb=stream_cb,
                            answer_options=answer_options)
                    else:
                        status, chat_body, diag = _proxy._orchestrate_core(
                            state, oai_messages, last_role, last_text, mt, resp_id,
                            tools, stream_cb=stream_cb)
                finally:
                    if (state.open_task() is not None
                            and state.pending_tool_route is None):
                        state.discard_open_task()

        if status == 200:
            resp_obj = chat_to_responses(
                chat_body, resp_id, prev_id, model, diag,
                request_payload=payload, item_layout=stream_cb.item_layout(),
                created_at=stream_cb.created_at)
            if should_store:
                try:
                    _store_then_emit(
                        resp_id, state, resp_obj, parent_id,
                        lambda: stream_cb.emit_completed(resp_obj),
                        routing_root_id=routing_root,
                        pending_ready=pending_ready)
                    stored = True
                except ResponseStoreCapacityError:
                    stream_cb.emit_failed(
                        "The response state store is at capacity",
                        "response_store_capacity", resp_obj)
            else:
                stream_cb.emit_completed(resp_obj)
    finally:
        if pending_ready is not None:
            _release_pending_response(resp_id, pending_ready)
        if parent_root is None and not stored:
            _proxy._release_routing_key(routing_root)
    return 200


def _make_responses_cb(write, resp_id, model, prev_id,
                       request_payload: dict | None = None):
    """Build a schema-complete Responses SSE adapter for answer-hop chunks."""
    request_payload = request_payload or {}
    seq = {"n": 0}
    created_at = _created_at()
    state_ = {"layout": [], "reasoning": None, "message": None,
              "tools": {}, "created_sent": False, "metadata": {}}

    def nxt():
        current = seq["n"]
        seq["n"] += 1
        return current

    def send(event_type: str, **fields) -> None:
        event = {"type": event_type, "sequence_number": nxt(), **fields}
        write(_proxy._sse_frame(event, event=event_type))

    def add_layout(kind: str, item_id: str, **extra) -> int:
        output_index = len(state_["layout"])
        state_["layout"].append({"kind": kind, "id": item_id,
                                 "output_index": output_index, **extra})
        return output_index

    def emit_created(metadata: dict | None = None) -> None:
        if state_["created_sent"]:
            return
        skel = _response_skeleton(
            resp_id, model, prev_id, "in_progress", request_payload,
            created_at=created_at, metadata=metadata)
        send("response.created", response=skel)
        state_["created_sent"] = True

    def ensure_reasoning() -> dict:
        item = state_["reasoning"]
        if item is None:
            item_id = _response_item_id(resp_id, "rs", 0)
            item = {"id": item_id,
                    "output_index": add_layout("reasoning", item_id),
                    "parts": [], "part_added": False}
            state_["reasoning"] = item
            send("response.output_item.added",
                 output_index=item["output_index"],
                 item={"id": item_id, "type": "reasoning", "summary": [],
                       "content": [], "status": "in_progress"})
        return item

    def ensure_message() -> dict:
        item = state_["message"]
        if item is None:
            item_id = _response_item_id(resp_id, "msg", 0)
            item = {"id": item_id,
                    "output_index": add_layout("message", item_id),
                    "parts": [], "part_by_kind": {}}
            state_["message"] = item
            send("response.output_item.added",
                 output_index=item["output_index"],
                 item={"id": item_id, "type": "message", "role": "assistant",
                       "status": "in_progress", "content": []})
        return item

    def ensure_message_part(kind: str) -> tuple[dict, dict]:
        message = ensure_message()
        part = message["part_by_kind"].get(kind)
        if part is None:
            content_index = len(message["parts"])
            part = {"kind": kind, "content_index": content_index, "parts": []}
            message["part_by_kind"][kind] = part
            message["parts"].append(part)
            state_["layout"][message["output_index"]].setdefault(
                "content_order", []).append(kind)
            wire_part = ({"type": "output_text", "text": "", "annotations": []}
                         if kind == "text"
                         else {"type": "refusal", "refusal": ""})
            send("response.content_part.added",
                 output_index=message["output_index"],
                 content_index=content_index, item_id=message["id"],
                 part=wire_part)
        return message, part

    def ensure_tool(chat_index: int, tc: dict, fn: dict) -> dict:
        item = state_["tools"].get(chat_index)
        if item is None:
            ordinal = len(state_["tools"])
            item_id = _response_item_id(resp_id, "fc", ordinal)
            call_id = tc.get("id") or _gen_tool_id()
            item = {"id": item_id, "call_id": call_id,
                    "name": fn.get("name") or "", "args": "",
                    "tool_ordinal": ordinal,
                    "output_index": add_layout(
                        "tool", item_id, tool_ordinal=ordinal,
                        chat_index=chat_index)}
            state_["tools"][chat_index] = item
            send("response.output_item.added",
                 output_index=item["output_index"],
                 item={"id": item_id, "type": "function_call",
                       "call_id": call_id, "name": item["name"],
                       "arguments": "", "status": "in_progress"})
        if tc.get("id"):
            item["call_id"] = tc["id"]
        if fn.get("name"):
            item["name"] = fn["name"]
        return item

    def close_items(finish_reason: str | None) -> None:
        item_status = ("incomplete" if finish_reason in
                       ("length", "content_filter") else "completed")
        if not state_["layout"]:
            ensure_message_part("text")
        reasoning = state_["reasoning"]
        if reasoning is not None:
            text = "".join(reasoning["parts"])
            if reasoning["part_added"]:
                send("response.reasoning_text.done",
                     output_index=reasoning["output_index"], content_index=0,
                     item_id=reasoning["id"], text=text)
                send("response.content_part.done",
                     output_index=reasoning["output_index"], content_index=0,
                     item_id=reasoning["id"],
                     part={"type": "reasoning_text", "text": text})
            send("response.output_item.done",
                 output_index=reasoning["output_index"],
                 item={"id": reasoning["id"], "type": "reasoning",
                       "summary": [],
                       "content": ([{"type": "reasoning_text", "text": text}]
                                   if reasoning["part_added"] else []),
                       "status": item_status})

        message = state_["message"]
        if message is not None:
            content = []
            for part in message["parts"]:
                full = "".join(part["parts"])
                if part["kind"] == "text":
                    send("response.output_text.done",
                         output_index=message["output_index"],
                         content_index=part["content_index"],
                         item_id=message["id"], text=full, logprobs=[])
                    wire_part = {"type": "output_text", "text": full,
                                 "annotations": []}
                else:
                    send("response.refusal.done",
                         output_index=message["output_index"],
                         content_index=part["content_index"],
                         item_id=message["id"], refusal=full)
                    wire_part = {"type": "refusal", "refusal": full}
                send("response.content_part.done",
                     output_index=message["output_index"],
                     content_index=part["content_index"],
                     item_id=message["id"], part=wire_part)
                content.append(wire_part)
            send("response.output_item.done",
                 output_index=message["output_index"],
                 item={"id": message["id"], "type": "message",
                       "role": "assistant", "status": item_status,
                       "content": content})

        for item in sorted(state_["tools"].values(),
                           key=lambda value: value["output_index"]):
            send("response.function_call_arguments.done",
                 output_index=item["output_index"], item_id=item["id"],
                 name=item["name"], arguments=item["args"])
            send("response.output_item.done",
                 output_index=item["output_index"],
                 item={"id": item["id"], "type": "function_call",
                       "call_id": item["call_id"], "name": item["name"],
                       "arguments": item["args"], "status": item_status})

    def cb(ev):
        kind, data = ev
        if kind == "meta_pre":
            metadata = _mol_meta(data, stringify=True)
            state_["metadata"] = metadata
            emit_created(metadata)
            skel = _response_skeleton(
                resp_id, model, prev_id, "in_progress", request_payload,
                created_at=created_at, metadata=metadata)
            send("response.in_progress", response=skel)
            return
        if kind == "chunk":
            choice = ((data.get("choices") or [{}])[0]
                      if isinstance(data, dict) else {})
            delta = choice.get("delta") or {}
            reasoning_delta = delta.get("reasoning")
            if reasoning_delta is None:
                reasoning_delta = delta.get("reasoning_content")
            if isinstance(reasoning_delta, str) and reasoning_delta:
                item = ensure_reasoning()
                if not item["part_added"]:
                    send("response.content_part.added",
                         output_index=item["output_index"], content_index=0,
                         item_id=item["id"],
                         part={"type": "reasoning_text", "text": ""})
                    item["part_added"] = True
                item["parts"].append(reasoning_delta)
                send("response.reasoning_text.delta",
                     output_index=item["output_index"], content_index=0,
                     item_id=item["id"], delta=reasoning_delta)
            content_delta = delta.get("content")
            if isinstance(content_delta, str) and content_delta:
                message, part = ensure_message_part("text")
                part["parts"].append(content_delta)
                send("response.output_text.delta",
                     output_index=message["output_index"],
                     content_index=part["content_index"],
                     item_id=message["id"], delta=content_delta, logprobs=[])
            refusal_delta = delta.get("refusal")
            if isinstance(refusal_delta, str) and refusal_delta:
                message, part = ensure_message_part("refusal")
                part["parts"].append(refusal_delta)
                send("response.refusal.delta",
                     output_index=message["output_index"],
                     content_index=part["content_index"],
                     item_id=message["id"], delta=refusal_delta)
            for tc in (delta.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                chat_index = tc.get("index")
                fn = tc.get("function")
                if (not isinstance(chat_index, int)
                        or isinstance(chat_index, bool) or chat_index < 0
                        or not isinstance(fn, dict)):
                    continue
                item = ensure_tool(chat_index, tc, fn)
                args = fn.get("arguments")
                if isinstance(args, str) and args:
                    item["args"] += args
                    send("response.function_call_arguments.delta",
                         output_index=item["output_index"],
                         item_id=item["id"], delta=args)
            return
        if kind == "meta_post":
            if isinstance(data, dict) and "error" in data:
                emit_created()
                error_body = data.get("error_body")
                error = (error_body.get("error")
                         if isinstance(error_body, dict) else None)
                if isinstance(error, dict):
                    message = str(error.get("message") or data.get("error"))
                    error_code = str(error.get("code") or "server_error")
                else:
                    message = str(data.get("error") or "answer hop failed")
                    error_code = "server_error"
                _write_failed_event(
                    write, resp_id, model, prev_id, request_payload,
                    message, nxt(), error_code=error_code, created_at=created_at,
                    metadata=state_["metadata"])
                return
            close_items(data.get("finish_reason") if isinstance(data, dict)
                        else None)

    def emit_completed(resp_obj):
        event_type = ("response.incomplete" if resp_obj.get("status") ==
                      "incomplete" else "response.completed")
        send(event_type, response=resp_obj)

    def emit_failed(
            message: str, code: str, response_obj: dict | None = None) -> None:
        if response_obj is None:
            response = _response_skeleton(
                resp_id, model, prev_id, "failed", request_payload,
                created_at=created_at, metadata=state_["metadata"])
        else:
            response = copy.deepcopy(response_obj)
            response.pop("incomplete_details", None)
            response["status"] = "failed"
        response["error"] = {"code": code, "message": message}
        send("response.failed", response=response)

    cb.emit_completed = emit_completed
    cb.emit_failed = emit_failed
    cb.item_layout = lambda: copy.deepcopy(state_["layout"])
    cb.created_at = created_at
    return cb


# --------------------------------------------------------------------------- small helpers

def _validate_request_capabilities(payload: dict) -> None:
    prev_id = payload.get("previous_response_id")
    if prev_id is not None and not isinstance(prev_id, str):
        raise ValueError("Responses `previous_response_id` must be a string")
    instructions = payload.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses `instructions` must be a string")
    for key in ("stream", "store", "parallel_tool_calls"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"Responses `{key}` must be a boolean")
    reasoning = payload.get("reasoning")
    effort = None
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise ValueError("Responses `reasoning` must be an object")
        unsupported = sorted(set(reasoning) - {"effort"})
        if unsupported:
            raise ValueError(
                "unsupported Responses reasoning option(s): "
                + ", ".join(unsupported))
        effort = _proxy._validate_reasoning_effort(
            reasoning.get("effort"), "Responses")
    _proxy._validate_chat_template_kwargs(payload, "Responses")
    if effort is not None and payload.get("chat_template_kwargs") is not None:
        raise ValueError(
            "Responses `reasoning.effort` cannot be combined with "
            "`chat_template_kwargs`")
    raw_tools = payload.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        raise ValueError("Responses tools must be an array")
    declared_names = {
        tool.get("name") for tool in (raw_tools or [])
        if isinstance(tool, dict) and tool.get("type") == "function"
        and isinstance(tool.get("name"), str) and tool.get("name")
    }
    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, str):
        if tool_choice not in ("none", "auto", "required"):
            raise ValueError("unsupported Responses tool_choice")
        if tool_choice == "required" and not declared_names:
            raise ValueError(
                "Responses `tool_choice` required needs declared tools")
    elif tool_choice is not None:
        name = (tool_choice.get("name")
                if isinstance(tool_choice, dict) else None)
        if (not isinstance(tool_choice, dict)
                or tool_choice.get("type") != "function"
                or not isinstance(name, str)
                or name not in declared_names):
            raise ValueError("invalid Responses function tool_choice")
    if payload.get("include"):
        raise ValueError("Responses `include` is not supported by this proxy")


def _metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        import json as _json
        return _json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _response_tools(tools: Any) -> list[dict]:
    """Return function tools in the schema used by a Response object."""
    if not isinstance(tools, list):
        return []
    out = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        normalized = {
            "type": "function",
            "name": tool.get("name") or "",
            "description": tool.get("description"),
            "parameters": tool.get("parameters"),
            "strict": tool.get("strict"),
        }
        out.append(normalized)
    return out


def _response_skeleton(resp_id: str, model: str, prev_id: str | None,
                       status: str, request_payload: dict,
                       *, created_at: int | None = None,
                       metadata: dict | None = None,
                       error: dict | None = None) -> dict:
    response = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at if created_at is not None else _created_at(),
        "model": model,
        "status": status,
        "previous_response_id": prev_id,
        "output": [],
        "parallel_tool_calls": bool(request_payload.get(
            "parallel_tool_calls", True)),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": _response_tools(request_payload.get("tools")),
        "metadata": metadata or {},
    }
    if error is not None:
        response["error"] = error
    return response


def _write_failed_event(write, resp_id: str, model: str,
                        prev_id: str | None, request_payload: dict,
                        message: str, sequence_number: int, *,
                        error_code: str = "server_error",
                        created_at: int | None = None,
                        metadata: dict | None = None) -> None:
    response = _response_skeleton(
        resp_id, model, prev_id, "failed", request_payload,
        created_at=created_at, metadata=metadata,
        error={"code": error_code, "message": message})
    event = {"type": "response.failed", "sequence_number": sequence_number,
             "response": response}
    write(_proxy._sse_frame(event, event="response.failed"))


def _response_item_id(resp_id: str, prefix: str, ordinal: int) -> str:
    digest = uuid.uuid5(
        uuid.NAMESPACE_URL, f"mol:{resp_id}:{prefix}:{ordinal}").hex[:24]
    return f"{prefix}_{digest}"


def _reasoning_item_text(item: dict) -> str:
    content = _flatten_content(item.get("content"))
    if content:
        return content
    return _flatten_content(item.get("summary"))


def _assistant_input_content(content: Any) -> tuple[str, str]:
    if content is None or isinstance(content, str):
        return content or "", ""
    if not isinstance(content, list):
        raise ValueError(
            f"unsupported assistant content value: {type(content).__name__}")
    text_parts = []
    refusal_parts = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type in (None, "text", "input_text", "output_text"):
            text_parts.append(str(part.get("text") or ""))
        elif part_type == "refusal":
            refusal_parts.append(str(part.get("refusal") or ""))
        else:
            raise ValueError(
                f"unsupported Responses content block type: {part_type!r}")
    return "".join(text_parts), "".join(refusal_parts)


def _flatten_input_content(content: Any) -> str:
    """Flatten text input blocks and reject unsupported multimodal blocks."""
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type not in (None, "text", "input_text", "output_text"):
                raise ValueError(
                    f"unsupported Responses content block type: {part_type!r}")
            parts.append(str(part.get("text") or ""))
        return "".join(parts)
    raise ValueError(
        f"unsupported Responses content value: {type(content).__name__}")


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("output_text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _role_or_user(item: dict) -> str:
    role = item.get("role") or "user"
    return "system" if role == "developer" else role


def _gen_tool_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


def _created_at() -> int:
    # local import to keep module-import side-effect-free in odd test envs
    import time as _t
    return int(_t.time())


__all__ = [
    "orchestrate_responses",
    "responses_input_to_oai_messages",
    "responses_tools_to_oai_tools",
    "chat_to_responses",
    "StoredResponse",
]
