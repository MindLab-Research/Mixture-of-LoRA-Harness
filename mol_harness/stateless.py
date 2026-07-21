"""Stateless state model for Chat Completions.

The dual-endpoint Proxy exposes two client-facing
endpoints with **independent state models**, both internally calling the engine
via ``/v1/chat/completions``:

  * ``/v1/chat/completions`` → **stateless** (this module): the agent resends
    the FULL message history each request (standard OAI client behavior). The
    proxy rebuilds the target specialist's *own view* from that resent history
    by folding segments handled by *other* specialists into summaries — a
    deterministic history partition. The side context (summaries) is keyed by a stable
    conversation fingerprint, so it survives across the agent's turn-by-turn
    requests.

  * ``/v1/responses`` → **stateful** (``session.ConvoState``): the proxy owns
    the timeline; the agent sends ``input`` + ``previous_response_id``.

``StatelessSideContext`` implements ``session.OrchestrationState`` so the shared
3-hop core (``proxy._orchestrate_core``) drives it with the same code path that
drives the stateful ``ConvoState``. Routing is query-only and identical on
both; the two own-view builders are not bit-identical across a specialist
switch because they use different state models, not because of a parity
bug.

Design note — why an ephemeral ``_current`` Task:
  The shared core records the specialist's answer via ``append_assistant`` and
  later hands the closed task to ``_summary_hop`` → ``summary_context_messages``.
  The stateful path stores the answer on ``Task.msgs``; the summary hop reads
  it back from there. To reuse the SAME core unchanged, the stateless path
  keeps a request-scoped ``_current`` Task whose ``msgs`` holds this request's
  produced assistant answer(s) (and, for sticky tool turns, the engine's
  answer after a tool result). The agent-resent history lives in
  ``agent_messages``; the own-view is rebuilt from that source of truth,
  not from ``_current``. ``_current`` is only consulted by the summary hop.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
from typing import Any

from .session import (
    Task,
    SUMMARY_SCAFFOLD,
    _adopt_system,
    _last_tool_call_ids,
)

# The router entry route (L0). Summaries are recorded for every non-entry
# specialist's finished segment.
ENTRY_ROUTE = "L0"


def _msg_role(msg: dict) -> str:
    return (msg.get("role") if isinstance(msg, dict) else "") or ""


def _content_to_text(content: Any) -> str:
    """Flatten an OAI message content (str | list[part]) to plain text.
    Normalize an OAI content value to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return " ".join(out)
    return str(content or "")


def _last_user_text(messages: list[dict]) -> str:
    """The current-turn user query (last user message in the agent history).
    Return the current-turn user query from the agent history."""
    for msg in reversed(messages or []):
        if _msg_role(msg) == "user":
            return _content_to_text(msg.get("content")).strip()
    return ""


def convo_key(messages: list[dict]) -> str:
    """Fingerprint the conversation by its FIRST user message so tool-turns of
    the same conversation map to the same sticky side-context slot."""
    first_user = ""
    for msg in messages or []:
        if isinstance(msg, dict) and _msg_role(msg) == "user":
            first_user = _content_to_text(msg.get("content"))[:200]
            break
    return hashlib.sha1(first_user.encode("utf-8", "replace")).hexdigest()[:16]


def build_own_view(messages: list[dict], summaries: list[dict], cur_route: str | None) -> list[dict]:
    """Rebuild the message list the CURRENT specialist (``cur_route``) sees.

    The own-view policy:

      * leading ``system`` messages (incl. tool defs) stay verbatim (head);
      * each recorded summary carries ``{route, end_idx}``; the ``end_idx``
        values partition the post-head history into contiguous segments;
      * a segment whose ``route == cur_route`` is kept **verbatim** (own KV
        prefix reuse — the specialist re-reads its own raw trace);
      * every other specialist's segment is folded to one assistant message
        carrying the natural prior-context summary block;
      * the not-yet-summarized tail (everything after the last ``end_idx``) is
        always kept verbatim — it is the current specialist's in-progress work.

    If nothing was folded (no cross-specialist segment, or stale/out-of-range
    boundaries), return the original list unchanged (zero cost on the common
    all-one-specialist path).

    Stale/out-of-range ``end_idx`` (``<= prev`` or ``> len(msgs)``) skips
    folding that segment — defensive against a reordered/trimmed agent history
    boundaries are skipped defensively.
    """
    msgs = list(messages or [])
    if not summaries:
        return msgs

    head = 0
    while head < len(msgs) and _msg_role(msgs[head]) in (
            "system", "developer"):
        head += 1

    own = list(msgs[:head])
    prev = head
    folded_any = False
    for sm in summaries:
        end = sm.get("end_idx")
        if not isinstance(end, int) or end <= prev or end > len(msgs):
            continue
        if sm.get("route") == cur_route:
            own += msgs[prev:end]
        else:
            own.append({
                "role": "assistant",
                "content": _summary_block_str(sm.get("summary")),
            })
            folded_any = True
        prev = end

    own += msgs[prev:]
    if not folded_any:
        return msgs
    return own


def _summary_block_str(summary: Any) -> str:
    """The cross-specialist fold form — matches ``session._summary_block`` and
    a natural prior-assistant context block, not a bare route marker."""
    return f"（前面的回合已由其他助手处理，结果摘要：{str(summary or '').strip()}）"


def _system_head(messages: list[dict]) -> list[dict]:
    """Leading system/tool-def messages — the stable head kept verbatim.
    Keep the leading system and tool-definition messages stable."""
    head = 0
    while head < len(messages) and _msg_role(messages[head]) in (
            "system", "developer"):
        head += 1
    return [messages[i] for i in range(head)] if head else []


def _sanitize_tool_transactions(messages: list[dict]) -> list[dict]:
    """Remove incomplete tool transactions from a view sent upstream."""
    rendered: list[dict] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        tool_calls = message.get("tool_calls")
        if (message.get("role") == "assistant"
                and "tool_calls" in message):
            end = index + 1
            while (end < len(messages)
                   and isinstance(messages[end], dict)
                   and messages[end].get("role") == "tool"):
                end += 1
            expected = []
            calls_valid = isinstance(tool_calls, list) and bool(tool_calls)
            for call in (tool_calls if isinstance(tool_calls, list) else []):
                function = call.get("function") if isinstance(call, dict) else None
                if (not isinstance(call, dict)
                        or not isinstance(call.get("id"), str)
                        or not call.get("id")
                        or call.get("type") != "function"
                        or not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or not function.get("name")
                        or not isinstance(function.get("arguments"), str)):
                    calls_valid = False
                    continue
                expected.append(call["id"])
            received = [messages[pos].get("tool_call_id")
                        for pos in range(index + 1, end)]
            complete = (
                calls_valid
                and len(expected) == len(tool_calls)
                and len(expected) == len(set(expected))
                and all(isinstance(call_id, str) and call_id
                        for call_id in received)
                and len(received) == len(set(received))
                and set(received) == set(expected)
            )
            if complete:
                rendered.extend(dict(messages[pos])
                                for pos in range(index, end))
            else:
                assistant = dict(message)
                assistant.pop("tool_calls", None)
                assistant.pop("function_call", None)
                if not isinstance(assistant.get("content"), str):
                    assistant["content"] = ""
                rendered.append(assistant)
            index = end
            continue
        if message.get("role") != "tool":
            rendered.append(dict(message))
        index += 1
    return rendered


class StatelessSideContext:
    """Per-conversation stateless side context — implements
    ``session.OrchestrationState``.

    Stores an append-only list of segment
    summaries (``{route, adapter, summary, end_idx}``) keyed across requests
    by the stable ``convo_key``, plus the sticky ``pending_tool_route``.

    Unlike the stateful ``ConvoState``, this context holds NO authoritative
    timeline — the agent resends the full message history each request. The
    own-view is rebuilt each hop from ``agent_messages`` + ``summaries`` via
    :func:`build_own_view`. ``agent_messages`` is set fresh every request via
    :meth:`set_request_messages` (the core calls it under the lock before the
    3-hop loop), so it is request-scoped; ``summaries`` persists across
    requests.

    ``_current`` (request-scoped ``Task``): holds this request's produced
    assistant answer(s) on ``.msgs`` so the summary hop can read them back,
    mirroring the stateful path. It is NOT the source of truth for own-views
    (``agent_messages`` is). Lazy-created by :meth:`open_task` on a sticky
    tool-result turn so the core's orphan-guard
    (``pending_tool_route and open_task() is None``) passes (risk R3).

    Tool stickiness (risk R3): tool results arrive in ``agent_messages`` (the
    agent resends them); :meth:`append_tool_result` is a no-op because
    :func:`build_own_view` renders the tail verbatim already. The specialist's
    answer (after the tool result) IS recorded on ``_current.msgs`` via
    :meth:`append_assistant`, so the summary hop sees the full tool loop.
    """

    # Attributes declared at class level for documentation; each instance sets
    # them in __init__. `lock` and `pending_tool_route` satisfy the
    # OrchestrationState Protocol's declared attributes.
    lock: threading.Lock
    async_lock: asyncio.Lock
    pending_tool_route: str | None

    def __init__(self, convo_key_id: str, system_msgs: list[dict] | None = None) -> None:
        self.convo_key_id = convo_key_id
        self.system_msgs: list[dict] = list(system_msgs or [])
        # append-only across the whole conversation (keyed by the stable
        # first-user fingerprint), so segment summaries from earlier turns
        # survive to inform the router/specialist on later turns.
        self.summaries: list[dict] = []
        self.pending_tool_route: str | None = None
        self.pending_tool_call_ids: list[str] = []
        self.pending_tool_generation: int = 0
        # request-scoped: the FULL agent-resent message history for the current
        # request. Set under the lock each request before the 3-hop loop.
        self.agent_messages: list[dict] = []
        # request-scoped ephemeral task holding this request's produced
        # assistant answer(s). See class docstring.
        self._current: Task | None = None
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()

    # -- OrchestrationState interface -------------------------------------

    def completed_count(self) -> int:
        return len(self.summaries)

    def open_task(self) -> Task | None:
        if self._current is not None:
            return self._current
        # Sticky tool-result turn arriving with no _current yet (the previous
        # request set pending_tool_route, ended, and this request reopened):
        # lazily create the ephemeral task so the core's sticky guard
        # (``pending_tool_route and open_task() is None``) passes (risk R3).
        if self.pending_tool_route is not None:
            self._current = Task(owner=self.pending_tool_route, init_user="")
            return self._current
        return None

    def begin_task(self, owner: str, user_text: str) -> Task:
        # Stateles: no authoritative task timeline — the own-view is rebuilt
        # from agent_messages. We create a fresh request-scoped _current
        # (clears any pending sticky route: a fresh user turn
        # abandons any open tool loop). Its init_user is unused (the agent's
        # resent history carries the user turn); it exists to receive the
        # specialist's produced answer for the summary hop.
        self.pending_tool_route = None
        self.pending_tool_call_ids = []
        self._current = Task(owner=owner, init_user=user_text)
        return self._current

    def append_assistant(self, message: dict) -> None:
        # Record the specialist's produced answer on _current.msgs so the
        # summary hop can read it back (mirrors the stateful path). The agent
        # will also resend this answer next request via agent_messages.
        if self._current is None:
            self._current = Task(owner="", init_user="")
        self._current.msgs.append(dict(message))

    def append_tool_result(self, message: dict) -> None:
        # Stateles: tool results are already in the agent's resent history
        # (agent_messages); build_own_view renders the tail verbatim, so they
        # reach the specialist without us touching the timeline. No-op.
        return

    def close_open_task(self, summarize: bool, summary: str = "") -> Task | None:
        # Close _current so its owner + produced answer are available to the
        # summary hop (the core reads task.msgs). Clearing _current + pending
        # matches the stateful path's close semantics.
        t = self._current
        if t is None:
            return None
        t.closed = True
        t.summary = summary if summarize else ""
        self._current = None
        self.pending_tool_route = None
        self.pending_tool_call_ids = []
        return t

    def discard_open_task(self) -> None:
        """Drop the just-opened failed task (pool-miss 400 retry). Stateful
        ConvoState pops ``tasks[-1]``; stateless just drops ``_current``.
        Never leaves a phantom segment (no summary is recorded for it)."""
        self._current = None
        if self.pending_tool_route is None:
            self.pending_tool_call_ids = []

    def stream_checkpoint(self) -> dict[str, Any]:
        """Snapshot persistent side-context before a streamed request."""
        return {
            "system_msgs": copy.deepcopy(self.system_msgs),
            "summaries": copy.deepcopy(self.summaries),
            "pending_tool_route": self.pending_tool_route,
            "pending_tool_call_ids": list(self.pending_tool_call_ids),
            "pending_tool_generation": self.pending_tool_generation,
            "agent_messages": copy.deepcopy(self.agent_messages),
            "current": copy.deepcopy(self._current),
        }

    def restore_stream_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.system_msgs = checkpoint["system_msgs"]
        self.summaries = checkpoint["summaries"]
        self.pending_tool_route = checkpoint["pending_tool_route"]
        self.pending_tool_call_ids = checkpoint["pending_tool_call_ids"]
        self.pending_tool_generation = checkpoint["pending_tool_generation"]
        self.agent_messages = checkpoint["agent_messages"]
        self._current = checkpoint["current"]

    def set_pending_tool_route(
            self, route: str, tool_call_ids: list[str] | None = None) -> None:
        self.pending_tool_route = route
        self.pending_tool_call_ids = list(
            tool_call_ids if tool_call_ids is not None
            else _last_tool_call_ids(self.open_task()))
        self.pending_tool_generation += 1

    def should_summarize(self, route: str) -> bool:
        # Summarize a finished segment only when the specialist was NOT the
        # entry router (L0 summaries would
        # just restate the user query — no cross-task handoff value).
        return route != ENTRY_ROUTE

    def own_view_messages(self, target_route: str, current_turn: list[dict] | None = None) -> list[dict]:
        # Fold other specialists' segments to summaries,
        # keep the target's own segments + the in-progress tail verbatim. The
        # agent's resent history IS the source of truth here (NOT _current).
        msgs = list(self.agent_messages)
        if current_turn:
            msgs = msgs + list(current_turn)
        return _sanitize_tool_transactions(
            build_own_view(msgs, self.summaries, target_route))

    def summary_context_messages(self, task: Task) -> list[dict]:
        # The summary input is the agent's
        # resent history (which ends in the current user turn — and, for a
        # sticky tool turn, the resent tool result) + the just-produced
        # specialist answer (task.msgs) + the summary scaffold. Pinned to the
        # task's owner (it summarized its own work). The scaffold never enters
        # any own-view (only its captured text is filed via record_summary).
        msgs = list(self.agent_messages)
        if task is not None:
            msgs = msgs + list(task.msgs)
        msgs = _sanitize_tool_transactions(msgs)
        msgs.append({"role": "user", "content": SUMMARY_SCAFFOLD})
        return msgs

    def record_summary(self, task: Task) -> None:
        # Append {route, adapter, summary, end_idx=len(messages)} so the NEXT
        # request's build_own_view can partition the resent history. end_idx is
        # the length of THIS request's agent_messages — everything up to here
        # belongs to this finished segment; the next turn's resent history
        # grows beyond it.
        if task is None or task.owner == ENTRY_ROUTE:
            return
        self.summaries.append({
            "route": task.owner,
            "adapter": task.owner,  # adapter name resolved by the proxy caller
            "summary": task.summary,
            "end_idx": len(self.agent_messages),
        })

    def set_request_messages(self, messages: list[dict]) -> None:
        # Stateles: the agent resent the FULL history. Adopt it (and the
        # system head) as this request's source of truth before the 3-hop loop.
        self.agent_messages = copy.deepcopy(list(messages or []))
        _adopt_system(self, _system_head(messages))
        # CRITICAL: clear the prior request's _current. The side context is
        # PERSISTENT across requests (keyed by convo_key), but _current is
        # request-scoped — it holds only THIS request's produced assistant
        # answer(s) for the summary hop. If a prior request left _current open
        # (a tool_call answer set pending_tool_route and returned before close),
        # its msgs would persist and the next request's summary hop would
        # duplicate the prior tool_call (once in agent_messages, resent; once
        # in the stale _current.msgs). pending_tool_route is PRESERVED so the
        # sticky-guard (open_task) lazily creates a fresh ephemeral _current.
        self._current = None

    def stage_external_history(self, messages: list[dict]) -> None:
        return

    # -- introspection ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "convo_key": self.convo_key_id,
            "n_summaries": len(self.summaries),
            "pending_tool_route": self.pending_tool_route,
            "pending_tool_call_ids": list(self.pending_tool_call_ids),
            "summarized_routes": [s["route"] for s in self.summaries],
        }


__all__ = [
    "StatelessSideContext",
    "convo_key",
    "build_own_view",
    "ENTRY_ROUTE",
]
