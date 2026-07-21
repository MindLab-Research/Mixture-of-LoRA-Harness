"""MoL conversation state and the Proxy-authoritative timeline.

The Proxy owns the append-only conversation timeline and reconstructs, for
each Engine hop, the target LoRA's own view of that history:

  * the target LoRA's own past task turns are kept **verbatim** (full trace),
  * every *other* LoRA's past task turns are collapsed to a 1-2 sentence summary,
  * the current user turn is appended verbatim.

Because the timeline is append-only and every own-view is rendered
deterministically, when a LoRA is re-entered its previously-seen prefix is
byte-identical to last time → the engine's native (LoRA-aware) prefix cache
hits and only the newly-appended tail is prefilled. Per-LoRA KV reuse is an
*emergent property* of stable per-LoRA prompts — no engine KV-reuse patch is
required.

A "task" is one routing decision: a user turn → route → (possibly several
tool-call rounds within the same LoRA) → final answer → summary. Tool calls do
not re-route: they extend the open task's trace in place, and
``pending_tool_route`` locks the next tool-result turn to the same LoRA.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
import threading
from typing import Any, Protocol

# OAI message: {"role": str, "content": str | list, ...}


@dataclass
class Task:
    """One routed task segment in the conversation timeline.

    Attributes:
        owner: route id of the specialist that owns this task (e.g. "L1").
        init_user: the user message text that initiated this task.
        msgs: the in-flight OAI messages produced *after* ``init_user`` while
            the task is open — specialist assistant answers, tool calls, and
            tool results. Once the task closes, this is the specialist's full
            trace (its own view keeps it verbatim; other LoRAs see only the
            summary).
        summary: 1-2 sentence summary, set when the task closes.
        closed: once True the task is immutable history.
    """

    owner: str
    init_user: str
    msgs: list[dict] = field(default_factory=list)
    context_msgs: list[dict] = field(default_factory=list)
    summary: str = ""
    closed: bool = False

    def trace_text(self) -> str:
        """Flatten the specialist's full trace to plain text (own-view verbatim)."""
        parts = [_msg_text(message) for message in self.context_msgs]
        parts.append(self.init_user)
        for m in self.msgs:
            parts.append(_msg_text(m))
        return "\n".join(p for p in parts if p)


@dataclass
class ConvoState:
    """Per-conversation proxy state (in-memory, append-only timeline).

    The proxy is the single source of truth for the timeline. The agent's
    inbound OAI message history is used only to extract (a) the leading
    system/tool-def messages and (b) the current turn's text; all prior task
    content is rebuilt from this state so own-views stay stable across turns.
    """

    convo_id: str
    system_msgs: list[dict] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    # Set when the last answer hop ended in a tool_call: the next tool-result
    # turn is locked to this LoRA (sticky), skipping routing + summary.
    pending_tool_route: str | None = None
    pending_tool_call_ids: list[str] = field(default_factory=list)
    pending_tool_generation: int = 0
    pending_context_msgs: list[dict] = field(default_factory=list)
    # Per-conversation lock — the proxy holds it across one full orchestration
    # so two concurrent requests on the same conversation (agent retry, dup)
    # serialize instead of racing on `tasks`/`pending_tool_route`.
    lock: threading.Lock = field(default_factory=threading.Lock)
    async_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False)

    # -- timeline queries -------------------------------------------------

    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.closed]

    def open_task(self) -> Task | None:
        if self.tasks and not self.tasks[-1].closed:
            return self.tasks[-1]
        return None

    # -- own-view construction (the per-LoRA prefix) ----------------------

    def own_view_messages(self, target_route: str, current_turn: list[dict] | None = None) -> list[dict]:
        """Build the OAI message list ``target_route`` sees right now.

        ``current_turn`` is the messages to append for the *current* hop —
        either ``[{"role": "user", "content": <new query>}]`` for a fresh
        routed turn, or the open task's continued messages
        (assistant tool_call + tool result) for a sticky tool-result turn.
        Rendered after the stable prefix so the engine prefills only this tail.

        Completed tasks render as:
          * own (owner == target_route): user turn + full trace messages verbatim,
          * other: user turn + a single assistant message carrying the summary.

        The open (in-flight) task renders only for its own owner, as its
        accumulated messages; for any other target it is absent (we never route
        to a different LoRA while a task is open — a stray new user turn first
        closes the open task via :meth:`close_open_task`).
        """
        msgs: list[dict] = [dict(m) for m in self.system_msgs]
        for t in self.tasks:
            if t.closed:
                if t.owner == target_route:
                    # own full trace, verbatim — this is the KV-reuse prefix
                    msgs.extend(copy.deepcopy(t.context_msgs))
                    msgs.append({"role": "user", "content": t.init_user})
                    msgs.extend(t.msgs)
                else:
                    msgs.append({"role": "user", "content": t.init_user})
                    msgs.append({
                        "role": "assistant",
                        "content": _summary_block(t),
                    })
            else:
                # open task — only relevant when we are continuing it (target
                # == owner). For a fresh route the open task is closed first.
                if t.owner == target_route:
                    msgs.extend(copy.deepcopy(t.context_msgs))
                    msgs.append({"role": "user", "content": t.init_user})
                    msgs.extend(t.msgs)
        if current_turn:
            msgs.extend(current_turn)
        return msgs

    def summary_context_messages(self, task: Task) -> list[dict]:
        """View for the summary hop: the just-finished task's full trace, plus
        a summary scaffold user message. Pinned to ``task.owner`` so the
        specialist that did the work summarizes it (it can see its own trace).

        The summary scaffold is NEVER fed into any own-view (only its captured
        text enters the timeline as ``task.summary``), so it cannot pollute a
        LoRA's reusable prefix (``mol_deployment_structure.md`` §3).
        """
        msgs: list[dict] = [dict(m) for m in self.system_msgs]
        # the task's own full trace, in context
        msgs.extend(copy.deepcopy(task.context_msgs))
        msgs.append({"role": "user", "content": task.init_user})
        msgs.extend(task.msgs)
        msgs.append({"role": "user", "content": SUMMARY_SCAFFOLD})
        return msgs

    # -- timeline mutations ----------------------------------------------

    def begin_task(self, owner: str, user_text: str) -> Task:
        """Open a new task for a freshly routed user turn. Any prior open task
        (abandoned tool loop) is force-closed first."""
        self.close_open_task(summarize=False)
        task = Task(
            owner=owner, init_user=user_text,
            context_msgs=copy.deepcopy(self.pending_context_msgs))
        self.pending_context_msgs = []
        self.tasks.append(task)
        self.pending_tool_route = None
        self.pending_tool_call_ids = []
        return task

    def append_assistant(self, message: dict) -> None:
        """Record a specialist assistant message on the open task (answer or
        tool_call)."""
        t = self.open_task()
        if t is None:
            raise RuntimeError("append_assistant with no open task")
        t.msgs.append(dict(message))

    def append_tool_result(self, message: dict) -> None:
        t = self.open_task()
        if t is None:
            raise RuntimeError("append_tool_result with no open task")
        t.msgs.append(dict(message))

    def close_open_task(self, summarize: bool, summary: str = "") -> Task | None:
        """Close the open task. If ``summarize`` is True, store the captured
        summary text; otherwise close with an empty summary (abandoned tool
        loop). Returns the closed task, or None if there was none."""
        t = self.open_task()
        if t is None:
            return None
        t.closed = True
        t.summary = summary if summarize else ""
        self.pending_tool_route = None
        self.pending_tool_call_ids = []
        return t

    def set_pending_tool_route(
            self, route: str, tool_call_ids: list[str] | None = None) -> None:
        self.pending_tool_route = route
        self.pending_tool_call_ids = list(
            tool_call_ids if tool_call_ids is not None
            else _last_tool_call_ids(self.open_task()))
        self.pending_tool_generation += 1

    def discard_open_task(self) -> None:
        """Drop the just-opened failed task (engine pool-miss 400 retry). Used
        by the shared core's pool-miss fallback so the discarded task never
        leaves a phantom empty segment in the timeline (a duplicate user turn
        + a false summary block in every other LoRA's own-view). Returns None
        if there was no open task."""
        if self.tasks and not self.tasks[-1].closed:
            discarded = self.tasks.pop()
            self.pending_context_msgs = copy.deepcopy(discarded.context_msgs)
        self.pending_tool_route = None
        self.pending_tool_call_ids = []

    def stream_checkpoint(self) -> dict[str, Any]:
        """Snapshot mutable conversation state before a streamed request.

        A client disconnect can happen after the answer has mutated the open
        task but before the terminal SSE frame is delivered. The shared core
        restores this checkpoint on any failed/aborted streamed turn.
        """
        return {
            "system_msgs": copy.deepcopy(self.system_msgs),
            "tasks": copy.deepcopy(self.tasks),
            "pending_tool_route": self.pending_tool_route,
            "pending_tool_call_ids": list(self.pending_tool_call_ids),
            "pending_tool_generation": self.pending_tool_generation,
            "pending_context_msgs": copy.deepcopy(self.pending_context_msgs),
        }

    def restore_stream_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.system_msgs = checkpoint["system_msgs"]
        self.tasks = checkpoint["tasks"]
        self.pending_tool_route = checkpoint["pending_tool_route"]
        self.pending_tool_call_ids = checkpoint["pending_tool_call_ids"]
        self.pending_tool_generation = checkpoint["pending_tool_generation"]
        self.pending_context_msgs = checkpoint["pending_context_msgs"]

    # -- OrchestrationState interface -------------------------------------
    # The shared 3-hop core (proxy._orchestrate_core) drives any state model
    # through this narrow interface. The stateful ConvoState stores a finished
    # task's summary directly on the Task (own_view_messages reads task.summary),
    # so should_summarize is unconditional and record_summary/set_request_messages
    # are no-ops. The stateless variant (stateless.StatelessSideContext) overrides
    # these to keep its append-only side-context of summaries + agent messages.

    def completed_count(self) -> int:
        return len(self.completed_tasks())

    def should_summarize(self, route: str) -> bool:
        # Stateful timeline: ALWAYS summarize a finished task — the summary is
        # folded into every OTHER LoRA's own-view (own_view_messages reads
        # task.summary), giving the router cross-task memory next turn.
        return True

    def record_summary(self, task: Task) -> None:
        # Stateful: summary already lives on task.summary (set by the core); the
        # timeline owns it directly. Nothing extra to record.
        return

    def set_request_messages(self, messages: list[dict]) -> None:
        # Stateful: the proxy-authoritative timeline ignores the agent's resent
        # history (only the system head + current turn are used). No-op.
        return

    def stage_external_history(self, messages: list[dict]) -> None:
        self.pending_context_msgs = copy.deepcopy(messages)

    # -- introspection ----------------------------------------------------

    # The router sees only the raw current query. Cross-task summaries are
    # supplied to the selected specialist, not to the route hop.

    def describe(self) -> dict[str, Any]:
        return {
            "convo_id": self.convo_id,
            "n_tasks": len(self.tasks),
            "n_completed": len(self.completed_tasks()),
            "pending_tool_route": self.pending_tool_route,
            "pending_tool_call_ids": list(self.pending_tool_call_ids),
            "pending_context_msgs": copy.deepcopy(self.pending_context_msgs),
            "owners": [t.owner for t in self.tasks],
        }


class OrchestrationState(Protocol):
    """The narrow interface the shared 3-hop core (route→answer→summary) uses
    to drive conversation state, so BOTH state models plug into the same loop:

      * ``ConvoState`` — the stateful proxy-authoritative timeline. Used by
        ``/v1/responses`` (proxy owns the store; agent sends ``input`` +
        ``previous_response_id``). Own-views are rebuilt from ``tasks``.
      * ``stateless.StatelessSideContext`` — rebuilt from the agent-resent
        message history each request. Used by ``/v1/chat/completions``
        Own-views are rebuilt via ``build_own_view`` using recorded summary
        boundaries.

    The two own-view builders are NOT bit-identical across a specialist switch
    (two different state models — see dual-endpoint plan risk R1), but routing
    is query-only and identical on both. Documented, not a parity bug.
    """

    lock: threading.Lock
    async_lock: asyncio.Lock
    pending_tool_route: str | None
    pending_tool_call_ids: list[str]
    pending_tool_generation: int

    def completed_count(self) -> int: ...
    def open_task(self) -> Task | None: ...
    def begin_task(self, owner: str, user_text: str) -> Task: ...
    def append_assistant(self, message: dict) -> None: ...
    def append_tool_result(self, message: dict) -> None: ...
    def close_open_task(self, summarize: bool, summary: str = "") -> Task | None: ...
    def discard_open_task(self) -> None: ...
    def set_pending_tool_route(
            self, route: str, tool_call_ids: list[str] | None = None) -> None: ...
    def should_summarize(self, route: str) -> bool: ...
    def own_view_messages(self, target_route: str, current_turn: list[dict] | None = None) -> list[dict]: ...
    def summary_context_messages(self, task: Task) -> list[dict]: ...
    def record_summary(self, task: Task) -> None: ...
    def set_request_messages(self, messages: list[dict]) -> None: ...
    def stage_external_history(self, messages: list[dict]) -> None: ...
    def stream_checkpoint(self) -> Any: ...
    def restore_stream_checkpoint(self, checkpoint: Any) -> None: ...


def _adopt_system(
        state: Any, system_msgs: list[dict], *, clear_empty: bool = False) -> None:
    """Adopt the latest system head onto a state object ONLY when its content
    actually changed. Works for any state model exposing a ``system_msgs``
    attribute (stateful ``ConvoState`` and stateless ``StatelessSideContext``).
    The system_msgs are the stable head of every own-view prefix; a blind
    replace on every request would mutate the prefix and silently break native
    prefix-cache reuse on re-entry (``mol_per_lora_kv_orchestration.md`` §1
    invariant). Shared by the stateful convo registry (proxy._get_convo), the
    Responses-path store, and the stateless side-context (set_request_messages)."""
    if not system_msgs:
        if clear_empty:
            state.system_msgs = []
        return
    cur = [(m.get("role"), m.get("content")) for m in state.system_msgs]
    new = [(m.get("role"), m.get("content")) for m in system_msgs]
    if cur != new:
        state.system_msgs = list(system_msgs)


SUMMARY_SCAFFOLD = (
    "请用一到两句话总结你刚刚完成了什么，供路由器在用户下一次请求时判断下一步。"
    "只输出总结本身，不要输出任何其他内容。"
)


def _summary_block(task: Task) -> str:
    """Render a cross-task segment as a natural prior-assistant context block.

    A prose block keeps the target specialist from treating a route marker as
    part of its answer. Empty summaries remain empty.
    """
    return f"（前面的回合已由其他助手处理，结果摘要：{(task.summary or '').strip()}）"


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return " ".join(out)
    return str(c or "")


# -- inbound message helpers (extract current turn from the agent request) ----

def split_system_and_turn(messages: list[dict]) -> tuple[list[dict], str, str]:
    """Split an inbound OAI message list into (system_msgs, last_role, last_text).

    Leading ``system``/``developer`` messages are the stable head
    kept verbatim across hops. The last message is the current turn: a new
    ``user`` turn (route) or a ``tool``/``assistant`` turn (sticky continuation).
    """
    head = 0
    while head < len(messages) and _role(messages[head]) in (
            "system", "developer"):
        head += 1
    system_msgs = [messages[i] for i in range(head)] if head else []
    last = messages[-1] if messages else {}
    role = _role(last)
    text = _msg_text(last)
    return system_msgs, role, text


def extract_tool_results(messages: list[dict]) -> list[dict]:
    """Pull the trailing ``tool`` messages off an inbound request (the tool
    results the agent is feeding back for a sticky continuation)."""
    out = []
    for m in reversed(messages):
        if _role(m) == "tool":
            out.append(m)
        else:
            break
    out.reverse()
    return out


def _last_tool_call_ids(task: Task | None) -> list[str]:
    if task is None:
        return []
    for message in reversed(task.msgs):
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(tool_calls, list):
            continue
        return [call["id"] for call in tool_calls
                if isinstance(call, dict)
                and isinstance(call.get("id"), str) and call["id"]]
    return []


def _role(m: dict) -> str:
    return (m.get("role") if isinstance(m, dict) else "") or ""
