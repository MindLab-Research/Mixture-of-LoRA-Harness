from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .library import LoRATask


@dataclass
class RouterDecision:
    route_id: str
    adapter_name: str
    decision: str
    raw_model_route: str | None = None
    diagnostics: dict[str, Any] | None = None


class RouterHarness:
    def __init__(
        self,
        tasks: dict[str, LoRATask],
        route_prompt: str,
        entry_route_id: str = "L0",
    ) -> None:
        if entry_route_id not in tasks:
            raise ValueError(f"Unknown entry_route_id={entry_route_id}")
        self.tasks = tasks
        self.route_prompt = route_prompt.strip()
        self.entry_route_id = entry_route_id

    def router_prompt(self, user_text: str) -> str:
        # The evaluated route hop sees only the raw current request followed by
        # this instruction. The trailing stem is moved into an assistant prefill
        # by proxy._route_chat.
        return f"{user_text.strip()}\n\n{self._router_instruction()}"

    def _router_instruction(self) -> str:
        labels = " ".join(task_id for task_id, _ in self._sorted_tasks())
        return (
            self.route_prompt
            .replace("{{LORA_DESCRIPTIONS}}", self._compact_definitions())
            .replace("{{ROUTE_LABELS}}", labels)
        )

    def router_instruction(self) -> str:
        return self._router_instruction()

    def _compact_definitions(self) -> str:
        lines: list[str] = []
        for task_id, task in self._sorted_tasks():
            definition = str(
                task.description or task.router_line or task.summary or ""
            ).strip()
            definition = " ".join(definition.split())
            lines.append(f"{task_id} = {definition}")
        return "\n".join(lines)

    def _sorted_tasks(self) -> list[tuple[str, LoRATask]]:
        return sorted(self.tasks.items())

    def parse_canonical_output(self, text: str) -> str | None:
        """Accept only a complete canonical model label."""
        match = _CANONICAL_ROUTE_RE.fullmatch(text or "")
        if not match:
            return None
        route = match.group(1).upper()
        return route if route in self.tasks else None

    def parse_router_output(self, text: str) -> str | None:
        # Accept the three canonical forms emitted by the route hop.
        # The proxy calls this with "model_id=" + completion (so ROUTE_RE fires
        # on the prefill+continuation), then with the bare completion (BARE/DIGIT).
        text = text or ""
        match = _ROUTE_RE.search(text)
        if match:
            return match.group(1).upper()
        match = _BARE_ROUTE_RE.search(text)
        if match:
            return match.group(0).upper()
        match = _DIGIT_ROUTE_RE.search(text)
        if match:
            return f"L{match.group(1)}"
        return None

    def route_by_library(self, user_text: str) -> RouterDecision:
        # Legacy library mode uses count-based scoring
        # (100*strong + 10*pos - 250*neg), priority is a TIEBREAKER only (not a
        # baseline — a zero-hit specialist must not beat the L0 default), drop
        # net-non-positive evidence, general-L0-prefix guard, ambiguity -> L0.
        if self._is_general_l0_request(user_text):
            return self._decision(self.entry_route_id, "general_l0_prefix")
        text_lower = user_text.lower()
        scores: dict[str, dict[str, Any]] = {}
        best_route = self.entry_route_id
        best_key = (0, -1)  # (signal_score, priority); L0 wins until beaten
        for task_id, task in self._sorted_tasks():
            if task_id == self.entry_route_id:
                continue
            strong_hits = [s for s in task.strong_signals if self._signal_matches(text_lower, s)]
            positive_hits = [s for s in task.positive_signals if self._signal_matches(text_lower, s)]
            negative_hits = [s for s in task.negative_signals if self._signal_matches(text_lower, s)]
            signal_score = 100 * len(strong_hits) + 10 * len(positive_hits) - 250 * len(negative_hits)
            priority = task.priority
            scores[task_id] = {
                "score": signal_score, "priority": priority,
                "strong_hits": strong_hits, "positive_hits": positive_hits,
                "negative_hits": negative_hits,
            }
            if signal_score <= 0:
                continue
            key = (signal_score, priority)
            if key > best_key:
                best_key = key
                best_route = task_id
        if best_route == self.entry_route_id:
            return self._decision(self.entry_route_id, "default_entry_lora",
                                  diagnostics={"scores": scores, "best_score": best_key[0]})
        return self._decision(best_route, "weighted_library",
                              raw_model_route=None,
                              diagnostics={"scores": scores, "best_score": best_key[0]})

    def apply_guardrail(self, raw_route: str | None, user_text: str) -> RouterDecision:
        # Legacy guardrail mode: a non-L0 library match WINS over the model route;
        # only when
        # the library stays on L0 do we accept the model's route.
        library_decision = self.route_by_library(user_text)
        library_decision.raw_model_route = raw_route
        if library_decision.route_id != self.entry_route_id:
            return library_decision
        if raw_route in self.tasks:
            return self._decision(raw_route, "model_route", raw_model_route=raw_route)
        return library_decision

    def route_to_adapter(self) -> dict[str, str]:
        return {
            task_id: task.adapter_name
            for task_id, task in self.tasks.items()
            if task.adapter_name
        }

    def _decision(
        self,
        route_id: str,
        decision: str,
        raw_model_route: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RouterDecision:
        return RouterDecision(
            route_id=route_id,
            adapter_name=self.tasks[route_id].adapter_name,
            decision=decision,
            raw_model_route=raw_model_route,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _signal_matches(text_lower: str, signal: str) -> bool:
        # Use word boundaries for ASCII tokens and substring matching for
        # phrases and CJK signals.
        signal = signal.strip().lower()
        if not signal:
            return False
        if re.fullmatch(r"[a-z0-9_+-]+", signal):
            return re.search(rf"(?<![a-z0-9_+-]){re.escape(signal)}(?![a-z0-9_+-])", text_lower) is not None
        return signal in text_lower

    @staticmethod
    def _is_general_l0_request(user_text: str) -> bool:
        return user_text.strip().lower().startswith(_GENERAL_L0_PREFIXES)


# Canonical route-output patterns (prefixed, bare, and digit-only forms).
_ROUTE_RE = re.compile(r"model_id\s*=\s*(L\d+)\b", re.IGNORECASE)
_BARE_ROUTE_RE = re.compile(r"\bL\d+\b", re.IGNORECASE)
_DIGIT_ROUTE_RE = re.compile(r"^\s*(\d+)\b")
_CANONICAL_ROUTE_RE = re.compile(r"\s*(L\d+)\s*", re.IGNORECASE)

_GENERAL_L0_PREFIXES = (
    "translate ", "rewrite ", "proofread ", "explain ", "summarize ",
    "compare ", "what is ", "why does ", "give me an overview",
)
