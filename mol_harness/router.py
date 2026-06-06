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
        entry_route_id: str = "L0",
    ) -> None:
        if entry_route_id not in tasks:
            raise ValueError(f"Unknown entry_route_id={entry_route_id}")
        self.tasks = tasks
        self.entry_route_id = entry_route_id

    def router_prompt(self, user_text: str) -> str:
        return (
            "User request:\n"
            f"{user_text.strip()}\n\n"
            f"{self.router_instruction()}"
        )

    def router_instruction(self) -> str:
        examples = self._routing_examples()
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
            f"{self._model_id_list()}\n\n"
            "Routing rules:\n"
            f"{self._routing_rules()}\n\n"
            f"{examples_block}"
            "Return exactly one line in this format and nothing else:\n"
            f"model_id=<{self._model_id_choices()}>\n\n"
            "model_id="
        )

    def parse_router_output(self, text: str) -> str | None:
        lower = text.lower()
        aliases = {self.entry_route_id.lower(): self.entry_route_id}
        for task_id, task in self.tasks.items():
            aliases[task_id.lower()] = task_id
            if task.adapter_name:
                aliases[task.adapter_name.lower()] = task_id
        match = re.search(r"model_id\s*=\s*([A-Za-z0-9_.:/-]+)", lower)
        if match:
            return aliases.get(match.group(1).strip().strip("`'\".,;:()[]{}<>").lower())
        stripped = lower.strip().strip("`'\".,;:()[]{}<>")
        if stripped in aliases:
            return aliases[stripped]
        mentions = []
        for alias, route_id in aliases.items():
            pattern = rf"(?<![A-Za-z0-9_.:/-]){re.escape(alias)}(?![A-Za-z0-9_.:/-])"
            if re.search(pattern, lower):
                mentions.append(route_id)
        unique_mentions = list(dict.fromkeys(mentions))
        if len(unique_mentions) == 1:
            return unique_mentions[0]
        return None

    def route_by_library(self, user_text: str) -> RouterDecision:
        candidates: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        for task_id, task in self.tasks.items():
            if task_id == self.entry_route_id:
                continue
            negative_hits = self._signal_hits(user_text, task.negative_signals)
            strong_hits = self._signal_hits(user_text, task.strong_signals)
            positive_hits = self._signal_hits(user_text, task.positive_signals)
            diagnostics[task_id] = {
                "negative_hits": negative_hits[:8],
                "strong_hits": strong_hits[:8],
                "positive_hits": positive_hits[:8],
            }
            if not strong_hits and not positive_hits:
                continue
            strong_score = self._weighted_signal_score(strong_hits)
            positive_score = self._weighted_signal_score(positive_hits)
            negative_score = self._weighted_signal_score(negative_hits)
            if strong_score == 0 and positive_score < 2:
                continue
            score = task.priority + strong_score * 100 + positive_score * 10 - negative_score * 250
            if score <= 0:
                continue
            candidates.append(
                {
                    "task_id": task_id,
                    "score": score,
                    "strong_count": len(strong_hits),
                    "strong_score": strong_score,
                    "positive_score": positive_score,
                    "negative_score": negative_score,
                    "strong_hits": strong_hits[:8],
                    "positive_hits": positive_hits[:8],
                    "negative_hits": negative_hits[:8],
                }
            )

        if not candidates:
            return self._decision(self.entry_route_id, "default_entry_lora", diagnostics=diagnostics)

        candidates.sort(key=lambda item: (item["score"], item["strong_count"]), reverse=True)
        top = candidates[0]
        if self._is_general_l0_request(user_text) and top["strong_score"] <= 1 and top["positive_score"] == 0:
            return self._decision(
                self.entry_route_id,
                "general_entry_lora_guard",
                diagnostics={"candidates": candidates, "signals": diagnostics},
            )
        if len(candidates) == 1 or top["score"] > candidates[1]["score"]:
            return self._decision(
                top["task_id"],
                "specialist_signal",
                diagnostics={"selected": top, "candidates": candidates, "signals": diagnostics},
            )
        return self._decision(
            self.entry_route_id,
            "ambiguous_specialist_signal_default_entry_lora",
            diagnostics={"candidates": candidates, "signals": diagnostics},
        )

    def apply_guardrail(self, raw_route: str | None, user_text: str) -> RouterDecision:
        library_decision = self.route_by_library(user_text)
        if library_decision.route_id != self.entry_route_id:
            library_decision.raw_model_route = raw_route
            return library_decision
        if raw_route in self.tasks:
            return self._decision(raw_route, "model_route", raw_model_route=raw_route)
        library_decision.raw_model_route = raw_route
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

    def _model_id_list(self) -> str:
        return "\n".join(
            f"- {task_id}: {task.description}"
            for task_id, task in self.tasks.items()
        )

    def _model_id_choices(self) -> str:
        return "|".join(self.tasks)

    def _routing_rules(self) -> str:
        rules: list[str] = []
        for task_id, task in self.tasks.items():
            for rule in task.routing_rules:
                rules.append(f"{task_id}: {rule}")
        return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))

    def _routing_examples(self) -> str:
        lines: list[str] = []
        for task_id, task in self.tasks.items():
            for example in task.examples:
                if "=>" in example:
                    user, route = example.split("=>", 1)
                    route = route.strip() or task_id
                else:
                    user, route = example, task_id
                lines.append(f"User: {user.strip()}\nmodel_id={route}")
        return "\n".join(lines)

    @staticmethod
    def _signal_matches(text: str, signal: str) -> bool:
        signal = signal.strip().lower()
        if not signal:
            return False
        if re.fullmatch(r"[a-z0-9_+-]+", signal):
            return re.search(rf"(?<![a-z0-9_+-]){re.escape(signal)}(?![a-z0-9_+-])", text) is not None
        return signal in text

    @classmethod
    def _signal_hits(cls, user_text: str, signals: list[str]) -> list[str]:
        lower = user_text.lower()
        return [signal for signal in signals if cls._signal_matches(lower, signal)]

    @staticmethod
    def _signal_weight(signal: str) -> int:
        signal = signal.strip()
        if not signal:
            return 0
        if any(ord(ch) > 127 for ch in signal):
            if len(signal) >= 6:
                return 4
            if len(signal) >= 4:
                return 3
        if len(signal) >= 40:
            return 6
        if len(signal) >= 20:
            return 4
        if any(ch in signal for ch in ("/", "_", "-", ":", "`", "*")):
            return 4
        if " " in signal:
            return 3
        return 1

    @classmethod
    def _weighted_signal_score(cls, hits: list[str]) -> int:
        return sum(cls._signal_weight(hit) for hit in hits)

    @staticmethod
    def _is_general_l0_request(user_text: str) -> bool:
        lower = user_text.strip().lower()
        return lower.startswith(
            (
                "translate ",
                "rewrite ",
                "proofread ",
                "explain ",
                "summarize ",
                "compare ",
                "what is ",
                "why does ",
                "give me an overview",
            )
        )
