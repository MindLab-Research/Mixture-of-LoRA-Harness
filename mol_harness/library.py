from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


@dataclass
class LoRATask:
    id: str
    task: str
    adapter_name: str
    source_path: str = ""
    level: str = ""
    priority: int = 0
    description: str = ""
    summary: str = ""
    router_line: str = ""
    routing_rules: list[str] = field(default_factory=list)
    strong_signals: list[str] = field(default_factory=list)
    positive_signals: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    config_path: str = ""

    @property
    def is_entry(self) -> bool:
        return self.level.upper() == "L0" or self.id.upper() == "L0"


_ROUTE_HEADING_RE = re.compile(r"^##\s+(L\d+)\s*$", re.IGNORECASE)
_ADAPTER_RE = re.compile(r"^adapter_name:\s*(\S+)\s*$")


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return text.split(". ", 1)[0][:200].strip()


def _route_blocks(text: str, path: Path) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    route_id: str | None = None
    lines: list[str] = []
    for raw_line in text.splitlines():
        match = _ROUTE_HEADING_RE.fullmatch(raw_line.strip())
        if match:
            if route_id is not None:
                blocks.append((route_id, lines))
            route_id = match.group(1).upper()
            lines = []
        elif route_id is not None:
            lines.append(raw_line.rstrip())
    if route_id is not None:
        blocks.append((route_id, lines))
    if not blocks:
        raise ValueError(f"No LoRA sections found in {path}; expected headings such as '## L0'")
    return blocks


def _parse_route(route_id: str, lines: list[str], path: Path) -> LoRATask:
    adapter_name = ""
    description_lines: list[str] = []
    in_description = False
    for raw_line in lines:
        line = raw_line.strip()
        adapter_match = _ADAPTER_RE.fullmatch(line)
        if adapter_match and not in_description:
            adapter_name = adapter_match.group(1)
            continue
        if line == "### Description":
            in_description = True
            continue
        if line.startswith("### ") and in_description:
            in_description = False
            continue
        if in_description:
            description_lines.append(raw_line.rstrip())

    description = "\n".join(description_lines).strip()
    if not adapter_name:
        raise ValueError(f"Missing adapter_name for {route_id} in {path}")
    if not description:
        raise ValueError(f"Missing Description for {route_id} in {path}")
    return LoRATask(
        id=route_id,
        task=route_id,
        level=route_id,
        adapter_name=adapter_name,
        description=description,
        summary=_first_sentence(description),
        router_line=description,
        config_path=str(path),
    )


def load_lora_definitions(lora_file: str | Path) -> dict[str, LoRATask]:
    path = Path(lora_file)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LoRA definition file: {path}")
    text = path.read_text(encoding="utf-8")
    tasks: dict[str, LoRATask] = {}
    for route_id, lines in _route_blocks(text, path):
        if route_id in tasks:
            raise ValueError(f"Duplicate LoRA route {route_id} in {path}")
        tasks[route_id] = _parse_route(route_id, lines, path)
    return dict(sorted(tasks.items()))


def library_to_jsonable(tasks: dict[str, LoRATask]) -> dict[str, Any]:
    return {
        "tasks": {
            task_id: task.__dict__
            for task_id, task in tasks.items()
        }
    }
