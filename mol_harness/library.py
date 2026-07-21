from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    library_path: str = ""

    @property
    def is_entry(self) -> bool:
        return self.level.upper() == "L0" or self.id.upper() == "L0"


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    try:
        _, raw_meta, body = text.split("---", 2)
    except ValueError:
        return {}, text
    meta: dict[str, str] = {}
    for raw_line in raw_meta.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def _parse_sections(body: str) -> dict[str, list[str]]:
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
    return sections


def _section_text(sections: dict[str, list[str]], name: str) -> str:
    return "\n".join(sections.get(name, [])).strip()


def _bullets(sections: dict[str, list[str]], name: str) -> list[str]:
    out: list[str] = []
    for line in sections.get(name, []):
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
    return out


def _comma_list(sections: dict[str, list[str]], name: str) -> list[str]:
    raw = _section_text(sections, name)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _section_items(sections: dict[str, list[str]], name: str) -> list[str]:
    out: list[str] = []
    for line in sections.get(name, []):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        out.append(stripped)
    return out


def _first_sentence(text: str) -> str:
    """Compact one-liner from a verbose description: collapse whitespace and take
    the first sentence, capped at 200 chars. Mirrors vllm _mol_routing so the
    summary fallback matches when a library has no explicit ## Summary section."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    cut = text.split(". ", 1)[0]
    return cut[:200].strip()


def parse_lora_markdown(path: Path) -> LoRATask:
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    sections = _parse_sections(body)
    task_id = meta.get("id") or path.stem
    datasets = _section_items(sections, "datasets")
    dataset = meta.get("dataset", "")
    if dataset and dataset not in datasets:
        datasets.insert(0, dataset)
    return LoRATask(
        id=task_id,
        task=meta.get("task", task_id),
        level=meta.get("level", ""),
        adapter_name=meta.get("adapter_name", ""),
        source_path=meta.get("source_path", ""),
        priority=int(meta.get("priority", "0") or 0),
        description=_section_text(sections, "description"),
        summary=_section_text(sections, "summary")
        or _first_sentence(_section_text(sections, "description")),
        router_line=_section_text(sections, "router_line"),
        routing_rules=_bullets(sections, "routing_rules"),
        strong_signals=_comma_list(sections, "strong_signals"),
        positive_signals=_comma_list(sections, "positive_signals"),
        negative_signals=_comma_list(sections, "negative_signals"),
        examples=_section_items(sections, "examples"),
        datasets=datasets,
        library_path=str(path),
    )


def load_lora_library(library_dir: str | Path) -> dict[str, LoRATask]:
    root = Path(library_dir)
    if not root.exists():
        raise FileNotFoundError(f"Missing LoRA Library directory: {root}")
    tasks = {
        task.id: task
        for task in (parse_lora_markdown(path) for path in sorted(root.glob("*.md")))
    }
    if not tasks:
        raise FileNotFoundError(f"No .md task files found in LoRA Library: {root}")
    if "L0" not in tasks:
        raise FileNotFoundError(f"LoRA Library must include an entry router file named L0.md: {root}")
    return tasks


def library_to_jsonable(tasks: dict[str, LoRATask]) -> dict[str, Any]:
    return {
        "tasks": {
            task_id: task.__dict__
            for task_id, task in tasks.items()
        }
    }
