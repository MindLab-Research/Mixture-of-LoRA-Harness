from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .library import LoRATask, load_lora_definitions


DEFAULT_MODEL_NAME = "Macaron-V1-Venti"
SUPPORTED_MODELS = {
    "Macaron-V1-Venti": "GLM-5.2",
    "Macaron-V1-Tall": "Qwen3.6-35B-A3B",
}
CANONICAL_ROUTES = ("L0", "L1", "L2", "L3")
REQUIRED_FILES = ("lora.md", "route.md", "intro.md", "summary.md")


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    base_model: str
    config_dir: Path
    tasks: dict[str, LoRATask]
    route_prompt: str
    intro_prompt: str
    summary_prompt: str


def _read_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model configuration file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Model configuration file is empty: {path}")
    return text


def _render_common(text: str, model_name: str, base_model: str) -> str:
    return (
        text.replace("{{MODEL_NAME}}", model_name)
        .replace("{{BASE_MODEL}}", base_model)
    )


def load_model_config(
    model_name: str,
    config_root: str | Path | None = None,
) -> ModelConfig:
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of: {supported}")

    root = Path(config_root) if config_root else Path(__file__).resolve().parent
    config_dir = root / model_name
    missing = [
        name for name in REQUIRED_FILES
        if not (config_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model configuration {config_dir} is missing: {', '.join(missing)}"
        )

    tasks = load_lora_definitions(config_dir / "lora.md")
    if tuple(tasks) != CANONICAL_ROUTES:
        raise ValueError(
            f"{config_dir / 'lora.md'} must define exactly: "
            + ", ".join(CANONICAL_ROUTES)
        )
    adapters = [tasks[route].adapter_name for route in CANONICAL_ROUTES]
    if len(set(adapters)) != len(adapters):
        raise ValueError(
            f"Adapter names must be unique in {config_dir / 'lora.md'}")

    base_model = SUPPORTED_MODELS[model_name]
    route_prompt = _render_common(
        _read_required(config_dir / "route.md"), model_name, base_model
    )
    for token in ("{{LORA_DESCRIPTIONS}}", "{{ROUTE_LABELS}}"):
        if token not in route_prompt:
            raise ValueError(f"{config_dir / 'route.md'} must contain {token}")

    return ModelConfig(
        model_name=model_name,
        base_model=base_model,
        config_dir=config_dir,
        tasks=tasks,
        route_prompt=route_prompt,
        intro_prompt=_render_common(
            _read_required(config_dir / "intro.md"), model_name, base_model
        ),
        summary_prompt=_render_common(
            _read_required(config_dir / "summary.md"), model_name, base_model
        ),
    )


ACTIVE_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", DEFAULT_MODEL_NAME)
ACTIVE_CONFIG_ROOT = os.environ.get("MOL_MODEL_CONFIG_ROOT") or None
ACTIVE_MODEL_CONFIG = load_model_config(ACTIVE_MODEL_NAME, ACTIVE_CONFIG_ROOT)
