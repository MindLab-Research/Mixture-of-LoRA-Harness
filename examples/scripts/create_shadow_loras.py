#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mol_harness import load_lora_library  # noqa: E402


def prepare_shadow_lora(source: Path, target: Path, copy_weights: bool) -> None:
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    target.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("peft_type", "LORA")
    config.setdefault("inference_mode", True)
    config.setdefault("lora_dropout", 0.0)
    config.setdefault("fan_in_fan_out", False)
    config.setdefault("modules_to_save", None)
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    dst_weights = target / "adapter_model.safetensors"
    if dst_weights.exists() or dst_weights.is_symlink():
        dst_weights.unlink()
    if copy_weights:
        shutil.copy2(weights_path, dst_weights)
    else:
        dst_weights.symlink_to(weights_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create SGLang-loadable shadow LoRA directories.")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/lora_library"))
    parser.add_argument("--output-dir", type=Path, default=Path("shadow_loras"))
    parser.add_argument("--copy-weights", action="store_true", help="Copy weights instead of symlinking them.")
    args = parser.parse_args()

    tasks = load_lora_library(args.library_dir)
    for task_id, task in tasks.items():
        if not task.adapter_name or not task.source_path:
            continue
        prepare_shadow_lora(Path(task.source_path), args.output_dir / task_id, args.copy_weights)
        print(f"{task_id}: {args.output_dir / task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
