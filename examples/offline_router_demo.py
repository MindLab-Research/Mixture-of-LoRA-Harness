#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mol_harness import RouterHarness, load_lora_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run metadata-only LoRA routing without a model server.")
    parser.add_argument("--library-dir", type=Path, default=Path("examples/lora_library"))
    parser.add_argument("prompt", nargs="?", default="Fix a failing pytest in this repository and run verification.")
    args = parser.parse_args()

    tasks = load_lora_library(args.library_dir)
    harness = RouterHarness(tasks)
    decision = harness.route_by_library(args.prompt)
    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "route_id": decision.route_id,
                "adapter_name": decision.adapter_name,
                "decision": decision.decision,
                "diagnostics": decision.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
