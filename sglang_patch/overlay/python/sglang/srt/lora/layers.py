"""Overlay shim for SGLang LoRA layer wrappers.

The upstream LoRA wrapper keeps the wrapped module under ``base_layer`` but does
not delegate unknown attributes. GLM/DeepSeek MoE code reads attributes such as
``moe_runner_config`` directly from ``mlp.experts``; after wrapping that object
becomes ``FusedMoEWithLoRA``. Delegate misses to the wrapped layer so the model
code keeps seeing the original FusedMoE surface.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_BASE_PYTHON = Path(os.environ.get("SGLANG_BASE_PYTHON", "/sgl-workspace/sglang/python"))
_BASE_FILE = _BASE_PYTHON / "sglang" / "srt" / "lora" / "layers.py"
_SPEC = importlib.util.spec_from_file_location("_sglang_base_lora_layers", _BASE_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load original SGLang LoRA layers from {_BASE_FILE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_sglang_base_lora_layers", _BASE)
_SPEC.loader.exec_module(_BASE)


def _delegating_getattr(self, name):
    try:
        return _BASE.nn.Module.__getattr__(self, name)
    except AttributeError as original_error:
        # Do not delegate parameter names during wrapper initialization.
        # torch.nn.Module.register_parameter calls hasattr(self, name);
        # delegating weight/bias would make registration fail with
        # "attribute already exists".
        if name in {"weight", "bias"}:
            raise original_error
        modules = self.__dict__.get("_modules")
        base_layer = modules.get("base_layer") if modules is not None else None
        if base_layer is not None:
            try:
                return getattr(base_layer, name)
            except AttributeError:
                pass
        raise original_error


_BASE.BaseLayerWithLoRA.__getattr__ = _delegating_getattr

for _name in dir(_BASE):
    if not _name.startswith("__") or _name in {"__all__", "__doc__"}:
        globals()[_name] = getattr(_BASE, _name)

__all__ = getattr(_BASE, "__all__", [name for name in globals() if not name.startswith("_")])
