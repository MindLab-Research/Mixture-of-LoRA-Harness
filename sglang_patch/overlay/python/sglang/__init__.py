"""Overlay package entrypoint for the local SGLang LoRA/KV-reuse patch.

Set PYTHONPATH to this overlay's python directory before /sgl-workspace/sglang/python.
Unmodified submodules are resolved from SGLANG_BASE_PYTHON, defaulting to the
remote source checkout used for this patch.
"""

from __future__ import annotations

import os as _os
import pkgutil as _pkgutil

_BASE_PYTHON = _os.environ.get("SGLANG_BASE_PYTHON", "/sgl-workspace/sglang/python")
_BASE_PACKAGE = _os.path.join(_BASE_PYTHON, "sglang")

__path__ = _pkgutil.extend_path(__path__, __name__)
if _os.path.isdir(_BASE_PACKAGE) and _BASE_PACKAGE not in __path__:
    __path__.append(_BASE_PACKAGE)

_ORIGINAL_INIT = _os.path.join(_BASE_PACKAGE, "__init__.py")
if _os.path.exists(_ORIGINAL_INIT):
    with open(_ORIGINAL_INIT, "rb") as _f:
        exec(compile(_f.read(), _ORIGINAL_INIT, "exec"), globals(), globals())
else:
    raise ImportError(f"Cannot find original SGLang package at {_BASE_PACKAGE!r}")
