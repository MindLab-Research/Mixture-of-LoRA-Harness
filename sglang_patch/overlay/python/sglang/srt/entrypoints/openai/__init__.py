"""Overlay package entrypoint for SGLang OpenAI entrypoint patches."""

from __future__ import annotations

import os as _os
import pkgutil as _pkgutil

_BASE_PYTHON = _os.environ.get("SGLANG_BASE_PYTHON", "/sgl-workspace/sglang/python")
_BASE_PACKAGE = _os.path.join(_BASE_PYTHON, "sglang", "srt", "entrypoints", "openai")

__path__ = _pkgutil.extend_path(__path__, __name__)
if _os.path.isdir(_BASE_PACKAGE) and _BASE_PACKAGE not in __path__:
    __path__.append(_BASE_PACKAGE)

