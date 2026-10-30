"""
Compatibility shim for in-repo SAM2 code.

This repo vendors Meta's SAM2 implementation under `sam/sam/`, but parts of the
code (and downstream modules) import it as a top-level package named `sam2`,
e.g. `from sam2.modeling...`.

To keep the original import paths working without rewriting the vendored files,
we expose a lightweight `sam2` package whose module search path points at
`<repo>/sam/sam`.
"""

from __future__ import annotations

import os

# Make `sam2.*` resolve to modules inside `<repo>/sam/sam/*`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDORED_SAM2_ROOT = os.path.join(_REPO_ROOT, "sam", "sam")

# `__path__` controls where Python looks for submodules of this package.
__path__ = [_VENDORED_SAM2_ROOT]  # type: ignore[name-defined]

