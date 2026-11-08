"""Path resolution for configs/manifests so the repo is machine-independent.

Committed configs and manifests use `${VAR}` placeholders instead of absolute
paths from the machine where the experiments ran. This module expands them
from, in order of precedence:

  1. explicit overrides passed to `resolve()`
  2. environment variables (e.g. PIERS_DATA_ROOT)
  3. `configs/piers/paths.yaml` if it exists (git-ignored, machine-local)
  4. `configs/piers/paths.example.yaml` (checked-in defaults)

Supported placeholders
----------------------
  ${PIERS_DATA_ROOT}        dataset root, e.g. /root/autodl-tmp/data
  ${PIERS_DATA_EXTRA_ROOT}  extra datasets, e.g. /root/autodl-tmp/data_extra
  ${PIERS_WEIGHTS_ROOT}     model weights, e.g. /root/autodl-tmp/weights
  ${PIERS_REPO_ROOT}        the piers checkout itself
  ${PIERS_RUNS_ROOT}        output/scratch root (default: <repo>/runs)

Any other `${NAME}` is left untouched (so shell-ish `${VAR}` in unrelated
files is not mangled) and reported by `unresolved()`.
"""
from __future__ import annotations

import os
import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

KNOWN_PLACEHOLDERS = {
    "PIERS_DATA_ROOT",
    "PIERS_DATA_EXTRA_ROOT",
    "PIERS_WEIGHTS_ROOT",
    "PIERS_REPO_ROOT",
    "PIERS_RUNS_ROOT",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATHS_YAML = os.path.join(REPO_ROOT, "configs", "piers", "paths.yaml")
PATHS_EXAMPLE = os.path.join(REPO_ROOT, "configs", "piers", "paths.example.yaml")

# Values recorded on the machine where the paper experiments were executed.
# These are the DEFAULTS so that a fresh clone reproduces the same layout
# when the user recreates it under the same root.
_SERVER_DEFAULTS = {
    "PIERS_DATA_ROOT": "/root/autodl-tmp/data",
    "PIERS_DATA_EXTRA_ROOT": "/root/autodl-tmp/data_extra",
    "PIERS_WEIGHTS_ROOT": "/root/autodl-tmp/weights",
}


def _read_simple_yaml(path: str) -> dict:
    """Minimal `key: value` YAML reader (avoids a PyYAML dependency)."""
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                out[key] = value
    return out


def default_paths() -> dict:
    """Resolve placeholder -> path using the documented precedence."""
    values: dict[str, str] = {}

    # 4. checked-in example file (lowest precedence for the machine-local keys)
    example = _read_simple_yaml(PATHS_EXAMPLE)
    for key in _SERVER_DEFAULTS:
        for candidate in (
            f"{key.lower()}",
            key.lower().replace("_root", ""),
        ):
            if candidate in example and example[candidate]:
                values[key] = example[candidate]
                break

    # 3. machine-local paths.yaml overrides the example
    local = _read_simple_yaml(PATHS_YAML)
    for key in _SERVER_DEFAULTS:
        for candidate in (
            f"{key.lower()}",
            key.lower().replace("_root", ""),
        ):
            if candidate in local and local[candidate]:
                values[key] = local[candidate]
                break

    # 2. environment variables win over files
    for key in _SERVER_DEFAULTS:
        env_value = os.environ.get(key)
        if env_value:
            values[key] = env_value

    # 1. hard defaults for anything still missing
    for key, value in _SERVER_DEFAULTS.items():
        values.setdefault(key, value)

    values["PIERS_REPO_ROOT"] = REPO_ROOT
    values.setdefault("PIERS_RUNS_ROOT", os.path.join(REPO_ROOT, "runs"))
    return values


def resolve_text(text: str, overrides: dict | None = None) -> str:
    """Expand `${PIERS_*}` placeholders inside a string."""
    if not isinstance(text, str) or "${" not in text:
        return text
    values = default_paths()
    if overrides:
        values.update({k: v for k, v in overrides.items() if v})

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        return str(values.get(name, match.group(0)))

    return PLACEHOLDER_RE.sub(_sub, text)


def resolve(obj: Any, overrides: dict | None = None) -> Any:
    """Recursively expand placeholders in dicts/lists/strings."""
    if isinstance(obj, str):
        return resolve_text(obj, overrides)
    if isinstance(obj, list):
        return [resolve(item, overrides) for item in obj]
    if isinstance(obj, dict):
        return {key: resolve(value, overrides) for key, value in obj.items()}
    return obj


def unresolved(obj: Any) -> list[str]:
    """Return the names of placeholders that could not be resolved."""
    found: list[str] = []

    def _walk(item: Any) -> None:
        if isinstance(item, str):
            for name in PLACEHOLDER_RE.findall(item):
                if name in KNOWN_PLACEHOLDERS and name not in found:
                    found.append(name)
        elif isinstance(item, list):
            for sub in item:
                _walk(sub)
        elif isinstance(item, dict):
            for sub in item.values():
                _walk(sub)

    _walk(obj)
    return found


def load_manifest(path: str, overrides: dict | None = None) -> dict:
    """Load a manifest JSON and expand its path placeholders."""
    import json

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return resolve(raw, overrides)
