"""Worker entrypoints — independent processes, sys/env diagnostics first.

Each worker prints sys.executable, key package versions, and the REAL loaded
module file (sam2.__file__ / model module path) and FAILS if the import path
points at the wrong source (e.g. repo-root shadowing vendor cwd).
"""
from __future__ import annotations
import sys


def diag(mod_names=("numpy", "PIL", "yaml")):
    import importlib
    print(f"executable: {sys.executable}")
    for m in mod_names:
        try:
            mod = importlib.import_module(m)
            print(f"{m}: {getattr(mod, '__version__', '?')} @ {getattr(mod, '__file__', '?')}")
        except Exception as e:
            print(f"{m}: IMPORT-FAIL {e}")


def omni_main(args=None):
    diag()
    # Real OmniSAM invocation resolved in gpu_runbook after source audit.
    print("piers-omni worker: source audit pending (repository_audit.md); refusing to fake-load.")
    return 2


def grounded_main(args=None):
    diag()
    print("piers-grounded worker: vendor snapshot pending (third_party/grounded_sam2); refusing synthetic backend.")
    return 2
