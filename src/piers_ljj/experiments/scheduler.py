"""Experiment scheduler: deps, resume, status machine. R5.

Status: not_run | passed_cpu | pending_gpu | passed_gpu_smoke |
        completed_real | blocked | failed.
Resume: a step is skipped only if status==completed_real (or passed_* for
CPU-only) AND config+dep hashes match. synthetic=true runs never aggregate.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys

from piers_ljj.experiments.suites import SUITES, DEPS


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_dir(root: str, run_id: str) -> str:
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def read_status(d: str) -> dict:
    fp = os.path.join(d, "status.json")
    if not os.path.exists(fp):
        return {"status": "not_run"}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def write_status(d: str, st: dict) -> str:
    fp = os.path.join(d, "status.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, sort_keys=True)
    return fp


def check_deps(suite: str, root: str) -> list:
    """Return missing dep run_ids for a suite (empty = ready).

    `suite` may be a canonical code ("T2") or a long name ("T2-semantic-to-text");
    both are canonicalized first. Dependency directories are resolved through
    SUITE_RUN_IDS so they match the configs' `run_id:` fields exactly.
    """
    from piers_ljj.experiments.suites import canonical_suite, run_id_for

    missing = []
    for dep in DEPS.get(canonical_suite(suite), []):
        dep_dir = os.path.join(root, run_id_for(dep))
        st = read_status(dep_dir)
        if st.get("status") not in ("completed_real", "passed_gpu_smoke", "passed_cpu"):
            missing.append(dep)
    return missing


def launch_worker(kind: str, args: list, log_path: str) -> int:
    """Launch omni/grounded worker as an independent process; stream to log."""
    cmd = [sys.executable, "-m", "piers_ljj.workers.entry", kind] + args
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"executable: {sys.executable}\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode
