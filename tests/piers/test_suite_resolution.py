"""Regression tests for the suite/dep-resolution bugs found on 2026-10-29.

Bug A: `cmd_run` passed the RAW config `suite` string to `check_deps`, but
       DEPS is keyed by canonical codes. A config with `suite: T2-...` never
       matched, so T2 silently ran even though its E0 dependency was missing.
Bug B: dependency directories were resolved as the canonical CODE ("E0")
       while configs write results under their `run_id` ("E0-effective-start"),
       so a completed dependency was never found.
Bug C: `effective_start.yaml` had no `suite:` field at all, so the E0
       baseline did not resolve to its registry entry.
"""
import json

import yaml

from piers_ljj.cli import main as cli_main
from piers_ljj.experiments.scheduler import check_deps
from piers_ljj.experiments.suites import SUITE_RUN_IDS, canonical_suite, run_id_for


def _write_cfg(tmp_path, cfg, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_canonical_suite_handles_long_names():
    assert canonical_suite("T2-semantic-to-text") == "T2"
    assert canonical_suite("T2") == "T2"
    assert canonical_suite("E0") == "E0"
    assert canonical_suite("") == ""


def test_run_id_for_matches_registry():
    assert run_id_for("T2") == "T2-semantic-to-text"
    assert run_id_for("T2-semantic-to-text") == "T2-semantic-to-text"
    assert run_id_for("E0") == "E0-effective-start"


def test_dep_lookup_uses_run_id_directory(tmp_path):
    """Bug B: dep dir must be the run_id, not the canonical code."""
    root = tmp_path / "runs"
    d = root / SUITE_RUN_IDS["E0"]
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": "completed_real"}))
    assert check_deps("T2", str(root)) == []
    # a stale dir named after the CODE must NOT satisfy the dependency
    (root / "E0").mkdir()
    (root / "E0" / "status.json").write_text(
        json.dumps({"status": "completed_real"}))
    (d / "status.json").write_text(json.dumps({"status": "not_run"}))
    assert check_deps("T2", str(root)) == ["E0"]


def test_long_suite_name_still_blocks_on_missing_dep(tmp_path):
    """Bug A: `suite: T2-semantic-to-text` must block when E0 is missing."""
    cfg = {"suite": "T2-semantic-to-text", "run_id": "T2-semantic-to-text",
           "runs_root": str(tmp_path), "manifest": "q.json",
           "fov": {"win_w": 32}}
    cp = _write_cfg(tmp_path, cfg)
    assert cli_main(["run", "--config", str(cp)]) == 3
    st = json.loads((tmp_path / "T2-semantic-to-text" / "status.json").read_text())
    assert st["status"] == "blocked"
    assert st["missing_deps"] == ["E0"]
    assert st["suite"] == "T2"


def test_long_suite_name_passes_once_dep_done(tmp_path):
    """Bug A + B together: after E0 completes, T2 must unblock."""
    root = tmp_path / "runs"
    d = root / SUITE_RUN_IDS["E0"]
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"status": "completed_real"}))
    cfg = {"suite": "T2-semantic-to-text", "run_id": "T2-semantic-to-text",
           "runs_root": str(root), "manifest": "q.json",
           "fov": {"win_w": 32}}
    cp = _write_cfg(tmp_path, cfg)
    assert cli_main(["run", "--config", str(cp), "--dry-run"]) == 0


def test_effective_start_config_declares_suite_e0():
    """Bug C: the E0 baseline config must declare `suite: E0`."""
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(repo, "configs", "piers", "effective_start.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert canonical_suite(cfg.get("suite", "")) == "E0"
    assert cfg.get("run_id") == SUITE_RUN_IDS["E0"]
