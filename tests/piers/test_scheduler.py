import json
import numpy as np
from piers_ljj.cli import main as cli_main
from piers_ljj.experiments.scheduler import config_hash, check_deps


def test_run_dry_run_validates_deps(tmp_path, monkeypatch):
    import yaml
    cfg = {"suite": "T2", "run_id": "T2", "runs_root": str(tmp_path),
           "manifest": "q.json", "fov": {"win_w": 32}}
    cp = tmp_path / "t2.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    # T2 dep E0 not done -> blocked
    assert cli_main(["run", "--config", str(cp)]) == 3
    st = json.loads((tmp_path / "T2" / "status.json").read_text())
    assert st["status"] == "blocked" and st["missing_deps"] == ["E0"]
    # fake E0 completed -> dry-run passes
    (tmp_path / "E0").mkdir()
    (tmp_path / "E0" / "status.json").write_text(json.dumps({"status": "completed_real"}))
    assert cli_main(["run", "--config", str(cp), "--dry-run"]) == 0


def test_evaluate_text_and_uda_null_rules(tmp_path):
    pred = np.zeros((2, 4, 4), int); pred[0, 0:2, 0:2] = 1
    gt = np.zeros((2, 4, 4), int); gt[0, 0:2, 0:2] = 1
    pp, gp = str(tmp_path / "p.npy"), str(tmp_path / "g.npy")
    np.save(pp, pred); np.save(gp, gt)
    out = str(tmp_path / "m.json")
    assert cli_main(["evaluate", "--pred-npz", pp, "--gt-npz", gp,
                     "--task", "text", "--out", out]) == 0
    m = json.loads(open(out).read())
    assert m["n"] == 2 and m["correct_reject"] == 1  # q1 both-empty -> reject
    pred2 = np.zeros((4, 4), int); gt2 = np.ones((4, 4), int) * 2
    pp2, gp2 = str(tmp_path / "p2.npy"), str(tmp_path / "g2.npy")
    np.save(pp2, pred2); np.save(gp2, gt2)
    out2 = str(tmp_path / "m2.json")
    assert cli_main(["evaluate", "--pred-npz", pp2, "--gt-npz", gp2,
                     "--task", "uda", "--n-cls", "13", "--out", out2]) == 0


def test_summarize_excludes_synthetic_and_incomplete(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "status.json").write_text(
        json.dumps({"status": "completed_real", "synthetic": False}))
    (tmp_path / "good" / "metrics.json").write_text(json.dumps({"miou": 0.5}))
    (tmp_path / "syn").mkdir()
    (tmp_path / "syn" / "status.json").write_text(
        json.dumps({"status": "completed_real", "synthetic": True}))
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "status.json").write_text(json.dumps({"status": "pending_gpu"}))
    out = str(tmp_path / "sum.json")
    assert cli_main(["summarize", "--runs-root", str(tmp_path), "--out", out]) == 0
    rows = json.loads(open(out).read())
    real = [r for r in rows if r.get("status") == "completed_real"]
    assert len(real) == 1 and real[0]["miou"] == 0.5


def test_config_hash_stable():
    assert config_hash({"b": 1, "a": 2}) == config_hash({"a": 2, "b": 1})
    assert check_deps("T2", "/nonexistent-root-xyz") == ["E0"]
