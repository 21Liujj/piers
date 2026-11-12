"""CPU-only: verify remote SPAN8_PREP integrity + build/validate split files.

Runs on the remote server. Validates that every manifest entry has both image and
label, that labels only contain {0..7, 255}, and that rgb/sem pixel counts match.
"""
import json, hashlib, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

ROOT = Path("/root/autodl-tmp/data_extra/SPAN8_PREP")
RUNS = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")
SPAN8 = ["ceiling","chair","door","floor","sofa","table","wall","window"]
ALLOWED = set(range(8)) | {255}

def check(rec):
    ip = ROOT/rec["area"]/rec["image"]
    lp = ROOT/rec["area"]/rec["label"]
    if not ip.exists() or not lp.exists():
        return {"id": rec["id"], "ok": False, "why": "missing file"}
    a = np.array(Image.open(lp))
    if a.ndim == 3: a = a[...,0]
    u = set(np.unique(a).tolist())
    bad = u - ALLOWED
    im = Image.open(ip)
    if im.size != (a.shape[1], a.shape[0]):
        return {"id": rec["id"], "ok": False, "why": f"size mismatch {im.size} vs {a.shape}"}
    if bad:
        return {"id": rec["id"], "ok": False, "why": f"bad labels {sorted(bad)}"}
    return {"id": rec["id"], "ok": True, "sha16": hashlib.sha256(lp.read_bytes()).hexdigest()[:16]}

recs = [json.loads(l) for l in open(ROOT/"manifest.jsonl", encoding="utf-8") if l.strip()]
with ThreadPoolExecutor(max_workers=48) as ex:
    results = list(ex.map(check, recs))
bad = [r for r in results if not r["ok"]]
ok = [r for r in results if r["ok"]]
report = {
    "root": str(ROOT),
    "total": len(recs),
    "ok": len(ok),
    "bad": len(bad),
    "bad_examples": bad[:10],
    "train": sum(1 for r in recs if r["fold"]=="train"),
    "heldout": sum(1 for r in recs if r["fold"]=="heldout"),
    "sha16_sample": [r["sha16"] for r in ok[:10]],
}
(RUNS/"SPAN8_INTEGRITY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(1 if bad else 0)
