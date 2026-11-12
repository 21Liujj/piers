"""CPU-only: per-class pixel statistics + rare-class analysis on SPAN8 splits."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("/root/autodl-tmp/data_extra/SPAN8_PREP")
RUNS = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")
SPAN8 = ["ceiling","chair","door","floor","sofa","table","wall","window"]

def stats_for(split_file):
    recs = [json.loads(l) for l in open(split_file, encoding="utf-8") if l.strip()]
    h = np.zeros(9, dtype=np.int64)
    present = np.zeros(8, dtype=bool)
    def one(r):
        a = np.array(Image.open(ROOT/r["area"]/r["label"]))
        if a.ndim == 3: a = a[...,0]
        u, c = np.unique(a, return_counts=True)
        return u, c
    with ThreadPoolExecutor(max_workers=32) as ex:
        for u, c in ex.map(one, recs):
            for k, v in zip(u.tolist(), c.tolist()):
                if k < 8:
                    h[k] += v; present[k] = True
                else:
                    h[8] += v
    return recs, h, present

out = {"classes": SPAN8}
for name in ["fit","calib","val","heldout"]:
    p = RUNS/f"split_{name}.jsonl"
    if not p.exists():
        continue
    recs, h, present = stats_for(p)
    tot = h.sum()
    valid = tot - h[8]
    pct = {SPAN8[i]: round(100.0*h[i]/tot, 4) for i in range(8)}
    out[name] = {
        "n_images": len(recs),
        "pixel_count": int(tot),
        "valid_ratio": round(float(valid/tot), 6),
        "class_pixel_pct": pct,
        "class_present": {SPAN8[i]: bool(present[i]) for i in range(8)},
        "rare_classes_under_1pct": [SPAN8[i] for i in range(8) if 0 < 100.0*h[i]/tot < 1.0],
        "absent_classes": [SPAN8[i] for i in range(8) if not present[i]],
    }
(RUNS/"SPAN8_CLASS_STATS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
