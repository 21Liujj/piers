"""Fold1 split + dataset index for indoor SPAN8 (CPU-only, no GPU needed)."""
from pathlib import Path
import json, random, hashlib
ROOT = Path("/root/autodl-tmp/data_extra/SPAN8_PREP")
OUT = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")
OUT.mkdir(parents=True, exist_ok=True)
SPAN8 = ["ceiling","chair","door","floor","sofa","table","wall","window"]
recs = []
with open(ROOT/"manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))
train = [r for r in recs if r["fold"] == "train"]
held = [r for r in recs if r["fold"] == "heldout"]
# room-level split for train (avoid same-room leakage): group by uuid prefix (before '_')
def room_of(r):
    rid = r["id"]
    return rid.split("_")[1] if "_" in rid else rid
train_rooms = sorted(set(room_of(r) for r in train))
rng = random.Random(42)
rng.shuffle(train_rooms)
n = len(train_rooms)
n_tr = int(round(n*0.8)); n_va = int(round(n*0.1))
splits = {
    "fit_rooms":   train_rooms[:n_tr],
    "calib_rooms": train_rooms[n_tr:n_tr+n_va],
    "val_rooms":   train_rooms[n_tr+n_va:],
}
def rooms_of(rs):
    return set(room_of(r) for r in rs)
fit = [r for r in train if room_of(r) in set(splits["fit_rooms"])]
calib = [r for r in train if room_of(r) in set(splits["calib_rooms"])]
val = [r for r in train if room_of(r) in set(splits["val_rooms"])]
# leakage check
def ov(a,b):
    return len(rooms_of(a) & rooms_of(b))
leak = {"fit∩calib": ov(fit,calib), "fit∩val": ov(fit,val), "calib∩val": ov(calib,val),
        "train∩heldout_rooms": len(set(room_of(r) for r in train) & set(room_of(r) for r in held))}
for name, rs in [("fit",fit),("calib",calib),("val",val),("heldout",held)]:
    p = OUT/f"split_{name}.jsonl"
    with open(p,"w",encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"{name}: {len(rs)} images -> {p}")
# per-image sha256 of label (audit trail) for a sample
sample = []
import hashlib
for r in (fit[:5] + held[:5]):
    lp = ROOT/r["area"]/r["label"]
    h = hashlib.sha256(lp.read_bytes()).hexdigest()[:16]
    sample.append({"id": r["id"], "area": r["area"], "label_sha256_16": h, "bytes": lp.stat().st_size})
out = {
    "protocol": "Fold1: train areas 1/2/3/4/6, heldout areas 5a/5b",
    "source_root": str(ROOT),
    "classes": SPAN8,
    "canvas": [1024, 512],
    "counts": {"all": len(recs), "train": len(train), "heldout": len(held),
               "fit": len(fit), "calib": len(calib), "val": len(val)},
    "rooms": {"total_train_rooms": n, **{k: len(v) for k,v in splits.items()}},
    "leakage_check_rooms": leak,
    "sample_sha256": sample,
}
(OUT/"INDOOR_SPLIT_PROTOCOL.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
