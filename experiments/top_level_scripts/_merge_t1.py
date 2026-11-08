
import json, os, numpy as np
P="/root/autodl-tmp/piers-main"
os.makedirs(f"{P}/runs/T1-100m", exist_ok=True)
all_preds=[]; qids=[]; masks=[]
for i in range(4):
    d=f"{P}/runs/T1-100m-s{i}"
    pr=json.load(open(f"{d}/predictions.json"))
    all_preds.extend(pr)
    z=np.load(f"{d}/masks.npz")
    qids.extend([str(x) for x in z["query_ids"]])
    masks.append(z["masks"])
masks=np.concatenate(masks, axis=0)
order=np.argsort(qids)
qids=[qids[i] for i in order]; masks=masks[order]
np.savez_compressed(f"{P}/runs/T1-100m/masks.npz", query_ids=np.array(qids), masks=masks)
pred_by={p["query_id"]:p for p in all_preds}
all_preds=[pred_by[q] for q in qids]
json.dump(all_preds, open(f"{P}/runs/T1-100m/predictions.json","w"), indent=2, default=str)
print(f"merged: {len(all_preds)} queries, masks {masks.shape}")
