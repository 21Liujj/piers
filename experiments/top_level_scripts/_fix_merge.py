
import json, os, numpy as np
P="/root/autodl-tmp/piers-main"
man=json.load(open(f"{P}/configs/piers/manifest.dev-calib-100.json"))
allq=[q["query_id"] for q in man["queries"]]
# --- rebuild T1-100m full 800 ---
pred_by={}; pmask={}
for i in range(4):
    d=f"{P}/runs/T1-100m-s{i}"
    for p in json.load(open(f"{d}/predictions.json")):
        pred_by[p["query_id"]]=p
    z=np.load(f"{d}/masks.npz")
    for k,m in zip([str(x) for x in z["query_ids"]], z["masks"]):
        pmask[k]=m.astype(bool)
full_preds=[pred_by[q] for q in allq]
os.makedirs(f"{P}/runs/T1-100m", exist_ok=True)
json.dump(full_preds, open(f"{P}/runs/T1-100m/predictions.json","w"), indent=2, default=str)
arr=np.stack([(pmask[q] if q in pmask else np.zeros((400,2048),bool)).astype(np.uint8) for q in allq])
np.savez_compressed(f"{P}/runs/T1-100m/masks.npz", query_ids=np.array(allq), masks=arr)
print("T1-100m fixed:", len(full_preds), arr.shape, "empty:", sum(q not in pmask for q in allq))
# --- rebuild T0-100m masks full 800 ---
z=np.load(f"{P}/runs/T0-100m/masks.npz")
pm0={str(k):m for k,m in zip(z["query_ids"], z["masks"])}
arr0=np.stack([(pm0[q] if q in pm0 else np.zeros((400,2048),bool)).astype(np.uint8) for q in allq])
np.savez_compressed(f"{P}/runs/T0-100m/masks.npz", query_ids=np.array(allq), masks=arr0)
print("T0-100m fixed:", arr0.shape)
