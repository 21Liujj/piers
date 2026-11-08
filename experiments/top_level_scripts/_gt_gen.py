
import json, os
import numpy as np
from PIL import Image
MAP=[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]
LUT=np.array(MAP, dtype=np.int64)
man=json.load(open("/root/autodl-tmp/piers-main/configs/piers/manifest.dev-calib-100.json"))
gtdir="/root/autodl-tmp/data/DensePASS/gtFine/val"
try:
    import cv2
    HAS_CV2=True
except Exception:
    HAS_CV2=False
def comps(binm):
    if HAS_CV2:
        n,lab,stats,cent=cv2.connectedComponentsWithStats(binm.astype(np.uint8), connectivity=8)
        out=[]
        for i in range(1,n):
            out.append({"area":int(stats[i,4]),"cx":float(cent[i][0]),"cy":float(cent[i][1]),"mask":(lab==i)})
        return out
    try:
        from scipy import ndimage
        lab,n=ndimage.label(binm, structure=np.ones((3,3)))
        out=[]
        for i in range(1,n+1):
            ys,xs=np.where(lab==i)
            if len(xs)==0: continue
            out.append({"area":int(len(xs)),"cx":float(xs.mean()),"cy":float(ys.mean()),"mask":(lab==i)})
        return out
    except Exception:
        return [{"area":int(binm.sum()),"cx":1e9,"cy":1e9,"mask":binm}]
img2lab={}
for im in man["images"]:
    src=im.get("src_file","")
    stemnum=src.split("_")[0]
    img2lab[im["image_id"]]=os.path.join(gtdir, stemnum+"_labelTrainIds.png")
n_ok=n_miss=0
for q in man["queries"]:
    gp=q.get("gt_mask")
    if gp and os.path.exists(gp):
        n_ok+=1; continue
    lab_p=img2lab.get(q["image_id"])
    if not lab_p or not os.path.exists(lab_p):
        print("MISS-LAB", q["query_id"], lab_p); n_miss+=1; continue
    g19=np.array(Image.open(lab_p))
    if g19.ndim==3: g19=g19[:,:,0]
    g19=np.clip(g19,0,18)
    m13=LUT[g19]
    rt=(q.get("raw_text") or "").strip().lower()
    tt=q.get("task_type")
    onto=(q.get("ontology") or {})
    if rt=="train" or q.get("negative"):
        binm=np.zeros_like(m13, bool)
    elif tt=="instance_referring" and "car" in rt:
        binm=(g19==13)
        cs=comps(binm)
        if cs:
            binm=min(cs,key=lambda d:d["cx"])["mask"]
        else:
            binm=np.zeros_like(m13,bool)
    elif tt=="instance_referring" and "person" in rt:
        binm=(g19==11)
        cs=comps(binm)
        if cs:
            binm=max(cs,key=lambda d:d["area"])["mask"]
        else:
            binm=np.zeros_like(m13,bool)
    elif "map13" in onto:
        binm=(m13==int(onto["map13"]))&(m13!=255)
    else:
        binm=np.zeros_like(m13,bool)
    os.makedirs(os.path.dirname(gp), exist_ok=True)
    Image.fromarray((binm.astype(np.uint8))*255).save(gp)
    n_ok+=1
print(f"GT done ok={n_ok} miss={n_miss} total={len(man['queries'])}")
