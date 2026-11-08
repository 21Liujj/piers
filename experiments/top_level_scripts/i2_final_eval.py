"""Run the frozen Innovation-2 final configuration on the remote GPU."""
import argparse, hashlib, json, sys, traceback
from pathlib import Path
ROOT=Path("/root/autodl-tmp/piers-main")
CONFIG=ROOT/"configs/piers/innovation2_final.json"
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""):
            h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",help="New remote directory for results")
    ap.add_argument("--profile",action="store_true")
    ap.add_argument("--verify-only",action="store_true")
    a=ap.parse_args()
    cfg=json.loads(CONFIG.read_text())
    if cfg["status"]!="final":
        raise RuntimeError("Final configuration has not been sealed")
    for path,digest in cfg["runtime_source_sha256"].items():
        if sha(path)!=digest:
            raise RuntimeError("Source changed after finalization: "+path)
    if sha(cfg["checkpoint_path"])!=cfg["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from final configuration")
    assert cfg["mode"]=="class_aware" and cfg["score"]=="uncertainty"
    assert cfg["rare_class_ids"]==[3,4,6,7] and cfg["rare_reserve_ratio"]==0.15
    assert cfg["budget"]==1024 and cfg["keep_ratio"]==1024/32768
    if a.verify_only:
        print(json.dumps({"status":"verified","config":str(CONFIG),
              "mode":cfg["mode"],"score":cfg["score"],"budget":cfg["budget"]},indent=2))
        return
    if not a.out:
        ap.error("--out is required unless --verify-only is used")
    sys.path.insert(0,str(ROOT))
    import i2_full_eval as ev
    ns=argparse.Namespace(mode=cfg["mode"],score=cfg["score"],budget=cfg["budget"],
        dataset=cfg["dataset"],limit=0,seed=cfg["seed"],threads=2,
        rare_ratio=cfg["rare_reserve_ratio"],boundary_ratio=0.125,
        profile=a.profile,out=str(Path(a.out).resolve()))
    try:
        ev.main(ns)
    except BaseException as exc:
        out=Path(ns.out);out.mkdir(parents=True,exist_ok=True)
        ev.atomic_json(out/"failure.json",{"status":"failed","error":str(exc),
                       "traceback":traceback.format_exc()})
        raise
if __name__=="__main__":
    main()
