"""Controlled, audited Innovation-2 evaluation. All execution is remote."""
import os, sys, json, time, argparse, hashlib, traceback
from pathlib import Path
from collections import Counter
ROOT = Path("/root/autodl-tmp/piers-main")
VENDOR = ROOT / "third_party/effective_omnisam"
sys.path.insert(0, str(VENDOR))
import numpy as np
import torch
import cv2
from scipy import ndimage
from PIL import Image
import utils.sam2_bootstrap
import validation as v

class AuditFailure(BaseException):
    pass

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def checksum(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def edges(a, valid):
    out = np.zeros_like(valid)
    e = (a[:, 1:] != a[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    out[:, 1:] |= e
    out[:, :-1] |= e
    e = (a[1:, :] != a[:-1, :]) & valid[1:, :] & valid[:-1, :]
    out[1:, :] |= e
    out[:-1, :] |= e
    return out & valid

def boundary_counts(gt, pred, nc):
    valid = (gt >= 0) & (gt < nc)
    valid = ndimage.binary_erosion(valid, structure=np.ones((3,3)), iterations=3, border_value=1)
    bg, bp = edges(gt, valid), edges(pred, valid)
    dg = ndimage.binary_dilation(bg, structure=np.ones((3,3)), iterations=2)
    dp = ndimage.binary_dilation(bp, structure=np.ones((3,3)), iterations=2)
    return [int(bp.sum()), int(bg.sum()), int((bp & dg).sum()), int((bg & dp).sum())]

def main(a):
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "result.json").exists():
        raise RuntimeError("Refusing to overwrite completed experiment")
    # Additional low/high-budget score controls must finish before isolated profiles.
    if a.profile:
        extra_path = Path(a.out).parent / "extra_accuracy_jobs.json"
        while extra_path.exists():
            extra = json.loads(extra_path.read_text())
            waiting = [x for x in extra if not (Path(x["out"])/"result.json").exists()
                       and not (Path(x["out"])/"failure.json").exists()]
            if not waiting:
                break
            dead = [x for x in waiting if not Path("/proc",str(x["pid"])).exists()]
            if dead:
                raise RuntimeError("Extra accuracy process exited without a result")
            time.sleep(2)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    v.setup_seed(a.seed)
    ns = argparse.Namespace(
        seed=a.seed, backbone="sam2_s", dataset=a.dataset, class_set=13,
        data_root="/root/autodl-tmp/data", wildpass_layout="flat",
        wildpass_processed_root="/root/autodl-tmp/data_extra/WildPASS13",
        val_list="", wildpass_image_dir="images", wildpass_label_dir="labels_raw",
        stanford_processed_root="", stanford_val_list="", stanford_legacy_area_crop=False,
        stanford_stride_w=256, batch_size=1,
        ckpt="/root/autodl-tmp/weights/omnisam_s.pth", ckpt_list=None, ckpt_names=None,
        metrics_json=str(outdir/"metrics.json"), log_file=None, log_append=False,
        enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9,
        use_mem_bank=True, save_pred_debug=0, pred_gt_histogram=False,
        debug_pred_vis_dir="", pred_gt_histogram_json="", pred_gt_histogram_md="",
        timing_json=None, run_label=outdir.name, _skip_per_run_json=False)
    if a.dataset == "WildPASS":
        import utils.wildpass_label_convert as wlc
        from utils.wildpass_label_convert import LUT_LABELID_TO_TRAINID, map19_trainid_to_train13
        def forced_labelid(raw):
            raw = np.asarray(raw, dtype=np.int64)
            tid = LUT_LABELID_TO_TRAINID[np.clip(raw, 0, 255)]
            return map19_trainid_to_train13(tid).astype(np.int32), "forced_labelId"
        wlc.convert_wildpass_raw_to_train13 = forced_labelid
        # WildPASS module imports the converter directly.
        import dataset.WildPASS as wildmod
        if hasattr(wildmod, "convert_wildpass_raw_to_train13"):
            wildmod.convert_wildpass_raw_to_train13 = forced_labelid
    loader, names, nc = v.build_dataset_and_loader(ns)
    files = list(loader.dataset.files)
    full_count = len(files)
    if a.limit:
        files = files[:a.limit]
    if a.profile:
        chosen = np.linspace(0, len(files)-1, min(10, len(files)), dtype=int)
        files = [files[0]] + [files[int(i)] for i in chosen]
    missing = [str(f[k]) for f in files for k in ("img", "label") if k in f and not Path(f[k]).is_file()]
    if missing:
        raise FileNotFoundError(str(missing[:4]))
    loader.dataset.files = files
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)
    ns.num_classes = nc
    source_paths = [ROOT/"m_eval2.py", Path(__file__), VENDOR/"model/memory_bank.py",
                    VENDOR/"model/modules/sparse_memory_retriever.py", VENDOR/"model/model.py",
                    VENDOR/"validation.py"]
    config = vars(a).copy()
    config.update(full_list_count=full_count, evaluated_count=len(files), class_names=names,
        checkpoint_sha256=checksum(ns.ckpt), source_sha256={str(p): checksum(p) for p in source_paths},
        file_names=[f.get("name", str(i)) for i,f in enumerate(files)],
        label_use="GT used for scoring only; routing masks derived exclusively from model predictions",
        boundary_definition="4-neighbor semantic edges; Chebyshev tolerance 2 pixels at 400x2048; ignore GT-void neighborhood",
        timing_note="Accuracy cases run concurrently. Only profile=true cases provide isolated timing.")
    atomic_json(outdir/"config.json", config)
    model = v.build_sam_model(ns, torch.device("cuda"), ckpt_path=ns.ckpt)
    mb = model.memory_bank
    mb.sparse_retriever.diagnostic_overlap = False
    hist = {}
    orig_fwd = model.forward
    audit = {"calls":0,"fallbacks":0,"shape_mismatches":0,"uncertainty_present":0,
             "rare_mask_present":0,"boundary_mask_present":0,"routes":Counter(),
             "effective_lengths":Counter(),"samples":[]}
    events = []
    if a.profile:
        def before_attention(mod, inp):
            ev = torch.cuda.Event(enable_timing=True); ev.record()
            mod._i2_event = ev
        def after_attention(mod, inp, output):
            end = torch.cuda.Event(enable_timing=True); end.record()
            events.append((mod._i2_event, end))
        mb.sam_model.memory_attention.register_forward_pre_hook(before_attention)
        mb.sam_model.memory_attention.register_forward_hook(after_attention)
    def wrapped(frame_idx, x, mem_type="source", historical_pseudo_labels=None, **kw):
        merged = dict(hist)
        if historical_pseudo_labels:
            merged.update(historical_pseudo_labels)
        output = orig_fwd(frame_idx, x, mem_type=mem_type,
            historical_pseudo_labels=merged or None, **kw)
        pred = output[0] if isinstance(output,(tuple,list)) else output
        hist[frame_idx] = pred.argmax(dim=1).detach()
        if frame_idx > 0:
            audit["effective_lengths"][int(getattr(mb,"last_effective_memory_length",0))] += 1
        return output
    model.forward = wrapped
    orig_init = mb._init_output_dict_source
    def init():
        hist.clear()
        return orig_init()
    mb._init_output_dict_source = init
    kw = dict(boundary_reserve_ratio=a.boundary_ratio, rare_class_ids=[3,4,6,7],
              rare_reserve_ratio=a.rare_ratio, boundary_score_source=a.score)
    mb.configure_sparse_retrieval(enabled=a.mode!="dense", mode=a.mode,
        keep_ratio=a.budget/32768, min_tokens=64, cache_enabled=False, fallback="feature_norm",
        fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True, **kw)
    if a.mode == "joint_guarded" and not mb.rare_class_ids:
        raise AuditFailure("joint configured with no rare class IDs")
    mb.set_runtime_context(stage="eval", iter_idx=0)
    orig_retrieve = mb.sparse_retriever.retrieve
    def retrieve(*args, **kw):
        output = orig_retrieve(*args, **kw)
        stats = output[3]
        audit["calls"] += 1
        audit["routes"][stats.selection_mode] += 1
        audit["fallbacks"] += int(stats.fallback_used)
        audit["shape_mismatches"] += int(stats.mask_shape_mismatch)
        for key, field in [("uncertainty_map","uncertainty_present"),
                           ("rare_class_mask","rare_mask_present"),("boundary_mask","boundary_mask_present")]:
            audit[field] += int(kw.get(key) is not None)
        if stats.fallback_used or stats.mask_shape_mismatch:
            raise AuditFailure("Unexpected fallback: "+str(stats))
        if stats.selection_mode != a.mode:
            raise AuditFailure("Wrong executed route: "+stats.selection_mode)
        if a.score == "uncertainty" and a.mode in {"boundary_guarded","class_aware","joint_guarded","uncertainty_topk"} and kw.get("uncertainty_map") is None:
            raise AuditFailure("Missing requested uncertainty score")
        if len(audit["samples"]) < 3:
            sample = {"memory_id":stats.memory_id, "selected":stats.selected_num_tokens,
                      "original":stats.original_num_tokens,"mode":stats.selection_mode}
            idx = output[2]
            if idx is not None:
                sample["unique_indices"] = int(torch.unique(idx).numel())
                for key in ["rare_class_mask","boundary_mask"]:
                    if kw.get(key) is not None:
                        m = kw[key].flatten()
                        sample[key+"_selected"] = int((m[idx] > 0).sum())
            audit["samples"].append(sample)
        return output
    if a.mode != "dense":
        mb.sparse_retriever.retrieve = retrieve
    orig_fast_hist = v.fast_hist
    per_hist, per_boundary, per_time = [], [], []
    start = time.perf_counter()
    last = [start]
    pred_dir = outdir/"predictions"
    if not a.profile:
        pred_dir.mkdir(exist_ok=True)
    def measured_hist(gt, pred, n):
        torch.cuda.synchronize()
        now = time.perf_counter()
        per_time.append(now-last[0])
        h = orig_fast_hist(gt,pred,n)
        per_hist.append(h.copy())
        if not a.profile:
            g = np.asarray(gt).reshape(400,2048)
            p = np.asarray(pred).reshape(400,2048)
            per_boundary.append(boundary_counts(g,p,n))
            Image.fromarray(p.astype(np.uint8)).save(pred_dir/(str(len(per_hist)-1).zfill(4)+".png"))
        if a.profile and len(per_hist) == 1:
            events.clear()
            torch.cuda.reset_peak_memory_stats()
        atomic_json(outdir/"progress.json",{"done":len(per_hist),"total":len(files),
            "elapsed_sec":time.perf_counter()-start,"route_calls":audit["calls"],
            "fallbacks":audit["fallbacks"]})
        last[0] = time.perf_counter()
        return h
    v.fast_hist = measured_hist
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    agg = v.validation(model,loader,torch.device("cuda"),nc,names,"sam2_s",9,a.dataset,ns)
    torch.cuda.synchronize()
    max_alloc = torch.cuda.max_memory_allocated()/2**20
    max_reserved = torch.cuda.max_memory_reserved()/2**20
    metrics = json.loads((outdir/"metrics.json").read_text())
    if len(per_hist) != len(files):
        raise AuditFailure("Image count mismatch")
    if a.mode != "dense" and not audit["calls"]:
        raise AuditFailure("Sparse route was never called")
    lengths = audit["effective_lengths"]
    if lengths and max(lengths) != (32768 if a.mode=="dense" else a.budget):
        raise AuditFailure("Effective memory budget mismatch: "+str(lengths))
    np.savez_compressed(outdir/"per_image.npz",hist=np.stack(per_hist),
        boundary=np.asarray(per_boundary,dtype=np.int64),image_time_sec=np.asarray(per_time))
    result = {"status":"complete","mode":a.mode,"score":a.score,"budget":a.budget,
        "dataset":a.dataset,"full_list_count":full_count,"num_eval_images":len(files),
        "class_names":names,"metrics":metrics,"audit":audit,
        "peak_allocated_mib":max_alloc,"peak_reserved_mib":max_reserved,
        "profile":a.profile}
    if per_boundary:
        c=np.asarray(per_boundary).sum(axis=0)
        precision=float(c[2]/max(c[0],1));recall=float(c[3]/max(c[1],1))
        result["boundary_f1_2px"]=200*precision*recall/max(precision+recall,1e-12)
    if a.profile:
        ts=np.asarray(per_time[1:])
        result["isolated_timing"]={"warmup_images":1,"measured_images":len(ts),
            "mean_image_sec":float(ts.mean()),"p50_image_sec":float(np.median(ts)),
            "p95_image_sec":float(np.percentile(ts,95)),
            "attention_cuda_ms_per_image":float(sum(s.elapsed_time(e) for s,e in events)/len(ts))}
    atomic_json(outdir/"result.json",result)
    print("COMPLETE",json.dumps({k:result[k] for k in ["mode","score","budget","dataset","status"]}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",required=True,choices=["dense","feature_norm","uncertainty_topk","spatial_pool","pool","boundary_guarded","class_aware","joint_guarded"])
    ap.add_argument("--score",default="uncertainty",choices=["feature_norm","uncertainty"])
    ap.add_argument("--budget",type=int,default=2048)
    ap.add_argument("--dataset",default="DensePASS",choices=["DensePASS","WildPASS"])
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--threads",type=int,default=2)
    ap.add_argument("--rare-ratio",type=float,default=0.15)
    ap.add_argument("--boundary-ratio",type=float,default=0.125)
    ap.add_argument("--profile",action="store_true")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    try:
        main(args)
    except BaseException as exc:
        Path(args.out).mkdir(parents=True,exist_ok=True)
        atomic_json(Path(args.out)/"failure.json",{"status":"failed","error":str(exc),"traceback":traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
