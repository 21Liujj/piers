"""Worker entrypoints — independent processes, sys/env diagnostics first.

Each worker prints sys.executable, key package versions, and the REAL loaded
module file (sam2.__file__ / model module path) and FAILS if the import path
points at the wrong source (e.g. repo-root shadowing vendor cwd).

Run from the vendor cwd so repo-root sam2()/utils() never shadows:
  cd third_party/grounded_sam2 && conda run -n piers-grounded \
    python -m piers_ljj.workers.entry grounded --config ... --manifest ... --out ...
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time


def diag(mod_names=("numpy", "PIL", "yaml")):
    import importlib
    print(f"executable: {sys.executable}")
    for m in mod_names:
        try:
            mod = importlib.import_module(m)
            print(f"{m}: {getattr(mod, '__version__', '?')} @ {getattr(mod, '__file__', '?')}")
        except Exception as e:
            print(f"{m}: IMPORT-FAIL {e}")


def _fail(msg: str) -> int:
    print(f"FATAL worker: {msg}")
    return 2


def _post_process_grounded(processor, outputs, input_ids, box_thr: float,
                           text_thr: float, target_sizes):
    """Call post_process_grounded_object_detection across transformers versions.

    The box-threshold kwarg was renamed `threshold` -> `box_threshold` between
    transformers releases. Resolve it from the live signature instead of
    hard-coding either name, so the worker does not break on a version bump.

    Falls back to trying both names when the signature only exposes **kwargs
    (e.g. a wrapper/mock), and raises loudly if neither is accepted.
    """
    import inspect
    fn = processor.post_process_grounded_object_detection
    kwargs = {"text_threshold": text_thr, "target_sizes": target_sizes}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}

    def _call():
        return fn(outputs, input_ids, **kwargs)

    if "box_threshold" in params:
        kwargs["box_threshold"] = box_thr
        return _call()
    if "threshold" in params:
        kwargs["threshold"] = box_thr
        return _call()

    # Signature is opaque (**kwargs only) -> try the modern name first, then
    # the legacy name, and report clearly if neither works.
    last_err = None
    for name in ("box_threshold", "threshold"):
        attempt = dict(kwargs)
        attempt[name] = box_thr
        try:
            return fn(outputs, input_ids, **attempt)
        except TypeError as e:
            last_err = e
    raise TypeError(
        "post_process_grounded_object_detection accepts neither "
        "'box_threshold' nor 'threshold'; transformers version unsupported "
        f"(last error: {last_err})")


def omni_main(args=None):
    diag()
    # Real OmniSAM invocation resolved in gpu_runbook after source audit.
    print("piers-omni worker: source audit pending (repository_audit.md); refusing to fake-load.")
    return 2


# ---------------------------------------------------------------------------
# piers-grounded real backend (HF Grounding-DINO + SAM2ImagePredictor)
# Mirrors third_party/grounded_sam2/grounded_sam2_hf_model_demo.py as an
# importable worker — NEVER imports the demo module (argparse+CUDA at import).
# ---------------------------------------------------------------------------

def _build_hf_backend(cfg, device: str):
    """Build (dino, sam) pair matching text.pipeline backend protocol.

    dino(pil_rgb, text, box_thr, text_thr) -> (boxes_xyxy_list, det_scores)
    sam.set_image(pil_rgb) -> enc ; sam.predict_boxes(enc, [box]) -> (masks, quals)
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    from piers_ljj.text.grounded_adapter import GroundedConfig
    assert isinstance(cfg, GroundedConfig)

    if not os.path.exists(cfg.sam2_ckpt):
        raise FileNotFoundError(
            f"sam2 checkpoint missing: {cfg.sam2_ckpt} (refusing synthetic fallback)")

    # Import sam2 lazily AFTER diag so the loaded path can be verified.
    import sam2 as _sam2mod
    print(f"sam2 backend: {_sam2mod.__file__}")
    repo_root_shadow = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "sam2"))
    if os.path.isfile(repo_root_shadow):
        pass  # a FILE named sam2 at repo root is harmless; a DIR shadowing is fatal
    if os.path.isdir(os.path.join(os.getcwd(), "..", "..", "sam2")) and \
            os.path.abspath(_sam2mod.__file__).startswith(os.path.abspath(os.path.join(os.getcwd(), "..", ".."))):
        # vendor cwd run: sam2 must come from vendor dir or site-packages, not repo-root copy
        print("WARNING worker: sam2 loaded from repo-root tree; "
              "run worker with cwd=third_party/grounded_sam2 to avoid shadowing")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(cfg.sam2_cfg, cfg.sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    processor = AutoProcessor.from_pretrained(
        cfg.grounding_model_id, local_files_only=cfg.local_files_only,
        trust_remote_code=False)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg.grounding_model_id, local_files_only=cfg.local_files_only,
        trust_remote_code=False).to(device)
    dino_model.eval()
    if getattr(cfg, "disable_custom_kernels", True):
        print("grounding-dino: disable_custom_kernels=True (transformers path, no custom ops)")

    use_amp = device.startswith("cuda") and cfg.dtype in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.dtype, torch.bfloat16)
    feature_cache = None
    if getattr(cfg, 'cache_prefusion_features', False):
        from piers_ljj.text.feature_cache import PreFusionFeatureCache
        feature_cache = PreFusionFeatureCache(dino_model, cfg.max_text_cache_entries)

    class _Dino:
        def __call__(self, pil_img: Image.Image, text: str, box_thr: float, text_thr: float):
            if feature_cache is not None:
                feature_cache.begin(pil_img, text)
            inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = dino_model(**inputs)
                else:
                    outputs = dino_model(**inputs)
            # (threshold -> box_threshold). Inspect the signature so the same
            # code runs on old and new transformers without guessing.
            results = _post_process_grounded(
                processor, outputs, inputs.input_ids, box_thr, text_thr,
                target_sizes=[pil_img.size[::-1]])
            r0 = results[0]
            import numpy as np
            boxes = [np.asarray(b, dtype=np.float64) for b in r0["boxes"].detach().cpu().numpy()]
            scores = [float(s) for s in r0["scores"].detach().cpu().numpy().tolist()]
            labels = r0.get("text_labels", r0.get("labels"))
            phrases = list(labels) if labels is not None else []
            return boxes, scores, phrases

        def cache_stats(self):
            return feature_cache.stats() if feature_cache is not None else {'enabled': False}

    class _Sam:
        def __init__(self):
            self.n_enc = 0
            self.n_dec = 0

        def set_image(self, pil_img: Image.Image):
            import numpy as np
            predictor.set_image(np.array(pil_img.convert("RGB")))
            self.n_enc += 1
            return object()

        def predict_boxes(self, enc, boxes):
            import numpy as np
            import torch
            arr = np.asarray(boxes, dtype=np.float32)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        masks, scores, _ = predictor.predict(
                            point_coords=None, point_labels=None,
                            box=arr, multimask_output=False)
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=None, point_labels=None,
                        box=arr, multimask_output=False)
            if masks.ndim == 4:
                masks = masks[:, 0]
            self.n_dec += len(boxes)
            return [np.asarray(m, dtype=bool) for m in masks], [float(s) for s in np.ravel(scores)]

    return _Dino(), _Sam()


def grounded_main(argv=None) -> int:
    import yaml
    ap = argparse.ArgumentParser(prog="piers_ljj grounded worker")
    ap.add_argument("--config", required=True, help="T0/T1/T2 yaml")
    ap.add_argument("--manifest", required=True, help="query manifest json (worker_view enforced)")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--max-queries", type=int, default=0, help="Debug-only query limit; zero means full manifest")
    a = ap.parse_args(argv)

    diag(("numpy", "PIL", "yaml", "torch", "transformers"))
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} cuda_available={torch.cuda.is_available()}")

    with open(a.config, encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    with open(a.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    from piers_ljj.text.grounded_adapter import GroundedConfig
    from piers_ljj.text.manifest import worker_view
    from piers_ljj.text.pipeline import run_erp_text
    from piers_ljj.collaboration.semantic_bridge import FrozenSemanticCache, guide_and_union
    from piers_ljj.geometry.erp import plan_fovs

    wman = worker_view(manifest)  # GT stripped even if present in file
    if a.max_queries:
        wman["queries"] = wman["queries"][:a.max_queries]
    g = run_cfg.get("grounded", {})
    gcfg = GroundedConfig(
        grounding_model_id=g.get("grounding_model_id", GroundedConfig.grounding_model_id),
        sam2_cfg=g.get("sam2_cfg", GroundedConfig.sam2_cfg),
        sam2_ckpt=g.get("sam2_ckpt", run_cfg.get("sam2_ckpt", "")),
        device=device,
        dtype=g.get("dtype", "bf16" if device.startswith("cuda") else "fp32"),
        box_thr=float(run_cfg.get("thresholds", {}).get("det", 0.35)),
        text_thr=float(run_cfg.get("thresholds", {}).get("text", 0.25)),
        disable_custom_kernels=bool(g.get("disable_custom_kernels", True)),
        local_files_only=bool(g.get("local_files_only", True)))

    try:
        backend = _build_hf_backend(gcfg, device)
    except Exception as e:
        return _fail(f"backend build failed: {e}")

    from PIL import Image
    import numpy as np
    os.makedirs(a.out, exist_ok=True)
    fov_cfg = run_cfg.get("fov", {})
    mode = "direct" if run_cfg.get("suite", "").startswith("T0") else "fov"
    collab = run_cfg.get("collab", "none")
    sem_cache = None
    if collab in {"semantic_to_text", "semantic_to_text_hard"}:
        try:
            sem_cache = FrozenSemanticCache(run_cfg["semantic_cache"], a.manifest)
            print(f"semantic cache: {sem_cache.directory}, images={len(sem_cache.ids)}")
        except Exception as e:
            return _fail(f"semantic cache load failed: {e}")
    elif collab != "none":
        return _fail(f"unsupported grounded collaboration mode: {collab}")

    t_all = time.perf_counter()
    all_preds, all_logs = [], []
    mask_store = {}          # query_id -> ERP-global binary mask (union of FoV cands)
    img_index = {im["image_id"]: im for im in wman["images"]}
    for q in wman["queries"]:
        im = img_index[q["image_id"]]
        pil = Image.open(im["path"]).convert("RGB")
        r = run_erp_text(pil, q["query_id"], q.get("norm_text") or q["raw_text"],
                         fov_cfg, gcfg,
                         backend_factory=lambda b=backend: b, mode=mode)
        raw_preds = r["predictions"]
        gm = r.pop("masks", [])
        H, W = pil.height, pil.width
        if mode == "direct":
            fovs = [{"view_id":"v000","x_start":0,"y_start":0,"win_w":W,"win_h":H}]
        else:
            fovs = plan_fovs(W,H,win_w=fov_cfg["win_w"],win_h=fov_cfg.get("win_h",H),
                            stride_x=fov_cfg.get("stride_x",fov_cfg["win_w"]),y_start=fov_cfg.get("y_start",0))
        valid_by_fov = {}
        for f in fovs:
            vv = np.zeros((H,W),bool)
            cols = (f["x_start"]+np.arange(f["win_w"]))%W
            vv[np.ix_(np.arange(f["y_start"],f["y_start"]+f["win_h"]),cols)] = True
            valid_by_fov[f["view_id"]] = vv
        probs = sem_cache.image(q["image_id"]) if sem_cache is not None else {}
        preds, union, log = guide_and_union(
            raw_preds, gm, [valid_by_fov[p["fov_id"]] for p in raw_preds],
            probs, q.get("ontology"), (H,W), mode=collab,
            alpha=float(run_cfg.get("collab_params",{}).get("alpha",0.5)),
            hard_thr=float(run_cfg.get("collab_params",{}).get("hard_thr") or 0.15))
        all_logs.extend(log)
        qid = q["query_id"]
        if os.path.basename(qid) != qid:
            raise ValueError("Invalid query identifier")
        candidate_dir = os.path.join(a.out,"candidates")
        os.makedirs(candidate_dir,exist_ok=True)
        packed = np.packbits(np.stack(gm).astype(bool),axis=2) if gm else np.zeros((0,H,(W+7)//8),np.uint8)
        candidate_path = os.path.join(candidate_dir,qid+".npz")
        np.savez_compressed(candidate_path,candidate_ids=np.asarray([p["id"] for p in raw_preds],dtype=str),
                            masks_packed=packed,shape=np.asarray([H,W]))
        mask_store[qid] = union
        r["candidate_mask_file"] = candidate_path
        r["candidate_masks_count"] = len(gm)
        all_preds.append({"query_id":qid,"image_id":q["image_id"],"mode":mode,
                          "result":r,"predictions":preds,"mask_selection":"kept_candidate_union"})
    dt = time.perf_counter() - t_all

    # Persist per-query binary masks for text IoU evaluation (evaluate reads
    # predictions only; never re-runs inference).
    try:
        import numpy as np
        if mask_store:
            qids = sorted(mask_store.keys())
            stack = np.stack([mask_store[k] for k in qids]).astype(np.uint8)
            np.savez_compressed(os.path.join(a.out, "masks.npz"),
                                query_ids=np.array(qids), masks=stack)
            print(f"mask store: {stack.shape} -> {os.path.join(a.out, 'masks.npz')}")
    except Exception as e:
        print(f"WARN mask store failed: {e}")

    with open(os.path.join(a.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_preds, f, indent=2, default=str)
    with open(os.path.join(a.out, "collab_log.json"), "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, default=str)
    with open(os.path.join(a.out, "status.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "completed_real" if device.startswith("cuda") else "passed_cpu",
                   "device": device, "suite": run_cfg.get("suite"),
                   "collab": collab, "latency_s": dt, "synthetic": False}, f, indent=2)
    print(f"grounded worker done: {len(all_preds)} queries in {dt:.1f}s -> {a.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="piers_ljj.worker")
    ap.add_argument("which", choices=["omni", "grounded"])
    a, rest = ap.parse_known_args(argv)
    if a.which == "omni":
        return omni_main(rest)
    return grounded_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
