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

    class _Dino:
        def __call__(self, pil_img: Image.Image, text: str, box_thr: float, text_thr: float):
            inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = dino_model(**inputs)
                else:
                    outputs = dino_model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_thr,
                text_threshold=text_thr, target_sizes=[pil_img.size[::-1]])
            r0 = results[0]
            import numpy as np
            boxes = [np.asarray(b, dtype=np.float64) for b in r0["boxes"].detach().cpu().numpy()]
            scores = [float(s) for s in r0["scores"].detach().cpu().numpy().tolist()]
            return boxes, scores

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
    from piers_ljj.collaboration.text_guide import apply_mode

    wman = worker_view(manifest)  # GT stripped even if present in file
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
    os.makedirs(a.out, exist_ok=True)
    fov_cfg = run_cfg.get("fov", {})
    mode = "direct" if run_cfg.get("suite", "").startswith("T0") else "fov"
    collab = run_cfg.get("collab", "none")
    sem_cache = {}
    if collab.startswith("semantic"):
        # frozen E0 semantic cache read (never recomputed here)
        import numpy as np
        cdir = run_cfg.get("semantic_cache", "")
        try:
            d = np.load(os.path.join(cdir, "probs.npz"))
            sem_cache = {k: d[k] for k in d.files}
            print(f"semantic cache: {cdir} keys={list(sem_cache)}")
        except Exception as e:
            return _fail(f"semantic cache load failed ({cdir}): {e}")

    t_all = time.perf_counter()
    all_preds, all_logs = [], []
    img_index = {im["image_id"]: im for im in wman["images"]}
    for q in wman["queries"]:
        im = img_index[q["image_id"]]
        pil = Image.open(im["path"]).convert("RGB")
        r = run_erp_text(pil, q["query_id"], q.get("norm_text") or q["raw_text"],
                         fov_cfg, gcfg,
                         backend_factory=lambda b=backend: b, mode=mode)
        preds = r["predictions"]
        if collab.startswith("semantic") and q["image_id"] in sem_cache:
            preds, log = apply_mode(preds, sem_cache[q["image_id"]], mode="semantic_to_text",
                                    alpha=float(run_cfg.get("collab_params", {}).get("alpha", 0.5)))
            all_logs.extend(log)
        elif collab.startswith("semantic"):
            preds, log = apply_mode(preds, {}, mode="none")
            all_logs.extend({**l, "reason": "no cache for image; kept original"} for l in log)
        all_preds.append({"query_id": q["query_id"], "image_id": q["image_id"],
                          "mode": mode, "result": r, "predictions": preds})
    dt = time.perf_counter() - t_all

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
