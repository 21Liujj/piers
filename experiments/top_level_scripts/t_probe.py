"""T probe: single image single query timing for T0 direct vs T1 fov (real backend)."""
import os, sys, time, json
sys.path.insert(0, '/root/autodl-tmp/piers-main/src')
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/grounded_sam2')
os.chdir('/root/autodl-tmp/piers-main/third_party/grounded_sam2')
from piers_ljj.text.grounded_adapter import GroundedConfig
from piers_ljj.text.pipeline import run_erp_text
from piers_ljj.workers.entry import _build_hf_backend
from PIL import Image
man = json.load(open('/root/autodl-tmp/piers-main/configs/piers/manifest.dev-calib-10.json'))
img = man['images'][0]
q = [x for x in man['queries'] if x['image_id']==img['image_id'] and x['raw_text']=='car'][0]
print('image:', img['path'], img['width'], img['height'], flush=True)
print('query:', q, flush=True)
gcfg = GroundedConfig(grounding_model_id='/root/autodl-tmp/weights/grounding-dino-tiny-hf', sam2_cfg='configs/sam2.1/sam2.1_hiera_s.yaml', sam2_ckpt='/root/autodl-tmp/weights/sam2.1_hiera_small.pt', device='cuda', dtype='bf16')
t0=time.perf_counter()
backend = _build_hf_backend(gcfg, 'cuda')
print(f'backend built in {time.perf_counter()-t0:.1f}s', flush=True)
pil = Image.open(img['path']).convert('RGB')
for mode, fov in [('direct', {}), ('fov', {'win_w':1024,'win_h':400,'stride_x':512,'y_start':0})]:
    t1=time.perf_counter()
    r = run_erp_text(pil, q['query_id'], q['raw_text'], fov, gcfg, backend_factory=lambda b=backend: b, mode=mode)
    dt=time.perf_counter()-t1
    print(f"MODE {mode}: fovs={r['fovs']} preds={len(r['predictions'])} groups={len(r['groups'])} calls={r['calls']} dt={dt:.1f}s", flush=True)
    for p in r['predictions'][:3]:
        print('  pred:', p['id'], p['detector_score'], p['sam_quality'], p['phrase'], 'segs=', len(p['box_segs']), flush=True)
print('PROBE-DONE', flush=True)
