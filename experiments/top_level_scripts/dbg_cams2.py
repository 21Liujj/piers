import sys, traceback
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader
import argparse
ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=None, log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True)
val_loader, class_names, num_classes = build_dataset_and_loader(ns)
ns.num_classes=num_classes
device=torch.device('cuda')
model=build_sam_model(ns,device,ckpt_path=ns.ckpt)
model.eval()
mb=model.memory_bank
mb.configure_sparse_retrieval(enabled=True, mode='cams', keep_ratio=0.0625, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True)
mb.set_runtime_context(stage='eval', iter_idx=0)
# direct unit test of _cams_score with realistic shapes
L_m, L_q, C = 4096, 4096, 64
mt = torch.randn(L_m,1,C,device=device); q = torch.randn(L_q,C,device=device)
try:
    s = mb.sparse_retriever._cams_score(mt, q)
    print('UNIT-CAMS-OK', tuple(s.shape), float(s.min()), float(s.max()), flush=True)
except Exception:
    traceback.print_exc(); print('UNIT-CAMS-FAIL', flush=True)
# hook retrieve to dump query info once
orig = mb.sparse_retriever.retrieve
state={'n':0}
def logged(*a, **k):
    state['n']+=1
    if state['n']<=2:
        qf = k.get('query_features')
        mt_ = k.get('memory_tokens', a[0] if a else None)
        print(f"[HOOK] call{state['n']} mode={k.get('mode')} mt_shape={None if mt_ is None else tuple(mt_.shape)} q_shape={None if qf is None else tuple(qf.shape)} q_numel={None if qf is None else qf.numel()}", flush=True)
        # manual resolve to see exception
        try:
            from model.modules.sparse_memory_retriever import SparseMemoryRetriever as R
            sc, fb, um = mb.sparse_retriever._resolve_score(k.get('mode'), mt_, None, None, None, None, qf, 'feature_norm')
            print(f"[HOOK] resolve used={um} fb={fb}", flush=True)
        except Exception:
            traceback.print_exc()
    return orig(*a, **k)
mb.sparse_retriever.retrieve = logged
val_loader.dataset.files = val_loader.dataset.files[:1]
import torch.utils.data as tud
vl = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
with torch.no_grad():
    for bi,(image,label,_,names,ori) in enumerate(vl):
        image=image.to(device)
        nf = val_loader.dataset._get_frame_cnt()
        for fi in range(min(nf,4)):
            model(fi, image[:,fi])
        print('last_sparse=', mb.last_sparse_stats.get('selection_mode'), 'fb=', mb.last_sparse_stats.get('fallback_used'), flush=True)
        break
print('DONE', flush=True)
