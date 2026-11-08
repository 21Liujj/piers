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
# monkeypatch _cams_score to show exception
orig_cams = mb.sparse_retriever._cams_score
def logged_cams(*a, **k):
    try:
        print(f"[CAMS-IN] mt={tuple(a[0].shape) if len(a)>0 else None} q={None if k.get('query_features') is None else tuple(k.get('query_features').shape)}", flush=True)
        return orig_cams(*a, **k)
    except Exception:
        traceback.print_exc(); print('[CAMS-EXC]', flush=True); raise
mb.sparse_retriever._cams_score = logged_cams
# also wrap _resolve_score cams branch manually
import model.modules.sparse_memory_retriever as R
orig_res = mb.sparse_retriever._resolve_score
def logged_res(mode, tokens_lbc, attn_scores=None, confidence_map=None, overlap_mask=None, boundary_mask=None, query_features=None, fallback='feature_norm'):
    if (mode or '').lower()=='cams':
        print(f"[RES-CAMS] L={tuple(tokens_lbc.shape)} q={'None' if query_features is None else tuple(query_features.shape)+' numel='+str(query_features.numel())} dev={tokens_lbc.device}", flush=True)
    return orig_res(mode, tokens_lbc, attn_scores, confidence_map, overlap_mask, boundary_mask, query_features, fallback)
mb.sparse_retriever._resolve_score = logged_res
val_loader.dataset.files = val_loader.dataset.files[:1]
import torch.utils.data as tud
vl = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
with torch.no_grad():
    for bi,(image,label,_,names,ori) in enumerate(vl):
        image=image.to(device)
        nf = val_loader.dataset._get_frame_cnt()
        print('nf=',nf,flush=True)
        for fi in range(min(nf,3)):
            model(fi, image[:,fi])
        print('last=', mb.last_sparse_stats, flush=True)
        break
print('DONE',flush=True)
