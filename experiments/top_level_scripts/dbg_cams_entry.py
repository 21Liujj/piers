import sys
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
print('sparse_enabled before:', mb.sparse_enabled, mb.sparse_mode, flush=True)
mb.configure_sparse_retrieval(enabled=True, mode='cams', keep_ratio=0.0625, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True)
mb.set_runtime_context(stage='eval', iter_idx=0)
print('after cfg:', mb.sparse_enabled, mb.sparse_mode, mb.runtime_stage, 'enabled_now=', mb._sparse_enabled_now(), flush=True)
# patch retrieve to log
orig_ret = mb.sparse_retriever.retrieve
def logged_ret(*a, **k):
    print(f"[RETRIEVE] mode={k.get('mode', a[5] if len(a)>5 else '?')} L={a[0].shape[0] if len(a)>0 else k.get('memory_tokens', '?')}", flush=True)
    try:
        out = orig_ret(*a, **k)
        print(f"[RETRIEVE-OK] sel={out[3].selection_mode} kept={out[3].selected_num_tokens} fb={out[3].fallback_used}", flush=True)
        return out
    except Exception as e:
        import traceback; traceback.print_exc(); print('[RETRIEVE-EXC]', repr(e), flush=True); raise
mb.sparse_retriever.retrieve = logged_ret
# patch memory_bank.forward to log prepared_chunks
orig_fwd = mb.forward
def logged_fwd(frame_idx, *a, **k):
    print(f'[MB-FWD] frame={frame_idx} stage={mb.runtime_stage} sparse_now={mb._sparse_enabled_now()}', flush=True)
    return orig_fwd(frame_idx, *a, **k)
mb.forward = logged_fwd
val_loader.dataset.files = val_loader.dataset.files[:2]
import torch.utils.data as tud
vl = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
with torch.no_grad():
    for bi,(image,label,_,names,ori) in enumerate(vl):
        image=image.to(device)
        nf = val_loader.dataset._get_frame_cnt()
        print(f'=== img {bi} nf={nf} ===', flush=True)
        for fi in range(nf):
            out,_,_ = model(fi, image[:,fi])
        print(f'last_sparse={mb.last_sparse_stats} eff={mb.last_effective_memory_length} before={mb.last_memory_tokens_before} after={mb.last_sparse_tokens_after}', flush=True)
        mb._init_output_dict_source()
print('DBG-DONE', flush=True)
