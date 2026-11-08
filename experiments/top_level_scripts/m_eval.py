import os, sys, json, time
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader, validation
import argparse
def run_variant(mode, kr, extra_cfg=None, n_img=20, tag='M'):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=f'/root/autodl-tmp/piers-main/runs/{tag}/metrics.json', log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True, save_pred_debug=0, pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='', pred_gt_histogram_md='', timing_json=None, run_label=tag)
    os.makedirs(f'/root/autodl-tmp/piers-main/runs/{tag}', exist_ok=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    # truncate to n_img for smoke
    val_loader.dataset.files = val_loader.dataset.files[:n_img]
    from torch.utils.data import DataLoader
    from functools import partial
    import torch.utils.data as tud
    val_loader = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank
    if mode=='dense':
        mb.configure_sparse_retrieval(enabled=False)
    else:
        kw={}
        if mode=='boundary_guarded':
            kw=dict(boundary_reserve_ratio=0.125)
        if mode=='class_aware':
            kw=dict(rare_class_ids=[3,4,5,6], rare_reserve_ratio=0.15)
        mb.configure_sparse_retrieval(enabled=True, mode=mode, keep_ratio=kr, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True, **kw)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    ns._skip_per_run_json=False
    t0=time.perf_counter()
    agg = validation(model=model, data_loader=val_loader, device=device, num_classes=num_classes, class_names=class_names, backbone_name='sam2_s', num_maskmem=9, dataset_name='DensePASS', args=ns)
    dt=time.perf_counter()-t0
    out={'tag':tag,'mode':mode,'kr':kr,'n_img':n_img,'eval_time_sec':round(dt,2),'mIoU':agg.get('mIoU'),'mAcc':agg.get('mAcc'),'PixAcc':agg.get('PixAcc'),'sparse_last':getattr(mb,'last_sparse_stats',{}).get('selection_mode'),'eff_len':int(getattr(mb,'last_effective_memory_length',0))}
    print('RESULT '+json.dumps(out), flush=True)
    json.dump(out, open(f'/root/autodl-tmp/piers-main/runs/{tag}/summary.json','w'), indent=2)
if __name__=='__main__':
    import sys as _s
    run_variant(_s.argv[1], float(_s.argv[2]), n_img=int(_s.argv[3]), tag=_s.argv[4])
