"""I2跨域修正版: 强制labelId解释 WildPASS raw (0=void->255), 绕开heuristic误判.
flat集 images/labels_raw 配对, labels_trainid作废不用.
只比 dense vs ca 主判线, N=2048.
"""
import os, sys, json, time, argparse
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch, numpy as np
import utils.sam2_bootstrap
import utils.wildpass_label_convert as wlc
from utils.wildpass_label_convert import LUT_LABELID_TO_TRAINID, map19_trainid_to_train13
def forced_labelid(raw):
    m = np.asarray(raw, dtype=np.int64)
    tid = LUT_LABELID_TO_TRAINID[np.clip(m, 0, 255)]
    out = map19_trainid_to_train13(tid)
    return out.astype(np.int32), "forced_labelId"
wlc.convert_wildpass_raw_to_train13 = forced_labelid
from validation import build_sam_model, build_dataset_and_loader, validation
def run_variant(mode, kr, n_img, tag):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='WildPASS', class_set=13,
        data_root='/root/autodl-tmp/data', wildpass_layout='flat',
        wildpass_processed_root='/root/autodl-tmp/data_extra/WildPASS13',
        val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_raw',
        stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256,
        batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth',
        ckpt_list=None, ckpt_names=None,
        metrics_json=f'/root/autodl-tmp/piers-main/runs/{tag}/metrics.json',
        log_file=None, log_append=False, enable_scm=False, scm_pad_width=512,
        enable_bepl=False, num_maskmem=9, use_mem_bank=True, save_pred_debug=0,
        pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='',
        pred_gt_histogram_md='', timing_json=None, run_label=tag)
    os.makedirs(f'/root/autodl-tmp/piers-main/runs/{tag}', exist_ok=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    print(f'[XFIX] total={len(val_loader.dataset.files)} take={n_img}', flush=True)
    val_loader.dataset.files = val_loader.dataset.files[:n_img]
    import torch.utils.data as tud
    val_loader = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank
    orig_fwd = model.forward
    hist = {}
    def wrapped(frame_idx, x, mem_type='source', historical_pseudo_labels=None, **kw):
        merged = dict(hist)
        if historical_pseudo_labels: merged.update(historical_pseudo_labels)
        out = orig_fwd(frame_idx, x, mem_type=mem_type, historical_pseudo_labels=merged if merged else None, **kw)
        try:
            pred = out[0] if isinstance(out,(list,tuple)) else out
            hist[frame_idx]=pred.argmax(dim=1).detach().cpu()
        except Exception: pass
        return out
    model.forward = wrapped
    if mode=='dense':
        mb.configure_sparse_retrieval(enabled=False)
    else:
        kw={} if mode!='class_aware' else dict(rare_class_ids=[3,4,6,7],rare_reserve_ratio=0.15,boundary_score_source='uncertainty')
        mb.configure_sparse_retrieval(enabled=True, mode=mode, keep_ratio=kr, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True, **kw)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    orig_init = mb._init_output_dict_source
    def wrapped_init():
        hist.clear()
        return orig_init()
    mb._init_output_dict_source = wrapped_init
    ns._skip_per_run_json=False
    t0=time.perf_counter()
    agg = validation(model=model, data_loader=val_loader, device=device, num_classes=num_classes, class_names=class_names, backbone_name='sam2_s', num_maskmem=9, dataset_name='WildPASS', args=ns)
    dt=time.perf_counter()-t0
    out={'tag':tag,'dataset':'WildPASS-fix','mode':mode,'kr':kr,'n_img':n_img,'eval_time_sec':round(dt,2),'mIoU':agg.get('mIoU'),'mAcc':agg.get('mAcc'),'PixAcc':agg.get('PixAcc'),'eff_len':int(getattr(mb,'last_effective_memory_length',0))}
    print('RESULT '+json.dumps(out), flush=True)
    json.dump(out, open(f'/root/autodl-tmp/piers-main/runs/{tag}/summary.json','w'), indent=2)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='dense');ap.add_argument('--kr',type=float,default=0.0625);ap.add_argument('--n_img',type=int,default=500);ap.add_argument('--tag',default='XWF-dense')
    a=ap.parse_args();run_variant(a.mode,a.kr,a.n_img,a.tag)
