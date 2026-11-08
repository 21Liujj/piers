import os, sys, json, time
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader, validation
import argparse
def run_variant(mode, kr, n_img=20, tag='M2x', extra=''):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=f'/root/autodl-tmp/piers-main/runs/{tag}/metrics.json', log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True, save_pred_debug=0, pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='', pred_gt_histogram_md='', timing_json=None, run_label=tag)
    os.makedirs(f'/root/autodl-tmp/piers-main/runs/{tag}', exist_ok=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    val_loader.dataset.files = val_loader.dataset.files[:n_img]
    import torch.utils.data as tud
    val_loader = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank
    # wrap model.forward to feed historical pseudo labels from previous frames
    orig_fwd = model.forward
    hist = {}
    cams_log = {'calls':0,'fallbacks':0,'first_exc':None}
    orig_resolve = mb.sparse_retriever._resolve_score
    def logged_resolve(mode_in=None, tokens_lbc=None, attn_scores=None, confidence_map=None, overlap_mask=None, boundary_mask=None, query_features=None, fallback='feature_norm', mode=None, **kw):
        _mode = mode if mode is not None else mode_in
        cams_log['calls']+=1
        try:
            out = orig_resolve(_mode, tokens_lbc, attn_scores, confidence_map, overlap_mask, boundary_mask, query_features, fallback)
        except Exception as e:
            cams_log['fallbacks']+=1
            if not cams_log['first_exc']: cams_log['first_exc']=repr(e)
            raise
        if out[1]: cams_log['fallbacks']+=1
        if cams_log['calls']<=3:
            print(f'[RESOLVE] in={_mode} used={out[2]} fb={out[1]} L={tokens_lbc.shape[0]} q={None if query_features is None else tuple(query_features.shape)} bmask={None if boundary_mask is None else tuple(boundary_mask.shape)}',flush=True)
        return out
    mb.sparse_retriever._resolve_score = logged_resolve
    def wrapped(frame_idx, x, mem_type='source', historical_pseudo_labels=None, **kw):
        # merge self-kept hist with any passed
        merged = dict(hist)
        if historical_pseudo_labels:
            merged.update(historical_pseudo_labels)
        out = orig_fwd(frame_idx, x, mem_type=mem_type, historical_pseudo_labels=merged if merged else None, **kw)
        # out is (pred, feat, feat_de); pred [B,K,H,W] -> pseudo [B,H,W]
        try:
            pred = out[0] if isinstance(out,(list,tuple)) else out
            import torch.nn.functional as F
            pl = pred.argmax(dim=1).detach().cpu()
            hist[frame_idx]=pl
        except Exception as e:
            pass
        return out
    model.forward = wrapped
    if mode=='dense':
        mb.configure_sparse_retrieval(enabled=False)
    else:
        kw={}
        if mode in ('boundary_guarded', 'joint_guarded'):
            kw=dict(boundary_reserve_ratio=0.125, boundary_score_source='uncertainty')
        if mode in ('class_aware', 'joint_guarded'):
            kw=dict(rare_class_ids=[3,4,6,7], rare_reserve_ratio=0.15, boundary_score_source='uncertainty')
        mb.configure_sparse_retrieval(enabled=True, mode=mode, keep_ratio=kr, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True, **kw)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    # clear hist per image: hook _init_output_dict_source
    orig_init = mb._init_output_dict_source
    def wrapped_init():
        hist.clear()
        return orig_init()
    mb._init_output_dict_source = wrapped_init
    ns._skip_per_run_json=False
    t0=time.perf_counter()
    agg = validation(model=model, data_loader=val_loader, device=device, num_classes=num_classes, class_names=class_names, backbone_name='sam2_s', num_maskmem=9, dataset_name='DensePASS', args=ns)
    dt=time.perf_counter()-t0
    out={'tag':tag,'mode':mode,'kr':kr,'n_img':n_img,'eval_time_sec':round(dt,2),'mIoU':agg.get('mIoU'),'mAcc':agg.get('mAcc'),'PixAcc':agg.get('PixAcc'),'sparse_last':getattr(mb,'last_sparse_stats',{}).get('selection_mode'),'eff_len':int(getattr(mb,'last_effective_memory_length',0)),'cams_log':cams_log}
    print('RESULT '+json.dumps(out), flush=True)
    json.dump(out, open(f'/root/autodl-tmp/piers-main/runs/{tag}/summary.json','w'), indent=2)
if __name__=='__main__':
    import sys as _s
    run_variant(_s.argv[1], float(_s.argv[2]), n_img=int(_s.argv[3]), tag=_s.argv[4])
