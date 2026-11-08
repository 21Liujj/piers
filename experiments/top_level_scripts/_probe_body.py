import sys
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import argparse, json, time, os
import torch
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader, validation
def probe(label_dir, tag):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='WildPASS', class_set=13,
        data_root='/root/autodl-tmp/data', wildpass_layout='flat',
        wildpass_processed_root='/root/autodl-tmp/data_extra/WildPASS13',
        val_list='', wildpass_image_dir='images', wildpass_label_dir=label_dir,
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
    import torch.utils.data as tud
    # first 5 files
    val_loader.dataset.files = val_loader.dataset.files[:5]
    val_loader = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank
    mb.configure_sparse_retrieval(enabled=False)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    ns._skip_per_run_json=False
    agg = validation(model=model, data_loader=val_loader, device=device, num_classes=num_classes, class_names=class_names, backbone_name='sam2_s', num_maskmem=9, dataset_name='WildPASS', args=ns)
    out={'tag':tag,'label_dir':label_dir,'mIoU':agg.get('mIoU'),'mAcc':agg.get('mAcc'),'PixAcc':agg.get('PixAcc')}
    print('PROBE_RESULT '+json.dumps(out), flush=True)
for ld, tg in [('labels_trainid','PW5-cur'),('labels_trainid.backup_20260511_175555','PW5-bak'),('labels_raw','PW5-raw')]:
    try:
        probe(ld, tg)
    except Exception as e:
        print(f'PROBE_FAIL {ld} {repr(e)}', flush=True)
print('probes done', flush=True)
