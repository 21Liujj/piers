import sys
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch
import utils.sam2_bootstrap
from validation import build_sam_model, build_dataset_and_loader
import argparse, model.modules.sparse_memory_retriever as R, model.memory_bank as MB
print("retriever file:", R.__file__, flush=True)
print("membank file:", MB.__file__, flush=True)
ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=None, log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True)
val_loader, class_names, num_classes = build_sam_model if False else build_dataset_and_loader(ns)
ns.num_classes=num_classes
device=torch.device('cuda')
model=build_sam_model(ns,device,ckpt_path=ns.ckpt)
model.eval()
mb=model.memory_bank
mb.configure_sparse_retrieval(enabled=True, mode='cams', keep_ratio=0.0625, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True)
mb.set_runtime_context(stage='eval', iter_idx=0)
val_loader.dataset.files = val_loader.dataset.files[:1]
import torch.utils.data as tud
vl = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
with torch.no_grad():
    for bi,(image,label,_,names,ori) in enumerate(vl):
        image=image.to(device)
        nf = val_loader.dataset._get_frame_cnt()
        for fi in range(min(nf,3)):
            model(fi, image[:,fi])
        print('last=', mb.last_sparse_stats, flush=True)
        break
print('DONE',flush=True)
