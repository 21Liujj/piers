import os, sys
sys.path.insert(0, '/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import torch
import utils.sam2_bootstrap  # noqa
from validation import build_sam_model, build_dataset_and_loader
import argparse
ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset='DensePASS', class_set=13, data_root='/root/autodl-tmp/data', wildpass_layout='cityscapes', wildpass_processed_root='', val_list='', wildpass_image_dir='images', wildpass_label_dir='labels_trainid', stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256, batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth', ckpt_list=None, ckpt_names=None, metrics_json=None, log_file=None, log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True)
val_loader, class_names, num_classes = build_dataset_and_loader(ns)
ns.num_classes = num_classes
device = torch.device('cuda')
model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
model.eval()
model.memory_bank.set_runtime_context(stage='eval', iter_idx=0)
# dense: 2 images
import itertools
from torch.utils.data import DataLoader
it = iter(val_loader)
for idx in range(2):
    image, label, _, names, ori = next(it)
    image=image.to(device)
    B=image.shape[0]; nf = val_loader.dataset._get_frame_cnt()
    with torch.no_grad():
        for fi in range(nf):
            model(fi, image[:,fi])
        mb=model.memory_bank
        print(f"img{idx} dense_effective_len={mb.last_effective_memory_length} before={mb.last_memory_tokens_before} after={mb.last_memory_tokens_after}", flush=True)
        mb._init_output_dict_source()
# now enable sparse feature_norm keep 0.5 and 0.25
for kr in [0.5, 0.25, 0.125]:
    model.memory_bank.configure_sparse_retrieval(enabled=True, mode='feature_norm', keep_ratio=kr, min_tokens=64, cache_enabled=False, fallback='feature_norm', fallback_full_when_uncertain=True, enable_in_train=True, enable_in_pseudo_label=True)
    model.memory_bank.set_runtime_context(stage='eval', iter_idx=0)
    model.memory_bank._init_output_dict_source()
    image, label, _, names, ori = next(it)
    image=image.to(device)
    with torch.no_grad():
        for fi in range(nf):
            model(fi, image[:,fi])
        mb=model.memory_bank
        print(f"kr={kr} eff_len={mb.last_effective_memory_length} before={mb.last_memory_tokens_before} after={mb.last_sparse_tokens_after} sparse={mb.last_sparse_stats.get('selection_mode')}", flush=True)
        mb._init_output_dict_source()
print("PROBE_DONE")
