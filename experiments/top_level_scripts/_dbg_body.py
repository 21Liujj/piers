import sys
sys.path.insert(0,'/root/autodl-tmp/piers-main/third_party/effective_omnisam')
import argparse, os
import torch, numpy as np
import utils.sam2_bootstrap
from validation import build_dataset_and_loader, build_sam_model
E='/root/autodl-tmp/envs/piers/bin/python'
def one(dataset, label_dir, idx=0):
    ns = argparse.Namespace(seed=1234, backbone='sam2_s', dataset=dataset, class_set=13,
        data_root='/root/autodl-tmp/data', wildpass_layout='flat',
        wildpass_processed_root='/root/autodl-tmp/data_extra/WildPASS13',
        val_list='', wildpass_image_dir='images', wildpass_label_dir=label_dir,
        stanford_processed_root='', stanford_val_list='', stanford_legacy_area_crop=False, stanford_stride_w=256,
        batch_size=1, ckpt='/root/autodl-tmp/weights/omnisam_s.pth',
        enable_scm=False, scm_pad_width=512, enable_bepl=False, num_maskmem=9, use_mem_bank=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    print(f'dataset {dataset} files {len(val_loader.dataset.files)} classes {num_classes}', flush=True)
    rec = val_loader.dataset.files[idx]
    print('rec', rec, flush=True)
    img, label, size, name, ori = val_loader.dataset[idx]
    print(f'img shape {tuple(img.shape)} label shape {tuple(label.shape)} ori uniq {np.unique(np.array(ori)).tolist()[:20]} size {size} name {name}', flush=True)
    # model forward first window only
    device=torch.device('cuda')
    ns.num_classes=num_classes
    model=build_sam_model(ns, device, ckpt_path=ns.ckpt)
    model.eval()
    with torch.no_grad():
        f0 = img[0:1].unsqueeze(0).to(device)  # [1,1,C,H,W]? dataset returns [F,C,H,W]
        print('f0 shape', tuple(f0.shape), flush=True)
        # dataset img: [F,C,1024,1024]? take frame0
        frame = img[0].unsqueeze(0).to(device)
        print('frame shape', tuple(frame.shape), flush=True)
        out, feat, feat_de = model(0, frame)
        print('out shape', tuple(out.shape), 'out min/max', float(out.min()), float(out.max()), flush=True)
        pred = out.argmax(dim=1).cpu().numpy()
        u,v=np.unique(pred, return_counts=True)
        print('pred uniq', dict(zip([int(x) for x in u],[int(x) for x in v])), flush=True)
        # full merge quick? run validation-style merge for this single image using model.memory_bank? skip, just frame-level
    print('done-one', flush=True)
try:
    one('WildPASS','labels_trainid',0)
except Exception as e:
    import traceback; traceback.print_exc()
try:
    one('DensePASS','',0)
except Exception as e:
    import traceback; traceback.print_exc()
