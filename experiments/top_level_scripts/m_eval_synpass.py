"""SynPASS 适配：室外街景全景，完整 13 类（与我们的模型头完全对齐）。

与 DensePASS / WildPASS 的差异
------------------------------
* 图像 2048x1024（ERP），SynPASS 原生 22 类语义。
* 目录组织：`img/<weather>/<split>/<MAP_xx_pointYY>/<id>.jpg`，
  标签 `semantic/<weather>/<split>/<MAP_xx_pointYY>/<id>_trainID.png`。
  文件名规则是 `img` -> `semantic`、`.jpg` -> `_trainID.png`。
* SynPASS 原生 23 标签值 -> Cityscapes 13 类的官方映射键（见
  dataset/SynPASS.py 的 `_key`）：
      [255,2,4,255,11,5,0,0,1,8,12,3,7,10,255,255,255,255,6,255,255,255,9]
  该键覆盖全部 13 个 Cityscapes 类（road/sidewalk/building/wall/fence/pole/
  tlight/tsign/veg/terrain/sky/person/car），因此 SynPASS 可以给出
  **完整 13 类**的室外全景泛化结果，弥补 WildPASS 只标注 8 类的不足。

* 评测协议与 DensePASS 一致：resize 到 4096x1024，滑窗 1024/stride384，双 pass。

用法
----
python m_eval_synpass.py --n_img 500 --tag GEN-SP500-dense
"""
import os
import sys
import json
import time
import argparse
import types
import importlib.util

import numpy as np

EFF = '/root/autodl-tmp/eff_omni'
sys.path.insert(0, EFF)


def _install_dynamic_fsmc_shim():
    """`model.model` imports `model.modules.dynamic_fsmc`, but that module was
    never vendored into this runtime tree.

    The evaluated checkpoint (`final.pth`, 1806 tensors) contains NO `fsmc`/`scm`
    parameters, so the SG-DRF feature-compression path is not part of the model
    being evaluated. We provide a stand-in that satisfies the import and raises
    loudly if it is ever actually used, so we can never silently evaluate a
    different model than the checkpoint describes.
    """
    try:
        if importlib.util.find_spec('model.modules.dynamic_fsmc') is not None:
            return False
    except (ImportError, ValueError, ModuleNotFoundError):
        pass

    class DynamicFSMC:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "DynamicFSMC required by model.model but absent from the evaluated "
                "checkpoint (no fsmc/scm tensors in final.pth) and its source was "
                "not vendored. Refusing to evaluate a model that does not match "
                "the checkpoint.")

        def __call__(self, *a, **kw):
            raise RuntimeError("DynamicFSMC placeholder invoked; aborting.")

    mod = types.ModuleType('model.modules.dynamic_fsmc')
    mod.DynamicFSMC = DynamicFSMC
    sys.modules['model.modules.dynamic_fsmc'] = mod
    return True


_install_dynamic_fsmc_shim()
import torch  # noqa: E402
import utils.sam2_bootstrap  # noqa: F401,E402  (must precede sam2 import)
from validation import build_sam_model  # noqa: E402
from utils.tools import merge_width_scatter_np  # noqa: E402
from utils.metrics import fast_hist, per_class_iu, per_class_dice  # noqa: E402

RUNS = '/root/autodl-tmp/piers-main/runs'
CKPT_DEFAULT = ('/root/autodl-tmp/piers-main/runs/I1-densepass-20261107_121457/'
                 'allhard_runs/R0_seed1234/training/final.pth')

# SynPASS 原生 23 值 -> Cityscapes 13 类（官方 _key，来自 dataset/SynPASS.py）
SYN_KEY = np.array([255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10,
                     255, 255, 255, 255, 6, 255, 255, 255, 9])

MODEL13 = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'car']


def build_synpass_dataset(root, list_path, crop_size=(4096, 1024),
                          sliding_window=(384, 1024), passes=2):
    """SynPASS val loader: 13-class trainId GT, DensePASS-style sliding window."""
    import torch as _t
    from PIL import Image
    from torchvision import transforms
    from torch.utils import data as _data

    class SynPASS13(_data.Dataset):
        def __init__(self):
            self.crop_size = crop_size
            self.stride, self.window_size = sliding_window
            self.passes = passes
            self.files = []
            names = [l.strip() for l in open(list_path) if l.strip()]
            for name in names:
                img_file = os.path.join(root, name)
                lbname = name.replace('img', 'semantic').replace('.jpg', '_trainID.png')
                label_file = os.path.join(root, lbname)
                if os.path.isfile(img_file) and os.path.isfile(label_file):
                    self.files.append({'img': img_file, 'label': label_file,
                                       'name': name})
                else:
                    print(f'[SynPASS] missing pair, skip: {name}', flush=True)

        def __len__(self):
            return len(self.files)

        def _get_frame_cnt(self):
            w, h = self.crop_size
            return (w - self.window_size) // self.stride + 1

        def _map23to13(self, mask):
            values = np.unique(mask)
            new_mask = np.ones_like(mask) * 255
            for value in values:
                if value >= len(SYN_KEY):
                    continue
                new_mask[mask == value] = SYN_KEY[value]
            return new_mask

        def _sliding_windows(self, image, label):
            h, w = label.shape
            pi, pl = [], []
            for x in range(0, w, self.stride):
                if x + self.window_size <= w:
                    pi.append(image[:, :, x:x + self.window_size])
                    pl.append(label[:, x:x + self.window_size])
            if self.passes == 2:
                pi.extend(reversed(pi))
                pl.extend(reversed(pl))
            return _t.stack(pi), _t.stack(pl)

        def __getitem__(self, index):
            d = self.files[index]
            raw = np.array(Image.open(d['label']))
            native = self._map23to13(raw)
            img = Image.open(d['img']).convert('RGB').resize(self.crop_size, Image.BICUBIC)
            lab = Image.fromarray(native.astype(np.uint8)).resize(self.crop_size, Image.NEAREST)
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            image = tf(img)
            label = _t.LongTensor(np.asarray(lab, dtype=np.int32))
            ori_label = _t.LongTensor(np.asarray(lab, dtype=np.int32))
            image, label = self._sliding_windows(image, label)
            size = np.asarray(label).shape
            return image, label, size, d['name'], ori_label

    return SynPASS13()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='dense')
    ap.add_argument('--kr', type=float, default=0.0625)
    ap.add_argument('--n_img', type=int, default=500)
    ap.add_argument('--tag', default='GEN-SP500-dense')
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--stride_w', type=int, default=384)
    ap.add_argument('--root', default='/root/autodl-tmp/data/SynPASS')
    ap.add_argument('--list', default='/root/autodl-tmp/dataset/synpass_list/val.txt')
    a = ap.parse_args()

    device = torch.device('cuda')
    ns = argparse.Namespace(
        seed=1234, backbone='sam2_s', dataset='SynPASS', class_set=13,
        data_root='/root/autodl-tmp/data', wildpass_processed_root='',
        wildpass_layout='flat', wildpass_image_dir='', wildpass_label_dir='',
        val_list='', stanford_processed_root='', stanford_val_list='',
        stanford_legacy_area_crop=False, stanford_stride_w=256,
        batch_size=1, ckpt=a.ckpt, ckpt_list=None, ckpt_names=None,
        metrics_json=f'{RUNS}/{a.tag}/metrics.json', log_file=None,
        log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False,
        num_maskmem=9, use_mem_bank=True, save_pred_debug=0,
        pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='',
        pred_gt_histogram_md='', timing_json=None, run_label=a.tag, num_classes=13,
    )

    ds = build_synpass_dataset(a.root, a.list, crop_size=(4096, 1024),
                                sliding_window=(a.stride_w, 1024), passes=2)
    total = len(ds)
    ds.files = ds.files[:a.n_img]
    print(f'[GEN] dataset=SynPASS mode={a.mode} total={total} take={len(ds.files)} '
          f'ckpt={a.ckpt}', flush=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=2, pin_memory=True)
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank

    if a.mode == 'dense':
        mb.configure_sparse_retrieval(enabled=False)
    else:
        kw = {}
        if a.mode == 'class_aware':
            kw = dict(rare_class_ids=[3, 4, 6, 7], rare_reserve_ratio=0.15,
                      boundary_score_source='uncertainty')
        elif a.mode == 'boundary_guarded':
            kw = dict(boundary_reserve_ratio=0.125, boundary_score_source='uncertainty')
        elif a.mode == 'joint_guarded':
            kw = dict(rare_class_ids=[3, 4, 6, 7], rare_reserve_ratio=0.15,
                      boundary_score_source='uncertainty')
        mb.configure_sparse_retrieval(
            enabled=True, mode=a.mode, keep_ratio=a.kr, min_tokens=64,
            cache_enabled=False, fallback='feature_norm',
            fallback_full_when_uncertain=True, enable_in_train=True,
            enable_in_pseudo_label=True, **kw)
    mb.set_runtime_context(stage='eval', iter_idx=0)

    num_frames = ds._get_frame_cnt()
    hist = np.zeros((13, 13), dtype=np.float64)
    t0 = time.perf_counter()

    import cv2
    model.eval()
    with torch.no_grad():
        from tqdm import tqdm
        for index, batch in enumerate(tqdm(loader, desc='Val', total=len(loader),
                                           mininterval=5.0)):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            ol = ori_label.detach().cpu().numpy().astype(np.int64)
            if ol.ndim == 2:
                ol = ol[np.newaxis, :, :]
            B = image.shape[0]
            H, W = image.shape[-2:]
            logits_maps = np.zeros((B, 2 * num_frames, 13, H, W))

            for fi in range(num_frames):
                out, _, _ = model(fi, image[:, fi])
                logits_maps[:, fi] = out.cpu().numpy()
            model.memory_bank._init_output_dict_source()
            for fi in range(num_frames):
                out, _, _ = model(fi, image[:, num_frames + fi])
                logits_maps[:, num_frames + fi] = out.cpu().numpy()
            model.memory_bank._init_output_dict_source()

            merged = merge_width_scatter_np(
                logits_np=logits_maps, orig_size=(1024, 4096), W_patch=W,
                stride_w=a.stride_w, passes=2, device=str(device),
                return_probs=False)
            pred = merged[0].astype(np.int64)
            pred = cv2.resize(pred.astype(np.uint8), (2048, 400),
                              interpolation=cv2.INTER_NEAREST).astype(np.int64)
            gt = ol[0]
            if gt.shape != (400, 2048):
                gt = cv2.resize(gt.astype(np.uint8), (2048, 400),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            hist += fast_hist(gt.reshape(-1), pred.reshape(-1), 13)

    dt = time.perf_counter() - t0
    iou = per_class_iu(hist)
    dice = per_class_dice(hist)
    total_px = hist.sum()
    gt_share = {MODEL13[i]: round(float(hist[i].sum() / max(total_px, 1)) * 100, 3)
                for i in range(13)}
    present = [i for i in range(13) if hist[i].sum() > 0]
    mIoU_present = float(np.nanmean(iou[present])) * 100 if present else float('nan')

    out = {
        'tag': a.tag, 'dataset': 'SynPASS', 'mode': a.mode, 'kr': a.kr,
        'n_img': len(ds.files), 'num_classes': 13, 'class_names': MODEL13,
        'label_kind': 'synpass_native23_to_cityscapes13_official_key',
        'ckpt': a.ckpt, 'eval_time_sec': round(dt, 2),
        'mIoU_pct': mIoU_present,
        'mIoU_all13_pct': round(float(np.nanmean(iou)) * 100, 2),
        'per_class_iou_pct': {MODEL13[i]: round(float(iou[i]) * 100, 2) for i in range(13)},
        'per_class_dice_pct': {MODEL13[i]: round(float(dice[i]) * 100, 2) for i in range(13)},
        'gt_pixel_share_pct': gt_share,
        'classes_with_gt': [MODEL13[i] for i in present],
        'images_per_sec': round(len(ds.files) / max(dt, 1e-6), 4),
        'protocol_note': (
            'SynPASS full 13-class protocol. SynPASS natively annotates 23 label '
            'values which the official key maps onto all 13 Cityscapes classes '
            '(road/sidewalk/building/wall/fence/pole/tlight/tsign/veg/terrain/sky/'
            'person/car). No class is dropped, so mIoU is directly comparable to '
            'the 13-class DensePASS table. GT is the official 23->13 trainID '
            'mapping, not a hand-written adaptation.'),
        'protocol': (
            'zero-shot inference, frozen weights, no target training, identical '
            'pipeline to DensePASS evaluation (sam2_s + num_maskmem=9 + sliding '
            'window 1024/stride384 + 2 passes)'),
    }
    print('RESULT ' + json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(f'{RUNS}/{a.tag}', exist_ok=True)
    with open(f'{RUNS}/{a.tag}/summary.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('[GEN] per-class IoU (13 classes):')
    for i in range(13):
        print('  %-14s IoU %6.2f  Dice %6.2f  GTshare %5.2f%%' % (
            MODEL13[i], iou[i] * 100, dice[i] * 100, gt_share[MODEL13[i]]))
    print('  mIoU(classes with GT) = %.2f' % mIoU_present)


if __name__ == '__main__':
    main()
