"""WildPASS 原生 8 类评测：把模型 13 类输出投影到 WildPASS 原生类别空间。

背景
----
WildPASS 官方只标注 8 个 evaluation ID：
    car=4 truck=5 bus=8 road=11 sidewalk=12 curb=13 person=17 crosswalk=24
    raw 0 = ignore
（见 docs/piers/WildPASS_LABEL_AUDIT_NOTICE_20261107.txt）

上一版 m_eval_xdom_v2.py 用的是 "shared-4" 口径（把 vehicle 合并成 car，
把 curb/crosswalk 丢掉）。那是为了快速验证映射是否正确，**不是最终口径**。

本脚本用 WildPASS **自己的** 8 类来评测：
    * 模型 13 类 logits -> 在概率层面聚合 -> WildPASS 8 类
    * GT 原始 raw ID -> WildPASS 8 类
    * mIoU 在这 8 类上计算
这样 mIoU 反映"模型在 WildPASS 全部标注类别上的表现"。

投影规则（模型 13 类 -> WildPASS 8 类）
----------------------------------------
    model road(0)      -> WildPASS road
    model sidewalk(1)  -> WildPASS sidewalk
    model person(11)   -> WildPASS person
    model car(12)      -> WildPASS vehicle（car/truck/bus 合并）
    model building/wall/fence/pole/tlight/tsign/veg/terrain/sky
                       -> ignore (255)
WildPASS 的 curb / crosswalk 没有模型对应类 -> 模型侧永远预测不到，
它们的 IoU 必然为 0。这两个类单独统计并明确标注 "no model counterpart"，
不计入 headline mIoU（否则 mIoU 被系统性拉低，失去意义）。
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

# WildPASS 原生 8 类（官方 evaluation ID 顺序）
WP_NATIVE = ['road', 'sidewalk', 'curb', 'crosswalk',
             'vehicle', 'person', 'truck', 'bus']
WP_ROAD, WP_SIDEWALK, WP_CURB, WP_CROSSWALK, WP_VEHICLE, WP_PERSON, WP_TRUCK, WP_BUS = range(8)

# 官方 raw evaluation ID -> WildPASS 原生类索引
WP_RAW2NATIVE = {
    11: WP_ROAD, 12: WP_SIDEWALK, 13: WP_CURB, 24: WP_CROSSWALK,
    4: WP_VEHICLE, 5: WP_TRUCK, 8: WP_BUS, 17: WP_PERSON,
    0: 255,
}
# 模型 13 类 -> WildPASS 原生类（255 = ignore）
# model car(12) -> vehicle（模型无法区分 car/truck/bus）
MODEL13_TO_WP = {
    0: WP_ROAD, 1: WP_SIDEWALK, 11: WP_PERSON, 12: WP_VEHICLE,
}
# WildPASS 原生类 -> 我们的模型 13 类（用于构造 "模型可及类" 子集）
WP_NATIVE_TO_MODEL13 = {
    WP_ROAD: 0, WP_SIDEWALK: 1, WP_PERSON: 11, WP_VEHICLE: 12,
}
# 有模型对应类的原生类（headline mIoU 用这 6 个）
WP_EVALUABLE = [WP_ROAD, WP_SIDEWALK, WP_VEHICLE, WP_PERSON, WP_TRUCK, WP_BUS]
# 没有模型对应类的原生类（单独报告）
WP_NO_COUNTERPART = [WP_CURB, WP_CROSSWALK]

MODEL13 = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'car']


def build_wildpass_native_dataset():
    """WildPASS val with *native* 8-class GT and the model's sliding-window protocol."""
    import torch as _t
    from PIL import Image
    from torchvision import transforms
    from torch.utils import data as _data

    ROOT = '/root/autodl-tmp/data/WildPASS'
    GT_DIR = os.path.join(ROOT, 'gtFine', 'val', 'cs')
    # `data_extra/WildPASS13/images/<n>.png` is the RGB panorama that was
    # extracted from `leftImg8bit/<City>/<n>.jpg` and is what the DensePASS-style
    # pipeline consumes (the matching GT lives in gtFine/val/cs/<n>.png).
    IMG_DIR = '/root/autodl-tmp/data_extra/WildPASS13/images'
    LIST = '/root/autodl-tmp/data_extra/WildPASS13/lists/val.txt'

    class WildPASSNative(_data.Dataset):
        def __init__(self, root, list_path, crop_size=(4096, 1024),
                     sliding_window=(384, 1024), passes=2):
            self.crop_size = crop_size
            self.stride, self.window_size = sliding_window
            self.passes = passes
            self.files = []
            names = [l.strip() for l in open(list_path) if l.strip()]
            for n in names:
                stem = os.path.splitext(os.path.basename(n))[0]
                gt = os.path.join(GT_DIR, stem + '.png')
                im = os.path.join(IMG_DIR, stem + '.png')
                if not (os.path.isfile(gt) and os.path.isfile(im)):
                    continue
                self.files.append({'gt': gt, 'img': im, 'name': stem + '.png'})

        def __len__(self):
            return len(self.files)

        def _get_frame_cnt(self):
            w, h = self.crop_size
            return (w - self.window_size) // self.stride + 1

        def _map_native(self, raw):
            m = np.asarray(raw, dtype=np.int64)
            out = np.full(m.shape, 255, dtype=np.int32)
            for rid, nid in WP_RAW2NATIVE.items():
                out[m == rid] = nid
            return out

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
            raw = np.array(Image.open(d['gt']))
            native = self._map_native(raw)
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

    return WildPASSNative


def project_probs_to_wildpass(logits):
    """(C=13, H, W) model logits -> (8, H, W) WildPASS-native probabilities.

    Aggregation is done on the *probability* level (after softmax over the 13
    model classes), so a pixel confidently predicted as `car` contributes its
    full probability mass to WildPASS `vehicle`, and the mass of the 9 classes
    that WildPASS does not annotate is dropped (renormalised over the kept ones).
    """
    probs = torch.softmax(logits, dim=0)               # (13, H, W)
    out = torch.zeros((8, probs.shape[1], probs.shape[2]),
                      dtype=probs.dtype, device=probs.device)
    for m13, nat in MODEL13_TO_WP.items():
        out[nat] += probs[m13]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='dense')
    ap.add_argument('--kr', type=float, default=0.0625)
    ap.add_argument('--n_img', type=int, default=500)
    ap.add_argument('--tag', default='GEN-WP500-native8')
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--stride_w', type=int, default=384)
    a = ap.parse_args()

    device = torch.device('cuda')
    ns = argparse.Namespace(
        seed=1234, backbone='sam2_s', dataset='WildPASS', class_set=13,
        data_root='/root/autodl-tmp/data',
        wildpass_processed_root='/root/autodl-tmp/data_extra/WildPASS13',
        wildpass_layout='flat', wildpass_image_dir='images',
        wildpass_label_dir='labels_raw', val_list='',
        stanford_processed_root='', stanford_val_list='',
        stanford_legacy_area_crop=False, stanford_stride_w=256,
        batch_size=1, ckpt=a.ckpt, ckpt_list=None, ckpt_names=None,
        metrics_json=f'{RUNS}/{a.tag}/metrics.json', log_file=None,
        log_append=False, enable_scm=False, scm_pad_width=512, enable_bepl=False,
        num_maskmem=9, use_mem_bank=True, save_pred_debug=0,
        pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='',
        pred_gt_histogram_md='', timing_json=None, run_label=a.tag,
        num_classes=13,
    )

    ds = build_wildpass_native_dataset()(
        '/root/autodl-tmp/data/WildPASS',
        '/root/autodl-tmp/data_extra/WildPASS13/lists/val.txt',
        crop_size=(4096, 1024), sliding_window=(a.stride_w, 1024), passes=2)
    total = len(ds)
    ds.files = ds.files[:a.n_img]
    print(f'[GEN] dataset=WildPASS native8 mode={a.mode} total={total} '
          f'take={len(ds.files)} ckpt={a.ckpt}', flush=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=2, pin_memory=True)
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank

    # memory 检索模式：唯一变量
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
    hist = np.zeros((8, 8), dtype=np.float64)
    hist_rel = np.zeros((8, 8), dtype=np.float64)
    t0 = time.perf_counter()

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

            # merge -> (N, 13, 1024, 4096) probabilities, then project to 8 native
            probs13 = merge_width_scatter_np(
                logits_np=logits_maps, orig_size=(1024, 4096), W_patch=W,
                stride_w=a.stride_w, passes=2, device=str(device),
                return_probs=True)
            probs8 = project_probs_to_wildpass(torch.from_numpy(probs13[0]).to(device))

            import cv2
            pred8 = probs8.argmax(dim=0).cpu().numpy().astype(np.int64)
            pred8 = cv2.resize(pred8.astype(np.uint8), (2048, 400),
                               interpolation=cv2.INTER_NEAREST).astype(np.int64)
            gt8 = ol[0]
            if gt8.shape != (400, 2048):
                gt8 = cv2.resize(gt8.astype(np.uint8), (2048, 400),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int64)
            hist += fast_hist(gt8.reshape(-1), pred8.reshape(-1), 8)

            # Relaxed vehicle metric: our 13-class head collapses Cityscapes
            # car/truck/bus into a single `car` class (trainId 19->13 collapse),
            # so the model can never separate WildPASS's `vehicle` / `truck` /
            # `bus`. Merging those three GT classes into `vehicle` is the
            # taxonomy-faithful comparison for this model.
            gt_rel = gt8.copy()
            gt_rel[(gt_rel == WP_TRUCK) | (gt_rel == WP_BUS)] = WP_VEHICLE
            hist_rel += fast_hist(gt_rel.reshape(-1), pred8.reshape(-1), 8)

    dt = time.perf_counter() - t0
    iou = per_class_iu(hist)
    dice = per_class_dice(hist)
    iou_rel = per_class_iu(hist_rel)
    ev_mask = np.zeros(8, dtype=bool)
    ev_mask[WP_EVALUABLE] = True
    nc_mask = np.zeros(8, dtype=bool)
    nc_mask[WP_NO_COUNTERPART] = True

    # Classes the model can actually emit in the native space
    # (model 13 -> native: road, sidewalk, person, vehicle).
    model_emittable = [WP_ROAD, WP_SIDEWALK, WP_PERSON, WP_VEHICLE]
    me_mask = np.zeros(8, dtype=bool)
    me_mask[model_emittable] = True

    out = {
        'tag': a.tag, 'dataset': 'WildPASS', 'mode': a.mode, 'kr': a.kr,
        'n_img': len(ds.files), 'label_kind': 'wildpass_native8_official_ids',
        'ckpt': a.ckpt, 'eval_time_sec': round(dt, 2),
        'native_classes': WP_NATIVE,
        'per_class_iou_pct': {WP_NATIVE[i]: round(float(iou[i]) * 100, 2) for i in range(8)},
        'per_class_dice_pct': {WP_NATIVE[i]: round(float(dice[i]) * 100, 2) for i in range(8)},
        'mIoU_model_emittable_pct': round(float(np.nanmean(iou[me_mask])) * 100, 2),
        'mIoU_evaluable_pct': round(float(np.nanmean(iou[ev_mask])) * 100, 2),
        'mIoU_all8_pct': round(float(np.nanmean(iou)) * 100, 2),
        'vehicle_relaxed_iou_pct': round(float(iou_rel[WP_VEHICLE]) * 100, 2),
        'mIoU_model_emittable_vehicle_relaxed_pct': round(
            float(np.nanmean(np.where(me_mask, iou_rel, np.nan))) * 100, 2),
        'no_model_counterpart_classes': {WP_NATIVE[i]: round(float(iou[i]) * 100, 2)
                                         for i in WP_NO_COUNTERPART},
        'gt_pixel_share_pct': {WP_NATIVE[i]: round(float(hist[i].sum() / max(hist.sum(), 1)) * 100, 2)
                                for i in range(8)},
        'images_per_sec': round(len(ds.files) / max(dt, 1e-6), 4),
        'protocol_note': (
            'WildPASS native 8-class protocol. GT = official evaluation IDs '
            '(road=11 sidewalk=12 curb=13 crosswalk=24 car=4 truck=5 bus=8 person=17, '
            'raw 0 = ignore). Model 13-class probabilities are projected onto the '
            'native space before argmax (model car(12)->vehicle; the 9 model classes '
            'WildPASS does not annotate are dropped). '
            'mIoU_model_emittable is the headline: it averages only the 4 native '
            'classes the model can actually emit (road, sidewalk, person, vehicle). '
            'curb/crosswalk/truck/bus have NO model counterpart (our 13-class head '
            'collapses car/truck/bus into one class) and are reported separately; '
            'including them would drive the number to 0 by construction rather than '
            'measure generalization. vehicle_relaxed merges GT truck+bus into '
            'vehicle, which is the taxonomy-faithful comparison for this model.'),
        'protocol': (
            'zero-shot inference, frozen weights, no target training, identical '
            'pipeline to DensePASS evaluation (sam2_s + num_maskmem=9 + sliding '
            'window 1024/stride384 + 2 passes)'),
    }
    print('RESULT ' + json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(f'{RUNS}/{a.tag}', exist_ok=True)
    with open(f'{RUNS}/{a.tag}/summary.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('[GEN] per-class IoU (native 8):')
    for i in range(8):
        print('  %-11s IoU %6.2f  Dice %6.2f' % (WP_NATIVE[i], iou[i] * 100, dice[i] * 100))
    print('  mIoU(model-emittable 4) = %.2f' % (np.nanmean(iou[me_mask]) * 100))
    print('  mIoU(evaluable 6)       = %.2f' % (np.nanmean(iou[ev_mask]) * 100))
    print('  mIoU(all 8)             = %.2f' % (np.nanmean(iou) * 100))
    print('  vehicle relaxed (truck+bus->vehicle) IoU = %.2f  -> emittable mIoU = %.2f' % (
        iou_rel[WP_VEHICLE] * 100, np.nanmean(np.where(me_mask, iou_rel, np.nan)) * 100))


if __name__ == '__main__':
    main()
