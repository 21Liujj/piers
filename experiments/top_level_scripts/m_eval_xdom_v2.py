"""室外全景数据集泛化评测（零样本，纯推理，不训练）。

目标
----
在 DensePASS 之外的全景街景数据集上评估冻结模型的泛化性。

数据集适配（关键，与 DensePASS 差异很大）
------------------------------------------
WildPASS (500 张, ERP 2048x400)
  * 标签体系完全不同。WildPASS 官方只用 8 个 raw evaluation ID：
        car=4, truck=5, bus=8, road=11, sidewalk=12, curb=13, person=17, crosswalk=24
        raw 0 = ignore
    见 docs/piers/WildPASS_LABEL_AUDIT_NOTICE_20261107.txt
  * 我们模型是 13 类街景语义头，两者只有 4 个语义可对齐：
        road(0) / sidewalk(1) / person(11) / vehicle(12)
    因此 WildPASS 只能做 **共享 4 类** 的补充评测，
    不能进 13 类主表（否则 building/sky 等无 GT 支撑的类会拉低/污染指标）。
  * `data_extra/WildPASS13/labels_trainid` 及其 `backup_*` 目录都使用了
    **不兼容的映射**（审计已确认），本脚本不使用它们；
    统一从 `labels_raw` + 官方映射现算。
  * 其余类（curb/crosswalk）无对应模型语义 -> 255 ignore。

Stanford360_SPAN8 (室内, 260 张)
  * 8 类室内语义 (ceiling/chair/door/floor/sofa/table/wall/window)，
    与 13 类街景头几乎没有同名类（仅 wall 与模型 wall(3) 同名）。
  * 不做数值 mIoU 对比；用于回答"室内是否需要微调"的定性分析。

统一协议
--------
* 冻结权重，纯推理，零样本，不用目标域标签训练。
* 与 DensePASS 评测完全一致的推理管线：
  sam2_s + num_maskmem=9 + 滑窗 1024/stride384 + 双 pass + 同一套 merge。
* 唯一变量是 memory 检索模式（dense = OmniSAM 标准；其余为 I2 各变体）。

用法
----
python m_eval_xdom_v2.py --dataset WildPASS --mode dense --n_img 500 --tag GEN-WP-dense
python m_eval_xdom_v2.py --dataset WildPASS --mode class_aware --kr 0.0625 --tag GEN-WP-ca
"""
import os
import sys
import json
import time
import argparse
import types
import importlib.util

# 运行时引擎位于 /root/autodl-tmp/eff_omni（piers-main/third_party 下只有 sam/checkpoints）
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
from validation import build_sam_model, build_dataset_and_loader, validation  # noqa: E402

RUNS = '/root/autodl-tmp/piers-main/runs'

# 论文对照臂的冻结权重（I1 实验产物，完整 state_dict，1806 tensors 全部匹配）
CKPT_DEFAULT = ('/root/autodl-tmp/piers-main/runs/I1-densepass-20261107_121457/'
                 'allhard_runs/R0_seed1234/training/final.pth')

# ---------------------------------------------------------------------------
# WildPASS 官方标签映射（依据 docs/piers/WildPASS_LABEL_AUDIT_NOTICE_20261107.txt）
# ---------------------------------------------------------------------------
# 官方 raw evaluation ID -> 语义
WILDPASS_RAW_SEMANTICS = {
    4: 'car', 5: 'truck', 8: 'bus',
    11: 'road', 12: 'sidewalk', 13: 'curb',
    17: 'person', 24: 'crosswalk',
    0: 'ignore',
}
# 共享 4 类协议：映射到我们 13 类模型的 trainId
#   road(0) sidewalk(1) person(11) vehicle(12)
#   vehicle = car+truck+bus 合并到模型 car(12)
WILDPASS_SHARED4_MAP = {11: 0, 12: 1, 17: 11, 4: 12, 5: 12, 8: 12}
WILDPASS_SHARED4_NAMES = ['road', 'sidewalk', 'person', 'vehicle']
# 模型 13 类中在 WildPASS 上有 GT 支撑的类（其余在共享协议下为 255 ignore）
WILDPASS_MODEL13_SUPPORTED = [0, 1, 11, 12]


def _install_wildpass_official_mapping():
    """Replace the stock converter with the official WildPASS evaluation-ID map.

    The stock `convert_wildpass_raw_to_train13` auto-detects the encoding and
    mis-maps these files (it treats them as Cityscapes trainIds), which is
    exactly the failure that produced the bogus 0.17 mIoU numbers.
    """
    import numpy as np
    import dataset.WildPASS as wds

    def official(raw):
        m = np.asarray(raw, dtype=np.int64)
        out = np.full(m.shape, 255, dtype=np.int32)
        for raw_id, train_id in WILDPASS_SHARED4_MAP.items():
            out[m == raw_id] = train_id
        return out, 'wildpass_official_shared4'

    wds.convert_wildpass_raw_to_train13 = official
    return True


def build_ns(dataset, mode, kr, tag, ckpt, stride_w=384,
             wildpass_label_dir='labels_raw'):
    ns = argparse.Namespace(
        seed=1234, backbone='sam2_s', dataset=dataset, class_set=13,
        data_root='/root/autodl-tmp/data', wildpass_layout='flat',
        wildpass_processed_root=('/root/autodl-tmp/data_extra/WildPASS13'
                                 if dataset == 'WildPASS' else ''),
        val_list='',
        wildpass_image_dir='images', wildpass_label_dir=wildpass_label_dir,
        stanford_processed_root=('/root/autodl-tmp/data_extra/Stanford360_SPAN8'
                                 if dataset == 'Stanford2D3D' else ''),
        stanford_val_list='', stanford_legacy_area_crop=False,
        stanford_stride_w=stride_w,
        batch_size=1, ckpt=ckpt, ckpt_list=None, ckpt_names=None,
        metrics_json=f'{RUNS}/{tag}/metrics.json',
        log_file=None, log_append=False, enable_scm=False, scm_pad_width=512,
        enable_bepl=False, num_maskmem=9, use_mem_bank=True, save_pred_debug=0,
        pred_gt_histogram=False, debug_pred_vis_dir='', pred_gt_histogram_json='',
        pred_gt_histogram_md='', timing_json=None, run_label=tag)
    return ns


def install_memory_mode(mb, mode, kr):
    """唯一变量：memory 读取方式。dense=OmniSAM标准(不压缩)，其余为I2各变体。"""
    if mode == 'dense':
        mb.configure_sparse_retrieval(enabled=False)
        return
    kw = {}
    if mode == 'class_aware':
        kw = dict(rare_class_ids=[3, 4, 6, 7], rare_reserve_ratio=0.15,
                  boundary_score_source='uncertainty')
    elif mode == 'boundary_guarded':
        kw = dict(boundary_reserve_ratio=0.125, boundary_score_source='uncertainty')
    elif mode == 'joint_guarded':
        kw = dict(rare_class_ids=[3, 4, 6, 7], rare_reserve_ratio=0.15,
                  boundary_score_source='uncertainty')
    mb.configure_sparse_retrieval(
        enabled=True, mode=mode, keep_ratio=kr, min_tokens=64, cache_enabled=False,
        fallback='feature_norm', fallback_full_when_uncertain=True,
        enable_in_train=True, enable_in_pseudo_label=True, **kw)


def run_variant(mode, kr, n_img, tag, dataset, ckpt,
                wildpass_label_dir='labels_raw'):
    label_kind = None
    if dataset == 'WildPASS':
        _install_wildpass_official_mapping()
        label_kind = 'wildpass_official_shared4'

    ns = build_ns(dataset, mode, kr, tag, ckpt,
                  stride_w=256 if dataset == 'Stanford2D3D' else 384,
                  wildpass_label_dir=wildpass_label_dir)
    os.makedirs(f'{RUNS}/{tag}', exist_ok=True)
    val_loader, class_names, num_classes = build_dataset_and_loader(ns)
    total = len(val_loader.dataset.files)
    val_loader.dataset.files = val_loader.dataset.files[:n_img]
    print(f'[GEN] dataset={dataset} mode={mode} total={total} '
          f'take={len(val_loader.dataset.files)} num_classes={num_classes} '
          f'label_dir={wildpass_label_dir} label_kind={label_kind} ckpt={ckpt}',
          flush=True)

    import torch.utils.data as tud
    val_loader = tud.DataLoader(val_loader.dataset, batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=True)
    ns.num_classes = num_classes
    device = torch.device('cuda')
    model = build_sam_model(ns, device, ckpt_path=ns.ckpt)
    mb = model.memory_bank

    # 历史伪标签记忆（与 OmniSAM 同流程）：跨 FoV 传递 argmax
    hist = {}
    orig_fwd = model.forward

    def wrapped(frame_idx, x, mem_type='source', historical_pseudo_labels=None, **kw):
        merged = dict(hist)
        if historical_pseudo_labels:
            merged.update(historical_pseudo_labels)
        out = orig_fwd(frame_idx, x, mem_type=mem_type,
                       historical_pseudo_labels=merged if merged else None, **kw)
        try:
            pred = out[0] if isinstance(out, (list, tuple)) else out
            hist[frame_idx] = pred.argmax(dim=1).detach().cpu()
        except Exception:
            pass
        return out

    model.forward = wrapped
    install_memory_mode(mb, mode, kr)
    mb.set_runtime_context(stage='eval', iter_idx=0)
    orig_init = mb._init_output_dict_source

    def wrapped_init():
        hist.clear()
        return orig_init()

    mb._init_output_dict_source = wrapped_init
    ns._skip_per_run_json = False

    t0 = time.perf_counter()
    agg = validation(model=model, data_loader=val_loader, device=device,
                     num_classes=num_classes, class_names=class_names,
                     backbone_name='sam2_s', num_maskmem=9,
                     dataset_name=dataset, args=ns)
    dt = time.perf_counter() - t0

    out = {
        'tag': tag, 'dataset': dataset, 'mode': mode, 'kr': kr,
        'n_img': len(val_loader.dataset.files), 'num_classes': num_classes,
        'class_names': class_names,
        'label_dir': wildpass_label_dir if dataset == 'WildPASS' else '',
        'label_kind': label_kind,
        'ckpt': ckpt,
        'eval_time_sec': round(dt, 2),
        'mIoU': agg.get('mIoU'), 'mAcc': agg.get('mAcc'),
        'PixAcc': agg.get('PixAcc'), 'FWIoU': agg.get('FWIoU'),
        'mDice': agg.get('mDice'),
        'per_class_iou_pct': agg.get('per_class_iou_pct'),
        'eff_len': int(getattr(mb, 'last_effective_memory_length', 0)),
        'images_per_sec': round(len(val_loader.dataset.files) / max(dt, 1e-6), 4),
    }
    if dataset == 'WildPASS':
        out['protocol_note'] = (
            'Shared-4-class protocol: WildPASS only annotates 8 raw IDs '
            '(car/truck/bus/road/sidewalk/curb/person/crosswalk). Mapped to the '
            '13-class model as road(0)/sidewalk(1)/person(11)/vehicle(12); '
            'curb/crosswalk and all other model classes are 255-ignore. '
            'These numbers are NOT comparable to the 13-class DensePASS table.')
        out['shared4_classes'] = WILDPASS_SHARED4_NAMES
    out['protocol'] = (
        'zero-shot inference, frozen weights, no target training, identical '
        'pipeline to DensePASS evaluation (sam2_s + num_maskmem=9 + sliding '
        'window 1024/stride384 + 2 passes)')
    print('RESULT ' + json.dumps(out, ensure_ascii=False), flush=True)
    with open(f'{RUNS}/{tag}/summary.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='WildPASS',
                    choices=['WildPASS', 'Stanford2D3D', 'DensePASS'])
    ap.add_argument('--mode', default='dense')
    ap.add_argument('--kr', type=float, default=0.0625)
    ap.add_argument('--n_img', type=int, default=500)
    ap.add_argument('--tag', default='GEN-WP-dense')
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--wildpass_label_dir', default='labels_raw')
    a = ap.parse_args()
    run_variant(a.mode, a.kr, a.n_img, a.tag, a.dataset, a.ckpt,
                a.wildpass_label_dir)
