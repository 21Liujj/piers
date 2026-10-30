"""
WildPASS raw 单通道语义图 → Cityscapes 13-class trainId（0..12, 255）。

要点：
- 纯 trainId：值 ∈ {0..18,255}，0 表示 road。
- 纯 labelId：0 多为 void（LUT→255），road 常为 7；值可 >18（如 24=person）。
- 混合格式（少数像素 >18、其余 ≤18）：整图走 labelId 会把大量 0 误当 void。
  采用 **hybrid**：默认 ≤18 按 trainId 解释，>18 的像素按 labelId→trainId→13；
  若存在典型 labelId 纹理（如大量 7 且无 0），则整图走 labelId。
"""
from __future__ import annotations

import numpy as np

KEY_19_TO_13 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12],
    dtype=np.int32,
)


def _build_labelid_to_trainid_lut() -> np.ndarray:
    from torchvision.datasets import Cityscapes

    lut = np.full(256, 255, dtype=np.int32)
    for c in Cityscapes.classes:
        if c.id is not None and c.id >= 0:
            tid = c.train_id
            lut[c.id] = 255 if tid in (-1, 255) else int(tid)
    return lut


LUT_LABELID_TO_TRAINID = _build_labelid_to_trainid_lut()


def map19_trainid_to_train13(mask: np.ndarray, key: np.ndarray = KEY_19_TO_13) -> np.ndarray:
    """输入为 19-class trainId（0..18,255）。"""
    m = mask.astype(np.int64)
    in_rng = (m >= 0) & (m < 19)
    mapped = key[np.clip(m, 0, 18)]
    return np.where(m == 255, 255, np.where(in_rng, mapped, 255)).astype(np.int32)


def _layout_score(train13: np.ndarray) -> float:
    m = train13.astype(np.int64)
    valid = m != 255
    if not valid.any():
        return 0.0
    n = float(valid.sum())
    sky = (m == 10).sum() / n
    bld = (m == 2).sum() / n
    road = (m == 0).sum() / n
    veg = (m == 8).sum() / n
    return float(3.0 * sky + 1.2 * bld + 1.0 * road + 0.9 * veg)


def infer_encoding_all_leq_18(m: np.ndarray) -> str:
    """仅当 raw 不含 (18,255) 内整数时调用。"""
    valid = (m >= 0) & (m != 255)
    if not valid.any():
        return "trainId"
    u = np.unique(m[valid])
    uset = set(u.tolist())
    f0 = float((m == 0).sum()) / float(m.size)
    f7 = float((m == 7).sum()) / float(m.size)
    out_t = map19_trainid_to_train13(m)
    tid = LUT_LABELID_TO_TRAINID[np.clip(m, 0, 255)]
    out_l = map19_trainid_to_train13(tid)
    s_t = _layout_score(out_t)
    s_l = _layout_score(out_l)
    sky_t = float((out_t == 10).sum())
    sky_l = float((out_l == 10).sum())

    if 7 in uset and f0 < 1e-6 and f7 > 1e-5:
        return "labelId"
    if 7 in uset and 0 not in uset:
        return "labelId"
    if 0 in uset and f0 > 5e-4:
        if sky_l > max(5000.0, sky_t * 4.0) and s_l > s_t + 0.08:
            return "labelId"
        return "trainId"

    if sky_l > max(2000.0, sky_t + 0.015 * m.size) and s_l > s_t + 0.02:
        return "labelId"
    if s_l > s_t + 0.05:
        return "labelId"
    return "trainId"


def convert_wildpass_raw_to_train13(raw: np.ndarray) -> tuple[np.ndarray, str]:
    m = np.asarray(raw, dtype=np.int64)
    valid = (m >= 0) & (m != 255)
    if not valid.any():
        return m.astype(np.int32), "empty"

    u = np.unique(m[valid])
    uset = set(u.tolist())
    if uset.issubset(set(range(13))):
        out = m.copy()
        out[~np.isin(out, list(range(13)) + [255])] = 255
        return out.astype(np.int32), "already_13"

    has_hi = (u > 18).any()
    tid_lut = LUT_LABELID_TO_TRAINID[np.clip(m, 0, 255)]
    out_l = map19_trainid_to_train13(tid_lut)
    out_t = map19_trainid_to_train13(m)

    if has_hi:
        f0 = float((m == 0).sum()) / float(m.size)
        has7 = bool((m == 7).any())
        # trainId 11=person 占比也可能很高，不能用「11 多 ⇒ labelId building」推断
        if has7:
            return out_l, "labelId_full"
        # 含 24 等但仍大量 0：0 更可能为 trainId road，仅对 >18 走 labelId
        if f0 > 0.02:
            out = out_t.copy()
            hi = (m > 18) & (m != 255)
            out[hi] = out_l[hi]
            return out, "hybrid_train_low_label_hi"
        return out_l, "labelId_full"

    enc = infer_encoding_all_leq_18(m)
    if enc == "labelId":
        return out_l, "labelId"
    return out_t, "trainId"
