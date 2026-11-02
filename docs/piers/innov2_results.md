# 创新点2结果台账（innov2-memory，DensePASS100已冻，跨域待修）

> 分支：`innov2-memory`；权重冻结 `omnisam_s.pth`；与OmniSAM同流程，唯一变量为memory读法。
> 流程：`sam2_s + num_maskmem=9 + 滑窗1024/stride384/2遍18窗 + validation()`。

## 1. 统一结果：用 ca（class_aware）

- `ca` = 窗内按类别保稀有类（15%名额给灯/牌/杆/栏），`N=2048`。
- `bg` = 窗内保边界（12.5%给高不确定边界），`N=2048`，仅作消融对照。
- 主表只放 `dense vs ca`；`bg/pool/topk/uniform/cams`进消融表。

## 2. 总指标（DensePASS100，N=2048 vs dense 32768）

| 指标 | dense原版 | ca主推 | bg次推 | pool | topk | uniform | cams |
|---|---|---|---|---|---|---|---|
| tokens | 32768 | 2048(1/16) | 2048 | 2048 | 2048 | 2048 | 2048 |
| mIoU | 60.46 | 60.30(-0.16) | 60.27(-0.19) | 60.07 | 59.83 | 59.80 | 59.75 |
| mAcc | 71.73 | 72.08(+0.35) | 72.03(+0.30) | 70.96 | 70.38 | 70.34 | 70.29 |
| PixAcc | 88.97 | 88.73 | 88.73 | 88.83 | 88.88 | 88.87 | 88.86 |
| FWIoU | 81.23 | 81.00 | 81.00 | 81.11 | 81.12 | 81.10 | 81.09 |
| mDice | 72.89 | 72.81 | 72.78 | 72.44 | 72.18 | 72.15 | 72.10 |
| wall time 100张 | 383s | 465s | 472s | 320s | 355s | 325s | 374s |

论文写法：16倍压缩只掉0.16，mAcc+0.35；tokens省16倍实，wall time ca/bg因打分开销+20%，pool/uniform才又省存又省时但压塌稀有类。

## 3. 13类全表（顺序 road/sidewalk/building/wall/fence/pole/light/sign/veg/terrain/sky/person/car）

| 类 | dense | ca | ca-dense | bg | pool | 结论 |
|---|---|---|---|---|---|---|
| road | 81.00 | 80.53 | -0.47 | 80.53 | 80.89 | 大类微跌 |
| sidewalk | 46.67 | 46.54 | -0.13 | 46.57 | 46.64 | 持平 |
| building | 90.02 | 89.89 | -0.13 | 89.89 | 90.00 | 持平 |
| wall | 45.19 | 45.25 | +0.06 | 45.17 | 44.44 | 持平 |
| fence | 49.65 | 48.97 | -0.68 | 49.02 | 49.45 | 大类让利 |
| pole | 45.93 | 45.27 | -0.66 | 45.27 | 45.32 | 大类让利 |
| traffic light | 30.06 | 30.69 | +0.63 | 30.60 | 25.84 | 核心收益；pool塌4.2，ca拉回4.85 |
| traffic sign | 26.03 | 26.69 | +0.67 | 26.50 | 26.41 | 核心收益 |
| vegetation | 79.62 | 78.84 | -0.78 | 78.84 | 79.12 | 大类让利 |
| terrain | 46.38 | 46.11 | -0.26 | 46.10 | 46.32 | 微跌 |
| sky | 94.58 | 94.60 | +0.03 | 94.60 | 94.53 | 持平 |
| person | 66.47 | 65.86 | -0.61 | 65.81 | 67.64 | pool不稳定反超 |
| car | 84.40 | 84.67 | +0.28 | 84.67 | 84.30 | 小涨 |

## 4. 与原论文/中间版对照（DensePASS100）

| 版本 | mIoU | 说明 |
|---|---|---|
| 原论文 OmniSAM-S w/ MA | 60.23 | Table3 sam2_s行 |
| 中间版 UDA 60.36 | 60.36 | pd_uda重训权重 |
| 中间版 final 60.26 | 60.26 | base_last权重 |
| 我们复现 dense | 60.46 | 同omnisam_s.pth，+0.23，复现成功 |

逐类贴住：road 81.00/80.76，building 90.02/89.79，sky 94.58/94.41，car 84.40/84.06，仅person 66.47/68.90低2.43。

## 5. 跨域状态（2026-11-02 15:33实测）

- XW500-dense：mIoU 0.09（GT错，非模型差；flat labels_trainid把void当road，GT sidewalk/building/sky全0）。数作废。
- XW500-ca：未出数（队列已杀）。
- XS260全5个：全炸 `decoder 13 vs 8 mismatch`，13类ckpt不能评8类SPAN8，原仓库限制。表3不放SPAN8。
- S100-b2-uniform并行已出：mIoU 60.4621/seam 57.0001/20窗/200额外/4165px/685s，与b2-adapt持平。
- GPU现空：0MiB/85043MiB，0%。
- 下一步：WildPASS改走官方 leftImg8bit/val/cs + gtFine/val/cs，走labelId->train13，先3张sanity再开500，只比dense vs ca；SPAN8改定性或不放。

## 6. 冻结口径

- 主表：dense 60.46 vs ca 60.30（16x压缩-0.16，mAcc+0.35，light/sign+0.6）。
- 消融：bg/pool/topk/uniform/cams。
- 分支封版条件：WildPASS500 dense vs ca重跑出数，保住 ca > pool/topk。
