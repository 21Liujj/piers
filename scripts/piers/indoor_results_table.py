"""Collect the indoor Fold-1 results into the final comparison table.

Reads the per-arm metrics JSON written by `indoor_eval_runner` and produces
  * `INDOOR_RESULTS_TABLE.json`  -- machine readable
  * `INDOOR_RESULTS_TABLE.md`    -- the thesis-ready table

The comparison pairs come from `indoor_experiment_matrix.write_plan` so the
table cannot silently disagree with the declared protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

from indoor_experiment_matrix import arm_specs, write_plan

RUNS = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")
EVAL = RUNS / "eval"
SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]

ARM_LABEL = {
    "Z0": "Original OmniSAM (frozen street 13-head, zero-shot)",
    "Z1": "Original OmniSAM + full I1/I2/I3 (no indoor adaptation)",
    "A0": "Adapted model (LoRA + 8-cls head), dense",
    "A1_B0": "Adapted model + I2, 18 forwards",
    "A1_B1": "Adapted model + I2, 19 forwards (1 I3 window)",
    "A1_B2": "Adapted model + I2, 20 forwards (full I3)",
}


PROTOCOL_NOTES = [
    "GT 只在 scorer 中读取，模型路径从不读取 GT。",
    "Z 组使用冻结的街道 13 类头，经官方 13->8 LUT 映射；"
    "无街道对应物的类别（beam/board/bookcase/clutter/column）结构上不可达，"
    "记为 255 并单独统计（不产生假阳性）。",
    "A 组使用室内微调的 8 类头；13 类分类器因形状不匹配被丢弃，8 类头从头训练。",
    "Memory bank 是时序模块：只有 frame_idx != 0 时才读历史。"
    "评测必须用递增 frame_idx 驱动窗口，否则 memory 被静默禁用。",
    "前向必须是顺序的（batch=1）。批量化会把整趟 history 压成同一个 frame_idx，"
    "静默关闭 memory attention（实测像素一致率仅 68%）。",
    "冻结的街道 checkpoint 不含任何 text 分支权重（1806 个张量中 0 个），"
    "因此 Z1 的 text 分支以 lambda_txt=0 运行（text logits 不参与融合），"
    "A1_* 组使用 stage-2 训练过的 text 分支（lambda_txt=0.1）。",
    "I2 的 rare-class 保护使用实测的 SPAN8 稀有类（sofa/window），"
    "而不是街道类别 id；配置其余参数与室外冻结配置一致。",
    "I3 接缝窗口带该趟 pass 的 memory 上下文重新观测（outdoor 同款），"
    "结果通过覆盖度加权融合进入基线预测，不覆盖基线。",
]


def load_metrics(split="heldout"):
    out = {}
    for arm in arm_specs():
        p = EVAL / f"{arm}_{split}_metrics.json"
        if p.is_file():
            out[arm] = json.loads(p.read_text(encoding="utf-8"))
    return out


def build_table(split="heldout"):
    metrics = load_metrics(split)
    plan = write_plan(RUNS / "INDOOR_EXPERIMENT_MATRIX.json")

    rows = []
    for arm, spec in arm_specs().items():
        m = metrics.get(arm)
        st = EVAL / f"{arm}_run_stats.json"
        run = json.loads(st.read_text(encoding="utf-8")) if st.is_file() else {}
        rows.append({
            "arm": arm,
            "label": ARM_LABEL.get(arm, arm),
            "description": spec["desc"],
            "head": spec["head"],
            "adapted": spec["adapted"],
            "memory": spec["memory"],
            "forwards": spec["semantic_forwards"],
            "i3_budget": spec["i3_budget"],
            "forwards_measured": run.get("forwards_per_image"),
            "wall_s": run.get("wall_s"),
            "memory_audit": run.get("memory_audit_final"),
            "lambda_txt": spec.get("lambda_txt", 0.0),
            "text_branch_trained": spec.get("text_branch_trained", False),
            "miou": round(m["miou"], 4) if m else None,
            "mAcc": round(m["mAcc"], 4) if m else None,
            "per_class_iou": {k: round(v, 4) for k, v in m["per_class_iou"].items()} if m else None,
            "n_images": m.get("n_images") if m else None,
        })

    pairs = []
    for cp in plan["compare_pairs"]:
        a, b = cp["a"], cp["b"]
        ma, mb = metrics.get(a), metrics.get(b)
        entry = {"a": a, "b": b, "claim": cp["claim"]}
        if ma and mb:
            entry["miou_a"] = round(ma["miou"], 4)
            entry["miou_b"] = round(mb["miou"], 4)
            entry["delta_miou"] = round(mb["miou"] - ma["miou"], 4)
            per = {}
            for c in SPAN8:
                va, vb = ma["per_class_iou"].get(c), mb["per_class_iou"].get(c)
                if va is not None and vb is not None and not (va != va and vb != vb):
                    per[c] = round(vb - va, 4)
            entry["delta_per_class_iou"] = per
        pairs.append(entry)

    return {
        "dataset": plan["dataset"],
        "split": plan["split"],
        "metric": plan["metric"],
        "guards": plan["guards"],
        "protocol_notes": PROTOCOL_NOTES,
        "arms": rows,
        "compare_pairs": pairs,
    }


def to_markdown(table) -> str:
    L = []
    L.append("# 室内 SPAN8 适配结果（Fold 1）")
    L.append("")
    L.append(f"数据集：{table['dataset']}")
    L.append(f"划分：{table['split']}")
    L.append(f"指标：{table['metric']}")
    L.append("")
    L.append("## 主表")
    L.append("")
    L.append("| Arm | 说明 | 头 | 适配 | Memory | Forwards | mIoU | mAcc |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in table["arms"]:
        mi = f"{r['miou']:.4f}" if r["miou"] is not None else "—"
        ma = f"{r['mAcc']:.4f}" if r["mAcc"] is not None else "—"
        L.append(f"| {r['arm']} | {r['label']} | {r['head']} | "
                 f"{'是' if r['adapted'] else '否'} | {r['memory']} | {r['forwards']} | {mi} | {ma} |")
    L.append("")
    L.append("## 逐类 IoU")
    L.append("")
    L.append("| Arm | " + " | ".join(SPAN8) + " |")
    L.append("|---" * (len(SPAN8) + 1) + "|")
    for r in table["arms"]:
        if not r["per_class_iou"]:
            continue
        cells = [f"{r['per_class_iou'].get(c, float('nan')):.4f}" for c in SPAN8]
        L.append(f"| {r['arm']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 对比")
    L.append("")
    L.append("| 对比 | 主张 | mIoU (a) | mIoU (b) | Δ mIoU |")
    L.append("|---|---|---|---|---|")
    for p in table["compare_pairs"]:
        if "delta_miou" in p:
            L.append(f"| {p['a']} → {p['b']} | {p['claim']} | {p['miou_a']:.4f} | "
                     f"{p['miou_b']:.4f} | {p['delta_miou']:+.4f} |")
        else:
            L.append(f"| {p['a']} → {p['b']} | {p['claim']} | — | — | 未完成 |")
    L.append("")
    L.append("## 协议保证")
    L.append("")
    for g in table["guards"]:
        L.append(f"- {g}")
    L.append("")
    L.append("## 协议细节（实现级）")
    L.append("")
    for n in table.get("protocol_notes", []):
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


def main():
    table = build_table()
    (EVAL / "INDOOR_RESULTS_TABLE.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (EVAL / "INDOOR_RESULTS_TABLE.md").write_text(to_markdown(table), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
