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
    return "\n".join(L)


def main():
    table = build_table()
    (EVAL / "INDOOR_RESULTS_TABLE.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (EVAL / "INDOOR_RESULTS_TABLE.md").write_text(to_markdown(table), encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
