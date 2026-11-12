"""Indoor experiment matrix: original OmniSAM vs our full model, and I3 ablations.

Arms (all on SPAN8 Fold1, heldout = area_5a+5b):

  Z0  original OmniSAM frozen 13-head, indoor 8 names, missing -> structural zeros
  Z1  original OmniSAM frozen 13-head + full I1/I2/I3 (no indoor adaptation)
  A0  adapted model (LoRA + 8-cls head), dense 18 semantic forwards
  A1  adapted model + full 20 forwards + real text branch
  I3 ablation inside A1:
      A1_B0  18 forwards (strict degradation to A0 geometry)
      A1_B1  19 forwards
      A1_B2  20 forwards (full I3)

Protocol guards:
  - GT is read only by the scorer, never by the model path
  - Z0/Z1 keep the frozen street 13-head; A* use the 8-class head
  - I2 memory config and I1/I3 source code are untouched
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109")
SPAN8 = ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"]
AREA_ROLES = {
    "area1_3_6": "train",
    "area2_4": "train",
    "area5": "heldout",
}


def arm_specs() -> dict:
    """Declarative spec of every arm to run."""
    return {
        "Z0": {
            "desc": "original OmniSAM, frozen street 13-head, no I1/I2/I3, no adaptation",
            "head": "street13",
            "adapted": False,
            "memory": "dense",
            "semantic_forwards": 18,
            "i3_budget": 0,
            "text_branch": False,
            "map13_to_span8": True,
        },
        "Z1": {
            "desc": "original OmniSAM + full I1/I2/I3 (no indoor adaptation)",
            "head": "street13",
            "adapted": False,
            "memory": "i2",
            "semantic_forwards": 20,
            "i3_budget": 2,
            "text_branch": True,
            # The frozen street checkpoint carries NO text-branch weights (0 of
            # its 1806 tensors), so its txt_head projector is random.  The Z
            # arm therefore runs the text branch *inert* (lambda_txt = 0, i.e.
            # the text logits are computed but not added to the visual logits).
            # Claiming a text contribution here would be unfounded.
            "lambda_txt": 0.0,
            "text_branch_trained": False,
            "map13_to_span8": True,
        },
        "A0": {
            "desc": "adapted model (LoRA r4 + 8-cls head), dense, 18 forwards",
            "head": "span8",
            "adapted": True,
            "memory": "dense",
            "semantic_forwards": 18,
            "i3_budget": 0,
            "text_branch": False,
            "map13_to_span8": False,
        },
        "A1_B0": {
            "desc": "adapted model + I2 + 18 forwards (strict degradation baseline)",
            "head": "span8",
            "adapted": True,
            "memory": "i2",
            "semantic_forwards": 18,
            "i3_budget": 0,
            "text_branch": True,
            # The A arms are fine-tuned WITH the indoor text branch, so the
            # text projector is trained rather than random.
            "lambda_txt": 0.1,
            "text_branch_trained": True,
            "map13_to_span8": False,
        },
        "A1_B1": {
            "desc": "adapted model + I2 + 19 forwards (1 I3 seam window)",
            "head": "span8",
            "adapted": True,
            "memory": "i2",
            "semantic_forwards": 19,
            "i3_budget": 1,
            "text_branch": True,
            "lambda_txt": 0.1,
            "text_branch_trained": True,
            "map13_to_span8": False,
        },
        "A1_B2": {
            "desc": "adapted model + I2 + 20 forwards (full I3, 2 seam windows)",
            "head": "span8",
            "adapted": True,
            "memory": "i2",
            "semantic_forwards": 20,
            "i3_budget": 2,
            "text_branch": True,
            "lambda_txt": 0.1,
            "text_branch_trained": True,
            "map13_to_span8": False,
        },
    }


def write_plan(path=RUNS / "INDOOR_EXPERIMENT_MATRIX.json"):
    specs = arm_specs()
    plan = {
        "dataset": "SPAN8 / Stanford2D3D via 360BEV mirror",
        "classes": SPAN8,
        "split": "fit 832 / calib 104 / val 104 (train, room-disjoint) / heldout 373 (area_5a+5b)",
        "canvas": [1024, 512],
        "semantic_geometry": {
            "window": [512, 512],
            "stride": 64,
            "starts": 9,
            "passes": 2,
            "base_forwards": 18,
        },
        "arms": specs,
        "compare_pairs": [
            {"a": "Z0", "b": "A0", "claim": "indoor adaptation gain (original vs our model)"},
            {"a": "Z1", "b": "A1_B2", "claim": "full-system gain with I1/I2/I3 both sides"},
            {"a": "A0", "b": "A1_B0", "claim": "I2 memory effect on adapted model"},
            {"a": "A1_B0", "b": "A1_B1", "claim": "I3 budget step 1"},
            {"a": "A1_B1", "b": "A1_B2", "claim": "I3 budget step 2"},
            {"a": "A1_B0", "b": "A1_B2", "claim": "total I3 effect at B=2"},
        ],
        "metric": "mIoU over 8 classes + per-class IoU, ignore=255, heldout only",
        "guards": [
            "GT read only by scorer",
            "frozen outdoor modules untouched",
            "Z arms use street 13-head with map13_to_span8 LUT",
            "A arms use adapted 8-cls head",
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def compare_table(metrics: dict) -> list:
    """Build the mIoU comparison rows from {arm: {miou, per_class_iou}}."""
    rows = []
    for arm, m in metrics.items():
        rows.append({
            "arm": arm,
            "miou": round(m["miou"], 4),
            "per_class_iou": {k: round(v, 4) for k, v in m.get("per_class_iou", {}).items()},
        })
    return sorted(rows, key=lambda r: r["miou"])


if __name__ == "__main__":
    plan = write_plan()
    print(json.dumps(plan, indent=2))
