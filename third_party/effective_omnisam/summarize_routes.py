import json
import os

results = []
routes = {
    "route1": [
        "boundary_0625_n500_seed1_20260826_101721",
        "boundary_125_n500_seed1_20260826_104642",
        "boundary_25_n500_seed1_20260826_111703",
        "boundary_125_reuse_seed1_FIXED",
        "boundary_25_reuse_seed1_20260826_154640",
    ],
    "route3": [
        "route3_rare010_n500_seed1_20260826_114654",
        "route3_rare015_n500_seed1_20260826_121710",
        "route3_rare020_n500_seed1_20260826_124736",
        "rare0125_reuse_seed1_FIXED",
        "rare020_reuse_seed1_RETRY",
    ],
}

for route, runs in routes.items():
    for run in runs:
        metrics_file = f"experiments/{route}/runs/{run}/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                lines = [l for l in f.readlines() if l.strip()]
                if lines:
                    last_line = json.loads(lines[-1])
                    print(f"{route} | {run}")
                    print(f"  mIoU={last_line.get('val_mIoU')} mAcc={last_line.get('val_mAcc')} PixAcc={last_line.get('val_PixAcc')} mDice={last_line.get('val_mDice')}")
                    print(f"  epoch={last_line.get('epoch')} step_time={last_line.get('step_time')}")
