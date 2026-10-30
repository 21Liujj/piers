# Pano-SMB Stage 2 Controlled Experiment

## Status

Stage 2A preparation: PENDING validation. Formal B0/B1 execution: PENDING.

## Experiment Protocol

The protocol is recovered from the clean fixed-pseudo-label run `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135`, which recorded mIoU 60.02 and BFScore 39.23. Its source config is `experiments/tecma/configs/A_Dtrain_Etrain.yaml`.

- Task: Cityscapes source to DensePASS target, 13 classes, `sam2_s`.
- Source list: `dataset/cityscapes_list/train.txt`.
- Target master list: `dataset/densepass_list/train.txt`.
- Frozen target subset: `experiments/tecma/runs/_subsets/Cityscapes_13cls_seed1_n2000.txt` (2000 entries).
- Validation list: `dataset/densepass_list/val.txt`.
- Fixed pseudo labels: `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`.
- Initialization: `/root/autodl-tmp/omnisam_s.pth` for source and target.
- Seed/batch: seed 1, train batch 4, PL batch 1, PL workers 2.
- Schedule: one full epoch, 500 target iterations, AdamW, LR `2e-6`, weight decay `1e-4`, polynomial scheduler with 1800 warmup iterations.
- Precision: AMP and TF32 enabled.
- Fixed-PL behavior: `--reuse_pl_dir`, `--no-pl_shuffle`; pseudo-label generation is skipped.
- Memory: MTC pool stride 2, then legacy feature-norm SMR at keep ratio 0.5; both apply only in training.
- Evaluation: the unchanged `engine.validate(epoch)` follows `engine.train(epoch)` in the historical integrated entry point. It computes mIoU, BFScore, per-class IoU, mAcc, PixAcc, FWIoU, and mDice.
- Profiling: existing `metrics/train_timing.jsonl` fields, including step time, epoch time, memory-attention time, memory tokens, and peak GPU memory.

## Frozen Configs

- B0: `experiments/pano_smb_stage2/B0_legacy/config.yaml`, `plugins.pano_smb.enabled=false`.
- B1: `experiments/pano_smb_stage2/B1_pano_smb/config.yaml`, `plugins.pano_smb.enabled=true` and `min_tokens=256`.
- Parsed config diff: `experiments/pano_smb_stage2/summary/config_diff.txt`.

Both configs explicitly snapshot the same pseudo confidence mask, MTC, SMR scorer, keep ratio, cache behavior, fallback behavior, and train/PL application gates. The allowed difference is only `plugins.pano_smb.enabled`.

## Fairness Checklist

- Dataset, source list, target list, fixed subset, fixed PL, seed, checkpoint, batch size, optimizer, LR, schedule, augmentations, MTC, SMR scorer, evaluation, precision, and GPU protocol: SAME.
- Total historical token budget: SAME (`512 x J`).
- Scientific variable: uniform per-FoV quota versus adaptive batch-shared per-FoV quota.
- Output directories: isolated and not shared with historical runs.
- B0 runs first; training, evaluation, metrics, and sanity failures stop B1.
- B0 sanity anchors are 60.02 mIoU and 39.23 BFScore with explicit tolerances 0.30 and 0.50.
- The sanity gate checks protocol reproduction only and does not make a scientific verdict.

## Scripts

- Master runner: `scripts/pano_smb_stage2/run_stage2.sh`.
- One-shot status: `scripts/pano_smb_stage2/status_stage2.sh`.
- Integrated train/eval stage recorder: `scripts/pano_smb_stage2/run_integrated.py`.
- Metrics extractor: `scripts/pano_smb_stage2/extract_metrics.py`.
- Config checker: `scripts/pano_smb_stage2/check_config_diff.py`.
- B0 gate: `scripts/pano_smb_stage2/check_b0_sanity.py`.
- Summary builder: `scripts/pano_smb_stage2/build_summary.py`.

The master runner is resumable through stage markers and artifact validation. The historical command combines training and validation in one Python process; the stage recorder detects the validation boundary and writes separate training/evaluation logs, timestamps, elapsed times, exit records, and markers without modifying the evaluator.

## Output Paths

- Root: `experiments/pano_smb_stage2`.
- B0 runtime artifacts: `experiments/pano_smb_stage2/B0_legacy/run`.
- B1 runtime artifacts: `experiments/pano_smb_stage2/B1_pano_smb/run`.
- Master log: `experiments/pano_smb_stage2/stage2_master.log`.
- Machine summary: `experiments/pano_smb_stage2/summary/stage2_summary.json`.
- Human summary: `experiments/pano_smb_stage2/summary/stage2_summary.md`.
- Per-class comparison: `experiments/pano_smb_stage2/summary/per_class_delta.csv`.
- B1 quota aggregate: `experiments/pano_smb_stage2/summary/B1_quota_summary.json`.

## Launch Command

```bash
tmux new-session -d -s pano_smb_stage2 'cd /root/autodl-tmp/OmniSAM-main && bash scripts/pano_smb_stage2/run_stage2.sh'
```

## Status Command

```bash
bash scripts/pano_smb_stage2/status_stage2.sh
```

## Final Summary

```bash
cat experiments/pano_smb_stage2/summary/stage2_summary.md
```

## Results

PENDING. Stage 2A does not launch B0 or B1 and makes no accuracy claim.
