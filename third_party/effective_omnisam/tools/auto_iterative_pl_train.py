import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("== CMD ==\n")
        f.write(" ".join(cmd) + "\n")
        f.write("== START ==\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
        return p.wait()


def _read_last_metrics(metrics_path: Path, n: int = 5) -> List[dict]:
    if not metrics_path.is_file():
        return []
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _parse_iou_from_ckpt_name(path: Path) -> Optional[float]:
    m = re.search(r"iou[_]?(\d+\.\d+)", path.name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _teacher_for_pl_generation(teacher_ckpt: Path, floor_ckpt: Path) -> Path:
    """
    Never use a weaker teacher (by filename mIoU) than floor_ckpt for pseudo-label generation.
    Protects against stale metrics pointing at a degraded best_*.pth.
    """
    if not floor_ckpt.is_file():
        return teacher_ckpt
    t_cur = _parse_iou_from_ckpt_name(teacher_ckpt)
    t_floor = _parse_iou_from_ckpt_name(floor_ckpt)
    if t_floor is None:
        return teacher_ckpt
    if t_cur is None or t_floor > t_cur:
        return floor_ckpt
    return teacher_ckpt


def _should_regen(miou_hist: List[float], plateau_eps: float, lookback: int) -> bool:
    if len(miou_hist) < lookback:
        return False
    recent = miou_hist[-lookback:]
    # Condition A: strictly decreasing
    if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
        return True
    # Condition B: plateau (small range)
    if (max(recent) - min(recent)) < plateau_eps:
        return True
    return False


@dataclass
class Paths:
    pd_train: Path
    pd_pl_base: Path
    ckpt_dir: Path
    metrics_jsonl: Path


def _resolve_paths(pd_train: Path, dataset: str, class_set: int, backbone: str) -> Paths:
    ckpt_dir = pd_train / "checkpoints" / f"{dataset}_{class_set}cls" / backbone
    return Paths(
        pd_train=pd_train,
        pd_pl_base=pd_train.parent / (pd_train.name + "_pl_rounds"),
        ckpt_dir=ckpt_dir,
        metrics_jsonl=ckpt_dir / "metrics.jsonl",
    )


def main():
    ap = argparse.ArgumentParser(description="Auto iterative pseudo-label refinement training driver.")
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="Cityscapes")
    ap.add_argument("--class_set", type=int, default=13, choices=[13, 19])
    ap.add_argument("--backbone", type=str, default="sam2_s")
    ap.add_argument("--pd_train", type=str, required=True, help="PD_ROOT used for checkpoints/resume state.")
    ap.add_argument("--teacher_ckpt_init", type=str, required=True, help="Initial teacher ckpt for round-1 PL generation.")
    ap.add_argument("--src_ckpt", type=str, default="/root/autodl-tmp/omnisam_s.pth")

    ap.add_argument("--total_epochs", type=int, default=12, help="Total target epochs to reach (1-based).")
    ap.add_argument("--chunk_epochs", type=int, default=3, help="Train epochs per PL round before checking triggers.")

    ap.add_argument("--warmup_pl", type=int, default=2000)
    ap.add_argument("--iters_pl", type=int, default=0)
    ap.add_argument("--pl_batch_size", type=int, default=1, help="Pseudo-label generation dataloader batch size.")
    ap.add_argument("--pl_num_workers", type=int, default=4, help="Pseudo-label generation dataloader workers.")
    ap.add_argument("--conf_thr", type=float, default=0.85)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--plateau_eps", type=float, default=0.10)
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--max_rounds", type=int, default=10)
    ap.add_argument("--sleep_s", type=int, default=2)
    ap.add_argument("--log_dir", type=str, default="logs")
    args = ap.parse_args()

    pd_train = Path(args.pd_train)
    paths = _resolve_paths(pd_train=pd_train, dataset=args.dataset, class_set=args.class_set, backbone=args.backbone)
    (Path(args.log_dir)).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_dir) / f"auto_iterative_pl_train_{stamp}.log"

    # Ensure ckpt dir exists (engine will create, but we want stable paths).
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpt = Path(args.teacher_ckpt_init)
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"teacher_ckpt_init not found: {teacher_ckpt}")
    teacher_floor = teacher_ckpt.resolve()

    # Track progression
    round_idx = 0
    while round_idx < args.max_rounds:
        round_idx += 1
        pd_pl = paths.pd_pl_base / f"round{round_idx}"
        pl_dir = pd_pl / "DensePASS" / args.backbone / f"{args.dataset}_{args.class_set}cls" / "epoch1"
        pl_dir_str = str(pl_dir)

        # (1) Generate pseudo-labels for epoch1 using current teacher (never weaker than teacher_ckpt_init)
        gen_teacher = _teacher_for_pl_generation(teacher_ckpt, teacher_floor)
        cmd_gen = [
            sys.executable,
            "run_adaptation.py",
            "--backbone", args.backbone,
            "--dataset", args.dataset,
            "--class_set", str(args.class_set),
            "--data_root", args.data_root,
            "--pd_root", str(pd_pl),
            "--src_ckpt", args.src_ckpt,
            "--tgt_ckpt", str(gen_teacher),
            "--warmup_pl", str(args.warmup_pl),
            "--iters_pl", str(args.iters_pl),
            "--pl_batch_size", str(args.pl_batch_size),
            "--pl_num_workers", str(args.pl_num_workers),
            "--batch_size", "1",
            "--num_epochs", "1",
            "--generate_only",
        ]
        if args.amp:
            cmd_gen.append("--amp")
        else:
            cmd_gen.append("--no-amp")
        if args.tf32:
            cmd_gen.append("--tf32")
        else:
            cmd_gen.append("--no-tf32")
        if gen_teacher.resolve() != teacher_ckpt.resolve():
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"\n[auto_iter_pl] round{round_idx}: PL gen teacher floored to {gen_teacher} "
                    f"(metrics had {teacher_ckpt})\n"
                )
        rc = _run(cmd_gen, log_file)
        if rc != 0:
            raise RuntimeError(f"Pseudo-label generation failed (round={round_idx}, rc={rc}). See {log_file}")

        # (2) Train for chunk_epochs on this fixed PL dir, using pd_train for checkpoints.
        # Determine current epoch start from metrics.jsonl if exists; else start at 1.
        last = _read_last_metrics(paths.metrics_jsonl, n=1)
        cur_epoch = int(last[-1]["epoch"]) if last else 0
        target_epoch = min(args.total_epochs, cur_epoch + args.chunk_epochs if cur_epoch else args.chunk_epochs)
        if target_epoch <= cur_epoch:
            break

        cmd_train = [
            sys.executable,
            "run_adaptation.py",
            "--backbone", args.backbone,
            "--dataset", args.dataset,
            "--class_set", str(args.class_set),
            "--data_root", args.data_root,
            "--pd_root", str(pd_train),
            "--src_ckpt", args.src_ckpt,
            "--tgt_ckpt", str(teacher_ckpt),
            "--reuse_pl_dir", pl_dir_str,
            "--num_epochs", str(target_epoch),
            "--batch_size", str(args.batch_size),
            "--conf_thr", str(args.conf_thr),
        ]
        if args.amp:
            cmd_train.append("--amp")
        else:
            cmd_train.append("--no-amp")
        if args.tf32:
            cmd_train.append("--tf32")
        else:
            cmd_train.append("--no-tf32")

        rc = _run(cmd_train, log_file)
        if rc != 0:
            raise RuntimeError(f"Training failed (round={round_idx}, rc={rc}). See {log_file}")

        # (3) Decide whether to regenerate pseudo-labels based on recent mIoU trend.
        recs = _read_last_metrics(paths.metrics_jsonl, n=max(10, args.lookback + 2))
        miou_hist = [float(r.get("val_mIoU", 0.0)) for r in recs]
        best_paths = [r.get("best_ckpt_path") for r in recs if r.get("best_ckpt_path")]
        best_ckpt_path = best_paths[-1] if best_paths else None

        if best_ckpt_path and os.path.isfile(best_ckpt_path):
            teacher_ckpt = Path(best_ckpt_path)

        # Stop if reached total_epochs
        if recs and int(recs[-1]["epoch"]) >= args.total_epochs:
            break

        # Trigger regen if plateau/decline
        if _should_regen(miou_hist, plateau_eps=args.plateau_eps, lookback=args.lookback):
            # regenerate next round with updated teacher_ckpt
            time.sleep(args.sleep_s)
            continue
        else:
            # Keep same pseudo-labels for next chunk by reusing same round index (do not regen).
            # Implement by decrementing round_idx so next loop repeats same PL dir.
            round_idx -= 1
            time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()

