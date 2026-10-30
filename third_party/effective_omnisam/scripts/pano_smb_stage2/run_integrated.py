#!/usr/bin/env python3
"""Run one integrated train/validate process and record its phase boundary."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VAL_MARKERS = ("Epoch 1 [Val]", "========== per-class IoU (0-100) ==========")
INTEGRATED_ARTIFACTS = (
    "launch_command.txt",
    "run.log",
    "training.log",
    "evaluation.log",
    "train_stage.json",
    "eval_stage.json",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def completed_successfully(run_dir):
    train_path = run_dir / "train_stage.json"
    eval_path = run_dir / "eval_stage.json"
    if not train_path.is_file() or not eval_path.is_file():
        return False
    try:
        train = json.loads(train_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        train.get("process_exit_code") == 0
        and train.get("boundary_observed") is True
        and evaluation.get("process_exit_code") == 0
    )


def archive_interrupted_attempt(run_dir):
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts_dir / f"attempt_{number:02d}").exists():
        number += 1
    archive_dir = attempts_dir / f"attempt_{number:02d}"
    archive_dir.mkdir()
    moved = []
    for name in INTEGRATED_ARTIFACTS:
        source = run_dir / name
        if source.exists():
            shutil.move(str(source), str(archive_dir / name))
            moved.append(name)
    write_json(
        archive_dir / "archive.json",
        {
            "archived_at": utc_now(),
            "reason": "interrupted integrated process resumed",
            "artifacts": moved,
        },
    )
    return archive_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--marker-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing command")

    run_dir = Path(args.run_dir)
    marker_dir = Path(args.marker_dir)
    marker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    prior = [run_dir / name for name in INTEGRATED_ARTIFACTS if (run_dir / name).exists()]
    if prior:
        raise SystemExit(f"refusing to overwrite prior integrated artifacts: {prior}")

    (run_dir / "launch_command.txt").write_text(
        subprocess.list2cmdline(command) + "\n", encoding="utf-8"
    )

    train_start_wall = utc_now()
    train_start = time.monotonic()
    eval_start_wall = None
    eval_start = None
    phase = "train"

    with (run_dir / "training.log").open("w", encoding="utf-8", buffering=1) as train_log, \
            (run_dir / "evaluation.log").open("w", encoding="utf-8", buffering=1) as eval_log, \
            (run_dir / "run.log").open("w", encoding="utf-8", buffering=1) as full_log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            full_log.write(line)
            if phase == "train" and any(marker in line for marker in VAL_MARKERS):
                train_end = time.monotonic()
                write_json(
                    run_dir / "train_stage.json",
                    {
                        "label": args.label,
                        "stage": "TRAIN",
                        "start": train_start_wall,
                        "end": utc_now(),
                        "elapsed_seconds": train_end - train_start,
                        "boundary_observed": True,
                        "process_exit_code": None,
                    },
                )
                (marker_dir / f".{args.label}_TRAIN_DONE").touch()
                phase = "eval"
                eval_start_wall = utc_now()
                eval_start = time.monotonic()
                eval_log.write(line)
            else:
                (eval_log if phase == "eval" else train_log).write(line)
        rc = process.wait()

    end_wall = utc_now()
    end_mono = time.monotonic()
    if phase == "train":
        write_json(
            run_dir / "train_stage.json",
            {
                "label": args.label,
                "stage": "TRAIN",
                "start": train_start_wall,
                "end": end_wall,
                "elapsed_seconds": end_mono - train_start,
                "boundary_observed": False,
                "process_exit_code": rc,
                "error": "validation boundary was never observed",
            },
        )
        return rc if rc else 3

    write_json(
        run_dir / "eval_stage.json",
        {
            "label": args.label,
            "stage": "EVAL",
            "start": eval_start_wall,
            "end": end_wall,
            "elapsed_seconds": end_mono - float(eval_start),
            "process_exit_code": rc,
        },
    )
    if rc:
        return rc

    train_record = json.loads((run_dir / "train_stage.json").read_text(encoding="utf-8"))
    train_record["process_exit_code"] = 0
    write_json(run_dir / "train_stage.json", train_record)
    (marker_dir / f".{args.label}_EVAL_DONE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
