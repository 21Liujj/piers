"""Verify every suite config resolves to a registered suite code."""
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))

from piers_ljj.experiments.suites import SUITES, canonical_suite  # noqa: E402

bad = 0
for f in sorted(glob.glob("configs/piers/*.yaml")):
    if "paths" in f:
        continue
    with open(f, encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    raw = c.get("suite", "")
    s = canonical_suite(raw)
    ok = s in SUITES
    if not ok:
        bad += 1
    print(f"{os.path.basename(f):28s} suite={raw:26s} -> {s:5s} "
          f"[{'OK' if ok else 'UNKNOWN'}] run_id={c.get('run_id')}")

print(f"\n{len(SUITES)} suites registered, {bad} unresolved config(s)")
sys.exit(1 if bad else 0)
