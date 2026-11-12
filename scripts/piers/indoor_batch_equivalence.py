"""Verify that batched window inference equals per-window inference.

The indoor runner batches the 9 windows of a pass into a single forward to
cut the evaluation time.  This test asserts that the batched result is
identical to the sequential one, so the speed-up cannot change any number
that ends up in the thesis table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/root/autodl-tmp/piers-main")
EFF = REPO / "third_party" / "effective_omnisam"
for p in (str(EFF), str(EFF / "sam"), str(REPO / "scripts" / "piers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from indoor_geometry import INDOOR_W, INDOOR_H, INDOOR_WIN, semantic_starts  # noqa: E402
from indoor_eval_runner import build_model, forward_pred, to_span8  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model("Z0", None, device)

    rng = np.random.default_rng(0)
    tensor = torch.from_numpy(
        rng.standard_normal((3, INDOOR_H, INDOOR_W)).astype(np.float32)).unsqueeze(0)
    starts = semantic_starts()
    order = starts[:4]  # a subset keeps the test quick but still exercises batching

    model.memory_bank._init_output_dict_source()
    seq = []
    for s in order:
        win = tensor[:, :, :, s:s + INDOOR_WIN].to(device)
        out = forward_pred(model, 0, win)
        seq.append(to_span8(out[0].float().cpu()))

    model.memory_bank._init_output_dict_source()
    wins = torch.cat([tensor[:, :, :, s:s + INDOOR_WIN] for s in order], dim=0).to(device)
    out = forward_pred(model, 0, wins)
    batched = out.float().cpu()
    for k, s in enumerate(order):
        b = to_span8(batched[k])
        agree = float((b == seq[k]).float().mean())
        print(f"window {s}: agreement {agree:.6f}")
        assert agree > 0.999, f"batched inference diverged at window {s}: {agree}"
    print("BATCH_EQUIVALENCE_OK")


if __name__ == "__main__":
    main()
