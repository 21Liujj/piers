"""Indoor A-stage supervised scaffold (blocked on official semantic GT).

Does NOT train or score on SPAN8 invalid labels. Verifies trainable set:
encoder trunk LoRA rank4 q/v (192 tensors) + decoder SegFormerHead (14 tensors,
8-cls head re-init recorded separately). Freezes encoder base, memory main,
DINO, text SAM2. Run --audit-only on server to record names/counts.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def audit_checkpoint(ckpt_path: str) -> dict:
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ks = list(ckpt.keys()) if isinstance(ckpt, dict) else []
    lora = sorted([k for k in ks if "linear_a_" in k or "linear_b_" in k])
    dec = sorted([k for k in ks if k.startswith("decoder")]
    )
    mem = sorted([k for k in ks if "memory" in k.lower()])
    return {
        "total": len(ks),
        "lora_tensors": len(lora),
        "decoder_tensors": len(dec),
        "memory_tensors": len(mem),
        "lora_sample": lora[:4],
        "decoder_sample": dec[:6],
        "expect_lora_192": len(lora) == 192,
        "expect_decoder_14": len(dec) == 14,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/autodl-tmp/weights/omnisam_s.pth")
    ap.add_argument("--out", default="/root/autodl-tmp/piers-main/runs/INDOOR-FOLD1-20261109/SUPERVISED_TRAINABLE_AUDIT.json")
    a = ap.parse_args()
    rep = audit_checkpoint(a.ckpt)
    rep.update({"schedule": "head200+joint1000, AdamW 1e-4/1e-5 wd0.01 clip1 bf16 mb1x4 seeds 1234/4321/2026",
                "status": "scaffold_only_no_training_on_invalid_GT"})
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
