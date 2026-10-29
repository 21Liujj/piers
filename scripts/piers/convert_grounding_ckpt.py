"""Convert the local grounding-dino checkpoint to the layout modern
transformers expects (fused `in_proj_weight` -> split query/key/value).

Why this is needed
------------------
The vendored `grounding-dino-tiny` checkpoint stores the decoder self-attention
and the text-enhancer attention as a FUSED projection:

    model.decoder.layers.{i}.self_attn.in_proj_weight      (768, 256)
    model.decoder.layers.{i}.self_attn.in_proj_bias        (768,)
    model.decoder.layers.{i}.encoder_attn_text.in_proj_*
    model.encoder.layers.{i}.text_enhancer_layer.self_attn.in_proj_*

Every transformers release we can install (>= 4.40) declares those layers as
THREE separate `nn.Linear` modules (`query`, `key`, `value`), so loading the
file directly silently drops ~36 attention tensors and randomly initialises the
corresponding weights -> the detector returns garbage (zero candidates).

The fused tensor is exactly the concatenation [query; key; value] along dim 0,
which is what transformers' own `convert_grounding_dino_to_hf.py` assumes in
`read_in_q_k_v_decoder` / `read_in_q_k_v_text_enhancer`. We reproduce that
conversion here, but as a standalone, verifiable step that runs offline.

The Swin backbone already stores split query/key/value and is left untouched.

Usage
-----
    python scripts/piers/convert_grounding_ckpt.py \
        --src  D:/piers2027/weights/grounding-dino-tiny/pytorch_model.bin \
        --dst  D:/piers2027/weights/grounding-dino-tiny-hf/pytorch_model.bin

The converted directory keeps the original config/tokenizer files (pass
--copy-aux) so it can be loaded with local_files_only=True.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

AUX_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.txt", "added_tokens.json",
             "special_tokens_map.json")


def _split_fused(sd: dict, prefix: str, hidden: int, report: list) -> None:
    """Split `<prefix>.in_proj_weight/bias` into query/key/value tensors."""
    wkey = f"{prefix}.in_proj_weight"
    bkey = f"{prefix}.in_proj_bias"
    if wkey not in sd:
        return
    w = sd.pop(wkey)
    b = sd.pop(bkey) if bkey in sd else None
    if w.shape[0] != 3 * hidden:
        raise ValueError(f"{wkey} has shape {tuple(w.shape)}, expected (3*{hidden}, ...)")
    q, k, v = w[:hidden], w[hidden:2 * hidden], w[2 * hidden:]
    sd[f"{prefix}.query.weight"] = q.clone()
    sd[f"{prefix}.key.weight"] = k.clone()
    sd[f"{prefix}.value.weight"] = v.clone()
    if b is not None:
        bq, bk, bv = b[:hidden], b[hidden:2 * hidden], b[2 * hidden:]
        sd[f"{prefix}.query.bias"] = bq.clone()
        sd[f"{prefix}.key.bias"] = bk.clone()
        sd[f"{prefix}.value.bias"] = bv.clone()
    report.append(prefix)


def convert(state_dict: dict, hidden: int = 256, encoder_layers: int = 6,
            decoder_layers: int = 6) -> tuple:
    sd = dict(state_dict)
    report: list = []
    for i in range(decoder_layers):
        _split_fused(sd, f"model.decoder.layers.{i}.self_attn", hidden, report)
        _split_fused(sd, f"model.decoder.layers.{i}.encoder_attn_text", hidden, report)
    for i in range(encoder_layers):
        _split_fused(sd, f"model.encoder.layers.{i}.text_enhancer_layer.self_attn",
                     hidden, report)
    return sd, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original pytorch_model.bin")
    ap.add_argument("--dst", required=True, help="destination directory")
    ap.add_argument("--copy-aux", action="store_true",
                    help="copy config/tokenizer files next to the converted bin")
    ap.add_argument("--hidden", type=int, default=256)
    a = ap.parse_args()

    if not os.path.exists(a.src):
        print(f"FATAL missing source checkpoint: {a.src}")
        return 2

    raw = torch.load(a.src, map_location="cpu", weights_only=True)
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]

    sd, report = convert(raw, hidden=a.hidden)
    print(f"converted {len(report)} fused attention layers")
    for p in report[:6]:
        print(f"  {p}")
    if len(report) > 6:
        print(f"  ... and {len(report) - 6} more")

    os.makedirs(a.dst, exist_ok=True)
    out_bin = os.path.join(a.dst, "pytorch_model.bin")
    torch.save(sd, out_bin)
    print(f"wrote {out_bin} ({os.path.getsize(out_bin)} bytes, {len(sd)} keys)")

    if a.copy_aux:
        src_dir = os.path.dirname(os.path.abspath(a.src))
        for f in AUX_FILES:
            p = os.path.join(src_dir, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(a.dst, f))
                print(f"copied {f}")

    # sanity: report any remaining fused keys
    leftover = [k for k in sd if "in_proj" in k]
    print(f"remaining fused in_proj keys: {len(leftover)}")
    if leftover:
        print(f"  {leftover[:8]}")
    meta = {"source": os.path.abspath(a.src), "hidden": a.hidden,
            "converted_layers": len(report), "keys": len(sd),
            "leftover_in_proj": leftover}
    with open(os.path.join(a.dst, "conversion_report.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
