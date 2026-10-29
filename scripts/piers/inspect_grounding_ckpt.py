"""Inspect the raw key names inside the local grounding-dino checkpoint.

Prints the attention-layer keys so we can tell which transformers layout the
file was saved with (fused `in_proj_weight` vs split `query/key/value`).
"""
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\piers2027\weights\grounding-dino-tiny\pytorch_model.bin"
sd = torch.load(path, map_location="cpu", weights_only=True)
if "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]

keys = sorted(sd.keys())
print(f"total keys: {len(keys)}")

print("\n--- attention-ish keys (decoder layer 0) ---")
for k in keys:
    if "decoder.layers.0." in k and ("attn" in k or "query" in k or "key" in k
                                    or "value" in k or "in_proj" in k):
        print(f"  {k}  shape={tuple(sd[k].shape)}")

print("\n--- text_enhancer layer 0 attention ---")
for k in keys:
    if "encoder.layers.0.text_enhancer_layer" in k and (
            "attn" in k or "in_proj" in k or "query" in k or "key" in k or "value" in k):
        print(f"  {k}  shape={tuple(sd[k].shape)}")

print("\n--- does it contain fused in_proj at all? ---")
fused = [k for k in keys if "in_proj" in k]
split = [k for k in keys if k.endswith(".query.weight") or k.endswith(".key.weight")
         or k.endswith(".value.weight")]
print(f"  fused in_proj keys : {len(fused)}")
print(f"  split q/k/v keys   : {len(split)}")
for k in split[:6]:
    print(f"    {k}")
