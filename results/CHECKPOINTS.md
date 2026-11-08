# Trained checkpoints (our weights)

Three checkpoints were trained by this project and are published as GitHub
Release assets so the published numbers can be reproduced without retraining.

Release: [`v0.2-paper-repro`](https://github.com/21Liujj/piers/releases/tag/v0.2-paper-repro)

| Asset | SHA256 | Role |
|---|---|---|
| `allhard_runs_R0_seed1234_final.pth` | `ff79b0b0f766dbe7ad695bd56535d11cade9b12f608ce0ddb13778ba49f9a1d3` | I1 main arm, seed 1234 |
| `allhard_runs_R1_seed1234_final.pth` | `07fdc14b7bf072edc78c402fd11e940d28a7e3dd73dcd3ff7f5f4c973b6a7232` | I1 main arm, seed 4321 |
| `dense_controls_R0_seed1234_final.pth` | `40159d944c8925cd1c61ce4f8929cbde1fdbda761d0c41fb3bdfe27fec8f5625` | dense-memory control |

```bash
gh release download v0.2-paper-repro --repo 21Liujj/piers --dir ./ckpt
```

Place them as:

```
ckpt/allhard_runs/R0_seed1234/training/final.pth
ckpt/allhard_runs/R1_seed1234/training/final.pth
ckpt/dense_controls/trained/R0_seed1234/training/final.pth
```

Verify a download:

```bash
python scripts/piers/verify_release_asset.py
```

---

## What is inside a checkpoint

Each file is a plain `OrderedDict` state dict, **1806 tensors / 140,126,644
parameters**, with four top-level groups:

| Prefix | Tensors | What it is |
|---|---|---|
| `sam_model.*` | 583 | the SAM2 model as used by the OmniSAM pipeline |
| `encoder.*` | 583 | the UDA source encoder |
| `decoder.*` | 14 | the 13-class semantic decoder |
| `memory_bank.*` | 626 | the memory bank (Innovation 2 lives here) |

### Which tensors actually changed by training

Comparing the three checkpoints tensor-by-tensor, **only 13 tensors differ**,
and all of them are in the decoder:

```
decoder.linear_c1.proj.{weight,bias}
decoder.linear_c2.proj.{weight,bias}
decoder.linear_c3.proj.{weight,bias}
decoder.linear_fuse.conv.weight
decoder.linear_fuse.bn.{weight,bias,running_mean,running_var}
decoder.linear_pred.{weight,bias}
```

The image encoder and the memory bank are **bit-identical** across all three
checkpoints (0 of 626 `memory_bank` tensors differ).

This is by design and matches the protocol in
`results/I1_densepass/protocol.json`:

> *training_scope*: Existing stage2 semantic decoder continuation

So the training continued **only the semantic decoder** on top of the frozen
baseline checkpoint, with the frozen-pseudo-label objective described in the
protocol. This is consistent with the recorded status
`training_gain_established: false` — the released Innovation-1 numbers are an
**inference-time** result, not a claim that training improved the model.

The differences between the two seeds (R0 vs R1) are tiny
(`maxabs ≈ 4e-4` on weights, `≈ 4e-2` on BatchNorm running stats), which is
what you expect from a 160-iteration, lr `2e-06` decoder-only continuation.

### The baseline checkpoint

All three runs were initialised from the OmniSAM-S baseline checkpoint
(`omnisam_s.pth`, referenced as `initial_checkpoint` in `protocol.json`).
That file is **not** in this repository — it belongs to the upstream
Effective-Omnisam / OmniSAM project and its expected checksum is recorded in
`results/I1_densepass/weights_checksums.txt`.

The 626 `memory_bank.*` tensors in our checkpoints carry that baseline's
memory-bank weights. They are identical in all three files because the memory
bank was frozen during training, and because Innovation 2 is implemented as a
**read policy** (how tokens are selected from the bank), not as a change to the
bank's parameters — see `configs/piers/innovation2_final.json` and
`src/piers_ljj/memory/bridge.py`.

---

## Not published

The following are deliberately **not** in the repository:

| Item | Why | Where it comes from |
|---|---|---|
| `omnisam_s.pth` (baseline init) | upstream artefact, not ours | upstream OmniSAM project |
| `sam2.1_hiera_small.pt` | third-party backbone | see `results/I1_densepass/weights_checksums.txt` |
| `grounding-dino-tiny-hf/` | third-party detector | convert with `scripts/piers/convert_grounding_ckpt.py` |
| per-image prediction caches | large, regenerable | rerun the pipeline |

Checksums for all of these are recorded in
`results/I1_densepass/weights_checksums.txt`.
