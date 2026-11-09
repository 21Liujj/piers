# Outdoor release status — 2026-11-09

This release publishes the existing frozen-v3 code and results. Efficiency optimization is a separate, unfinished task. Indoor work belongs to the user's Cursor workflow and is not included here.

| Item | Recorded scope | Status |
|---|---|---|
| Original/full outdoor accuracy | Dense100, Wild500 native8, Syn1690 native22 and separate mapped13 | Complete; raw integer ledgers included |
| I2 matched accuracy | Dense100 original, I2-only, I2+I3, full | Complete |
| Full-system I2 switch | Dense100 I1/I3/OV with/without I2 | Complete; -0.0663pp with I2, paired CI crosses zero |
| I2 component timing | Fixed-affinity ABBA,3 images,6 measurements/variant | Complete; mean4.512->4.290s, descriptive |
| Full-system I2 timing | Same real-text function, ABBA3 images | Complete; mean13.457->13.382s, median essentially equal; device memory rises |
| All factorial and budget/seam controls | Registered in matrix.json | Commands supplied; not all newly measured |
| Historical18-row memory selection | Older records cite selection on Dense100 | Raw full matrix unavailable; not independently reverified |
| Historical I1 calibrator | Original trained bytes | Unavailable; current release uses source-only rebuilt calibrators |
| Text prefusion cache | Three fixed images,2 measured repeats per image | Exact sampled boxes/scores/masks;31–36% text-branch reduction; optional/off by default |
| Whole-system faster than original OmniSAM | Same hardware/input/counting scope | Not achieved by this release |
| Jetson final campaign / indoor adaptation | Separate work | Not claimed complete here |

Current DensePASS gain is+0.5594pp and its paired95% interval crosses zero. SynPASS native22 gain is+0.5440pp, mapped13 gain+0.2854pp. WildPASS native8 gain is+20.0098pp; four novel categories have no original source-head channel. Do not attribute all gains to I2. Dense history stays resident, so1/32 attention tokens is not1/32 RAM, VRAM or latency.

The optional DynamicFSMC module is an explicit fail-fast import guard inherited from the recovered runtime. This source checkpoint has no active FSMC/SCM branch; attempting to construct that unavailable optional component aborts rather than fabricating behavior. The original/full released inference configurations keep it disabled.

Fresh-checkout validation and exact tested scopes are recorded in `results/paper20261109/release_validation.json`. Reproduction guide: [REPRODUCE.md](../../REPRODUCE.md).
