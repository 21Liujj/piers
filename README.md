# PIERS panoramic segmentation

This repository records the outdoor frozen-weight comparison and the completed indoor supervised adaptation comparison between original OmniSAM and the complete I1/I2/I3 pipeline. Start with [outdoor reproduction](REPRODUCE.md) or [indoor reproduction and web setup](REPRODUCE_INDOOR.md). Code, result ledgers, calibration models and figures are versioned; large weights and complete evidence archives are attached to their matching GitHub Releases.

| Evaluation | Images | Original OmniSAM mIoU | Complete v3 mIoU | Difference |
|---|---:|---:|---:|---:|
| DensePASS standard mapped13 | 100 | 60.2961 | 60.8555 | +0.5594pp |
| WildPASS native8 | 500 | 37.5572 | 57.5670 | +20.0098pp |
| SynPASS native22 | 1690 | 25.0723 | 25.6163 | +0.5440pp |
| SynPASS mapped13 | 1690 | 43.5811 | 43.8665 | +0.2854pp |
| Stanford 2D-3D-S SPAN8 indoor adaptation | 373 | 37.6424 | 46.5591 | +8.9167pp |

Outdoor rows use frozen source weights. The indoor pair shares one Stage1 checkpoint, rank4 LoRA and 8-class decoder trained on fit832; full-model I1 calibrators use training-area calib104. It uses Grounding DINO + SAM2, not CLIP. Indoor is supervised cross-scene adaptation, not zero-shot transfer. The indoor I2+I3 core obtains42.0019% mIoU; full-minus-original image-bootstrap95% CI is[+8.3126,+9.4728]pp. See [the indoor records](results/indoor_matched20261112), [protocol](configs/piers/INDOOR_Matched_Fold1.yaml), and [v0.4 assets](https://github.com/21Liujj/piers/releases/tag/v0.4-indoor-20261112).

Full per-class IoU, mAcc, pixel accuracy, FWIoU, mDice, confidence intervals and recorded pixel counts are in [results/paper20261109](results/paper20261109). Run `python scripts/paper/reproduce.py verify-ledgers` to reconstruct the four main accuracy rows from integer counts on CPU.

I1 is source-calibrated text/semantic cooperation. I2 selects uncertainty-ranked, class-protected memory tokens. I3 adds two adaptive seam windows. The complete pipeline also has explicit open-vocabulary category competition, including requested categories absent from the original13-channel head. Both use the same original checkpoint; extra pretrained DINO/SAM2 resources and compute are disclosed.

**Efficiency remains an open objective.** The current complete pipeline is slower and uses more device memory than the original baseline. Reducing the maximum Memory Attention input from32768 to1024 tokens does not compress resident dense history or demonstrate whole-system speedup. The optional prefusion-cache candidate is disabled by default and has only a three-image text-branch validation so far.

DensePASS100 is development validation. Target-set failures informed the v2/v3 repairs, so WildPASS/SynPASS results are diagnostic development evidence rather than a fresh untouched holdout. Source calibrators were rebuilt from Cityscapes; historical calibrator bytes were unavailable. Indoor has one heldout area/fold and one training seed; training-room disjointness is unverified, and heldout was used for earlier smoke diagnostics. Its fixed1200-step LoRA is not claimed optimal. Older six-arm indoor and CLIP-alignment runs are historical and must not replace the matched373-image table. Older directories under `results/` and the v0.2 release retain their historical protocols.

See [the release status](docs/piers/PAPER_RELEASE_STATUS_20261109.md) for completed checks and remaining work. The upstream/vendor licenses remain in their source directories.

## Local panoramic segmentation system

[The web workbench](apps/panorama_web/README.md) runs the interface on a local computer and performs inference on a GPU server over SSH. It supports scene defaults without text, requested text targets, global instance IDs, visual comparison, result export, task history and the archived experiment dashboard. Outdoor inference uses the complete frozen v3 model. Indoor SPAN8 defaults/subsets use the completed same-checkpoint I1/I2/I3 model; novel indoor categories use the separately labelled generic text route. Three fixed indoor web predictions match the formal experiment pixel-for-pixel. Runtime data and credentials are excluded from the repository.

See [today's experiment and web acceptance record](docs/piers/INDOOR_WEB_RELEASE_20261112.md), [the earlier web record](docs/piers/WEB_WORKBENCH_STATUS_20261111.md), and [the efficiency handoff](docs/piers/CURSOR_EFFICIENCY_HANDOFF_20261111.md) for the separate optimization work.
