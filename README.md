# PIERS panoramic segmentation

This release records the completed outdoor, frozen-weight comparison between original OmniSAM and the complete I1/I2/I3 + open-vocabulary pipeline. **Start with [REPRODUCE.md](REPRODUCE.md).** It contains fresh-server setup, verified inference-weight downloads, exact dataset manifests, parallel inference and scoring commands.

| Evaluation | Images | Original OmniSAM mIoU | Complete v3 mIoU | Difference |
|---|---:|---:|---:|---:|
| DensePASS standard mapped13 | 100 | 60.2961 | 60.8555 | +0.5594pp |
| WildPASS native8 | 500 | 37.5572 | 57.5670 | +20.0098pp |
| SynPASS native22 | 1690 | 25.0723 | 25.6163 | +0.5440pp |
| SynPASS mapped13 | 1690 | 43.5811 | 43.8665 | +0.2854pp |

Full per-class IoU, mAcc, pixel accuracy, FWIoU, mDice, confidence intervals and recorded pixel counts are in [results/paper20261109](results/paper20261109). Run `python scripts/paper/reproduce.py verify-ledgers` to reconstruct the four main accuracy rows from integer counts on CPU.

I1 is source-calibrated text/semantic cooperation. I2 selects uncertainty-ranked, class-protected memory tokens. I3 adds two adaptive seam windows. The complete pipeline also has explicit open-vocabulary category competition, including requested categories absent from the original13-channel head. Both use the same original checkpoint; extra pretrained DINO/SAM2 resources and compute are disclosed.

**Efficiency remains an open objective.** The current complete pipeline is slower and uses more device memory than the original baseline. Reducing the maximum Memory Attention input from32768 to1024 tokens does not compress resident dense history or demonstrate whole-system speedup. The optional prefusion-cache candidate is disabled by default and has only a three-image text-branch validation so far.

DensePASS100 is development validation. Target-set failures informed the v2/v3 repairs, so WildPASS/SynPASS results are diagnostic development evidence rather than a fresh untouched holdout. Source calibrators were rebuilt from Cityscapes; historical calibrator bytes were unavailable. Indoor adaptation is outside this release. Older directories under `results/` and the v0.2 release remain historical records, not interchangeable current-main results.

See [the release status](docs/piers/PAPER_RELEASE_STATUS_20261109.md) for completed checks and remaining work. The upstream/vendor licenses remain in their source directories.

## Local panoramic segmentation system

[The web workbench](apps/panorama_web/README.md) runs the interface on a local computer and performs inference on a GPU server over SSH. It supports scene defaults without text, requested text targets, global instance IDs, visual comparison, result export, task history and the archived experiment dashboard. Outdoor inference uses the complete frozen v3 model; indoor inference is a generic zero-shot text route, with adapted indoor weights still pending. Runtime data and credentials are excluded from the repository.

See [the web acceptance record](docs/piers/WEB_WORKBENCH_STATUS_20261111.md) for verification and [the updated efficiency handoff](docs/piers/CURSOR_EFFICIENCY_HANDOFF_20261111.md) for the separate optimization work.
