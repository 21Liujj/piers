# architecture

Three envs, one repo, separate processes, file-based protocol:

```
[piers-omni] OmniSAM train/infer ─┐
                                   ├─ masks/semantics/pseudo files ─ evaluation/summarize
[piers-grounded] DINO+SAM2 FoV ───┘        (run manifests, metrics, timing)
[piers-core] geometry/protocol/fusion/metrics/synth tests (CPU)
```

- Text path: FoV crop (wrap) → DINO boxes (orig-size xyxy, no 2nd scaling) → SAM2
  Image Predictor (one image-encode per FoV) → ERP remap → assoc → optional
  semantic rerank → output.
- UDA path: effective_start weights → FoV sequence w/ MemoryBank → SCM →
  semantics + pseudo → optional text_to_pseudo revision (versioned, then train reads it).
- Workers print `sys.executable` + `sam2.__file__`; wrong-path import = hard fail.
- Single-GPU: sequential, explicit load/unload boundary.
