# data_protocol

- UDA: Cityscapes→DensePASS, class space `densepass13@v1`.
  19→13 map: `[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]`
  (person+rider→11; car/truck/bus/motorcycle/bicycle→12; train→255).
  Alignment tags: exact | coarse_compatible | unmapped.
- Text queries: `category_union` vs `instance_referring` + negatives;
  manifest fixes image/query/split/GT-type. GT read by evaluator ONLY.
- Splits: train / dev-calib / final-eval; never tune on final-eval labels.
- Cache keys: image+query content, model/processor/cfg, thresholds, FoV geo,
  dtype/backend, semantic ckpt. Any change → no stale reuse.
