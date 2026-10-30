# Module Map

## Geo Memory (`plugins.geo_memory`)

- Status: default `enabled: false`; Task 0-3/2-2 implementation is diagnosis-only.
- Purpose: log geometry metadata and capped token/summary diagnostics for Geo-TeCMA without changing memory attention, sparse retrieval, losses, or selection.
- Required metadata fields distinguish `tensor_order`, `frame_idx`, `true_erp_x_range`, and `true_erp_y_range`.
- Horizontal/seam position must not be inferred from tensor order when random reverse may be active; records set `true_erp_x_range_reliable=false` in that case.
- Latitude/polar diagnostics use `mem_hw` token-row mapping: `token_row = token_idx // mem_w`.
- Default config: `mode=diagnosis`, `seam_selection_enabled=false`, `allow_tensor_order_inference=false`, `latitude_source=mem_hw_token_row`, `token_writer.enabled=false`, `token_writer.max_records=5`, `token_writer.first_n_batches=1`.
- Code paths: `adaptation/strategies.py` builds frame/token summary records, `model/memory_bank.py` stores latest memory/grid metadata, `adaptation/trainer.py` caps records by batch/record limits, `adaptation/engine.py` writes JSONL.
