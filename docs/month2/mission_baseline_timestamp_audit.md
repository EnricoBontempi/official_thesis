# Mission M1 — Timestamp audit for `40NH3_1.2.h5`

**Date:** 2026-11-16 (America/New_York)  
**Shard:** `data/raw/combustion/hf_dataset/numerical/data-00022-of-00030.arrow`  
**sim_id:** `40NH3_1.2.h5`

## Identity vs remote inventory

| Field | Local | `data/manifests/remote_file_inventory.json` |
|---|---|---|
| path | `.../data-00022-of-00030.arrow` | `combustion/hf_dataset/numerical/data-00022-of-00030.arrow` |
| size_bytes | 2098221056 | 2098221056 |
| sha256 | `3d908c0dc2913dc658fd227c5750d99a7f4c4293ae29a634cd96d02129e4dbdb` | identical |

SHA/size match the Month-1 remote inventory. Raw bytes were not modified.

## Raw packaging (no ShardStore padding)

- `shape_t = 2001`, `numerical` reshapes cleanly to `(2001, 128, 128, 15)`.
- `t_shape = 2000`, `t` length = 2000, dtype float64, all finite.
- `t` spans **1.5 → 2.0**, matching Month-1 inventory `t_start`/`t_end`.
- `t` is strictly increasing; median `diff(t) ≈ 2.5e-4`.
- Exactly **one** irregular step: `diff[1098] ≈ 5e-4` (double median). Relative to a uniform 2001-grid `1.5 + k·2.5e-4`, this matches deletion of uniform index **1099** (`t=1.77475`) — an **interior** missing sample, **not** a missing final timestamp.
- Field comparison: frames **1999 and 2000 are bitwise identical** for T/vi/vj (and the full channel vector). Frames 0→1 and 1998→1999 are not duplicates.

## Rejection of the previous “pad the last t” story

The Step-2.3 reader extrapolated `t[2000]=2.00025` beyond audit `t_end=2.0`, assuming the short array lacked the **last** sample. Evidence contradicts that:

1. Validated `t` already ends at 2.0 with 2000 samples.
2. The irregular gap is interior (index 1098/1099 of the uniform grid).
3. Numerical index 2000 is a **duplicate field snapshot** of index 1999, not a new physical time.

## Policy adopted for single-frame Month-2 use

- Generic short-`t` padding / extrapolation is **refused** (`TimelineError`).
- Non-finite or decreasing timelines are **refused**.
- For this identity-bound anomaly only (`sim_id`, filename, size, sha256):
  - indices `0…1999`: serve fields with original `t[i]`, `time_validated=True`;
  - index `2000`: serve fields with `time_validated=False`, `t=NaN`, **no invented coordinate**, only if `numerical[2000]==numerical[1999]`.
- Month-2 primary baselines remain **single-frame by index**; they do not use absolute time. Temporal forecasting / multi-frame alignment stays **unauthorized** for this shard until a scientific decision is made.

Machine-readable companion: `results/baseline_consolidation/timestamp_diagnosis_40NH3_1.2.json`.
