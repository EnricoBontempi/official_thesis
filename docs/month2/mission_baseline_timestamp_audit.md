# Mission M1 — Timestamp audit for `40NH3_1.2.h5`

**Date:** 2026-11-16 (America/New_York); **metadata correction:** 2026-11-17  
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

1. The `t` array already ends at 2.0 with 2000 samples.
2. The irregular gap is interior (index 1098/1099 of the uniform grid).
3. Numerical index 2000 is a **duplicate field snapshot** of index 1999, not a new physical time.

## Post-replica metadata correction (2026-11-17)

Opus review of consolidation commit `129f298` found residual over-claim: `_resolve_time` had set `time_validated=True` for indices `0…1999` merely because `t[i]` exists. **Presence of a coordinate ≠ verified field↔time association**, given the interior gap and duplicate packing.

### Definitive flags (`combustion_recon.data.temporal_quality`)

| Flag | Meaning on this shard |
|---|---|
| `has_original_time_coordinate` | `True` for `i < 2000` (archival `t[i]` retained); `False` for `i=2000` (no invented coordinate) |
| `timeline_regular` | `False` (documented interior gap) |
| `field_time_association_verified` | `False` for **all** indices |
| `time_validated` | Compatibility alias; also `False` for **all** indices |
| `duplicate_of` | `1999` when `i=2000`; else `None` |
| `time_coordinate_original` / `t` | Archival original when present; **not** presented as validated physical time of the field |

`require_verified_field_time_association(...)` refuses multi-frame / forecasting use when association is unverified. **Single-frame-by-index remains allowed**; samples are not silently dropped or deduplicated.

### What was not changed

- No raw rewrite; no protocol frame list change; no baseline rerun.
- Historical CSV / `consolidation_v1` artifacts unchanged.
- Freeze commit of the replica remains `2c800ce`; reader SHA changes only because metadata logic changed after that freeze.

Machine-readable companion: `results/baseline_consolidation/timestamp_diagnosis_40NH3_1.2.json`.
