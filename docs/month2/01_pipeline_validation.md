# Month 2 — Pipeline validation (Step 2.2)

**Date:** 2026-11-16 12:33:27 -0500
**Host:** local workspace (WSL). Engaging GPU/allocation **not verified**.
**Commit base expected:** `5bce7142d7dbaf9e5ea6bc1e3c8dffe75e066ce9`

## Environment
- CPU count: 16
- RAM available (from /proc or free): None GiB
- Disk free on workspace: None GiB
- Raw shards: `/home/enric/projects/official_thesis/data/raw/combustion/hf_dataset/numerical` exists=True, n=30
- Peak RSS observed during this script (MB): 6482.6

## Artifacts written
- `data/manifests/train_norm_stats_month2_fit.json`
- `data/manifests/month2_frames.json`
- `data/manifests/month2_observations.json`
- `docs/month2/01_pipeline_validation.md`

## Frame counts
```json
{
  "train_fit": {
    "n_samples": 760,
    "n_conditions": 20,
    "n_frames_per_condition": 38,
    "split": "train",
    "payload_access": "allowed"
  },
  "seen_train_eval": {
    "n_samples": 220,
    "n_conditions": 20,
    "n_frames_per_condition": 11,
    "split": "train",
    "payload_access": "allowed"
  },
  "val_eval": {
    "n_samples": 123,
    "n_conditions": 3,
    "n_frames_per_condition": 41,
    "split": "val",
    "payload_access": "allowed"
  },
  "test_interp_eval": {
    "n_samples": 123,
    "n_conditions": 3,
    "n_frames_per_condition": 41,
    "split": "test_interp",
    "payload_access": "requires_selection_freeze"
  },
  "test_extrap_eval": {
    "n_samples": 164,
    "n_conditions": 4,
    "n_frames_per_condition": 41,
    "split": "test_extrap",
    "payload_access": "requires_selection_freeze"
  }
}
```

## Fit-only stats
- Path: `data/manifests/train_norm_stats_month2_fit.json`
- Conditions: 20
- Frames/condition: 38
- Temperature mean/std (K): 2039.508661 / 331.737843
- Month-1 stats file preserved: True
- Independent subset mean/std abs diff: 4.547e-13 / 2.842e-14

## Real-data batch check
```json
{
  "sample0_sim_id": "0NH3_0.85.h5",
  "sample0_frame": 0,
  "n_train_fit": 760,
  "coord_match_raw": true,
  "temperature_match_raw": true,
  "velocity_sites_match_raw": true,
  "denorm_roundtrip_ok": true,
  "batch_keys": [
    "K",
    "anchor_coords",
    "anchor_site_index",
    "anchor_valid",
    "anchor_values",
    "dense_channel_names",
    "dense_values",
    "mask_temperature",
    "mask_velocity",
    "query_coords",
    "scalar_coords",
    "scalar_valid",
    "scalar_values",
    "scalar_var_ids",
    "site_fraction",
    "task_mode",
    "velocity_coords",
    "velocity_site_index",
    "velocity_valid",
    "velocity_values"
  ],
  "batch_velocity_valid_shape": [
    2,
    819
  ],
  "dataloader_workers": 2,
  "dataloader_batch_wall_s": 16.587583541870117,
  "rss_mb_before_loader": 6482.5546875,
  "rss_mb_after_loader": 6482.5546875,
  "store_stats": {
    "shard_decodes": 1,
    "cache_hits": 1,
    "cache_misses": 1,
    "frames_materialized": 1
  },
  "compact_cache_frames": 1,
  "compact_cache_bytes_est": 262144,
  "test_access_denied_without_freeze": true,
  "evaluator_rmse_mean_predictor": 242.66588055932934,
  "selection_table_rows": 1,
  "p_off_has_op_params": false,
  "vk_in_model_inputs": false
}
```

## Memory / I/O notes
- Arrow full-trajectory decode peaks near ~2.1 GiB per shard (float32 numerical blob).
- Month-2 store keeps only compact T+vi+vj(+vk) frames in an LRU cache; full arrays are discarded after extract.
- Compact cache estimate after validation: 262144 bytes (1 frames).
- No persistent heavy conversion of the 59 GiB dataset was created.

## Limits / not claimed
- Engaging access/GPU allocation not verified in this step.
- No baseline/POD/ParamMean fitting, no neural training.
- U-Net/GLU forward isolation still to be repeated when models exist (probe only).
- Pseudo-extrap folds not fitted; fold lock still open.

## Overall status (primary path, 2026-11-16 initial Step 2.2): **PASS**

---

## Corrective API guards (post-`1fc150f`, before Step 2.3)

These harden **misuse of the API**; they are not a new reconstruction result. The primary real-data path above remains the evidence for batch correctness.

### 1. Role membership (`build_index`)
- Custom `sim_ids` must be a unique subset of the role’s operating-condition split (TRAIN subsets allowed for folds).
- TEST IDs on `train_fit` / `seen_train_eval` now raise `RoleMembershipError` **before** any shard I/O.
- Covered by `tests/test_month2_api_guards.py`.

### 2. Fit-only stats provenance
- Production `Month2FitOnlyNormalizer` verifies: full TRAIN set (unless fold `expected_sim_ids`), exact `train_fit` frame indices, `splits_sha256`, finite/positive field std, and **`protocol_scientific_sha256`** (binds `protocol_id` + split ref + frames; ignores descriptive YAML churn).
- Full-file `protocol_sha256` may differ after status-only YAML edits; that mismatch alone does **not** invalidate numbers. Existing field statistics were **not** recomputed; meta was patched with the scientific fingerprint.
- `compute_month2_fit_stats` validates TRAIN membership and canonical frames before reads; non-canonical frames require explicit `provenance_mode='debug_subset'` (rejected by production normalizer unless `allow_noncanonical_frame_indices=True`).

### 3. Aggregation homogeneity
- `aggregate_frame_then_condition_then_split` refuses mixed `role` / method / p / K / coverage / seeds when those keys are present (`MixedConfigurationError`).
- Call **once per configuration**. `train_fit` and `seen_train_eval` are not fused via shared `split=train`.
- Mean order: frame → `(split, sim_id)` → split; selection helper is eligible only for `val_eval`.
- Month-1 `evaluate_per_condition` unchanged for direct callers.

### 4. Prefetch I/O (small control, not a full scan)
| Path | `shard_decodes` for 5 frames of one TRAIN traj |
|---|---|
| `prefetch` then `get_frame` | **1** |
| Naive `get_frame` loop with `max_cached_frames=1` (3 frames) | **3** |

Runner implication for Step 2.3: warm with `prefetch(sim_id, needed_frames)` and set `max_cached_frames` ≥ frames/trajectory needed (e.g. ≥38 for `train_fit`) to avoid re-decoding.

### Suite after guards
- **55 passed** (`pytest tests/`)
- `--validate-only` **PASS** (no heavy stats recompute)

