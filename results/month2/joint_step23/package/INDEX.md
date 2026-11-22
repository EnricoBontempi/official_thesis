# Joint delivery, steps 2 and 3: K corrections (Cursor) and temperature-assisted training (Sonnet)

Status: complete. Every blocking compatibility check passed (table below). No new training, no TEST, no winner, no ranking.

## Inputs, exact references

- Step 2 as published by Cursor: branch `sweep/c149-temperature-k`, SHA `be18ad007e66fbf2565ad3d5486c6513fc5c3e4c`, index `results/month2/temperature_k_sweep/INDEX.md`, report `docs/month2/11_temperature_k_sweep.md`; the tables used here are byte-identical to the published ones (hashes in `joint_references.json`).
- Step 2 package (Cursor), `/home/bontempi/official_thesis_k_sweep_package_v1`: forward/correction SHA `54a3b4d3a177b731ed7eef45030c3a49b0986c16`, aggregation SHA `30593985337f6367f5e974807cb54865b00d678f`. Tables `by_seed.csv`, `across_seeds.csv`, `per_condition.csv`, `versus_free.csv`, `checks.json`, `INDEX.md`; file hashes in `joint_references.json`. Per-frame CSV: `/home/bontempi/official_thesis_k_sweep_v1/<run>/per_frame/<ckpt>/<role>/rep<r>.csv`.
- Step 3 package v2 (Sonnet), `/home/bontempi/official_thesis_t_assisted_eval_v2/package`: matrix v2 sha256 `3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7`, chain v2 eval array `30083486`, aggregation `30083487`, chain SHA `a86e39c61731586a3343d9d1489b4d3b22fa0547`. Per-frame CSV: `/home/bontempi/official_thesis_t_assisted_eval_v2/runs/<run>/per_frame/<ckpt>/<role>/rep<r>.csv`. The v1 package published at `aa8f65f` is unchanged and remains the reference for the 1000-epoch state of the three extended runs.
- Anchor manifest `configs/observations/month2_temperature_anchors_nested_v1.json`, file sha256 `a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54`, identical on both sides.
- Joint post-processing code SHA `682fe29496d9852c82af2a5d6fb59f05eb329006` (`scripts/57_t_joint_package.py`).

## What is compared

Same 12 (family, p, seed) cells, K in {0, 1, 4, 9, 16, 32}, checkpoints best and last, roles seen220 and VAL123, replicas 0/1/2 (layout nested_r<replica>), 5% velocity. Macro RMSE = mean over conditions of the per-condition mean, in kelvin, as in S7 and in the K sweep.

- free: S7 model, no temperature (Cursor `free`, constant in K on the common domain).
- free+mean, free+median: the same free prediction shifted by the mean/median of the K anchor residuals (Cursor).
- assisted: the separate model trained with K temperature anchors in input (Sonnet).

Domains: full field; common domain (grid minus the 32 layout sites, the same for every K); off-anchor(K); anchor-site error. Primary table: common domain.

## Caveats that decide how the numbers may be read

- The assisted models are different trainings with a different selection criterion (mean over the six K of VAL 5% common-domain macro RMSE, single best); `best` and `last` are not the same epochs as the free S7 checkpoints (`tables/assisted_checkpoint_epochs.csv`). At K=0 the assisted model is not the free model.
- The three runs TA-glu-on-s1, TA-glu-off-s1, TA-glu-off-s2 were continued from epoch 1000 up to the limit set in the authorisation; the others are unchanged and were re-attested by hash.
- Mean/median corrections are post hoc shifts and do not change the model; they move the full-field RMSE through the anchor residuals only.
- Seeds: three per cell; dispersion in `joint_across_seeds_K.csv`. The GLU free reference has known top-k tie sensitivity to batching (S7 residual note).
- No estimator is selected; paired contrasts are descriptive.

## Tables (`tables/`)

- `joint_by_run_replica_K.csv`: one row per (family, p, seed, checkpoint, role, replica, K), four domains for each of free, mean, median, assisted, and the paired differences.
- `joint_by_seed_K.csv`: the same averaged over the three replicas.
- `joint_across_seeds_K.csv`: mean/min/max/std across the three seeds.
- `joint_paired_contrasts.csv`: assisted-free, mean-free, median-free, assisted-mean, assisted-median, with the number of seeds below zero.
- `reference_baselines_from_cursor_by_seed.csv`: GlobalTrainMean, ParamConditionedMean and GappyPOD rows as published by Cursor (no assisted counterpart).
- `compatibility_checks.csv`, `assisted_checkpoint_epochs.csv`.

## Common-domain macro RMSE (K), VAL123, best checkpoint, mean over 3 seeds and 3 replicas

| family | p_conditioning | estimator | K=0 | K=1 | K=4 | K=9 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|
| adapted_glu | False | free | 44.48 | 44.48 | 44.48 | 44.48 | 44.48 | 44.48 |
| adapted_glu | False | mean | 44.48 | 71.67 | 45.36 | 42.41 | 42.87 | 41.62 |
| adapted_glu | False | median | 44.48 | 71.67 | 41.73 | 41.41 | 41.22 | 41.16 |
| adapted_glu | False | assisted | 44.37 | 43.19 | 41.91 | 41.90 | 41.76 | 41.43 |
| adapted_glu | True | free | 42.72 | 42.72 | 42.72 | 42.72 | 42.72 | 42.72 |
| adapted_glu | True | mean | 42.72 | 66.61 | 45.09 | 42.62 | 42.95 | 41.85 |
| adapted_glu | True | median | 42.72 | 66.61 | 41.64 | 41.38 | 41.27 | 41.24 |
| adapted_glu | True | assisted | 43.91 | 43.93 | 43.94 | 43.93 | 43.92 | 43.78 |
| sparse_unet | False | free | 40.57 | 40.57 | 40.57 | 40.57 | 40.57 | 40.57 |
| sparse_unet | False | mean | 40.57 | 70.86 | 40.94 | 36.89 | 37.39 | 35.95 |
| sparse_unet | False | median | 40.57 | 70.86 | 35.84 | 35.57 | 35.45 | 35.40 |
| sparse_unet | False | assisted | 39.75 | 39.75 | 39.74 | 39.74 | 39.73 | 39.71 |
| sparse_unet | True | free | 39.70 | 39.70 | 39.70 | 39.70 | 39.70 | 39.70 |
| sparse_unet | True | mean | 39.70 | 68.53 | 40.32 | 36.32 | 36.96 | 35.45 |
| sparse_unet | True | median | 39.70 | 68.53 | 34.85 | 34.69 | 34.58 | 34.54 |
| sparse_unet | True | assisted | 37.66 | 37.66 | 37.65 | 37.63 | 37.62 | 37.55 |

## Same, last checkpoint

| family | p_conditioning | estimator | K=0 | K=1 | K=4 | K=9 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|---|
| adapted_glu | False | free | 45.00 | 45.00 | 45.00 | 45.00 | 45.00 | 45.00 |
| adapted_glu | False | mean | 45.00 | 71.35 | 45.57 | 42.66 | 43.24 | 41.85 |
| adapted_glu | False | median | 45.00 | 71.35 | 41.92 | 41.59 | 41.42 | 41.35 |
| adapted_glu | False | assisted | 44.26 | 43.26 | 42.14 | 42.19 | 42.06 | 41.68 |
| adapted_glu | True | free | 44.19 | 44.19 | 44.19 | 44.19 | 44.19 | 44.19 |
| adapted_glu | True | mean | 44.19 | 65.10 | 45.50 | 43.24 | 43.63 | 42.59 |
| adapted_glu | True | median | 44.19 | 65.10 | 42.38 | 42.18 | 42.08 | 42.07 |
| adapted_glu | True | assisted | 44.93 | 44.93 | 44.95 | 44.95 | 44.92 | 44.79 |
| sparse_unet | False | free | 42.95 | 42.95 | 42.95 | 42.95 | 42.95 | 42.95 |
| sparse_unet | False | mean | 42.95 | 74.84 | 42.28 | 37.86 | 38.39 | 36.71 |
| sparse_unet | False | median | 42.95 | 74.84 | 36.59 | 36.33 | 36.21 | 36.17 |
| sparse_unet | False | assisted | 43.68 | 43.68 | 43.68 | 43.67 | 43.66 | 43.62 |
| sparse_unet | True | free | 40.72 | 40.72 | 40.72 | 40.72 | 40.72 | 40.72 |
| sparse_unet | True | mean | 40.72 | 69.92 | 41.25 | 37.09 | 37.37 | 35.98 |
| sparse_unet | True | median | 40.72 | 69.92 | 35.51 | 35.34 | 35.26 | 35.22 |
| sparse_unet | True | assisted | 39.79 | 39.79 | 39.79 | 39.78 | 39.77 | 39.76 |

## Compatibility checks

| check | pass | detail |
|---|---|---|
| manifest file in repo is the frozen one | PASS | a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54 |
| Cursor: manifest sha256 on all 12 runs | PASS | 12 runs, mismatches [] |
| assisted: matrix v2 refers to the same manifest sha256 | PASS | matrix_v2 sha256 3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7 |
| assisted: v2 package built on the v2 matrix, 12 runs, not smoke, no ranking | PASS | package matrix 3663d33820aac446b9621bf35ae49dec26ee4cc6a8379ad5363b0a81e8d922f7, n_runs 12, gate max 0.0 |
| assisted: 12 COMPLETE.json with gate ok | PASS | gate macro tolerance 0.05 K |
| Cursor: 12 COMPLETE.json with gate ok | PASS | free-gate vs S7 CSV tolerance 0.05 K |
| Cursor: K=0 identities and K=1 mean=median at 0 | PASS | {"k0_ao_full": 0.0, "k0_mean_full": 0.0, "k0_median_full": 0.0, "k0_ao_centered": 0.0, "k0_mean_centered": 0.0, "k0_median_centered": 0.0, "k1_mean_median": 0.0} |
| Cursor: SHAs readable from its INDEX | PASS | forward 54a3b4d3a177b731ed7eef45030c3a49b0986c16, aggregation 30593985337f6367f5e974807cb54865b00d678f |
| Cursor package file equals the version published at be18ad0: INDEX.md | FAIL | cluster 6b6582a14193 vs published 1747679256a5 (text/figure regenerated after the tables; tables identical) |
| Cursor package file equals the version published at be18ad0: across_seeds.csv | PASS | identical |
| Cursor package file equals the version published at be18ad0: by_seed.csv | PASS | identical |
| Cursor package file equals the version published at be18ad0: checks.json | PASS | identical |
| Cursor package file equals the version published at be18ad0: figures/val_common_rmse_vs_K.png | FAIL | cluster f4bde939dd66 vs published 76f1257355e8 (text/figure regenerated after the tables; tables identical) |
| Cursor package file equals the version published at be18ad0: per_condition.csv | PASS | identical |
| Cursor package file equals the version published at be18ad0: versus_free.csv | PASS | identical |
| K sets equal | PASS | Cursor [0, 1, 4, 9, 16, 32], assisted [0, 1, 4, 9, 16, 32] |
| replicas equal | PASS | 0/1/2 on both sides |
| Cursor layout is nested_r<replica> (manifest layout paired to the velocity replica) | PASS | {"0": ["nested_r0"], "1": ["nested_r1"], "2": ["nested_r2"]} |
| frame counts per role equal (seen220, VAL123) | PASS | Cursor {'seen_train_eval': 220, 'val_eval': 123}, assisted {'seen_train_eval': 220, 'val_eval': 123} |
| frame keys (sim_id, frame_index) identical per role | PASS | seen_train_eval: identical; val_eval: identical |
| velocity mask hash identical per (role, replica, frame) | PASS | 1029 matched, 0 different, 0 unmatched |
| common domain = 16384 - 32 layout sites on both sides | PASS | Cursor [16352], assisted [16352] |
| Cursor off-anchor count is 16384-K and anchor count K | PASS | {"0": [16384], "1": [16383], "4": [16380], "9": [16375], "16": [16368], "32": [16352]} |
| units: both sides RMSE in kelvin on the same order (median frame RMSE 5-300 K) | PASS | Cursor median 39.8 K, assisted 39.1 K |
| macro definition reproduces Cursor's by_seed.csv (full and common) | PASS | 4320 of 4320 cells, max|diff| full 0.00e+00 K, common 0.00e+00 K |
| macro definition reproduces the assisted v2 global_by_replica.csv (full and common) | PASS | 864 of 864 cells, max|diff| full 2.13e-14 K, common 2.13e-14 K |

