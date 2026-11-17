# Dense velocity → temperature diagnostic (Step 2.4)

**Project:** `EnricoBontempi/official_thesis` only (MIT/Deng Lab).  
**Roadmap:** [roadmap_unet.md](../../roadmap_unet.md)  
**Phases:** U0–U3 **executed** (STOP B). U4 **runner prepared, not executed**.  
**Not claimed:** Step 2.4 complete; scientific U4 training; budget definitively approved; GLU.

## Status legend

| Label | Meaning |
|---|---|
| Implemented / Executed / Verified / Pending | as in STOP A |

## Git / dependencies

| Item | Value |
|---|---|
| Branch | `mission/dense-unet` |
| STOP A | `3643ce5` |
| Corrective + U3 code (pre-experiment) | `11aae27` |
| STOP B results commit | `64186a77c8603e617aed50c793a72e7f39059adb` |
| Dependencies | consolidation `3cf7bf0` + roadmap `adabacc` |
| Merge to main | **not** done |

---

## A — Corrective completion U1–U2 (verified)

### A1 Isolation
Replaced the vacuous same-`x` mutation test with `test_mutation_isolation_sample_path_to_forward`: synthetic cache-like fields → `dense_fields_to_input_tensor` → eval forward. Mutating T / T_norm / vk / metadata leaves input+pred unchanged; p_off ignores source p; allowed vi/p changes move predictions. Distinguishes coverage from upstream `InputIsolationProbe` / sparse pack tests.

### A2 Cache validation
`validate_dense_cache_manifest` now checks: allowed roles, required role presence, protocol `sim_id`/`frame_index`/`role` order, shapes/dtype/lengths, stats/protocol/splits/geometry fingerprints. **Data fingerprints** separated from **model fingerprints** (width/budget/config sha do not invalidate data reuse). Shard SHA policy explicit: inventory LFS + local size — **not** local content re-hash. Metadata refreshed in place via `refresh_dense_cache_manifest_metadata` (payloads untouched). Rejection tests for altered frames/stats.

### A3 YAML contract
`assert_dense_scientific_contract` consumes/refuses: 2C, single-frame, temperature_free, K=0, coverage 1.0, channel order, stats/protocol paths. Profiling records `effective_config.json` including overrides.

### A4 Numeric safety
Trainer refuses non-finite parameter gradients / grad-norm **before** `optimizer.step` (finite loss alone insufficient). Loss/lr/clip defaults unchanged despite high clip rates in profiling.

### Tests executed
- `tests/test_unet_dense.py` — 16 passed  
- Regression: `test_model_input_isolation`, `test_month2_api_guards`, `test_temporal_quality`, `test_month2_pipeline`, `test_baselines` — passed  
- **Not repeated:** 500-update overfit (no numerical path change requiring it)  
- **Missing / not claimed:** local re-hash of all raw arrows; MAC/FLOP tooling

---

## U3 — Profiling (executed)

| Item | Value |
|---|---|
| Code SHA at run | `11aae27` |
| Weights | **new** (seed 0); overfit **not** reused |
| Device | CPU; torch threads 8; CUDA false |
| b / batch / epochs | 32 / 8 / 2 |
| Updates | 190 (95/epoch × 2) |
| Train wall | 93.4 s |
| VAL eval wall (timing only) | 4.9 s |
| Peak RSS | ~1.74 GiB |
| TRAIN loss | 0.877 → 0.101 (finite); clip freq ~0.74 |
| Projected 200 ep + eval/10 | **~2.62 h** CPU |
| Logged VAL RMSE (non-decisive) | 80.9 K |

**Budget proposal for U4 (not launched):** b=32, batch=8, device=CPU (until GPU verified), E_max=200, lr=1e-3, eval every 10, wall cap ≥24 h (measured need ≪24 h). VAL metrics **not** used to choose width/lr. High clip frequency noted for monitoring in U4 — no lr change from this alone.

Artifacts: `results/month2_dense_unet/u3_profile/`

---

## U3 — Dense GappyPOD (executed)

| Item | Value |
|---|---|
| experiment_id | `month2_dense_linear_v1` |
| Config | `configs/experiments/month2_dense_linear.yaml` |
| R / K / coverage | 32 / 0 / 1.0 (single all-True mask) |
| Frames | 760 / 220 / 123; fit-only stats |
| ParamMean | full TRAIN 20 conditions, basis `piecewise_phi1`, α=0.01 (**not** 8-frame overfit) |
| Wall / RSS | ~51 s / ~2.61 GiB |
| Sparse contract / consolidation_v1 | **untouched** |

### λ search (VAL mean-over-conditions RMSE_K)

| p | λ | VAL |
|---|---:|---:|
| off | 1e-4 | 104.20 |
| off | **1e-3** ★ | **93.66** |
| off | 1e-2 | 96.67 |
| off | 1e-1 | 123.05 |
| off | 1 | 136.93 |
| on | 1e-4 | 82.44 |
| on | 1e-3 | 71.66 |
| on | **1e-2** ★ | **70.46** |
| on | 1e-1 | 80.85 |
| on | 1 | 86.88 |

### Selected + matched references (VAL)

| Method | VAL mean-cond RMSE_K |
|---|---:|
| GlobalMean | 139.70 |
| ParamMean (full TRAIN) | 88.01 |
| GappyPOD dense p_off λ=1e-3 | 93.66 |
| GappyPOD dense p_on λ=1e-2 | **70.46** |

Per-condition (selected GappyPOD dense p_on): `0NH3_1.05` 94.0; `20NH3_1.1` 69.4; `40NH3_0.9` 47.9 K.

Artifacts: `results/month2_dense_linear/dense_linear_u3_v1/`

---

## U4 — Runner prepared (not executed)

| Distinction | Status |
|---|---|
| Epoch runner + checkpoint/resume path | **Implemented / tested** |
| Candidate config | `configs/experiments/month2_dense_unet_u4_candidate.yaml` |
| Scientific U4 training | **Not started** (`u4_executed: false`) |
| Budget definitively approved | **No** (candidate only) |

**Candidate settings (not a final approval):** b=32, batch 8, CPU fp32, p_on, seed 0, 200 epochs, lr=1e-3, AdamW+warmup/cosine as roadmap, wall cap **6 h** (configurable/recorded), **no** automatic lr=3e-4.

**Behaviour implemented:** fresh init (no overfit/profile weights); each epoch visits all 760 train_fit with reproducible shuffle; scheduler over total updates; seen+VAL eval every 10 epochs and last; VAL mean-condition RMSE selection with first-minimum ties; best+last checkpoints; explicit stop for non-finite / wall / error (status never falsely `completed`); train mode restored after eval; full run identity; refuse overwrite of existing run dirs; CLI requires `--authorize-scientific-u4`.

**Resume:** epoch-boundary via `checkpoints/last.pt` (model/optim/sched/epoch/RNG). Mid-epoch shuffle continuity **not** claimed.

**Tests:** `tests/test_unet_epoch_trainer.py` (+ existing dense suite) — visits, eval cadence, selection/tie-break, Kelvin aggregation path, train/eval restore, checkpoint reload, wall/non-finite stop, CLI refuse without auth.

### Exact command for a later authorized run

```bash
python scripts/07_train_reconstructor.py train-u4 \
  --authorize-scientific-u4 \
  --config configs/experiments/month2_dense_unet_u4_candidate.yaml \
  --dense-config configs/experiments/month2_dense_unet.yaml \
  --run-id u4_pon_seed0_v1 \
  --cache-root data/processed/month2_dense_unet_cache \
  --wall-hours 6
```

Without `--authorize-scientific-u4` the process exits and does nothing.

## Checklist

- [x] U0–U2 (STOP A) + corrective A1–A4  
- [x] U3 profiling + dense GappyPOD — **STOP B**  
- [x] U4 runner prepared (code/tests/config) — **training not executed**  
- [ ] U4 scientific p_on run — **pending authorization**  
- [ ] U5–U7 pending  

## Decisions still needed before executing U4

1. Confirm or revise budget (b=32, batch=8, CPU, E_max=200, wall 6 h candidate).  
2. Whether high TRAIN grad-clip frequency warrants a separate lr=3e-4 attempt — not automatic.  
3. Explicit go-ahead to run the command above.

## Explicit non-claims

- **U4 scientific training was not started.**  
- Step 2.4 not complete.  
- No neural p_off, GLU, sparse U-Net, T sensors, temporal windows, TEST, cluster, or heavy downloads.
