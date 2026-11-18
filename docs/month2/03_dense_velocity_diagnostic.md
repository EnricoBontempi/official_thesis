# Dense velocity → temperature diagnostic (Step 2.4)

**Project:** `EnricoBontempi/official_thesis` only (MIT/Deng Lab).  
**Roadmap:** [roadmap_unet.md](../../roadmap_unet.md)  
**Phases:** U0–U5 **executed** (U5 = snapshot-swap control on U4 best). U6–U7 not started.  
**Not claimed:** Step 2.4 / mission complete; supervisor ratification; other seeds beyond U6 p_off seed 0; GLU; sparse success.

**Opus external review (U4/U5, reported by Enrico):** usable with reserves; no invalidating defect; CSV metrics recomputed; no checkpoint/cache/raw re-inference. Confirmed aligned replay, swap pairing/transform, VAL non-independence, best 26.931 K @160 / last 28.055 K @730. U5 baseline refit noted as mandate deviation but per-frame ≈ U3 (~5.5e-12). Not supervisor ratification.

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

## U4 — Engaging p_on seed 0 (executed)

| Item | Value |
|---|---|
| Decision | Enrico 2026-11-17 (operational; no supervisor ratification claimed) |
| Commit | `50bda7fe42323d2abd8ef46763be65ccbaf3f008` |
| Job | `29465586` · `mit_amf_advanced_gpu` · `mit_normal_gpu` · QoS AMF · **L40S** · `node3008` |
| Run dir | `results/month2_dense_unet/u4_pon_seed0_engaging_v1/` (+ Engaging pool twin) |
| Status | `stopped_early_plateau` at **730**/1000 epochs |
| Best VAL macro RMSE | **26.931 K** @ epoch **160** (first absolute min) |
| Last VAL | 28.055 K @ 730 |
| Seen / TRAIN @ best | 37.675 K / 13.652 K |
| Queue / compute | ≈132 s / ≈28.5 min (smoke gate ≈0.85 h ≪ 4.5 h) |
| Dense GappyPOD p_on VAL (U3) | ≈70.46 K (matched contract; linear comparator) |
| Checkpoint SHA | see `checkpoint_hashes.json` (`.pt` on Engaging, gitignored) |

Full curves, per-frame best/last, and scheduler counters: `auto_report.md`, `eval_history.json`, `evals/epoch_0160`, `evals/epoch_0730`.

**Env delta vs local WSL:** Engaging `torch 2.8.0+cu128` (local `.venv` was `2.15.0+cu132`); dedicated `~/envs/official_thesis` (HONF env untouched).

## U5 — Snapshot-swap control (executed)

| Item | Value |
|---|---|
| Code SHA | `8549a9b900a038d41da7513831a90908c49b87aa` |
| Job | `29471596` · AMF · `mit_normal_gpu` · **L40S** · `node3102` |
| Checkpoint | U4 `best.pt` @160 · sha256 `6d456abc…a03634` |
| Aligned replay | VAL/seen macros **exact match** to U4 (Δ=0; atol declared 0.05 K) |
| VAL macro | aligned **26.93** → swapped **96.55** K (Δ **+69.6**) |
| Seen macro | aligned **37.68** → swapped **150.30** K (Δ **+112.6**) |
| Matched VAL | GappyPOD 70.46 · ParamMean 88.01 · GlobalMean 139.70 K |
| Run dir | `results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/` |

**Reading (see `interpretation.md`):** swap hurts; mostly structure (Centered-RMSE), with a much larger hit on `0NH3_1.05`. Evidence that instantaneous velocity–temperature alignment matters for this seed — **not** causality, ceiling, or sparse transfer. VAL remains selection-contaminated.

## Checklist

- [x] U0–U2 (STOP A) + corrective A1–A4  
- [x] U3 profiling + dense GappyPOD — **STOP B**  
- [x] U4 Engaging p_on seed 0  
- [x] U5 snapshot-swap control — **STOP C**  
- [~] U6 p_off seed 0 (+ snapshot control) — this mandate; other seeds deferred  
- [ ] U7 pending  

## Explicit non-claims

- Step 2.4 / Month-2 not complete.  
- Single paired seed 0 for p_on/p_off; VAL used for selection/scheduler/stop.  
- No other seeds / GLU / sparse / T-sensors / TEST in this mandate.  
- Swap or p_off gap ≠ physical mechanism proof, information ceiling, or universal uselessness of parameters.
