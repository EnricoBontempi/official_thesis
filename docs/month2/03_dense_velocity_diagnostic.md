# Dense velocity → temperature diagnostic (Step 2.4)

**Project:** `EnricoBontempi/official_thesis` only (MIT/Deng Lab).  
**Roadmap:** [roadmap_unet.md](../../roadmap_unet.md)  
**Phases:** U0–U3 **executed** (STOP B). U4 **authorized by Enrico 2026-11-17** for Engaging prep + smoke/profile + one p_on seed-0 run (max 1000).  
**Not claimed:** Step 2.4 complete; supervisor ratification; U5+; GLU.

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

## U4 — Authorized Engaging run (prep)

| Distinction | Status |
|---|---|
| Epoch runner + epoch-boundary resume | **Hardened / tested** (complete vs partial ckpt) |
| Config | `month2_dense_unet_u4_engaging_v1` in `month2_dense_unet_u4_candidate.yaml` |
| Decision | **Enrico 2026-11-17** (operational; no supervisor ratification claimed) |
| Scientific U4 training | **Authorized** for one p_on seed-0 Engaging run after smoke/gate |
| Replaces | CPU/200 candidate; **not** Opus WSD |

**Authorized settings:** b=32, batch 8, CUDA fp32 (TF32 off), p_on, seed 0, **max 1000 epochs**, AdamW lr=1e-3, warmup **950** updates then ReduceLROnPlateau on VAL RMSE, eval train/seen/VAL every 10, best=first VAL min, early stop ≥200 ep / 15 eval / 0.1% rel (reset on LR drop), wall ≤6 h on `mit_normal_gpu` AMF.

**Resume policy:** only `dense_unet_epoch_complete_v2`; mid-epoch interrupt → `partial_interrupt.pt` without bumping completed epochs or overwriting `last.pt`; incompatible data/arch/science rejected; n_epochs extension requires explicit flag.

**Cache:** payload SHA256 map in `data/manifests/month2_dense_unet_cache_payload_hashes.json` (aggregate `d0b1d068…`); verify after Engaging transfer.

### Engaging command (after smoke gate)

```bash
python scripts/07_train_reconstructor.py train-u4 \
  --authorize-scientific-u4 \
  --config configs/experiments/month2_dense_unet_u4_candidate.yaml \
  --dense-config configs/experiments/month2_dense_unet.yaml \
  --run-id u4_pon_seed0_engaging_v1 \
  --cache-root "$CACHE" \
  --expect-cache-hashes "$HASHES" \
  --device cuda \
  --wall-hours 6
```

## Checklist

- [x] U0–U2 (STOP A) + corrective A1–A4  
- [x] U3 profiling + dense GappyPOD — **STOP B**  
- [x] U4 pre-run corrections + Engaging decision recorded  
- [ ] U4 scientific p_on Engaging run — **in progress this mandate**  
- [ ] U5–U7 pending  

## Explicit non-claims

- Step 2.4 not complete until U4 finishes and is evaluated.  
- No U5/p_off/other seeds/GLU/sparse/T-sensors/TEST in this mandate.  
- No automatic continuation past 1000 epochs.
