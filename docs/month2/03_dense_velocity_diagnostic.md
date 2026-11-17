# Dense velocity → temperature diagnostic (Step 2.4) — STOP A delivery

**Project:** `EnricoBontempi/official_thesis` only (MIT/Deng Lab).  
**Roadmap:** [roadmap_unet.md](../../roadmap_unet.md)  
**Phase delivered:** **U0–U2 only**. **STOP A.**  
**Not claimed:** Step 2.4 complete; U3+; scientific training; VAL selection; GLU.

## Status legend

| Label | Meaning |
|---|---|
| Implemented | Code/config present on the branch |
| Executed | Ran locally in this mission |
| Verified | Checks/tests passed with recorded evidence |
| Pending | Not part of this STOP A delivery |

## U0 — Preflight

| Item | Value | Status |
|---|---|---|
| Work branch | `mission/dense-unet` | verified |
| Dependencies | consolidation `3cf7bf0` + roadmap `adabacc` (merged on branch) | verified |
| Python / torch | 3.13.13 / 2.15.0+cu132 | verified |
| CUDA | **False** (CPU used) | verified |
| Host RAM / disk | ~25 GiB RAM; ~819 GiB free | inspected |
| Raw local | `data/raw/combustion/hf_dataset/numerical` (~59 GB) | verified |
| TEST payloads | not accessed | verified |
| HONF / thesis-research | not used | verified |

Document merge conflicts (README, RESEARCH_LOG, roadmap_month2) resolved by preserving consolidation evidence **and** the published U-Net plan.

## U1 — Dense contract and cache

**Implemented**
- `configs/experiments/month2_dense_unet.yaml` — `month2_dense_unet_v1` (distinct from sparse baselines)
- `configs/models/unet_small.yaml` — b=32 candidate; concat-p; head std=1e-3
- `src/combustion_recon/training/dense_cache.py`
- Compact versioned stub: `data/manifests/month2_dense_unet_cache_manifest.json`
- Build log: `results/month2_dense_unet/u1_cache_build.json`

**Executed / verified**
| Role | Count | On-disk (role dir) |
|---|---:|---:|
| train_fit | 760 | ~191 MiB |
| seen_train_eval | 220 | ~56 MiB |
| val_eval | 123 | ~31 MiB |
| **Total payload** | **1103** | **~277 MiB** (`bytes_total` 289,774,422) |

- Build wall ≈ **493 s**; peak RSS ≈ **6.5 GiB**
- Geometry shared; channels `[vi_norm, vj_norm, x_scaled, y_scaled, nh3_norm, phi_norm]`; targets separate
- Dense mask all-True; K=0; site_fraction=1.0; fit-only stats reused (T_norm TRAIN mean≈0, std≈1)
- Shard provenance: `remote_file_inventory.json` LFS sha256 + local size verify (23 shards); no TEST
- Temporal flags preserved; single-frame; `40NH3_1.2` association remains unverified (warning on seen index 2000 path as expected)
- Spot checks vs `Month2ReconDataset` passed (pytest)

VAL is cached for later declared selection only; **not used for U2 model decisions**.

## U2 — Network, trainer, tests, tiny overfit

**Implemented**
- `src/combustion_recon/models/unet.py` — DenseUNet
- `src/combustion_recon/training/trainer.py`
- `src/combustion_recon/training/snapshot_swap.py` (U5 pairing helper; **not executed on VAL**)
- `scripts/07_train_reconstructor.py` (`build-cache`, `overfit`; `--scientific` refused)
- `tests/test_unet_dense.py`

**Architecture (effective)**  
Encoder 128:b → 64:2b → 32:4b → 16:8b → bottleneck 8:8b; four bilinear ups + concat skips; 1×1 linear head → T_norm. GroupNorm(8)+SiLU; conv bias=True. Condizionamento = sola concatenazione canali p.  
**Parameters (trainable):** **4,902,753** (`sum(p.numel() for p in parameters if requires_grad)`), b=32.  
Head init N(0, 1e-3), bias 0 (non all-zero).

**Tests executed:** `pytest tests/test_unet_dense.py` → **11 passed** (isolation, gradients, checkpoint reload, evaluator fail-fast, cache contract, dataset spot-check, swap pairing).  
**Not executed here:** U3 profiling, dense GappyPOD, scientific VAL metrics for selection, MAC/FLOP tooling (not required for STOP A).

**Overfit diagnostic (executed)** — `results/month2_dense_unet/u2_overfit/overfit_summary.json`

| Item | Value |
|---|---|
| Frames (cache indices) | 84, 102, 228, 249, 347, 356, 393, 398 |
| Conditions | `0NH3_1.h5`, `0NH3_1.25.h5`, `20NH3_1.05.h5`, `20NH3_1.2.h5` |
| Updates | 500 (max_updates); samples_seen 4000 |
| Device | CPU |
| Initial MSE_norm / RMSE_K | 0.909 / 316.2 K |
| Final MSE_norm / RMSE_K | 0.00162 / 13.4 K |
| ParamMean same subset (diag.) | mean RMSE ≈ 106.4 K |
| Wall / peak RSS | ≈ 231 s / ≈ 1.58 GiB |
| Checkpoint reload max \|Δpred\| | 0 |
| Learning demonstrated | **true** (final MSE < 0.5× initial) |

Checkpoint `overfit_last.pt` and maps `.npz` remain **out of Git**.

## Checklist (roadmap_unet)

- [x] U0 — Checkout with roadmap + consolidation — **verified**
- [x] U1 — Dense contract + cache — **verified**
- [x] U2 — Implementation/tests/overfit — **verified** → **STOP A**
- [ ] U3 — Profiling + dense GappyPOD — **pending**
- [ ] U4–U7 — **pending**

## Issues / decisions before U3

1. Fix **b ∈ {32,16}**, batch, device, E_max after measured profiling (CPU-only host today; GPU not available in this environment).
2. Grad-clip fired on **21%** of overfit updates (norm>1) — monitor in profiling; not a reason to change lr yet.
3. ParamMean comparison on the 8-frame subset is **diagnostic only** (refit on those conditions); scientific matched ParamMean uses the consolidated TRAIN fit.
4. Cache sha256 provenance is inventory-backed + size-checked, not re-hashed locally (intentional cost control); re-hash optional if auditors require byte-identity beyond size+inventory.

## Explicit non-claims

- Step **2.4 not complete**.
- **U3 and scientific training were not started.**
- No full-epoch profiling, no dense GappyPOD fit, no VAL-guided selection, no GLU / masked U-Net / T sensors / temporal windows, no TEST access, no cluster work.
