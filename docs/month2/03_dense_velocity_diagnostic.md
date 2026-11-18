# Dense velocity → temperature diagnostic (Step 2.4) — mission closure report

**Project:** `EnricoBontempi/official_thesis` only (MIT/Deng Lab).  
**Roadmap:** [roadmap_unet.md](../../roadmap_unet.md) (detail of Month-2 Step 2.4 only).  
**Branch:** `mission/dense-unet` (merge to `main` **not** done).  
**Phases:** **U0–U7 closed** in the executed perimeter (U6 = p_off seed 0 + snapshot control; further seeds **deferred**).  
**Not claimed:** Month-2 complete; supervisor ratification; multi-seed robustness; sparse/GLU success; physical causality or information ceiling.

**External Opus reviews (reported by Enrico; not supervisor ratification):**
- U4/U5: usable with reserves; CSV metrics recomputed; no checkpoint/cache/raw re-inference.
- U6 @ `a3a4d8f`: valid; p_on/p_off comparison usable with reserves; no blocking protocol/optimization defect; ready for U7 / later sparse prep; no new experiment required before closure.

---

## 1. Question and contract

**Question:** With full instantaneous planar velocity (vi, vj) on the grid, does a compact U-Net reconstruct instantaneous T better than ParamMean and dense GappyPOD? How much depends on velocity–temperature alignment of the *same* snapshot? How much do explicit NH3/phi channels add?

**Contract (executed):**
- Task: temperature-free, **K=0**, site fraction **100%**, **single-frame**, planar **2C** (vi/vj).
- Input tensor channels: `[vi_norm, vj_norm, x_scaled, y_scaled, nh3_norm, phi_norm]` → direct normalized T.
- **p_on:** NH3/phi from fit-only global stats, constant maps. **p_off:** same architecture with channels 4/5 **exactly zero** (no ParamMean prior in the forward).
- Roles: **train_fit 760**, **seen_train_eval 220**, **val_eval 123**; TEST payloads denied.
- Stats: `Month2FitOnlyNormalizer` (global fit-only); same cache/geometry for p_on and p_off.
- Loss: MSE on normalized T; AdamW lr=1e-3, wd=1e-4, grad clip 1; fp32; no AMP/TF32/`torch.compile`.
- Schedule: warmup 950 updates; ReduceLROnPlateau on VAL macro-condition RMSE; eval every 10 epochs; best = first absolute VAL min; early stop after ≥200 epochs and 15 evals without 0.1% relative improvement (counter resets on LR drop).
- Architecture: U-Net `base_channels=32`, ~4.90M params; seed **0** only in this mission.

---

## 2. Data, baseline references, and costs

| Item | Value |
|---|---|
| Cache payload SHA | `d0b1d068…a69c725` (Engaging + tip) |
| Dense GappyPOD (U3) | R=32, K=0; λ p_off=1e-3 (**93.66 K** VAL); λ p_on=1e-2 (**70.46 K** VAL) |
| ParamMean VAL | **88.01 K** (full TRAIN fit) |
| GlobalMean VAL | **139.70 K** |
| U3 artifacts | `results/month2_dense_linear/dense_linear_u3_v1/` |
| Profiling (U3) | CPU path; Engaging later measured ~28.5 min / 730 ep (p_on) and ~24 min / 600 ep (p_off) on **L40S** |

**Baseline handling note:** U5 *refit* GlobalMean/ParamMean/GappyPOD p_on at U3-selected hyperparameters (mandate deviation). Opus verified per-frame numeric equivalence to U3 within ~5.5e-12. **U6 reused U3 tables without refit.**

---

## 3. Executed sequence and SHAs

| Phase | Commit (executed / delivery) | Run / job | Hardware | Outcome |
|---|---|---|---|---|
| U0–U2 + fixes | STOP A `3643ce5` (+ later fixes) | local | CPU | contract, cache, overfit |
| U3 | results `64186a7…` | local | CPU | profile + dense GappyPOD |
| U4 p_on seed 0 | code `50bda7f` · delivery `f670bc3` | `u4_pon_seed0_engaging_v1` · job `29465586` | L40S `node3008` | best **26.931 K @160**; last 28.055 @730; stop @730 |
| U5 swap (p_on best) | code `8549a9b` · delivery `faa34d6` | `u5_pon_seed0_snapshot_control_v1` · `29471596` | L40S `node3102` | VAL 26.93→96.55 (Δ+69.6) |
| U6 p_off seed 0 | train `2511280` · eval `815bb7a` · delivery `a3a4d8f` | train `29476814` · eval `29479532` | L40S | best **39.669 K @30**; last 44.102 @600; swap VAL +62.7 |
| U7 docs | *(this closure commit)* | — | — | documentary consolidation only |

**Checkpoints (Engaging pool, gitignored; hashes versioned):**
- U4 best: `6d456abcb5653c16a7a90df40eda377caf42ffb94fd829a65f4c06be58a03634`
- U6 best: `8330730817dae5f674d981cdcd006e7cf97fa1d5f90a5365510c20b4099f8e01`

**Reproduce (already used; no new runs in U7):**
```bash
# U4 train (historical)
python scripts/07_train_reconstructor.py train-u4 --authorize-scientific-u4 ...
# U5 / U6 eval
python scripts/09_dense_snapshot_control.py ...
python scripts/10_u6_poff_eval_and_swap.py ...
# Engaging wrappers: scripts/engaging/u4_*.sbatch, u5_eval.sbatch, u6_*.sbatch
```

---

## 4. Main numeric results (verified from artifacts)

### 4.1 Best vs last — RMSE, |bias|, Centered-RMSE (VAL macro = mean over 3 conditions)

Derived table (no new inference): [`results/month2_dense_unet/u7_closure_val_level_structure.csv`](../../results/month2_dense_unet/u7_closure_val_level_structure.csv), built from existing `evals/epoch_*/val_eval_per_frame.csv.gz`.  
**Do not** recover level/structure by quadratically combining already-averaged macros; the RMSE²≈bias²+C-RMSE² identity holds per frame (or per spatial field), not after condition/role averaging.

| Checkpoint | RMSE | \|bias\| | Centered-RMSE |
|---|---:|---:|---:|
| p_on best @160 | **26.93** | 7.09 | 25.15 |
| p_off best @30 | **39.67** | 15.98 | 34.57 |
| p_on last @730 | 28.06 | 8.91 | 24.44 |
| p_off last @600 | 44.10 | 30.45 | 28.94 |

### 4.2 Per-condition VAL RMSE (best checkpoints)

| Condition | p_on @160 | p_off @30 | Δ (off−on) |
|---|---:|---:|---:|
| 0NH3_1.05 | 32.01 | 45.68 | +13.7 |
| 20NH3_1.1 | 27.61 | 35.65 | +8.0 |
| 40NH3_0.9 | 21.17 | 37.68 | +16.5 |

Macro averages hide that `0NH3_1.05` is consistently the hardest / most swap-sensitive condition.

### 4.3 Snapshot-swap (same U5 pairings)

| Model | Role | aligned | swapped | Δ |
|---|---|---:|---:|---:|
| p_on (U5) | VAL | 26.93 | 96.55 | +69.6 |
| p_on (U5) | seen | 37.68 | 150.30 | +112.6 |
| p_off (U6) | VAL | 39.67 | 102.41 | +62.7 |
| p_off (U6) | seen | 51.03 | 153.10 | +102.1 |

### 4.4 Matched dense ranking (VAL macro RMSE)

| Method | VAL K |
|---|---:|
| U-Net p_on best | **26.93** |
| U-Net p_off best | **39.67** |
| GappyPOD dense p_on | 70.46 |
| ParamMean | 88.01 |
| GappyPOD dense p_off | 93.66 |
| GlobalMean | 139.70 |
| U-Net p_on swapped | 96.55 |
| U-Net p_off swapped | 102.41 |

**Verified statement:** U-Net p_off (39.67 K) is **lower** than every dense baseline listed above, including GappyPOD p_on (70.46 K), while remaining **worse** than U-Net p_on (26.93 K).

---

## 5. Interpretation (plausible, not proven)

**Empirical (artifact-backed):**
1. Dense U-Net p_on shows a large advantage vs dense linear/mean baselines on this VAL set and seed.
2. Snapshot swap strongly degrades both p_on and p_off → predictions are snapshot-specific for this protocol/seed.
3. Removing explicit p channels costs ~+12.7 K VAL RMSE at the selected best checkpoints; both |bias| and Centered-RMSE worsen in the best-vs-best comparison.
4. Along p_off training, Centered-RMSE at last (28.94) is **better** than at best (34.57), while |bias| **worsens** (15.98→30.45) and RMSE worsens (39.67→44.10). Selection on RMSE picking epoch 30 is consistent with the selection rule; it does **not** by itself prove a bug or insufficient budget.
5. Last-vs-last: the p_on/p_off gap is especially a **level** gap (|bias| 8.91 vs 30.45) while structure is closer (24.44 vs 28.94). Last remains an **additional diagnostic**, not a retroactive replacement of the selected checkpoint.

**Not demonstrated:** causality of NH3/phi physics; information ceiling; necessity of non-linearity in general; sparse success; that p_on remains better with few sensors; multi-seed robustness (only paired seed 0).

**VAL caveat:** VAL drove selection, LR plateau, and early stop for U4 and U6 — not an independent final test.

---

## 6. Traceability notes (documentation only; manifests not rewritten retroactively)

1. **U6 `working_tree.clean: false`:** `dirty_entries` was **empty**; unclean status came from **untracked** prior outputs (e.g. U4 `evals/epoch_*`, `.git_commit`) on the Engaging tip. Tracked scientific sources used for the run were committed.
2. **U6 swapped per-frame tables lack `src_frame_index`:** pairing remains reconstructible from `swap_pairs_reused_from_u5.json` / U5 `swap_pairs.json`. **Recommendation for future runs:** emit donor indices on swapped metric rows.
3. **U5 vs U6 baselines:** U5 refit at fixed U3 hyperparameters (deviation; Opus ≈U3 per-frame). U6 reused U3 tables without refit.

---

## 7. Figures (presentation-ready, already generated)

**U5:** `results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/figures/`  
**U6:** `results/month2_dense_unet/u6_poff_seed0_snapshot_control_v1/figures/` (includes shared-scale p_on/p_off panels and tight error views; U5 originals untouched).

---

## 8. Mission outcome and Month-2 status

**Dense U-Net mission (Step 2.4 / U0–U7):** closed **technically and experimentally** in the executed perimeter (p_on seed 0, snapshot control, p_off seed 0 + control). Outcome: **positive empirical dense results**, **preliminary** p_on/p_off comparison (single paired seed), variability across seeds **not measured**.

**Still open for Month 2:** Step 2.5 GLU audit; Step 2.6 sparse neural; 2.7–2.9; supervisor ratification; merge to main.

**Next orientation (not authorization):** prepare sparse U-Net protocol (2.6) and finish GLU audit (2.5); keep p_on as initial sparse candidate with planned p_off comparison; report best/last with level/structure from frame-level metrics.

---

## Checklist

- [x] U0–U2 (STOP A) + corrective A1–A4  
- [x] U3 profiling + dense GappyPOD — **STOP B**  
- [x] U4 Engaging p_on seed 0  
- [x] U5 snapshot-swap control — **STOP C**  
- [x] U6 p_off seed 0 + snapshot control (other seeds deferred)  
- [x] U7 documentary closure (this report + roadmap/diary/README/registry sync)

## Explicit non-claims

- Month-2 not complete.  
- Single paired seed 0; VAL selection-contaminated.  
- No sparse / GLU / T-sensors / TEST / extra seeds in this closed mission.  
- Empirical dense gains ≠ physical mechanism proof, information ceiling, or guaranteed sparse transfer.
