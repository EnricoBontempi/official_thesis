# Step 1.4 — Locked Validation Protocol, Splits & Observation Layouts

**Lock date:** 2026-11-14  
**Proposal reference:** [`docs/proposal.md`](../proposal.md) §5  
**Configs:**  
- [`configs/splits/splits_interp_extrap.yaml`](../../configs/splits/splits_interp_extrap.yaml)  
- [`configs/splits/evaluation_protocol.yaml`](../../configs/splits/evaluation_protocol.yaml)  
- [`configs/masks/observation_layouts.yaml`](../../configs/masks/observation_layouts.yaml)  
**Stats:** [`data/manifests/train_norm_stats.json`](../../data/manifests/train_norm_stats.json)  
**Figure:** `results/figures/month1_audit/fig06_locked_operating_condition_splits.png`

---

## 1. Operating-condition split (strict withholding)

Entire trajectories (all 2001 frames) belonging to a `sim_id` live in **exactly one** of:

| Split | n | Role |
|-------|--:|------|
| `train` | 20 | Fit normalizers / baselines / neural models |
| `val` | 3 | Hyperparameter & model selection only |
| `test_interp` | 3 | Final reporting — inside training \((\mathrm{NH}_3\%,\phi)\) hull |
| `test_extrap` | 4 | Final reporting — outside / frontier of training range |

**Partition check:** \(20+3+3+4=30\), zero overlap (enforced by `tests/test_no_data_leakage.py`).

### Train (20)

`0NH3`: 0.85, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25  
`20NH3`: 0.85, 0.9, 1.05, 1.2, 1.25  
`40NH3`: 0.85, 1.0, 1.1, 1.2  
`80NH3`: 0.9, 1.05, 1.15, 1.2  

Train ranges: \(\mathrm{NH}_3\in\{0,20,40,80\}\%\), \(\phi\in[0.85,1.25]\).

### Val (3)

`0NH3_1.05.h5`, `20NH3_1.1.h5`, `40NH3_0.9.h5`

### Test-Interp (3)

| `sim_id` | Rationale |
|----------|-----------|
| `20NH3_1.h5` | Interior in both NH₃% and φ |
| `40NH3_1.15.h5` | φ bracketed by train 1.10 & 1.20 at 40% NH₃ |
| `60NH3_0.95.h5` | Fuel-blend interpolation between 40% and 80% NH₃ |

### Test-Extrap (4)

| `sim_id` | Rationale |
|----------|-----------|
| `20NH3_0.75.h5` | Leanest φ in dataset (\< train min 0.85) |
| `0NH3_1.3.h5` | Richest φ at pure CH₄ (\> train max 1.25) |
| `20NH3_1.3.h5` | Richest φ at 20% NH₃ |
| `80NH3_1.25.h5` | High-NH₃ + rich frontier corner |

---

## 2. Training-only normalization

Computed by `scripts/03_lock_splits_and_stats.py` **exclusively** on the 20 train trajectories:

- Fields: Temperature (ch 10), Velocity[i/j/k], recomputed \(\lvert\mathbf{u}\rvert=\sqrt{v_i^2+v_j^2+v_k^2}\)
- Operating parameters: `nh3_fraction` \(\in[0,0.8]\), `equivalence_ratio` φ

Loader: `combustion_recon.data.normalizer.TrainOnlyNormalizer` refuses stats whose `meta.split != "train"` or whose `sim_ids` disagree with the locked YAML.

**Rule:** never refit mean/std on val/test; never pool all 30 conditions for normalization.

---

## 3. High-temperature / flame-zone definitions

Locked in `configs/splits/evaluation_protocol.yaml`:

| Region | Definition | Role |
|--------|------------|------|
| \(\Omega_{\mathrm{high}}\) | \(T \ge 2100\,\mathrm{K}\) | **Primary** high-T mask (IoU, Dice, conditional RMSE/MAE) |
| \(\Omega_{\mathrm{peak}}\) | \(T \ge 2200\,\mathrm{K}\) | Peak thermal core diagnostic |
| \(\Omega_{\mathrm{jet}}\) | \(T < 1800\,\mathrm{K}\) | Cold jet / reaction–mixing layer diagnostic |

Thresholds are **fixed a priori** from training physics (burnt-dominated domain).  

**Train-only Temperature (20 conditions):** mean \(2039.3\,\mathrm{K}\), \(\sigma_T=331.6\,\mathrm{K}\), \(p_{50}=2122.9\,\mathrm{K}\), \(p_{95}=2256.5\,\mathrm{K}\).

**Mean coverage of regions (fraction of space–time voxels):**

| Split | \(\Omega_{\mathrm{high}}\) (\(T\ge2100\)) | \(\Omega_{\mathrm{peak}}\) (\(T\ge2200\)) | \(\Omega_{\mathrm{jet}}\) (\(T<1800\)) |
|-------|------------------------------------------:|------------------------------------------:|---------------------------------------:|
| train | 0.602 | 0.185 | 0.092 |
| val | 0.859 | 0.319 | 0.069 |
| test_interp | 0.824 | 0.254 | 0.088 |
| test_extrap | 0.116 | 0.001 | 0.122 |

Extrapolation conditions are much colder on average (smaller \(\Omega_{\mathrm{high}}\)) — an important stress test for thermal-level transfer.

---

## 4. Observation layouts & task modes

See `configs/masks/observation_layouts.yaml`.

- **Task A (`temperature_free`):** sparse Velocity[i], Velocity[j] only → full \(T\).  
- **Task B (`temperature_assisted`):** same velocities + \(K\in\{1,4,9,16,32\}\) sparse \(T\) anchors.

Layouts: random sparse (0.5–20%), axial/radial rakes, localized missing boxes (`missing_shear_layer_box`, `missing_irz_core_box`).

**Token rule:** observations are emitted as \((x,y,\mathrm{var\_id},value)\) **only** at observed sites (`combustion_recon.data.masks`). Unobserved locations are absent — never filled with `0.0`.

Geometry reminder (Step 1.3): cold jets enter from the **right** (\(x+\)); symmetry about **row ≈ 64**; \(v_j\) axial, \(v_i\) transverse, \(v_k\) swirl.

---

## 5. Metric suite (always per condition)

Implemented in `combustion_recon.evaluation.metrics`:

1. **Field error:** RMSE, MAE, Rel L2, NRMSE (by train \(\sigma_T\)), \(R^2\)  
2. **Spatial bias:** mean signed error + optional 2D bias map  
3. **High-T region:** conditional RMSE/MAE in \(\Omega_{\mathrm{high}}\), IoU/Dice of predicted high-T mask, rel. area error; also jet-zone conditional RMSE  
4. **Temperature level:** \(|\langle\hat T\rangle-\langle T\rangle|\), \(|P_{95}(\hat T)-P_{95}(T)|\), peak error  

`evaluate_per_condition(...)` returns a DataFrame grouped by `(split, sim_id)` — never a single pooled number as the primary report.

Orthogonal axes: change observation layout **or** operating condition, not both without an explicit factorial design.

---

## 6. Automated guardrails

```bash
pytest tests/test_no_data_leakage.py tests/test_masks_and_tokens.py tests/test_metrics.py -q
```

- `test_no_data_leakage.py` — partition, overlap, interp-inside / extrap-frontier, train-only stats  
- `test_masks_and_tokens.py` — tokens match observed sites only; no zero-fill of missing  
- `test_metrics.py` — four metric families + per-condition table  

---

## 7. Code entry points

| Module | Role |
|--------|------|
| `combustion_recon.data.dataset.CombustionReconDataset` | Arrow reader, split filter, sparse tokens, full 128×128 |
| `combustion_recon.data.normalizer.TrainOnlyNormalizer` | Train-only z-score |
| `combustion_recon.data.masks` | Layouts + tokens |
| `combustion_recon.evaluation.metrics` | Proposal §5 metrics |
