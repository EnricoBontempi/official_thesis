# Step 1.5 — Inexpensive Feasibility Checks: Does Velocity Carry Temperature Information?

**Run date:** 2026-11-14  
**Proposal reference:** [`docs/proposal.md`](../proposal.md) Month 1 + §5  
**Script:** [`scripts/04_feasibility_checks.py`](../../scripts/04_feasibility_checks.py)  
**Baselines:** [`src/combustion_recon/models/baselines.py`](../../src/combustion_recon/models/baselines.py)  
**Splits / masks / stats:** locked artifacts from Step 1.4  
**Sampling:** stride-50 frames → 41 snapshots/trajectory; held-out train times `arange(25, 2000, 100)` → `seen_train_eval`

---

## 0. Protocol reminders (anti-leakage)

| Guardrail | Enforcement |
|-----------|-------------|
| Entire operating-condition withholding | Metrics reported per `sim_id` and per split only |
| Train-only statistics | Mean predictors / POD / HGB fit on the 20 train conditions only |
| Missing ≠ physical zero | Gappy POD uses only masked observation rows (verified in `tests/test_baselines.py`) |
| Temperature-free vs assisted separated | Separate `task_mode` columns in CSVs and figures |

**Frame counts:** train fit bank 820; `seen_train_eval` 400; `val`/`test_interp` 123 each; `test_extrap` 164.  
**POD rank selected on val (temp-free, `random_5pct`):** \(R=32\) (val RMSE 102.7 K vs 109.4 K for \(R=64\)).  
**Train \(\sigma_T\):** 331.6 K.

---

## 1. Check 1 — Kinematic & physical coupling

### 1.1 Pooled correlations with \(T\) (20 train conditions)

| Feature | Pearson \(r\) | Spearman | MI |
|---------|-------------:|---------:|---:|
| \(v_i\) | +0.004 | +0.013 | 0.079 |
| \(v_j\) | −0.327 | −0.212 | 0.125 |
| \(v_k\) | +0.011 | +0.014 | 0.081 |
| \(\|\mathbf{u}\|\) | −0.379 | −0.268 | 0.114 |
| \(\nabla_{\mathrm{2D}}\cdot\mathbf{u}\) (preferred) | −0.014 | +0.012 | 0.013 |
| \(\nabla_{\mathrm{2D}}\cdot\mathbf{u}\) (alt pairing) | −0.009 | −0.009 | 0.022 |
| \(\|\omega_z\|\) | −0.056 | −0.062 | 0.000 |
| \(\|\nabla\mathbf{u}\|_F\) | −0.118 | −0.124 | 0.026 |

Within-condition \(\|\mathbf{u}\|\)–\(T\) Pearson averages \(-0.48\) (std \(0.11\)) — stronger than the pooled figure, consistent with **cross-condition confounding**.

### 1.2 Interpretation

- Instantaneous velocity magnitude and axial \(v_j\) carry the strongest linear/rank association with \(T\), but the relationship is **negative** (high-speed jet / shear vs hot products) and only moderate.
- Dilatation / vorticity are **not** strong pooled linear predictors of \(T\) on this \(128\times128\) V-flame slice (flame-front expansion is spatially thin relative to the domain).
- **Cross-condition ambiguity is real:** \(\mathbb{E}[T\mid\|\mathbf{u}\|]\) curves for \(0\%\), \(40\%\), and \(80\%\) NH₃ separate by hundreds of kelvin at matched velocity magnitude (Fig. 07). Velocity alone cannot pin down the thermal state across fuel blends.

**Artifact:** `results/tables/month1_check1_kinematic_correlations.csv`, Fig. 07.

---

## 2. Checks 2–3 — Mean fields & local regressors

Split-mean metrics (temperature-free):

| Method | Split | RMSE [K] | Mean signed bias [K] | High-T IoU | Level \|error\| [K] |
|--------|-------|---------:|---------------------:|-----------:|--------------------:|
| GlobalTrainMean | seen | 176 | +1 | 0.58 | 71 |
| | val | 139 | −61 | 0.81 | 69 |
| | interp | 133 | −20 | 0.84 | 36 |
| | extrap | **223** | **+114** | **0.12** | **115** |
| ParamConditionedMean | seen | 139 | −0.2 | 0.61 | 38 |
| | val | 95 | −35 | 0.90 | 35 |
| | interp | 120 | −40 | 0.80 | 40 |
| | extrap | 190 | +88 | 0.16 | 88 |
| LocalHGB Uncond. (full vel) | seen | 166 | −0.2 | 0.58 | 73 |
| | val | 140 | −65 | 0.83 | 75 |
| | interp | 118 | −22 | 0.84 | 34 |
| | extrap | 215 | +119 | 0.11 | 124 |
| LocalHGB Cond. (full vel) | seen | **116** | +2 | 0.62 | **12** |
| | val | 97 | +6 | **0.94** | 20 |
| | interp | **95** | +6 | **0.91** | **10** |
| | extrap | **159** | +66 | 0.12 | 67 |

### Takeaways

1. **Parametric mean** beats the global train mean on every split — a large fraction of error is the **condition-mean flame state**, recoverable from \((\mathrm{NH}_3\%,\phi)\) alone (zero instantaneous sensors).
2. **Unconditioned local regression** barely beats GlobalMean on transfer splits → local \(v\to T\) without parameters largely rediscovers the train mean thermal bias.
3. **Conditioning** (`nh3_fraction`, \(\phi\)) collapses level error on seen/val/interp (level abs error \(10\)–\(20\) K) and cuts extrap RMSE from \(215\to159\) K, but **High-T IoU on extrap remains ~0.12** — parameters fix the mean level more than the frontier high-\(T\) topology.

---

## 3. Check 4 — Gappy POD (joint \([v_i,v_j,v_k,T]\), \(R=32\))

### 3.1 Temperature-free sparsity (`random_*`, conditioned POD)

| Coverage | val RMSE / IoU | interp RMSE / IoU | extrap RMSE / IoU |
|---------:|---------------:|------------------:|------------------:|
| 1% | 83 / 0.91 | 105 / 0.74 | 178 / 0.16 |
| 2% | 79 / 0.90 | 101 / 0.73 | 172 / 0.15 |
| 5% | **79 / 0.89** | **100 / 0.69** | **168 / 0.14** |
| 10% | 82 / 0.88 | 101 / 0.66 | 166 / 0.14 |
| 20% | 85 / 0.88 | 103 / 0.65 | 166 / 0.14 |

Sparse velocity already saturates near ~2–5% coverage for this linear modal estimator; denser observations do not fix extrapolation.

At `random_5pct`, **conditioned** Gappy POD improves val RMSE \(101\to79\) K and extrap \(181\to168\) K vs unconditioned (Fig. 08).

### 3.2 Temperature-assisted (\(K\) anchors @ vel 5%, conditioned POD)

| \(K\) | val RMSE (level) | interp RMSE (level) | extrap RMSE (level) |
|------:|-----------------:|--------------------:|--------------------:|
| 0 | 80 (28) | 98 (45) | 168 (81) |
| 1 | 80 (28) | 98 (45) | 168 (81) |
| 4 | 80 (28) | 98 (45) | 168 (80) |
| 9 | 79 (27) | 98 (44) | 167 (80) |
| 16 | 79 (27) | 97 (44) | 165 (79) |
| 32 | 78 (27) | 96 (43) | 163 (76) |

A few sparse \(T\) anchors give only **marginal** linear-POD gains (\(\sim5\) K on extrap from \(K=0\to32\)). Cross-condition thermal ambiguity is **not** resolved by \(K\le32\) point sensors under this modal model — a strong motivation for nonlinear, parameter-conditioned neural reconstructors in Month 2.

---

## 4. Deliverables

| Path | Content |
|------|---------|
| `results/tables/month1_check1_kinematic_correlations.csv` | Within-condition + pooled Pearson/Spearman/MI |
| `results/tables/month1_check2_3_4_per_condition_metrics.csv` | Per-frame / per-condition metrics (all methods) |
| `results/tables/month1_feasibility_split_summary.csv` | Split aggregates; `temperature_free` / `temperature_assisted` separated |
| `results/tables/month1_feasibility_meta.json` | Selected \(R\), \(\sigma_T\), frame counts |
| `results/figures/month1_feasibility/fig07_*.png` | Coupling maps + ambiguity curves |
| `results/figures/month1_feasibility/fig08_*.png` | Uncond. vs cond. transfer |
| `results/figures/month1_feasibility/fig09_*.png` | Sparsity + \(K\)-assisted curves |
| `results/figures/month1_feasibility/fig10_*.png` | Qualitative interp (`60NH3_0.95`) vs extrap (`20NH3_0.75`) |
| `tests/test_baselines.py` | Mask hygiene for GlobalMean / GappyPOD |

---

## 5. Implications for Month 2

1. Velocity **does** carry usable instantaneous structure (local HGB + Gappy POD beat mean fields on seen/val/interp), but **not** a condition-invariant thermometer.
2. Operating-parameter conditioning is **first-order** for reducing systematic thermal bias — must be a modular architectural toggle, not an afterthought.
3. Extrapolation remains hard for all inexpensive baselines (High-T IoU \(\sim0.11\)–\(0.16\)); Month 2 neural work should treat `test_extrap` as the primary stress test.
4. Temperature-assisted linear POD with \(K\le32\) is a weak upper bound under the current basis — keep temperature-free vs assisted as **separate** evaluation tracks for the neural suite.
5. Recommended non-neural baselines to retain: **ParamConditionedMean** (zero sensors) and **Condition-Aware Gappy POD** (\(R=32\)).
