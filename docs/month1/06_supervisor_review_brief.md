# Month 1 Executive Summary & Supervisor Review Brief

**Project:** Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion  
**Student:** Enrico Bontempi (MSc Thesis, MIT — Deng Lab)  
**Date:** November 15, 2026  
**Status:** **Month 1 Complete (Steps 1.1–1.6, Verified & Corrected)**

---

## 1. Executive Summary

All Month 1 objectives defined in the thesis proposal have been completed and verified:
1. **Full Dataset Staged & Audited:** All 30 numerical Large Eddy Simulation (LES) operating conditions of the swirl-stabilized $\text{NH}_3/\text{CH}_4/\text{air}$ flame from **RealPDEBench** (~58.6 GB, $2001 \text{ frames} \times 128 \times 128 \times 15 \text{ channels}$ per condition) have been downloaded, checksum-verified, and audited at full $128 \times 128$ resolution ([`02_data_audit_report.md`](02_data_audit_report.md)).
2. **Leakage-Free Validation Protocol Locked:** Because the official RealPDEBench release provides empty validation/test splits for numerical combustion data, we designed and locked a strict **operating-condition split** (`20 train`, `3 val`, `3 test_interp`, `4 test_extrap`), training-only normalizers, regional error decomposes (`High-T` $\ge 2100\text{ K}$, `Jet/Front` $< 1800\text{ K}$, `Bias` vs. `Centered-RMSE`, and `Top-25% Core IoU`), deterministic `crc32` masks with nested $K$-anchor subsets, and **25 automated `pytest` guardrails** ([`03_validation_and_splits_protocol.md`](03_validation_and_splits_protocol.md)).
3. **Inexpensive Feasibility Checks & Corrected Baselines Completed:** We quantified the kinematic coupling between velocity and temperature and evaluated non-neural baseline families (`GlobalTrainMean`, `ParamConditionedMean`, `LocalHGB`, and variance-scaled MAP `GappyPOD` with $R=64$ modes, plus `Anchor Offset` thermal calibration for $K \in \{1, 4, 9, 16, 32\}$ thermocouples) across seen, validation, interpolation, and extrapolation conditions ([`04_feasibility_check_report.md`](04_feasibility_check_report.md)).
4. **Literature & GLU Architectural Blueprint:** We aligned our findings with the exact formulation of **GLU** (`arXiv:2603.26023`), **RealPDEBench** (`arXiv:2601.01829`), and **RF-PINNs** (*JCP* 2025) and defined three structured neural architecture candidates for Month 2–3 ([`05_literature_and_glu_notes.md`](05_literature_and_glu_notes.md)).

---

## 2. Key Findings from the Data Audit (Steps 1.2–1.3)

### 2.1 Geometric & Physical Conventions (`fig01`–`fig05`)
- **Domain & Orientation:** Each trajectory is a 2D diametral slice ($x \in [-36.9, +36.9]\text{ mm}$, $y \in [62.9, 159.1]\text{ mm}$, $\Delta t = 0.25\text{ ms}$, $4000\text{ Hz}$). In array coordinates `(row, col)`, the flame is symmetric about the horizontal midline (**row 64**). Two cold unburnt reactant jets ($T \sim 250\text{--}400\text{ K}$) enter from the **right boundary (`col = 127`)** and diverge in a V-shape toward the left, enclosing a hot central **Inner Recirculation Zone (IRZ)** ($T \sim 1900\text{--}2280\text{ K}$).
- **Velocity Channels (2-Component Planar PIV vs. 3-Component Stereo-PIV):**
  - `Ch 12 (Velocity[j])`: **Axial velocity** ($+25\text{ m/s}$ along the V-jets; $-10\text{ m/s}$ in the central IRZ).
  - `Ch 11 (Velocity[i])`: **In-plane radial/transverse velocity** (antisymmetric about row 64).
  - `Ch 13 (Velocity[k])`: **Out-of-plane azimuthal swirl velocity** (opposite signs above and below row 64).
  - *Note:* Exported `Ch 14 (Velocity_Magnitude)` contains minor interpolation artifacts (values down to $-8\text{ m/s}$); our pipeline always recomputes Euclidean speed $|\mathbf{u}|$ from Channels 11–13.
- **Parameter Space Coverage (`30 conditions`):**
  - $\text{NH}_3 \in \{0\%, 20\%, 40\%, 60\%, 80\%\}$ and equivalence ratio $\phi \in [0.75, 1.30]$.
  - While mean temperature drops systematically with increasing $\text{NH}_3\%$ (from $\sim 2187\text{ K}$ at $0\%\text{ NH}_3$ to $\sim 1864\text{ K}$ at $80\%\text{ NH}_3$) and cold-jet penetration ($T < 1200\text{ K}$) more than doubles ($3.5\% \rightarrow 8.5\%$), **mean velocity magnitude remains nearly constant ($11\text{--}14\text{ m/s}$) across 28 of the 30 conditions** (`fig05`).
- **Temporal Autocorrelation:** High-variance flame-brush points remain correlated over **50–100+ frames** ($\tau_c \approx 14\text{ ms}$), confirming that entire operating conditions must be withheld to prevent data leakage (and seen-condition evaluation uses a temporal buffer of $\ge 100$ frames / $\ge 2\tau_c$).

---

## 3. Locked Experimental Protocol (Step 1.4)

### 3.1 Operating-Condition Splits (`fig06`)
| Split | Count | Conditions (`sim_id`) | Scientific Role |
|-------|------:|-----------------------|-----------------|
| **`train`** | **20** | `0NH3` (7), `20NH3` (5), `40NH3` (4), `80NH3` (4) spanning $\phi \in [0.85, 1.25]$ | Fit normalizers ($\mu_T = 2039.3\text{ K}, \sigma_T = 331.6\text{ K}$), POD modes, and neural weights |
| **`val`** | **3** | `0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9` | Hyperparameter & architecture selection (supplemented by leave-edge-condition-out CV within `train` for extrapolation tuning) |
| **`test_interp`** | **3** | `20NH3_1.0`, `40NH3_1.15`, `60NH3_0.95` | Held-out evaluation **inside** the training $(\text{NH}_3\%, \phi)$ hull (including unseen $60\%\text{ NH}_3$ blend) |
| **`test_extrap`** | **4** | `20NH3_0.75` (leanest), `0NH3_1.3`, `20NH3_1.3` (richest), `80NH3_1.25` (high-$\text{NH}_3$ rich frontier) | Held-out evaluation **outside** the training parameter range |

### 3.2 Regional & Decomposed Evaluation Metrics
To avoid threshold saturation and separate global thermal scale from spatial flame structure, all methods are evaluated on:
- **Full-Field Decomposition:** $\text{RMSE}^2 = \text{Bias}^2 + \text{Centered-RMSE}^2$ (separating mean thermal level shift from spatial pattern accuracy).
- **Regional Errors:** **High-Temperature Region ($\Omega_{\text{high}}: T \ge 2100\text{ K}$)** RMSE & IoU, **Jet / Flame-Front Mixing Region ($\Omega_{\text{jet}}: T < 1800\text{ K}$)** RMSE, and **Per-Frame Top-25% Hot-Core IoU** (invariant to uniform temperature offsets).

---

## 4. Feasibility Check Results & Corrected Baseline Benchmarks (Step 1.5)

### 4.1 Kinematic-Thermal Coupling & Cross-Condition Ambiguity (`fig07`)
- **Within a fixed operating condition**, $|\mathbf{u}|$ and axial velocity $v_j$ correlate negatively with $T$ (within-condition Pearson $r = -0.48 \pm 0.11$; pooled $r = -0.38$), locating the cold reactant jets and shear layers.
- **Across operating conditions**, conditional curves $\mathbb{E}[T \mid |\mathbf{u}|]$ for $0\%, 40\%, 80\%\text{ NH}_3$ are separated by **$200\text{--}360\text{ K}$ at identical flow speeds** (`fig07`). Velocity provides strong **local spatial structure**, but is **thermodynamically ambiguous** across mixtures without operating-condition or thermal anchor information.
- **Why Naive 2D PDE Residuals Are Excluded:** Planar 2D divergence $\nabla_{\text{2D}} \cdot \mathbf{u}$ has near-zero pointwise correlation with $T$ ($r = -0.014$) because low-Mach thermal dilatation is confined to the thin reaction sheet (zero across the $\sim 90\%$ burnt volume) and 3D azimuthal swirl makes $\partial_z v_k$ non-zero on a 2D slice.

### 4.2 Temperature-Free Baseline Comparison (`fig08`)

Split-averaged metrics across our locked splits (`seen_train_eval`, `val`, `test_interp`, `test_extrap`):

| Baseline Method | Inputs | `seen` RMSE / High-T / Jet | `val` RMSE / Bias / CtrRMSE / IoU | `test_interp` RMSE / Bias / CtrRMSE / IoU | `test_extrap` RMSE / High-T / Jet / Bias / CtrRMSE / IoU |
|---|---|---:|---:|---:|---:|
| **GlobalTrainMean** | None | 174 / 101 / 406 K | 140 / −61 / 115 K / 0.81 | 134 / −20 / 123 K / 0.84 | 223 / 194 / 507 / +114 / 175 K / 0.12 |
| **ParamConditionedMean** | $(\text{NH}_3\%, \phi)$ only (LOCO `piecewise_phi1`) | 129 / 75 / 335 K | 88 / −17 / 85 K / 0.90 | 107 / −13 / 105 K / 0.89 | 172 / 223 / 414 / +43 / 159 K / 0.20 |
| **LocalHGB (Uncond, 3C)** | Velocity $(v_i, v_j, v_k)$ | 164 / 87 / 381 K | 139 / −60 / 112 K / 0.84 | 118 / −17 / 109 K / 0.86 | 219 / 131 / 505 / +124 / 166 K / 0.12 |
| **LocalHGB (Cond, 3C)** | Velocity 3C + $(\text{NH}_3\%, \phi)$ | 116 / 66 / 313 K | 93 / +9 / 91 K / **0.94** | 96 / +4 / 94 K / **0.91** | 159 / **136** / 383 / +69 / **133 K** / 0.12 |
| **GappyPOD (Uncond, 3C, 5%)** | 5% velocity 3C | 123 / 72 / 276 K | 95 / −46 / 72 K / 0.88 | 98 / −33 / 87 K / 0.84 | 183 / 157 / 382 / +93 / 139 K / 0.12 |
| **GappyPOD (Cond, 3C, 5%)** | 5% vel 3C + $(\text{NH}_3\%, \phi)$ | **108** / **65** / **269 K** | **69** / −15 / **67 K** / **0.94** | **88** / −14 / **86 K** / 0.90 | **153** / 203 / **360** / +41 / 139 K / **0.17** |
| **OraclePOD (True $T$ Projection)** | Full true $T$ projected on $R=32$ POD basis | 54 / 39 / 119 K | 36 / −2 / 36 K / 0.97 | 47 / −2 / 46 K / 0.93 | 75 / 85 / 164 / +6 / 75 K / 0.30 |

- **2-Component vs. 3-Component Velocity:** Out-of-plane swirl velocity $v_k$ changes baseline RMSE by $\le 4\text{ K}$; standard 2-component planar PIV $(v_i, v_j)$ captures nearly all linear/pointwise velocity-temperature information.

### 4.3 Temperature-Assisted Mode: Nested $K$ Thermocouples Fix Level Bias, Not Flame Topology (`fig09b`, `fig10`)

With variance-scaled MAP regularization and explicit **Anchor Offset (`+AO`)** calibration on nested thermocouple sets ($1 \subset 4 \subset 9 \subset 16 \subset 32$):

| $K$ Thermocouples | `GappyPOD Cond + AO` (`val` RMSE / \|Bias\| / High-T) | `GappyPOD Cond + AO` (`interp` RMSE / \|Bias\| / High-T) | `GappyPOD Cond + AO` (`extrap` RMSE / \|Bias\| / High-T / IoU) | `ParamMean + AO` (`extrap` RMSE / Bias / High-T / IoU) |
|---:|---:|---:|---:|---:|
| **$K = 0$** | 69 K / 15 K / 35 K | 88 K / 21 K / 38 K | 153 K / 43 K / 203 K / 0.17 | 172.1 K / +42.6 K / 223.2 K / 0.195 |
| **$K = 1$** | 72 K / 17 K / 38 K | 92 K / 24 K / 43 K | 159 K / 53 K / 213 K / 0.19 | 182.9 K / +16.1 K / 247.2 K / 0.202 |
| **$K = 4$** | 72 K / 19 K / 39 K | 93 K / 25 K / 45 K | 150 K / 44 K / 211 K / 0.19 | 172.5 K / +0.9 K / 248.8 K / 0.197 |
| **$K = 9$** | 70 K / 16 K / 36 K | 89 K / 18 K / 40 K | 145 K / 33 K / 209 K / 0.20 | 166.7 K / +0.3 K / 246.2 K / 0.205 |
| **$K = 16$** | 69 K / 14 K / 35 K | 88 K / 15 K / 38 K | 142 K / 26 K / 206 K / 0.20 | 163.5 K / +1.0 K / 244.4 K / 0.201 |
| **$K = 32$** | **68 K / 9 K / 33 K** | **87 K / 11 K / 36 K** | **140 K / 19 K / 206 K / 0.20** | **161.6 K / −0.4 K / 245.8 K / 0.200** |

- **Core Scientific Takeaway for Month 2–3:** Sparse thermocouples ($K = 4\text{--}32$) successfully eliminate global extrapolation temperature bias ($86\text{ K} \rightarrow 15\text{ K}$ for `GappyPOD+AO`; $+89\text{ K} \rightarrow +0.7\text{ K}$ for `ParamMean+AO`). **However, even when thermal bias is zeroed out, High-T RMSE remains $\sim 204\text{--}226\text{ K}$, Jet/Flame-Front RMSE remains $\sim 370\text{ K}$, and High-T IoU remains $0.16\text{--}0.22$ (`fig10`).** Classical linear bases cannot deform the instantaneous turbulent flame front across chemical regimes—defining the exact target for our GLU-based neural reconstructors in Month 2–3.

---

## 5. Questions & Proposed Architecture Roadmap for Supervisor Sign-Off

1. **Split & Metric Protocol Approval:** Confirm the locked 30-condition split (`20 train`, `3 val`, `3 test_interp`, `4 test_extrap`) and the decomposed evaluation suite (`Full RMSE`, `Bias`, `Centered-RMSE`, `High-T RMSE/IoU` at $2100\text{ K}$, `Jet/Front RMSE` at $<1800\text{ K}$, and `Top-25% Core IoU`).
2. **Default Velocity Modality:** Since 2-component in-plane velocity $(v_i, v_j)$ matches standard planar PIV (`observation_layouts.yaml`) and performs within $\le 4\text{ K}$ of 3-component velocity on all baselines, should we lock **2-component $(v_i, v_j)$ as primary** and keep **3-component Stereo-PIV $(v_i, v_j, v_k)$ as an ablation**?
3. **Month 2–3 Neural Architecture Progression (Detailed in `05_literature_and_glu_notes.md` §5.2):**
   - **Candidate 1 (`GLU-Adapted`):** Direct adaptation of the lab's GLU architecture (`arXiv:2603.26023`) with `variable_id` token embeddings, FiLM operating-parameter conditioning on $\mathbf{z}_{\text{global}}$ and $\varphi(\mathbf{x}, \mathbf{p})$, and modality-separated k-NN retrieval.
   - **Candidate 2 (`GLU-Residual`, recommended Month 2 primary):** Two-stage model where the FiLM-conditioned GLU velocity backbone predicts the temperature-free field $\hat{T}_0(\mathbf{x})$, and when $K > 0$ thermocouples are present, a residual cross-attention head assimilates exact anchor discrepancies $e_k = T(\mathbf{x}_k) - \hat{T}_0(\mathbf{x}_k)$.
   - **Candidate 3 (`GLU-Assimilator`, candidate Month 3 methodological contribution):** Neural prior + learned $r$-mode spatial correction basis $\mathbf{b}_\theta(\mathbf{x}) \in \mathbb{R}^r$ updated in closed form at inference time using GLU's predicted local variance $\hat{\sigma}^2(\mathbf{x}_k)$ to weight each thermocouple.
