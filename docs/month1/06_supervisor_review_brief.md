# Month 1 Executive Summary & Supervisor Review Brief

**Project:** Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion  
**Student:** Enrico Bontempi (MSc Thesis, MIT — Deng Lab)  
**Date:** November 14, 2026  
**Status:** **Month 1 Complete (Steps 1.1–1.6)**

---

## 1. Executive Summary

All Month 1 objectives defined in the thesis proposal have been completed:
1. **Full Dataset Staged & Audited:** All 30 numerical Large Eddy Simulation (LES) operating conditions of the swirl-stabilized $\text{NH}_3/\text{CH}_4/\text{air}$ flame from **RealPDEBench** (~58.6 GB, $2001 \text{ frames} \times 128 \times 128 \times 15 \text{ channels}$ per condition) have been downloaded, checksum-verified, and audited at full $128 \times 128$ resolution ([`02_data_audit_report.md`](02_data_audit_report.md)).
2. **Leakage-Free Validation Protocol Locked:** Because the official RealPDEBench release provides empty validation/test splits for numerical combustion data, we designed and locked a strict **operating-condition split** (`20 train`, `3 val`, `3 test_interp`, `4 test_extrap`), training-only normalizers, fixed high-temperature region thresholds, and 19 automated `pytest` guardrails ([`03_validation_and_splits_protocol.md`](03_validation_and_splits_protocol.md)).
3. **Inexpensive Feasibility Checks & Baselines Completed:** We quantified the kinematic coupling between velocity and temperature and evaluated four fast baseline families (`GlobalTrainMean`, `ParamConditionedMean`, `LocalHGB`, and `GappyPOD` with $R=32$ modes) across seen, validation, interpolation, and extrapolation conditions under both **temperature-free** and **temperature-assisted** observation layouts ([`04_feasibility_check_report.md`](04_feasibility_check_report.md)).
4. **Literature & GLU Alignment:** We reviewed **GLU** (`arXiv:2603.26023`), **RealPDEBench** (`arXiv:2601.01829`), and **RF-PINNs** (*JCP* 2025) and translated our Month 1 empirical findings into the architectural specification for Month 2 ([`05_literature_and_glu_notes.md`](05_literature_and_glu_notes.md)).

---

## 2. Key Findings from the Data Audit (Steps 1.2–1.3)

### 2.1 Geometric & Physical Conventions (`fig01`–`fig05`)
- **Domain & Orientation:** Each trajectory is a 2D diametral slice ($x \in [-36.9, +36.9]\text{ mm}$, $y \in [62.9, 159.1]\text{ mm}$, $\Delta t = 0.25\text{ ms}$, $4000\text{ Hz}$). In array coordinates `(row, col)`, the flame is symmetric about the horizontal midline (**row 64**). Two cold unburnt reactant jets ($T \sim 250\text{--}400\text{ K}$) enter from the **right boundary (`col = 127`)** and diverge in a V-shape toward the left, enclosing a hot central **Inner Recirculation Zone (IRZ)** ($T \sim 1900\text{--}2280\text{ K}$).
- **Velocity Channels:**
  - `Ch 12 (Velocity[j])`: **Axial velocity** (positive $+25\text{ m/s}$ along the V-jets; negative $-10\text{ m/s}$ in the central IRZ).
  - `Ch 11 (Velocity[i])`: **In-plane radial/transverse velocity** (antisymmetric about row 64 as the annular jet expands).
  - `Ch 13 (Velocity[k])`: **Out-of-plane azimuthal swirl velocity** (opposite signs above and below row 64).
  - *Note:* Exported `Ch 14 (Velocity_Magnitude)` contains small numerical interpolation artifacts (including negative values down to $-8\text{ m/s}$); our pipeline always recomputes Euclidean speed $|\mathbf{u}| = \sqrt{v_i^2 + v_j^2 + v_k^2}$ from Channels 11–13.
- **Parameter Space Coverage (`30 conditions`):**
  - $\text{NH}_3 \in \{0\%, 20\%, 40\%, 60\%, 80\%\}$ and equivalence ratio $\phi \in [0.75, 1.30]$.
  - While mean temperature drops systematically with increasing $\text{NH}_3\%$ (from $\sim 2187\text{ K}$ at $0\%\text{ NH}_3$ to $\sim 1864\text{ K}$ at $80\%\text{ NH}_3$) and cold-jet penetration ($T < 1200\text{ K}$) more than doubles ($3.5\% \rightarrow 8.5\%$), **mean velocity magnitude remains nearly constant ($11\text{--}14\text{ m/s}$) across 28 of the 30 conditions** (`fig05`).
- **Temporal Autocorrelation:** While domain-averaged $T$ decorrelates in $\sim 10\text{--}20$ frames, high-variance flame-brush points remain correlated over **50–100+ frames** ($\tau_c \approx 14\text{ ms}$), confirming that entire operating conditions must be withheld to prevent data leakage.

---

## 3. Locked Experimental Protocol (Step 1.4)

### 3.1 Operating-Condition Splits (`fig06`)
| Split | Count | Conditions (`sim_id`) | Scientific Role |
|-------|------:|-----------------------|-----------------|
| **`train`** | **20** | `0NH3` (7), `20NH3` (5), `40NH3` (4), `80NH3` (4) spanning $\phi \in [0.85, 1.25]$ | Fit normalizers ($\mu_T = 2039.3\text{ K}, \sigma_T = 331.6\text{ K}$), POD modes, and neural weights |
| **`val`** | **3** | `0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9` | Hyperparameter & architecture selection |
| **`test_interp`** | **3** | `20NH3_1.0`, `40NH3_1.15`, `60NH3_0.95` | Held-out evaluation **inside** the training $(\text{NH}_3\%, \phi)$ convex hull (including unseen $60\%\text{ NH}_3$ blend) |
| **`test_extrap`** | **4** | `20NH3_0.75` (leanest), `0NH3_1.3`, `20NH3_1.3` (richest), `80NH3_1.25` (high-$\text{NH}_3$ rich frontier) | Held-out evaluation **outside** the training parameter range |

### 3.2 High-Temperature Region Definition
Because burnt products ($T > 1800\text{ K}$) fill $\sim 91\%$ of the cropped recirculation window, we locked:
- **Primary High-Temperature Region ($\Omega_{\text{high}}$):** $T(x, y) \ge \mathbf{2100\text{ K}}$ (covers $60\%$ of voxels in `train`, $82\text{--}86\%$ in near-stoichiometric `val`/`test_interp`, and drops to **$12\%$** in cooler `test_extrap` conditions).
- **Secondary Diagnostic Regions:** Peak core $\Omega_{\text{peak}}$ ($T \ge 2200\text{ K}$) and cold jet / reaction mixing layer $\Omega_{\text{jet}}$ ($T < 1800\text{ K}$).

---

## 4. Feasibility Check Results & Baseline Benchmarks (Step 1.5)

### 4.1 Does Velocity Carry Temperature Information? (`fig07`)
- **Within a fixed operating condition**, velocity magnitude $|\mathbf{u}|$ and axial velocity $v_j$ correlate negatively with temperature (within-condition Pearson $r = -0.48 \pm 0.11$), clearly delineating the cold reactant jets and shear layers from the hot recirculation core.
- **Across operating conditions**, conditional curves $\mathbb{E}[T \mid |\mathbf{u}|]$ for $0\%, 40\%, 80\%\text{ NH}_3$ are separated by **$200\text{--}360\text{ K}$ at identical flow speeds** (`fig07`). Thus, velocity provides strong **spatial structure** information, but is **thermodynamically ambiguous** across mixtures without operating-condition or thermal anchor information.
- **Note on 2D PDE Residuals:** Planar 2D divergence $\nabla_{\text{2D}} \cdot \mathbf{u}$ has near-zero pointwise correlation with $T$ ($r = -0.014$) because strong out-of-plane azimuthal swirl ($v_k$) makes $\partial_z v_k$ non-negligible on a 2D slice. Consequently, naive 2D continuity/energy PDE penalties would be physically inconsistent.

### 4.2 Quantitative Baseline Comparison (`fig08`–`fig10`)

Split-averaged metrics across our locked splits (`seen_train_eval`, `val`, `test_interp`, `test_extrap`):

| Baseline Method | Observation Input | `seen` RMSE / Bias / IoU | `val` RMSE / Bias / IoU | `test_interp` RMSE / Bias / IoU | `test_extrap` RMSE / Bias / IoU |
|-----------------|-------------------|-------------------------:|------------------------:|--------------------------------:|--------------------------------:|
| **GlobalTrainMean** | None (0 sensors) | 176 K / +1 K / 0.58 | 139 K / −61 K / 0.81 | 133 K / −20 K / 0.84 | 223 K / +114 K / 0.12 |
| **ParamConditionedMean** | $(\text{NH}_3\%, \phi)$ only (0 sensors) | 139 K / 0 K / 0.61 | 95 K / −35 K / 0.90 | 120 K / −40 K / 0.80 | 190 K / +88 K / 0.16 |
| **LocalHGB (Unconditioned)** | Full velocity $(v_i, v_j, v_k)$ | 166 K / 0 K / 0.58 | 140 K / −65 K / 0.83 | 118 K / −22 K / 0.84 | 215 K / +119 K / 0.11 |
| **LocalHGB (Conditioned)** | Full velocity + $(\text{NH}_3\%, \phi)$ | **116 K** / +2 K / 0.62 | 97 K / +6 K / **0.94** | **95 K** / +6 K / **0.91** | **159 K** / +66 K / 0.12 |
| **GappyPOD (Unconditioned)** | 5% sparse velocity (`random_5pct`) | 127 K / +4 K / 0.61 | 101 K / −50 K / 0.86 | 98 K / −31 K / 0.80 | 181 K / +98 K / 0.12 |
| **GappyPOD (Conditioned)** | 5% sparse velocity + $(\text{NH}_3\%, \phi)$ | **113 K** / 0 K / 0.60 | **79 K** / −25 K / 0.89 | 100 K / −45 K / 0.69 | 168 K / +79 K / 0.14 |

### 4.3 Three Key Methodological Takeaways Leading Into Month 2
1. **Central Hypothesis Confirmed on Interpolation vs. Extrapolation:**
   - Adding operating-condition parameters $(\text{NH}_3\%, \phi)$ to velocity observations reduces temperature-level error from $34\text{--}75\text{ K}$ down to **$10\text{--}20\text{ K}$** on `val` and `test_interp` (achieving High-T IoU = **0.91–0.94** with `LocalHGB Conditioned` and **79 K RMSE** with `GappyPOD Conditioned` at just 5% velocity coverage).
   - However, on **extrapolation (`test_extrap`)**, all simple baselines suffer severe systematic over-prediction (bias $+66\text{ to }+119\text{ K}$, High-T IoU $\approx 0.12\text{--}0.16$).
2. **Sharp Turbulent Flame-Brush Errors in Linear POD (`fig10`):**
   - Even when the global temperature level is well estimated, linear modal reconstruction (`GappyPOD`) blurs the wrinkled instantaneous shear layers around the V-jets, leaving localized errors of **$\pm 200\text{ K}$** along the flame front.
3. **Why Naive Multi-Variable Pooling Fails in `temperature_assisted` Mode (`fig09b`):**
   - Adding $K = 1\text{--}32$ sparse temperature sensors to `GappyPOD` at $5\%$ velocity coverage reduced `test_extrap` RMSE by only $\sim 5\text{ K}$ ($168 \rightarrow 163\text{ K}$) because the $2,457$ velocity scalar observations overwhelm the $K \le 32$ temperature observations in unweighted least-squares fitting.
   - **Targeted Opportunity for Month 2–3:** Our neural reconstructor (informed by **GLU**) should decouple **local spatial structure encoding** (driven by sparse velocity tokens) from **global thermal scale calibration** (driven by operating parameters $(\text{NH}_3\%, \phi)$ and, in temperature-assisted mode, a dedicated calibration pathway for sparse $T$ anchors).

---

## 5. Questions for Supervisor Sign-Off (Transition to Month 2)

1. **Split & Threshold Approval:** Do we confirm the locked 30-condition split (`20 train`, `3 val`, `3 test_interp`, `4 test_extrap`) and the primary high-temperature threshold $T_{\text{high}} = 2100\text{ K}$ (supplemented by $\Omega_{\text{jet}}: T < 1800\text{ K}$)?
2. **Velocity Channel Selection:** Since all 3 velocity components $(v_i, v_j, v_k)$ are available on the 2D diametral slice (analogous to Stereo-PIV), should the default `temperature_free` configuration observe all 3 components $(v_i, v_j, v_k)$ or strictly the 2 in-plane components $(v_i, v_j)$ (with 3-component stereo velocity as an ablation)?
3. **Initial Neural Architecture (Month 2):** Confirm starting Month 2 with the deterministic GLU-inspired token encoder + coordinate-conditioned decoder (`GLURec`), evaluated alongside `ParamConditionedMean` and `ConditionedGappyPOD` ($R=32$).
