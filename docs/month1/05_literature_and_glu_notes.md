# Step 1.6 — Literature Review & GLU Architecture Alignment

**Date:** 2026-11-14  
**Author:** Enrico Bontempi (MSc Thesis — Deng Lab, MIT)  
**Proposal Reference:** [`docs/proposal.md`](../proposal.md) §3, §4, §6 & Initial References [1–4]

---

## 1. Purpose of This Document

This note synthesizes the three core scientific references cited in the thesis proposal and connects their methodologies directly to the empirical findings from our Month 1 audit and feasibility study (Steps 1.2–1.5):

1. **GLU (Deng Lab, MIT):** Wang, L., Chen, J., Tricard, N., Chen, Z., & Deng, S. *GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting.* `arXiv:2603.26023` (2026).
2. **RealPDEBench:** Hu, P., Feng, H., Liu, H., et al. *RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data.* `arXiv:2601.01829` / ICLR 2026 Oral (2026).
3. **RF-PINNs:** Yadav, V., Casel, M., & Ghani, A. *RF-PINNs: Reactive flow physics-informed neural networks for field reconstruction of laminar and turbulent flames using sparse data.* *Journal of Computational Physics*, 524, 113698 (2025).

---

## 2. Analysis of Reference 1 — GLU (`arXiv:2603.26023`, Deng Lab)

### 2.1 Core Problem & Paradigm
Traditional neural operators and encoder-decoder architectures either compress sparse observations into a single **global latent vector** (which smooths out sharp local gradients and high-frequency turbulent structures) or rely purely on **local neighborhood interpolation** (which fails when observations are sparse or when global system-level constraints govern the state). Furthermore, most frameworks treat **sparse field reconstruction** and **temporal forecasting** as separate pipelines.

**GLU (Global-Local-Uncertainty Fusion)** unifies sparse reconstruction and forecasting via three core architectural pillars:
1. **Structured Latent Representation (Global Summary + Local Tokens):**
   - Each available sensor observation at location $\mathbf{x}_i$ with variable identity $c_i$ and value $v_i$ is encoded as a **local measurement token** anchored to its physical coordinate $\mathbf{x}_i$ (avoiding any grid zero-filling of missing sites).
   - A **Global Summary** branch aggregates system-wide context across all available measurements to maintain global field coherence and large-scale organization.
   - A **Local Branch** preserves fine-scale, high-frequency spatial fidelity around observed locations via adaptive neighborhood selection.
2. **Uncertainty-Driven Importance Weighting / Fusion:**
   - Rather than blending global predictions and local sensor tokens uniformly across space, GLU estimates an **uncertainty / importance field** that dynamically weights the contribution of global context versus local tokens at each query coordinate $\mathbf{x}_q$.
3. **Coordinate-Conditioned Decoding & Leader-Follower Dynamics:**
   - For spatial reconstruction, a coordinate-conditioned decoder queries the fused Global-Local latent representation at arbitrary coordinates $\mathbf{x}_q \in \Omega$.
   - For temporal forecasting, a hierarchical *Leader-Follower* dynamics module evolves the latent state forward in time without quadratic attention bottlenecks.

### 2.2 Alignment with Our Thesis Scope & Guardrails
As explicitly stated in `docs/proposal.md` (§4):
> *"Existing laboratory approaches, including GLU, will inform the choice of representation and architecture. GLU already addresses sparse reconstruction and forecasting, so the thesis will not claim that flexible sparse-field reconstruction is itself a new capability."*

How our thesis adapts and extends the GLU paradigm for combustion diagnostics:
- **Instantaneous Cross-Variable Reconstruction ($\mathbf{u} \rightarrow T$):** We focus on the **spatial reconstruction pathway** (predicting instantaneous temperature $T$ at all $128 \times 128$ query coordinates from sparse observations of velocity $\mathbf{u}$, with or without sparse $T$ anchors), rather than multi-step temporal rollouts.
- **What Month 1 Revealed About Global vs. Local Roles in Reacting Flows:**
  Our Step 1.5 feasibility checks (`04_feasibility_check_report.md`) demonstrated a sharp physical separation between **spatial structure** and **thermal scale**:
  1. **Local / Structural role of velocity observations:** Sparse velocity measurements $(v_i, v_j, v_k)$ locate the instantaneous shear layers, the V-shaped cold reactant jets, and the Inner Recirculation Zone (IRZ).
  2. **Global / Thermodynamic role of operating parameters & $T$ anchors:** Because identical velocity magnitudes correspond to burnt-gas temperatures that differ by $200\text{--}360\text{ K}$ across $\text{NH}_3/\text{CH}_4$ blends (`fig07`), velocity tokens alone cannot determine the global thermal level on unseen mixtures.
  3. **Why naive token pooling fails in Temperature-Assisted mode (`fig09b`):** When $2,457$ velocity scalars ($5\%$ coverage) are pooled equally with $K=4\text{--}32$ temperature sensors, the velocity tokens overwhelm the sparse temperature anchors. In a GLU-inspired architecture, **operating parameters $(\text{NH}_3\%, \phi)$ and sparse $T$ anchors must explicitly condition the Global Summary / thermal baseline**, while sparse velocity tokens drive the **Local / spatial fluctuation decoding**!

---

## 3. Analysis of Reference 2 — RealPDEBench (`arXiv:2601.01829`)

### 3.1 Physical & Numerical Setup of the Combustion Benchmark
- **Physical Configuration:** 3D swirl-stabilized $\text{NH}_3/\text{CH}_4/\text{air}$ co-firing burner exhibiting thermoacoustic oscillations and complex turbulence-chemistry interactions.
- **Numerical Simulations (Our Core Dataset):** Unsteady 3D Large Eddy Simulations (LES) solved with detailed chemical kinetics (**38 species, 184 elementary reactions**) and the Eddy Dissipation Concept (EDC) turbulence-chemistry interaction model, exported onto a 2D diametral slice ($128 \times 128$ grid, 15 physical channels, $\Delta t = 2.5 \times 10^{-4}\text{ s}$ at $4000\text{ Hz}$ for $2001$ frames per condition across 30 operating conditions).
- **Experimental Data:** High-speed line-of-sight $\text{OH}^*$ chemiluminescence intensity images ($4000\text{ fps}$).

### 3.2 Benchmark Task vs. Our Thesis Task
- **RealPDEBench Task:** Autoregressive **temporal forecasting** ($20 \rightarrow 20$ frames) of fields at half spatial resolution (`sub_s = 2`, i.e., $64 \times 64$), where all 30 numerical trajectories are pooled into `train_index_numerical.json` (`val` and `test` numerical indices are empty `[]`) to pretrain models before fine-tuning on real $\text{OH}^*$ sequences.
- **Our Thesis Task:** Instantaneous **cross-variable sparse reconstruction** at full **$128 \times 128$** resolution (`sub_s = 1`), holding out **entire operating conditions** into `val` (3), `test_interp` (3), and `test_extrap` (4) (`configs/splits/splits_interp_extrap.yaml`).
- **Physical Guardrail Confirmed:** In RealPDEBench, Channel 8 (`Mole_Fraction_of_OH`) is a 2D planar LES species mole fraction, whereas `observed` is line-of-sight integrated $\text{OH}^*$ chemiluminescence (or its numerical surrogate). Our pipeline strictly isolates numerical velocity (Ch 11–13) and numerical Temperature (Ch 10) and never equates $\text{OH}$ with $\text{OH}^*$.

---

## 4. Analysis of Reference 3 — RF-PINNs (*JCP* 2025, Yadav, Casel, & Ghani)

### 4.1 Methodology & Governing Equations
RF-PINNs reconstructs steady/mean and unsteady reacting flow fields from sparse PIV velocity or temperature measurements by penalizing PDE residuals of the low-Mach reactive Navier-Stokes conservation laws:
1. **Continuity:** $\partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0$
2. **Momentum:** $\rho (\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u}) = -\nabla p + \nabla \cdot \boldsymbol{\tau}$
3. **Species Transport:** $\rho (\partial_t Y_k + (\mathbf{u} \cdot \nabla) Y_k) + \nabla \cdot (\rho Y_k \mathbf{V}_k) = \dot{\omega}_k$
4. **Energy:** $\rho c_p (\partial_t T + (\mathbf{u} \cdot \nabla) T) = \dot{\omega}_T + \nabla \cdot (\lambda \nabla T) - \rho \nabla T \cdot \sum_k c_{p,k} Y_k \mathbf{V}_k$

To make optimization tractable without supervising all 38 species, RF-PINNs relies on steady-state assumptions, axisymmetric/2D closures, or reaction progress variable reductions.

### 4.2 Critical Assessment for Our Dataset (Why Pure Physics Residuals Are Restricted)
In `docs/proposal.md` (§4), we established a strict rule:
> *"Physics-based penalties will be considered only if their required variables and derivatives can be evaluated consistently."*

Our Month 1 audit (`02_data_audit_report.md` and `04_feasibility_check_report.md`) provides decisive empirical evidence on why naive PDE residuals from RF-PINNs **cannot** be applied blindly to RealPDEBench 2D slices:
1. **Missing 3D Out-of-Plane Derivatives ($\partial_z$ / $\partial_\theta$):** The RealPDEBench combustion grid is a single 2D diametral slice $(x, y)$ of a strongly 3D swirling turbulent flow ($v_k \sim \pm 15\text{ m/s}$ out of the plane). Instantaneous turbulent eddies cross the plane continuously, so the out-of-plane convective derivative $v_k \partial_z T$ and out-of-plane velocity divergence $\partial_z v_k$ are non-zero and unobserved in a single 2D slice!
   - This explains why our Step 1.5 Check 1 found that the in-plane 2D divergence $\nabla_{\text{2D}} \cdot \mathbf{u} = \partial_x v_j + \partial_y v_i$ has near-zero pointwise correlation with $T$ ($r = -0.014$): in a 3D swirling turbulent slice, $\partial_x v_j + \partial_y v_i \neq \nabla_{\text{3D}} \cdot \mathbf{u}$!
2. **Unobserved Chemical Source Term $\dot{\omega}_T$ at Inference Time:** In the energy equation, the dominant source term across the flame front is the chemical heat release rate $\dot{\omega}_T$ (Channel 1), which is **not observed** at inference time when only sparse velocity measurements are available.
3. **Conclusion on Physics Penalties:** Enforcing a naive 2D steady or unsteady energy/continuity PDE residual on instantaneous 3D swirl-slice snapshots would introduce an **inconsistent physical bias** (penalizing valid 3D out-of-plane transport and unknown chemical source terms). Therefore, our Month 2–3 methodology will focus on **data-driven, token-based neural reconstruction (informed by GLU) with explicit operating-condition conditioning**, considering lightweight spatial regularity or state-consistency priors only where variables are self-contained.

---

## 5. Proposed Model Specification for Month 2 (Informed by GLU & Step 1.5 Findings)

Based on the literature synthesis and our Step 1.5 baselines, we specify the exact model suite for Month 2 implementation:

| Tier | Model Name | Inputs at Inference | Role in Thesis |
|------|------------|---------------------|----------------|
| **Baseline 1a** | `GlobalTrainMeanPredictor` | None | Lower-bound reference (already implemented in Step 1.5) |
| **Baseline 1b** | `ParamConditionedMeanPredictor` | $(\text{NH}_3\%, \phi)$ only | Isolates variance explained by time-averaged condition state (implemented in Step 1.5) |
| **Baseline 2** | `ConditionedGappyPOD` ($R=32$) & `LocalHGB` | Sparse $(v_i, v_j, v_k)$ [+ optional $(\text{NH}_3\%, \phi)$ + optional $K$ sparse $T$] | Compact non-neural reconstruction baselines (implemented in Step 1.5) |
| **Neural Model (Month 2)** | **GLU-Inspired Coordinate-Conditioned Reconstructor (`GLURec`)** | Sparse measurement tokens $\{(x_m, y_m, c_m, v_m)\}_{m=1}^M$ + optional operating parameters $\mathbf{p} = (\text{NH}_3\%, \phi)$ | Primary deterministic neural reconstructor |

### Architectural Blueprint of the Neural Reconstructor (`GLURec`)
1. **Sparse Token Encoder (Measurement Encoder):**
   - Each observed datum $m \in \{1, \dots, M\}$ is represented as a tuple $(x_m, y_m, c_m, v_m)$, where $(x_m, y_m)$ are normalized physical coordinates (using Fourier / sinusoidal positional encodings), $c_m \in \{\text{vel}_i, \text{vel}_j, \text{vel}_k, T\}$ is a learnable variable-identity embedding, and $v_m$ is the train-normalized scalar value.
   - Missing grid points are **never** passed as zero-valued tokens.
2. **Dual-Branch Latent Assembly (Global Context + Local Spatial Features):**
   - **Local Branch:** Projects sparse velocity tokens onto a latent spatial feature grid or k-NN local attention neighborhood around query coordinates $(x_q, y_q)$ to capture sharp instantaneous shear-layer and V-jet positions.
   - **Global Branch & Operating-Parameter Conditioning Module:** Aggregates global summary tokens across the measurement set and integrates operating parameters $\mathbf{p} = (\text{NH}_3\%, \phi)$ via a modular conditioning toggle (`none` vs. `FiLM` feature-wise affine modulation / cross-attention).
   - **Separated Thermal Anchor Pathway (for `temperature_assisted` mode):** To prevent the $O(10^3)$ velocity tokens from drowning out $K \in \{1, \dots, 32\}$ sparse temperature sensors (the exact failure mode we diagnosed in Gappy POD in Step 1.5!), any available $T$ tokens are routed through a dedicated **Thermal Calibration Attention / Residual Head** in addition to the encoder.
3. **Coordinate-Conditioned Decoder:**
   - Queries the fused global-local representation at all $128 \times 128$ target coordinates $(x_q, y_q)$ to predict normalized temperature $\hat{T}(x_q, y_q)$.
