# Step 1.6 — Literature Review & GLU Architecture Alignment

**Date:** 2026-11-15 (Updated after Step 1.5 methodological fix & multi-agent review)  
**Author:** Enrico Bontempi (MSc Thesis — Deng Lab, MIT)  
**Proposal Reference:** [`docs/proposal.md`](../proposal.md) §3, §4, §6 & Initial References [1–4]

---

## 1. Purpose of This Document

This note synthesizes the three core scientific references cited in the thesis proposal and connects their methodologies directly to the empirical findings from our Month 1 audit and corrected feasibility study (Steps 1.2–1.5):

1. **GLU (Deng Lab, MIT):** Wang, L., Chen, J., Tricard, N., Chen, Z., & Deng, S. *GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting.* `arXiv:2603.26023` (2026).
2. **RealPDEBench:** Hu, P., Feng, H., Liu, H., et al. *RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data.* `arXiv:2601.01829` / ICLR 2026 Oral (2026).
3. **RF-PINNs:** Yadav, V., Casel, M., & Ghani, A. *RF-PINNs: Reactive flow physics-informed neural networks for field reconstruction of laminar and turbulent flames using sparse data.* *Journal of Computational Physics*, 524, 113698 (2025).

---

## 2. Analysis of Reference 1 — GLU (`arXiv:2603.26023`, Deng Lab)

### 2.1 Exact Mathematical Formulation in the Original GLU Paper
Traditional neural operators and encoder-decoder architectures either compress sparse observations into a single **global latent bottleneck** (which smooths out sharp local gradients and high-frequency turbulent structures) or rely purely on **local neighborhood interpolation** (which fails when observations are sparse or when system-wide constraints govern the state).

**GLU (Global-Local-Uncertainty Fusion)** addresses sparse reconstruction (and temporal forecasting via a Leader-Follower dynamics module) through three exact architectural components (Eqs. 6–20 of `arXiv:2603.26023`):

1. **Co-Located Vector Token Embedding & Bidirectional Cross-Attention Encoder:**
   - In the original GLU formulation, each sparse sensor at location $\mathbf{x}_i \in \mathbb{R}^d$ ($i \in \{1, \dots, N\}$) observes **all $c$ physical channels simultaneously** ($\mathbf{u}_i \in \mathbb{R}^c$). Its initial token embedding combines learnable Fourier positional features with a linear value projection:
     $$\mathbf{h}_i = \text{LN}\!\left(W_p [\cos(2\pi B \mathbf{x}_i),\, \sin(2\pi B \mathbf{x}_i)]\right) + \text{LN}\!\left(W_v \mathbf{u}_i\right)$$
     where $B$ is a learnable frequency matrix (note that in original GLU, there is no `variable_id` token because all channels are co-located at each sensor $\mathbf{x}_i$).
   - Rather than $O(N^2)$ self-attention across all $N$ sensors, GLU uses $S$ learnable latent queries $L \in \mathbb{R}^{S \times D}$ ($S \ll N$) in a **two-stage bidirectional cross-attention block** ($O(NS)$ complexity):
     $$L' = \text{CrossAttn}(Q = L,\, K = H,\, V = H), \qquad Z_{\text{local}} = \text{CrossAttn}(Q = H,\, K = L',\, V = L') + H$$
   - The encoder retains **all $N$ sensor tokens** $Z_{\text{local}} = \{\mathbf{z}_i\}_{i=1}^N$ (enriched with global context) alongside a **single pooled global context vector** $\mathbf{z}_{\text{global}} \in \mathbb{R}^D$.

2. **Coordinate-Only Beta Importance Field $\varphi(\mathbf{x})$ & Warped k-NN Retrieval:**
   - GLU parameterizes a static spatial importance field $\varphi(\mathbf{x}) \sim \text{Beta}(\alpha(\mathbf{x}), \beta(\mathbf{x}))$ that depends **only on spatial coordinates $\mathbf{x}$** (via an MLP over coordinate embeddings).
   - For any target query coordinate $\mathbf{y}_j \in \Omega$, GLU retrieves the $k$ nearest observed sensors using an **importance-warped distance metric**:
     $$d_\varphi(\mathbf{y}_j, \mathbf{x}_i) = \frac{\|\mathbf{y}_j - \mathbf{x}_i\|_2}{\varphi(\mathbf{x}_i)^\gamma + \epsilon}$$
     Sensors located in high-importance regions (large $\varphi(\mathbf{x}_i)$) have a smaller effective distance and thus a wider spatial radius of influence.
   - The retrieved $k$ neighbor tokens are aggregated with distance and importance weights:
     $$\mathbf{f}_j = \sum_{i \in \mathcal{N}_k(\mathbf{y}_j)} w_{ij}\, \varphi(\mathbf{x}_i)\, \text{Proj}(\mathbf{z}_i)$$

3. **Concatenative Global-Local Decoder & Heteroscedastic Uncertainty Supervision:**
   - At each query coordinate $\mathbf{y}_j$, the local feature $\mathbf{f}_j$, coordinate encoding $\mathbf{p}(\mathbf{y}_j)$, and global token $\mathbf{z}_{\text{global}}$ are fused via **concatenation inside an MLP**:
     $$\mathbf{h}_j = \text{MLP}\!\left(\left[\mathbf{f}_j + \mathbf{p}(\mathbf{y}_j) \;;\; \mathbf{z}_{\text{global}}\right]\right)$$
   - Two output heads predict the field mean $\hat{\mathbf{u}}(\mathbf{y}_j)$ and local predictive variance $\hat{\sigma}^2(\mathbf{y}_j)$, trained with a heteroscedastic Gaussian negative log-likelihood (NLL) loss plus an alignment penalty linking $\varphi(\mathbf{x})$ to high-variance / high-gradient regions.

### 2.2 What Original GLU Tested on Combustion vs. Our Thesis Problem
- **Combustion in Original GLU (`arXiv:2603.26023`):** Evaluated on a **single fixed operating condition** (stoichiometric $\text{CH}_4/\text{air}$ combustion over a backward-facing step in OpenFOAM with 3 channels: $\text{CO}$, $T$, velocity), where the $1\%$ sparse sensors measured **all channels simultaneously (including Temperature $T$)** at every sensor location.
- **Why Original GLU Cannot Be Applied As-Is to Our Diagnostic Problem:**
  1. **Cross-Variable Unobserved Target ($\mathbf{u} \rightarrow T$):** In our primary `temperature_free` regime, temperature $T$ is **completely unobserved** at inference time ($0\%$ $T$ sensors); in our `temperature_assisted` regime, velocity sensors ($M_u \sim 10^2\text{--}10^3$ optical PIV vectors) and temperature sensors ($K \in \{1, \dots, 32\}$ thermocouples) are **non-colocated and extreme in cardinality ratio ($M_u \gg K$)**.
  2. **Multi-Condition Parametric Transfer ($\text{NH}_3\%, \phi$):** Across the 30 $\text{NH}_3/\text{CH}_4$ operating conditions, our Step 1.5 Check 1 (`fig07`) proved that identical velocity magnitudes map to temperatures differing by $200\text{--}360\text{ K}$, and the flame brush shifts spatially with $(\text{NH}_3\%, \phi)$—meaning a static coordinate-only importance field $\varphi(\mathbf{x})$ and unconditioned global token $\mathbf{z}_{\text{global}}$ are insufficient for unseen mixtures.
  3. **Scalar Value Normalization Pitfall:** Applying zero-bias $\text{LN}(W_v v_m)$ to a 1D scalar observation $v_m \in \mathbb{R}^1$ would collapse its magnitude to $\text{sign}(v_m) \in \{-1, +1\}$; scalar token values must therefore be embedded via an MLP without 1D LayerNorm collapse.

---

## 3. Analysis of Reference 2 — RealPDEBench (`arXiv:2601.01829`)

### 3.1 Physical & Numerical Setup of the Combustion Benchmark
- **Physical Configuration:** 3D swirl-stabilized $\text{NH}_3/\text{CH}_4/\text{air}$ co-firing burner exhibiting thermoacoustic oscillations and turbulence-chemistry interactions across 30 operating conditions ($\text{NH}_3 \in \{0, 20, 40, 60, 80\}\%$, $\phi \in [0.75, 1.30]$).
- **Numerical Simulations (Our Core Dataset):** Unsteady 3D Large Eddy Simulations (LES) solved with a **reduced chemical mechanism (38 species, 184 reactions)** and the Eddy Dissipation Concept (EDC) turbulence-chemistry interaction model, exported onto a 2D diametral slice ($128 \times 128$ grid, 15 physical channels, $\Delta t = 2.5 \times 10^{-4}\text{ s}$ at $4000\text{ Hz}$ for $2001$ frames per condition).
- **Experimental Modality & Surrogate Channel:** High-speed line-of-sight $\text{OH}^*$ chemiluminescence images ($4000\text{ fps}$). In RealPDEBench, a trained U-Net surrogate maps numerical states to the experimental $\text{OH}^*$ observation space (stored in the `observed` array of the Arrow files).

### 3.2 Benchmark Task vs. Our Thesis Task
- **RealPDEBench Task:** Autoregressive **temporal forecasting** ($20 \rightarrow 20$ frames) at half spatial resolution (`sub_s = 2`, $64 \times 64$), where all 30 numerical trajectories are pooled into `train_index_numerical.json` (`val` and `test` numerical index files are empty `[]`) to pretrain temporal operators before fine-tuning on real $\text{OH}^*$.
- **Our Thesis Task:** Instantaneous **cross-variable sparse spatial reconstruction** at full **$128 \times 128$** resolution (`sub_s = 1`), holding out **entire operating conditions** into `val` (3), `test_interp` (3), and `test_extrap` (4) (`configs/splits/splits_interp_extrap.yaml`).
- **Physical Guardrail Confirmed:** Channel 8 (`Mole_Fraction_of_OH`) is a 2D planar LES species mole fraction, whereas `observed` is line-of-sight $\text{OH}^*$ chemiluminescence. Our pipeline strictly isolates numerical velocity (Ch 11–13) and numerical Temperature (Ch 10) and never equates $\text{OH}$ with $\text{OH}^*$.

---

## 4. Analysis of Reference 3 — RF-PINNs (*JCP* 2025, Yadav, Casel, & Ghani)

### 4.1 What RF-PINNs Actually Does vs. Why We Do Not Use PDE Residual Losses
A natural question is: *Why is a PINNs paper (which puts Navier-Stokes PDE residuals into the loss function) in our bibliography if our thesis does not train a PINN?*
1. **Why RF-PINNs is cited:** RF-PINNs is the most recent state-of-the-art paper (2025) attempting **cross-variable reconstruction ($\mathbf{u} \rightarrow T$) in reacting flames from sparse velocity data**. Citing and analyzing it proves why existing physics-informed optimization methods cannot solve our problem on instantaneous turbulent 2D slices.
2. **How RF-PINNs works:** RF-PINNs reconstructs **time-averaged (steady) mean fields** (or laminar flames) by optimizing a coordinate MLP per flame condition using low-Mach reactive Navier-Stokes residuals, reducing the chemistry to a single progress variable $\theta = (T - T_u)/(T_b - T_u)$ whose source term is closed via Cantera lookup tables or an Eddy Break-Up (EBU) mean closure.
3. **Three Physical Reasons Why Naive PDE Residual Losses Fail on Our Dataset (`docs/proposal.md` §4 Guardrail):**
   - **Concentrated Thin-Front Dilatation + 3D Out-of-Plane Swirl:** In a low-Mach flame, thermal dilatation ($\nabla \cdot \mathbf{u} = -\frac{1}{\rho}\frac{D\rho}{Dt}$) is non-zero **only inside the thin reaction sheet** where density drops across the flame front, and is nearly zero across the $\sim 90\%$ burnt-gas volume. Moreover, on a 2D diametral slice of a strongly 3D swirling burner ($v_k \sim \pm 15\text{ m/s}$), out-of-plane derivatives ($\partial_z v_k$, $v_k \partial_z T$) are unobserved. This explains why our Step 1.5 Check 1 found near-zero pointwise correlation between planar $\nabla_{\text{2D}} \cdot \mathbf{u}$ and $T$ ($r = -0.014$).
   - **Unobserved Instantaneous Chemical Heat Release $\dot{\omega}_T$:** In the instantaneous energy equation, the chemical source term $\dot{\omega}_T$ (Ch 1) is unobserved at inference time when only sparse velocity sensors are given.
   - **Per-Condition Optimization vs. Instantaneous Multi-Condition Generalization:** RF-PINNs requires re-optimizing PDE residuals for a specific flame, whereas we require fast feed-forward inference across arbitrary instantaneous turbulent snapshots and unseen $(\text{NH}_3\%, \phi)$ mixtures.

---

## 5. Candidate Neural Architectures for Month 2–3 (Informed by GLU & Step 1.5 Corrected Findings)

### 5.1 What Step 1.5 (Corrected) Proved About Level vs. Topology
After fixing the MAP regularization and modality weighting in Step 1.5 (`04_feasibility_check_report.md`):
- **Level Error (Bias) is Solved by Conditioning or $K$ Anchors:** On `test_extrap`, adding $K=32$ sparse temperature anchors (`GappyPOD Cond + Anchor Offset` or `ParamMean + Anchor Offset`) slashes global thermal bias from **$+86\text{ K}$ down to $15\text{ K}$** (and **$+0.7\text{ K}$** for `ParamMean + AO`), reducing full-field RMSE from $175\text{ K} \rightarrow 136\text{ K}$.
- **Flame-Front Topology & High-T Shape Are NOT Solved by Linear Baselines:** Even when thermal bias is eliminated ($K=32$), `test_extrap` **High-T RMSE stays locked at $204\text{ K}$**, **Jet/Flame-Front RMSE stays at $\sim 370\text{ K}$**, and **High-T IoU remains $0.16\text{--}0.22$** (`fig08`–`fig10`). Linear POD modes learned on training mixtures cannot deform the V-jet shear layers and flame brush into new chemical regimes.

### 5.2 Three Structured Neural Architecture Candidates for Month 2–3
Note on naming: The names below (**Candidate 1, 2, 3**) are **descriptive working names created for our thesis design** to clearly distinguish the three progressive ways of adapting the lab's **GLU** backbone (`arXiv:2603.26023`) to our heterogeneous cross-variable combustion problem:

| Candidate Architecture (Working Name) | How It Adapts GLU (`arXiv:2603.26023`) | Target Role in Thesis |
|---|---|---|
| **Candidate 1: Direct Multi-Modal GLU Adaptation (`GLU-Adapted`)** | • Replaces GLU's co-located vector input $\mathbf{u}_i$ with heterogeneous scalar/vector tokens carrying an explicit `variable_id` embedding and MLP scalar value encoder (avoiding 1D LayerNorm collapse).<br>• Injects operating parameters $\mathbf{p} = (\text{NH}_3\%, \phi)$ via **FiLM** (Feature-wise Linear Modulation) into the global latent token $\mathbf{z}_{\text{global}}$, decoder MLP, and a **parameter-conditioned importance field** $\varphi(\mathbf{x}, \mathbf{p}) \sim \text{Beta}(\alpha(\mathbf{x}, \mathbf{p}), \beta(\mathbf{x}, \mathbf{p}))$.<br>• **Modality-Separated k-NN Retrieval:** Queries the $k$ nearest velocity tokens $\mathcal{N}_k^{(u)}(\mathbf{y}_j)$ separately from all $K$ available temperature anchors $\mathcal{O}_T$ so $K \ll M_u$ anchors are never dropped by top-$k$ spatial pruning. | Direct lab baseline adaptation (Month 2 baseline neural model). |
| **Candidate 2: Two-Stage Residual-Calibrated GLU (`GLU-Residual`)** | • **Stage 1 (Temperature-Free Backbone):** A FiLM-conditioned GLU encoder-decoder processes sparse velocity tokens $\mathcal{O}_u$ and $\mathbf{p} = (\text{NH}_3\%, \phi)$ to predict the temperature-free field $\hat{T}_0(\mathbf{x}) = m_\theta(\mathbf{x}, \mathbf{p}) + r_\theta(\mathbf{x} \mid \mathcal{O}_u, \mathbf{p})$ and local variance $\hat{\sigma}_0^2(\mathbf{x})$.<br>• **Stage 2 (Residual Anchor Assimilation for $K > 0$):** Evaluates exact anchor residuals $e_k = T(\mathbf{x}_k) - \hat{T}_0(\mathbf{x}_k)$ at the $K$ thermocouples and passes $\{(x_k, y_k, e_k)\}_{k=1}^K$ through a lightweight residual cross-attention head that outputs a global level offset $\hat{b}$ plus a zero-mean spatial front deformation $\hat{d}(\mathbf{x})$. Guarantees exact reduction to $\hat{T}_0(\mathbf{x})$ when $K=0$ and prevents $M_u$ velocity tokens from diluting $K$ anchors. | Recommended primary Month 2 neural architecture (cleanly unifies temperature-free and temperature-assisted tracks). |
| **Candidate 3: Uncertainty-Weighted Low-Rank Neural Assimilation (`GLU-Assimilator`)** | • **Neural Prior + Learned Correction Basis:** From sparse velocity $\mathcal{O}_u$ and parameters $\mathbf{p}$, the GLU-inspired decoder outputs: (1) a prior temperature field $m_\theta(\mathbf{x})$, (2) predictive uncertainty $\hat{\sigma}^2(\mathbf{x})$ (via GLU's heteroscedastic head), and (3) a small snapshot-dependent basis of $r$ spatial correction functions $\mathbf{b}_\theta(\mathbf{x}) \in \mathbb{R}^r$ ($r \approx 4\text{--}16$, with $b_1(\mathbf{x}) \equiv 1$ for level offset and $b_{2\dots r}(\mathbf{x})$ concentrated on the V-jet shear layers and flame brush; optionally scaled by theoretical adiabatic flame temperature $T_{\text{ad}}(\mathbf{p})$).<br>• **Closed-Form Differentiable Anchor Update ($K > 0$):** Given $K$ sparse temperature anchors with residuals $\mathbf{e} = \mathbf{T}_K - \mathbf{m}_K$, updates the basis coefficients $\hat{\mathbf{a}} \in \mathbb{R}^r$ in closed form:<br>$$\hat{\mathbf{a}} = \left(\mathbf{\Lambda}^{-1} + B_K^\top \mathbf{R}^{-1} B_K\right)^{-1} B_K^\top \mathbf{R}^{-1} \mathbf{e}, \quad \hat{T}(\mathbf{x}) = m_\theta(\mathbf{x}) + \mathbf{b}_\theta(\mathbf{x})^\top \hat{\mathbf{a}}$$<br>where $\mathbf{R} = \text{diag}(\hat{\sigma}^2(\mathbf{x}_1), \dots, \hat{\sigma}^2(\mathbf{x}_K))$ weights each thermocouple by GLU's predicted local variance (down-weighting anchors that fall inside noisy turbulent shear layers!). | Targeted methodological contribution for Month 3 (`docs/proposal.md` §6 Month 3 objective). |
