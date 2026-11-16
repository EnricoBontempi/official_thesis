# Month 1 Detailed Roadmap — Data Audit, Feasibility Checks & Problem Confirmation

**Project:** Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion  
**Author:** Enrico Bontempi (MSc Thesis — Deng Lab, MIT)  
**Reference Proposal:** [docs/proposal.md](docs/proposal.md)

**Transition note (2026-11-16):** The Month 1 artifacts and corrected baseline runs are available. The supervisor sign-off component of Step 1.6 remains open; completion marks below do not establish technical approval. The latest evidence, known discrepancies in older summaries, and proposed next steps are tracked in [the Month 2 context](docs/month2/00_context_and_decisions.md) and [roadmap_month2.md](roadmap_month2.md).

---

## Purpose of Month 1

Month 1 establishes the empirical and methodological foundation for the entire six-month thesis. **No deep neural networks are trained at scale before completing this phase.** Instead, Month 1 answers four decisive questions:

1. **What is actually inside the RealPDEBench Combustion dataset?** (Exact operating conditions, variable definitions, coordinate grids, trajectory lengths, storage footprint, and data integrity.)
2. **How do we split the dataset without leakage?** (Which operating conditions define *Train*, *Validation*, *Test-Interpolation*, and *Test-Extrapolation*, and how is the *high-temperature region* formally defined?)
3. **Do sparse velocity observations contain usable information about the temperature field?** (Inexpensive physical, kinematic, and linear/modal checks of $u, v \rightarrow T$ within and across operating conditions, plus the effect of adding a few sparse $T$ anchors.)
4. **What exact baseline and initial neural architecture (informed by GLU) will we build in Month 2?** (Documented and agreed upon with the supervisors.)

---

## Core Scientific Guardrails (Enforced Across All Steps)

- **Numerical Target Scope:** Core experiments use numerical velocity observations and numerical temperature targets ($128 \times 128$ grid). Experimental $\text{OH}^*$ chemiluminescence images are **never** treated as experimental temperature maps, and numerical $\text{OH}$ species concentration is **never** equated with $\text{OH}^*$ intensity.
- **Strict Operating-Condition Withholding:** Entire operating conditions (all trajectories and correlated time snapshots belonging to that condition) are withheld for validation and testing. No snapshot from a withheld trajectory may ever appear in training.
- **Orthogonal Evaluation Axes:** Changes in *observation layout/sparsity* and changes in *operating condition* (interpolation vs. extrapolation) are controlled and evaluated separately.
- **Training-Only Statistics:** All normalization and preprocessing statistics (mean, standard deviation, min/max, POD modes) are fitted **exclusively on the training operating conditions**.
- **Missing Values Are Not Physical Zeros:** Observation representations and masks must never conflate unobserved grid locations with a physical value of $0$.
- **Separate Task Reporting:** **Temperature-Free** ($u, v \rightarrow T$) and **Temperature-Assisted** ($u, v + \text{sparse } T \rightarrow T$) configurations are always tracked and reported in separate tables/plots.
- **Disaggregated Metrics:** Every evaluation reports metrics **per operating condition** (never only as a pooled average across conditions) and covers all four required error dimensions:
  1. Full-field reconstruction error (e.g., normalized L2 / RMSE / MAE),
  2. Spatial bias (systematic spatial over-/under-prediction maps),
  3. High-temperature region error (localization, extent/IoU, and conditional RMSE inside the flame zone),
  4. Temperature-level error (peak temperature error, high-percentile error, and spatially integrated thermal level bias).

---

## Step-by-Step Execution Plan

### Step 1.1 — Repository Scaffolding & Environment Setup
- **Status:** `[x] Completed`
- **Objective:** Create a clean, modular Python package structure (`combustion_recon`) and reproducible environment so every subsequent script writes standardized manifests, tables, and plots.
- **Concrete Actions:**
  1. Create the repository directory skeleton:
     - `data/{raw,interim,processed,manifests}`
     - `configs/{splits,masks,models,experiments}`
     - `src/combustion_recon/{data,models,training,evaluation,utils}`
     - `scripts/`
     - `results/{tables,figures}`
     - `docs/month1/`
     - `tests/`
  2. Create `pyproject.toml` with core scientific dependencies (`numpy`, `scipy`, `h5py`, `torch`, `huggingface_hub`, `pyyaml`, `matplotlib`, `scikit-learn`, `pytest`) and install package in editable mode (`pip install -e .`).
  3. Run a quick hardware/storage check (`df -h`, GPU check) to verify available local disk capacity before downloading data.
- **Deliverables:**
  - Installable `pyproject.toml` and `src/combustion_recon/__init__.py`
  - Directory tree initialized with `.gitkeep` / `README.md` files

---

### Step 1.2 — Remote Reconnaissance of RealPDEBench (Pre-Download Inspection)
- **Status:** `[x] Completed`
- **Objective:** Inspect the structure, file sizes, licensing, and loader conventions of the RealPDEBench Combustion dataset *before* committing to a multi-gigabyte data transfer.
- **Concrete Actions:**
  1. Query the HuggingFace Hub API (`AI4Science-WestlakeU/RealPDEBench`) to list all files associated with the `combustion` subset, recording exact paths, file formats (`.h5`, `.npy`, `.pt`, etc.), and byte sizes.
  2. Inspect the official RealPDEBench GitHub repository (`https://github.com/AI4Science-WestlakeU/RealPDEBench`) and documentation (`https://realpdebench.github.io/datasets/combustion/`) to extract:
     - How operating conditions are named and parameterized (e.g., $\text{NH}_3/\text{CH}_4$ fuel fraction, equivalence ratio $\phi$, inlet velocity/swirl parameters),
     - Exact order and physical meaning of tensor channels,
     - Dataset license and citation requirements.
  3. Determine the exact storage footprint of the numerical combustion dataset versus the experimental $\text{OH}^*$ dataset.
- **Deliverables:**
  - `scripts/01_remote_recon.py`: Script that queries HF metadata and writes the file inventory.
  - `data/manifests/remote_file_inventory.json`: Complete list of remote combustion files and sizes.
  - `docs/month1/01_remote_reconnaissance.md`: Summary of dataset structure, channel specs from official docs, licensing, and download plan.

---

### Step 1.3 — Pilot Data Transfer & Trajectory Deep Audit
- **Status:** `[x] Completed` (pilot + full 30-shard numerical staging & audit)
- **Objective:** Download representative trajectories first to verify tensor layouts and physical consistency, then stage the usable numerical combustion trajectories and produce a complete inventory of operating conditions.
- **Concrete Actions:**
  1. **Pilot download:** Download 1–2 numerical trajectories into `data/raw/` and inspect their internal structure (HDF5 keys/attributes or array shapes).
  2. **Verify physical & numerical conventions:**
     - Spatial grid dimensions ($128 \times 128$), coordinate axes $(x, y \text{ or } r, z)$, physical domain extent, and burner nozzle orientation.
     - Channel inventory: identify exact indices for axial/radial velocity components $(u, v)$, temperature $(T)$, and chemical species; verify units (e.g., $\text{m/s}$ for velocity, $\text{K}$ for temperature, mass/mole fractions for species).
     - Temporal structure: number of trajectories per operating condition, snapshots per trajectory ($N_t$), time step $\Delta t$, and temporal correlation decay (autocorrelation time $\tau_c$ of $T$ and $u, v$).
  3. **Data quality check:** Scan for `NaN`/`Inf` values, corrupted snapshots, constant/dead channels, or unphysical temperature spikes/drops.
  4. **Full numerical staging:** Once the pilot check succeeds and disk budget is confirmed, download the full set of numerical combustion operating conditions and compute SHA256/size checksums.
  5. **Visual inspection:** Plot instantaneous snapshots, temporal mean fields, and standard deviation fields of $(u, v, |\mathbf{u}|, T)$ for each operating condition.
- **Deliverables:**
  - `scripts/02_audit_trajectories.py`: Automated audit & visualization script.
  - `data/manifests/operating_conditions.yaml`: Authoritative table mapping every operating condition ID to its physical parameters (fuel composition, equivalence ratio, trajectory count, snapshot count, value ranges).
  - `results/figures/month1_audit/`: Multi-panel visualizations of representative snapshots, mean fields, and fluctuation intensity across conditions.
  - `docs/month1/02_data_audit_report.md`: Complete data audit report.

---

### Step 1.4 — Operating-Condition Splits, Observation Masks & Evaluation Protocol Lock
- **Status:** `[x] Completed`
- **Objective:** Lock the experimental protocol before any model development so that all future baseline and neural experiments are strictly comparable and leakage-free.
- **Concrete Actions:**
  1. **Map the operating-condition parameter space:** Plot the available operating conditions in parameter space (e.g., equivalence ratio $\phi$ vs. $\text{NH}_3/\text{CH}_4$ ratio).
  2. **Define and lock the Operating-Condition Split (`configs/splits/splits_interp_extrap.yaml`):**
     - **Training Set:** Operating conditions used to fit normalizers and train models.
     - **Validation Set:** Entire operating condition(s) withheld for hyperparameter tuning and model selection.
     - **Test Set — Interpolation:** Entire operating condition(s) withheld inside the parameter range spanned by the training set.
     - **Test Set — Extrapolation:** Entire operating condition(s) withheld outside/at the frontier of the training parameter range (e.g., extreme fuel blend or equivalence ratio).
  3. **Establish the fixed High-Temperature Region definition:**
     - Analyze the temperature histograms across all training conditions (identifying the bimodal/multi-modal separation between unburnt reactants and burnt products/reaction zone).
     - Fix a primary physical threshold $T_{\text{high}}$ (and check sensitivity to a normalized thermal progress variable threshold) that defines the high-temperature region $\Omega_{\text{high}} = \{(x, y) \mid T(x, y) \ge T_{\text{high}}\}$.
  4. **Define standardized Observation Layouts (`configs/masks/`):**
     - *Random uniform sparse sampling* at fixed sparsity levels (e.g., $0.5\%, 1\%, 2\%, 5\%, 10\%, 20\%$ of the $128 \times 128$ grid).
     - *Structured probe layouts* (e.g., 1D axial/radial line cuts or sparse sensor arrays).
     - *Localized occlusion / missing-block masks* (e.g., missing flame core or shear-layer box).
     - *Task configurations:*
       - **Task A (Temperature-Free):** Sparse $(u, v)$ only $\rightarrow$ full $T$.
       - **Task B (Temperature-Assisted):** Sparse $(u, v)$ + $K$ sparse temperature sensors ($K \in \{1, 4, 9, 16, 32\}$) $\rightarrow$ full $T$.
  5. **Write automated anti-leakage tests (`tests/test_no_data_leakage.py`):**
     - Assert zero overlap of condition IDs and trajectory IDs across Train, Val, Test-Interp, and Test-Extrap.
     - Assert normalization statistics match strictly the training split.
- **Deliverables:**
  - `configs/splits/splits_interp_extrap.yaml`
  - `configs/masks/observation_layouts.yaml`
  - `src/combustion_recon/evaluation/metrics.py` (implementing all 4 required metric families + per-condition aggregation)
  - `tests/test_no_data_leakage.py` & `tests/test_metrics.py`
  - `docs/month1/03_validation_and_splits_protocol.md`

---

### Step 1.5 — Inexpensive Feasibility Checks: Does Velocity Carry Temperature Information?
- **Status:** `[x] Completed`
- **Objective:** Quantify how much thermal information is physically and statistically recoverable from velocity observations $(u, v)$ alone, where that relationship breaks down across operating conditions, and how much sparse $T$ anchors help.
- **Concrete Actions:**
  1. **Kinematic & Physical Coupling Analysis (Full-Grid Sanity Check):**
     - Compute spatial derivatives of velocity on the $128 \times 128$ grid:
       - 2D planar divergence / dilatation: $\nabla_{\text{2D}} \cdot \mathbf{u} = \partial_x u + \partial_y v$ (direct kinematic signature of gas expansion across the flame front),
       - Vorticity: $\omega_z = \partial_x v - \partial_y u$,
       - Velocity magnitude $|\mathbf{u}|$ and strain-rate magnitude.
     - Compute pointwise and spatial cross-correlations and mutual information between $\{u, v, |\mathbf{u}|, \nabla_{\text{2D}} \cdot \mathbf{u}, |\omega_z|\}$ and $T$ within each condition and across conditions.
  2. **Inexpensive Predictive Checks (Fast Non-Deep Baselines):**
     - **Check 1 — Mean-Field Predictors:**
       - Global training mean field $\bar{T}_{\text{train}}(x, y)$ evaluated on seen and unseen conditions.
       - Condition-interpolated mean field $\bar{T}(x, y \mid \text{params})$ using operating parameters only (zero instantaneous measurements) to isolate how much variance comes from time-averaged flame shape vs. instantaneous turbulent fluctuations.
     - **Check 2 — Local / Patch Regression ($u, v \rightarrow T$):**
       - Fit fast Ridge / Kernel / LightGBM or small MLP regressors mapping local velocity (and spatial coordinates $x, y$) to $T(x, y)$, both **without** and **with** operating-condition parameters.
       - Compare performance on **seen conditions (held-out trajectories)** vs. **withheld conditions (interpolation & extrapolation)**.
     - **Check 3 — Sparse Observation Reconstruction (Gappy POD / Linear Modal Estimator):**
       - Compute POD basis modes on training snapshots.
       - Reconstruct $T$ from sparse velocity measurements (**Temperature-Free**) vs. sparse velocity + $K$ temperature measurements (**Temperature-Assisted**).
  3. **Diagnose the Ambiguity Mechanism:**
     - Specifically test whether two different operating conditions (e.g., different $\text{NH}_3/\text{CH}_4$ ratios or $\phi$) can exhibit similar velocity fields while having systematically different adiabatic flame temperatures or flame lift-off heights.
- **Deliverables:**
  - `scripts/03_feasibility_checks.py`
  - `results/tables/month1_feasibility_summary.csv` (per-condition metrics for all inexpensive checks)
  - `results/figures/month1_feasibility/` (joint PDFs, dilatation vs. $T$ maps, error vs. sparsity curves for temperature-free vs. temperature-assisted)
  - `docs/month1/04_feasibility_check_report.md`

---

### Step 1.6 — Literature Synthesis, GLU Architecture Alignment & Supervisor Sign-Off
- **Status:** `[x] Completed (2026-11-14)` — `docs/month1/05_literature_and_glu_notes.md`, `docs/month1/06_supervisor_review_brief.md`
- **Objective:** Translate the empirical findings from Steps 1.2–1.5 and the laboratory's prior work (`GLU`) into a concrete, supervisor-approved model design for Month 2.
- **Concrete Actions:**
  1. **Focused technical review of the 3 anchor papers:**
     - **GLU** (*Wang et al., Deng Lab, 2026*, `arXiv:2603.26023`): Document how GLU encodes irregular/sparse observations and fuses global-local representations, and specify how our deterministic reconstructor adapts this paradigm for cross-variable reconstruction ($u, v \rightarrow T$) and operating-parameter conditioning.
     - **RealPDEBench** (*Hu et al., 2026*, `arXiv:2601.01829`): Document benchmark protocols and any baseline numbers reported on the combustion dataset.
     - **RF-PINNs** (*Yadav et al., 2025*, *JCP*): Document what variables/governing equations are required for reactive-flow physics losses and verify whether RealPDEBench provides sufficient variables/derivatives or if purely data-driven + parameter-conditioned reconstruction is more rigorous.
  2. **Specify the Month 2 Model Suite:**
     - **Baseline 1:** Training Mean-Temperature-Field Predictor.
     - **Baseline 2:** Best compact non-neural baseline identified in Step 1.5 (e.g., Gappy POD or Spatial Kernel/RBF Regression).
     - **Initial Neural Reconstructor:**
       - *Input token representation:* $\mathbf{m}_i = (x_i, y_i, \text{var\_id}_i, v_i)$ for each available measurement $i \in \{1, \dots, M\}$ (never passing masked pixels as zeros).
       - *Measurement Encoder:* Processes the variable-length set/grid of observed tokens into a latent spatial/global representation.
       - *Operating-Condition Conditioning Module:* Modular toggle (None vs. FiLM / Cross-Attention / Token-Append of fuel composition & equivalence ratio).
       - *Coordinate-Conditioned Decoder:* Predicts $\hat{T}(x_q, y_q)$ at arbitrary query coordinates $(x_q, y_q)$ on the $128 \times 128$ grid.
  3. **Compile the Month 1 Supervisor Review Brief:**
     - A clean, 3–5 page executive summary (Markdown/PDF) presenting:
       - Dataset audit & split table,
       - Feasibility check results ($u, v \rightarrow T$ vs. $u, v + \text{sparse } T \rightarrow T$),
       - Proposed baseline & neural architecture diagram for Month 2 sign-off.
- **Deliverables:**
  - `docs/month1/05_literature_and_glu_notes.md`
  - `docs/month1/06_supervisor_review_brief.md`

---

## Summary Checklist for Month 1 Completion

- [x] **Step 1.1:** Repo directory structure, `pyproject.toml`, and local environment verified.
- [x] **Step 1.2:** Remote inspection of `RealPDEBench` combustion files, sizes, and channel specs completed.
- [x] **Step 1.3:** All 30 numerical shards staged & audited; `operating_conditions.yaml`, `month1_30_conditions_inventory.csv`, `fig05`, and `docs/month1/02_data_audit_report.md` written.
- [x] **Step 1.4:** Operating-condition splits (`Train` / `Val` / `Test-Interp` / `Test-Extrap`), high-temperature region threshold, observation masks, and anti-leakage pytest suite locked.
- [x] **Step 1.5:** Inexpensive velocity-to-temperature feasibility checks (kinematic proxies, local regression, Gappy POD, temperature-free vs. temperature-assisted) executed and documented.
- [x] **Step 1.6:** GLU architecture review completed and Month 1 Supervisor Review Brief ready for Deng Lab meeting.
