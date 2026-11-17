# Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion

**Author:** Enrico Bontempi  
**Host Group:** Deng Lab, Massachusetts Institute of Technology (MIT)  
**Thesis Proposal:** [docs/proposal.md](docs/proposal.md)  
**Month 1 Roadmap:** [roadmap_month1.md](roadmap_month1.md)  
**Month 2 Roadmap:** [roadmap_month2.md](roadmap_month2.md) — operationally authorized by Enrico (2026-11-16); Step 2.1 protocol concretized; supervisor technical ratification tracked separately  
**Month 2 Protocol:** [configs/splits/month2_frame_protocol.yaml](configs/splits/month2_frame_protocol.yaml), [configs/experiments/month2_pilot.yaml](configs/experiments/month2_pilot.yaml)  
**Baseline Consolidation Mission:** [roadmap_missione.md](roadmap_missione.md) — completed on `mission/baseline-consolidation` at `3cf7bf0`; report [docs/month2/mission_baseline_consolidation_report.md](docs/month2/mission_baseline_consolidation_report.md); integration pending  
**U-Net Mission:** [roadmap_unet.md](roadmap_unet.md) — Step 2.4; U0–U3 done on `mission/dense-unet`; U4 **runner ready, training not started** ([report](docs/month2/03_dense_velocity_diagnostic.md))
**Research Log:** [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)  
**Current Evidence & Open Decisions:** [docs/month2/00_context_and_decisions.md](docs/month2/00_context_and_decisions.md)

## Overview

This repository contains the codebase, experimental configurations, data audit protocols, evaluation suites, and documentation for the MSc Thesis:

> *How accurately can a neural model reconstruct an instantaneous temperature field from spatially incomplete observations of other physical variables, and how does its performance change at operating conditions excluded from training?*

### Repository Structure
- `src/combustion_recon/`: Core thesis package (data loaders, masks, evaluation metrics, non-neural baselines, and adapted neural models).
- `external/GLU_Reconstruction/`: **Untouched upstream reference implementation** of **GLU** (`arXiv:2603.26023`, [cosmos2w/GLU_Reconstruction](https://github.com/cosmos2w/GLU_Reconstruction)) provided by the Deng Lab authors. Kept strictly unmodified as the laboratory reference baseline; all thesis-specific adaptations live in `src/combustion_recon/`.
- `configs/`: Locked operating-condition splits (`splits_interp_extrap.yaml`), evaluation protocol, and observation layouts.
- `docs/month1/` & `overleaf_month1_report/`: Month 1 audit reports, baseline feasibility study, and LaTeX progress report.
- `docs/month2/` & `roadmap_month2.md`: Decision register and Month-2 plan. Step 2.1 protocol configs and Step 2.2 data pipeline (`Month2ReconDataset`, fit-only stats, manifests) are in place; Step 2.3 baseline results are available; their post-review consolidation is on `mission/baseline-consolidation` (`3cf7bf0`, not yet integrated at planning). `roadmap_unet.md` specifies the next dense single-frame diagnostic. Neural work has not started.
- `scripts/` & `tests/`: Reproducible execution scripts and automated `pytest` guardrails.

### Key Scientific Constraints
1. **No Data Leakage Across Conditions or Trajectories:** Entire operating conditions are withheld for validation/testing (interpolation vs. extrapolation). Correlated snapshots from a withheld trajectory never appear in training.
2. **Strict Separation of Observation Layouts & Operating Conditions:** Observation masks/layouts and operating-condition transfers are controlled independently.
3. **Training-Only Statistics:** Normalization and preprocessing statistics are fitted strictly on training splits.
4. **Separate Reporting:** Temperature-free (`u, v -> T`) and temperature-assisted (`u, v + sparse T -> T`) tasks are tracked and reported separately, broken down per operating condition.
5. **Physical Fidelity:** Numerical species concentrations are never equated with experimental chemiluminescence ($\text{OH} \neq \text{OH}^*$).
