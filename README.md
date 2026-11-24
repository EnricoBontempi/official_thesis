# Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion

**Author:** Enrico Bontempi  
**Host Group:** Deng Lab, Massachusetts Institute of Technology (MIT)  
**Thesis Proposal:** [docs/proposal.md](docs/proposal.md)  
**Current priority (24 Nov 2026):** [roadmaptests.md](roadmaptests.md) — robustness to non-ideal velocity observations (audit → frozen-model evaluation → conditional retraining). Roadmaps A/B/C are **suspended, not executed**. No job launched by this planning update.  
**Month 1 Roadmap:** [roadmap_month1.md](roadmap_month1.md)  
**Month 2 Roadmap:** [roadmap_month2.md](roadmap_month2.md) — updated plan approved by Enrico on **2026-11-20**: C149 multi-seed U-Net/GLU comparison → matched K sweep of mean/median corrections → learned T-input experiment → final review. Supervisor ratification tracked separately; Month 2 remains open  
**Month 2 Protocol:** [configs/splits/month2_frame_protocol.yaml](configs/splits/month2_frame_protocol.yaml), [configs/experiments/month2_pilot.yaml](configs/experiments/month2_pilot.yaml)  
**Baseline Consolidation Mission:** [roadmap_missione.md](roadmap_missione.md) — completed on `mission/baseline-consolidation` at `3cf7bf0`; report [docs/month2/mission_baseline_consolidation_report.md](docs/month2/mission_baseline_consolidation_report.md); integration pending  
**U-Net dense mission (Step 2.4):** [roadmap_unet.md](roadmap_unet.md) — **U0–U7 closed** on `mission/dense-unet` (p_on/p_off seed 0 + snapshot controls; further seeds deferred). Report: [docs/month2/03_dense_velocity_diagnostic.md](docs/month2/03_dense_velocity_diagnostic.md). Results: `results/month2_dense_unet/`. Month 2 **not** complete (S7 confirmation and Steps 2.7–2.9 open).  
**Sparse mission (Steps 2.5–2.6):** [roadmapsparse.md](roadmapsparse.md) — S0–S6 screening delivered at STOP E; S7 closed at STOP F (temperature-free, 21 Nov 2026; report `docs/month2/10_s7_c149_model_comparison.md`, index `results/month2/s7_c149_comparison/package/INDEX.md`); Month 2 remains open. **Approved S7:** C149 (149 frames/condition, 2980 total), TRAIN velocity 5%, masked U-Net and adapted deterministic GLU × p_on/off × seeds 0/1/2. **10 new trainings + 2 existing U-Net C149 seed-0 runs**; preparation and submission are the next Sonnet assignment. No S7 job submitted by this documentation update. No arbitrary scientific hour cap.
**Completed diagnostic studies:** [Opus roadmap](roadmapopus.md) (data/training variability; O6 adopts C149) and [Astra roadmap](roadmapastra.md) (S5 level/structure and median9 exploratory diagnostic). Sources remain on their dedicated branches; closure is not an integration claim. Their future use is incorporated into the current Month-2 plan.
**Temperature experiments after S7:** first K=0/1/4/9/16/32 mean/median correction sweep with matched baselines; then learned T inputs on the selected backbone, or **both families if the comparison is inconclusive**, using exactly the same K/sites/observations. Learned-training matrix is specified before execution. No current TEST authorization.  
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
- `docs/month2/` & `roadmap_month2.md`: Decision register and Month-2 plan. Steps 2.1–2.3 done; dense U-Net diagnostic (Step 2.4 / U0–U7) closed on `mission/dense-unet` with report `03_dense_velocity_diagnostic.md` and artifacts under `results/month2_dense_unet/`. GLU audit/adaptation and A38 sparse screening S0–S6 delivered; the C149 multi-seed comparison S7 and subsequent correction/T-input studies remain open. Baseline consolidation lives on `mission/baseline-consolidation` (`3cf7bf0`; main integration pending).
- `scripts/` & `tests/`: Reproducible execution scripts and automated `pytest` guardrails.

### Key Scientific Constraints
1. **No Data Leakage Across Conditions or Trajectories:** Entire operating conditions are withheld for validation/testing (interpolation vs. extrapolation). Correlated snapshots from a withheld trajectory never appear in training.
2. **Strict Separation of Observation Layouts & Operating Conditions:** Observation masks/layouts and operating-condition transfers are controlled independently.
3. **Training-Only Statistics:** Normalization and preprocessing statistics are fitted strictly on training splits.
4. **Separate Reporting:** Temperature-free (`u, v -> T`) and temperature-assisted (`u, v + sparse T -> T`) tasks are tracked and reported separately, broken down per operating condition.
5. **Physical Fidelity:** Numerical species concentrations are never equated with experimental chemiluminescence ($\text{OH} \neq \text{OH}^*$).
