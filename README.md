# Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion

**Author:** Enrico Bontempi  
**Host Group:** Deng Lab, Massachusetts Institute of Technology (MIT)  
**Thesis Proposal:** [docs/proposal.md](docs/proposal.md)  
**Month 1 Roadmap:** [roadmap_month1.md](roadmap_month1.md)

## Overview

This repository contains the codebase, experimental configurations, data audit protocols, evaluation suites, and documentation for the MSc Thesis:

> *How accurately can a neural model reconstruct an instantaneous temperature field from spatially incomplete observations of other physical variables, and how does its performance change at operating conditions excluded from training?*

### Key Scientific Constraints
1. **No Data Leakage Across Conditions or Trajectories:** Entire operating conditions are withheld for validation/testing (interpolation vs. extrapolation). Correlated snapshots from a withheld trajectory never appear in training.
2. **Strict Separation of Observation Layouts & Operating Conditions:** Observation masks/layouts and operating-condition transfers are controlled independently.
3. **Training-Only Statistics:** Normalization and preprocessing statistics are fitted strictly on training splits.
4. **Separate Reporting:** Temperature-free (`u, v -> T`) and temperature-assisted (`u, v + sparse T -> T`) tasks are tracked and reported separately, broken down per operating condition.
5. **Physical Fidelity:** Numerical species concentrations are never equated with experimental chemiluminescence ($\text{OH} \neq \text{OH}^*$).
