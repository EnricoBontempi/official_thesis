# Proposal 1 — Neural Reconstruction of Hidden Temperature Fields from Sparse Observations in Turbulent Combustion

- **Author:** Enrico Bontempi — MSc Thesis Proposal
- **Proposed research period:** Six months
- **Host group:** Deng Lab, Massachusetts Institute of Technology

---

## 1. Engineering motivation

The proposed application is combustion diagnostics under limited measurement access: estimating the spatial temperature distribution of a reacting flow when the available observations do not provide a complete temperature map. The information sought includes the location and extent of high-temperature regions, rather than only a spatially averaged temperature.

The intended role of the neural model is to supplement available measurements with an estimate of an unobserved physical field. This would be useful only if the reconstruction remains informative when operating conditions and observation coverage change. A model that reproduces training conditions accurately but fails without warning under a different fuel mixture would have limited diagnostic value.

The thesis will address this problem through controlled experiments on an accessible reacting-flow dataset. Its immediate deliverable will be a quantitatively validated reconstruction methodology, rather than a deployed engine diagnostic system. Engine-specific data from Politecnico di Milano may provide an additional application, but will not be required to complete the thesis.

## 2. Research question and proposed contribution

The main research question is:

> **How accurately can a neural model reconstruct an instantaneous temperature field from spatially incomplete observations of other physical variables, and how does its performance change at operating conditions excluded from training?**

The initial configuration will use incomplete observations of velocity, with temperature available as a supervised training target and evaluation reference but absent from the inference inputs. Additional observation configurations will be examined only when supported by the dataset and a clear measurement interpretation.

The central hypothesis is that reconstruction performance depends on both observation coverage and knowledge of the operating condition. Conditioning the model on known fuel composition and equivalence ratio may reduce systematic errors when transferring between operating conditions, but cannot compensate for information that is absent from the observations.

The proposed ML contribution is a controlled investigation of this dependence, using a neural reconstruction model selected with the laboratory. The initial comparison will quantify the effects of operating-condition information and observation sparsity. Based on the identified limitations, the methodological objective will be to develop a targeted improvement in observation-conditioned reconstruction, particularly in how incomplete measurements and operating information are integrated. The specific mechanism will be determined after the initial baseline study, in consultation with the supervisors.

Parameter conditioning and measurement masking are not claimed as new methods in themselves. The contribution will depend on establishing a relevant limitation of existing approaches and showing, through appropriate comparisons, what the proposed treatment improves and where it fails.

## 3. Dataset and scope

The initial dataset will be the combustion component of **RealPDEBench**, which provides numerical fields for a swirl-stabilized ammonia–methane–air flame across multiple operating conditions. Its documentation lists velocity, temperature and thermochemical variables on a $128 \times 128$ grid, with time-resolved trajectories. Experimental observations are $\text{OH}^*$ chemiluminescence images; they are not experimental temperature maps.

- **Dataset documentation:** https://realpdebench.github.io/datasets/combustion/
- **Data repository:** https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench

The core experiments will use numerical observations and numerical temperature targets. Numerical species concentrations will not be presented as directly measured experimental quantities, and $\text{OH}$ concentration will not be equated with $\text{OH}^*$ image intensity.

The first data audit will establish the actual operating-condition combinations, coordinate conventions, channel definitions, usable trajectories and licensing conditions. Selected trajectories will be inspected before committing to the full data transfer.

## 4. Initial methodology

Each available observation will be represented by its position, variable identity and value. A measurement encoder will produce a representation that a coordinate-conditioned decoder uses to predict temperature at query locations. This allows the model to receive different observation layouts without requiring unavailable values to be treated as physical zeros.

A deterministic neural reconstructor will provide the initial implementation. Existing laboratory approaches, including GLU, will inform the choice of representation and architecture. GLU already addresses sparse reconstruction and forecasting, so the thesis will not claim that flexible sparse-field reconstruction is itself a new capability.

Training will initially minimize normalized temperature reconstruction error. Random observation masking will expose the model to different spatial coverage during training. Physics-based penalties will be considered only if their required variables and derivatives can be evaluated consistently.

An early comparison will include a small number of temperature observations alongside the incomplete velocity measurements. This will serve two purposes: testing how much direct thermal information changes the problem, and providing a predefined fallback if velocity alone cannot support useful reconstruction. The temperature-free and temperature-assisted tasks will be reported separately.

## 5. Validation strategy

Entire operating conditions will be withheld from training. Where the available combinations permit it, evaluation will distinguish interpolation within the training parameter range from extrapolation beyond it. Correlated snapshots from a withheld trajectory will not appear in training.

Observation-layout changes will be evaluated separately from operating-condition changes. All normalization and preprocessing statistics will be fitted using training data only.

Essential comparisons will include:
- a mean-temperature-field predictor;
- a compact non-neural reconstruction baseline appropriate to the selected observations;
- the neural model with and without operating-parameter conditioning;
- temperature-free and temperature-assisted observation configurations.

Evaluation will report field error, spatial bias, errors in the high-temperature region and errors in temperature level. A fixed definition of the high-temperature region will be established before final testing. Results will be reported per operating condition, rather than only as a pooled average.

Failure will be informative if it shows that a configuration that works within one condition does not transfer, or that a small amount of direct temperature information is necessary. Failure of the tested networks will not establish a fundamental impossibility of reconstruction.

## 6. Six-month work plan

- **Month 1 — Data audit and problem confirmation.**
  Inspect representative trajectories and metadata; review the closest reconstruction studies; define operating-condition splits and observation configurations. Run inexpensive checks of whether velocity observations contain useful temperature information. Agree with the supervisors on the initial model and engineering interpretation.
- **Month 2 — Baselines and first neural implementation.**
  Implement the data pipeline, reference predictors and neural reconstructor. Establish performance on seen operating conditions and compare temperature-free observations with a temperature-assisted configuration. Select the main configuration based on these results.
- **Month 3 — Controlled methodological comparison.**
  Implement operating-parameter conditioning and observation masking. Train matched models with the same data and comparable computational budgets. Select model settings using validation conditions only.
- **Month 4 — Generalization experiments.**
  Evaluate withheld operating conditions and distinguish interpolation from extrapolation. Test reduced observation coverage and localized missing regions as separate experiments.
- **Month 5 — Reliability and engineering interpretation.**
  Analyze spatial errors, high-temperature-region reconstruction and failure cases. Repeat essential experiments to assess variability. Consider a small external dataset only if access and compatibility are already established.
- **Month 6 — Consolidation and writing.**
  Complete final comparisons, document the reproducible evaluation protocol, write the thesis and prepare a manuscript draft if the results support one. Reserve time for supervisor review and revision.

## 7. Risks and expected outcome

The main risk is that velocity alone does not sufficiently determine temperature across different mixtures. The predefined fallback is to retain sparse-state reconstruction while admitting a limited number of temperature measurements. This preserves the ML question while changing the available information explicitly.

Other risks include insufficient coverage of operating conditions and substantial storage requirements. These will be addressed through an early metadata audit and a staged use of the dataset.

The expected outcome is an evidence-based assessment of neural temperature reconstruction under changing conditions, together with a tested treatment of operating information and incomplete observations. Transfer to actual engine diagnostics will remain a subsequent validation step.

## Initial references

1. RealPDEBench, combustion dataset documentation and repository:
   - https://realpdebench.github.io/datasets/combustion/
   - https://github.com/AI4Science-WestlakeU/RealPDEBench
2. Wang, L., Chen, J., Tricard, N., Chen, Z., and Deng, S. *GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting.* arXiv:2603.26023, 2026. https://arxiv.org/abs/2603.26023
3. Hu, P., Feng, H., Liu, H., Yan, T., Deng, W., Gao, T., Zheng, R., Zheng, H., Yu, C., Wang, C., Li, K., Ma, Z.-M., Zhou, D., Lu, X., Fan, D., and Wu, T. *RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data.* arXiv:2601.01829, 2026. https://arxiv.org/abs/2601.01829
4. Yadav, V., Casel, M., and Ghani, A. *RF-PINNs: Reactive flow physics-informed neural networks for field reconstruction of laminar and turbulent flames using sparse data.* Journal of Computational Physics, 524, 113698, 2025. https://doi.org/10.1016/j.jcp.2024.113698
