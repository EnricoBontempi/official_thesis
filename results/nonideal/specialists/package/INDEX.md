# Specialist retraining — recovery, clean cost and transfer

Synthetic regimes (not a PIV validation). Selection = ORIGINAL clean VAL macro (first absolute minimum); the specialist VAL metric is only recorded. VAL123 was used for selection. No tests, no thresholds. Baselines = original clean models of the same family/seed (not retrained).

Rows: 370440 specialist frame evaluations (30 runs x 2 checkpoints x 6174); baselines 37044.

## Own scenario, best checkpoint: gain = RMSE_original - RMSE_specialist (K), mean of 3 seeds [min, max]

### seen220

| family | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|
| U-Net | -1.83 [-7.07, +2.40] | -1.44 [-5.51, +2.69] | +0.73 [-2.94, +7.43] | +0.32 [-3.67, +5.24] | +3.00 [+0.65, +7.02] |
| GLU | +0.35 [-4.12, +4.83] | -2.03 [-6.91, +1.06] | -1.58 [-10.91, +7.74] | +0.22 [-1.40, +2.30] | -5.41 [-21.60, +5.34] |

### VAL123

| family | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|
| U-Net | -0.85 [-1.02, -0.70] | -2.65 [-3.84, -1.80] | -1.88 [-4.77, +0.48] | -0.42 [-1.80, +2.06] | +0.29 [-1.19, +3.24] |
| GLU | -1.26 [-3.85, +1.52] | +0.10 [-2.67, +2.67] | -2.18 [-7.01, +2.63] | -1.90 [-4.23, -0.12] | -2.81 [-9.53, +1.92] |

## Clean-input cost (specialist - original RMSE on R0, K), best

| family | role | N1 | N2 | G1 | G1-eq | N1G1 |
|---|---|---|---|---|---|---|
| U-Net | seen220 | +1.83 [-2.37, +7.08] | +2.64 [-1.37, +6.45] | +2.96 [-2.47, +5.74] | +0.51 [-3.74, +3.99] | +1.21 [-1.75, +3.09] |
| U-Net | VAL123 | +0.86 [+0.63, +1.08] | +3.61 [+2.33, +5.00] | +3.90 [+1.98, +6.47] | +0.53 [-1.50, +1.92] | +1.85 [+0.20, +3.21] |
| GLU | seen220 | -0.31 [-4.84, +4.17] | +4.37 [+1.18, +9.96] | +4.98 [-2.80, +14.10] | +0.51 [-1.28, +2.39] | +8.53 [-0.68, +22.78] |
| GLU | VAL123 | +1.32 [-1.37, +3.86] | +2.48 [-0.47, +4.77] | +4.09 [+1.15, +7.48] | +2.35 [+0.35, +4.54] | +4.83 [+0.69, +9.51] |

## Training outcomes (read before interpreting the tables)

- 30 runs stopped by the operational early-stop rule (plateau); 0 runs reached 1200 epochs without the plateau after the single authorised continuation (`budget_exhausted`).
- `budget_exhausted` runs are valid artifacts (finite, checkpoints read back, gates passed, continuation registered) and are INCLUDED, labelled in `tables/training_diagnostics.csv`, `per_run_scenario.csv`, `fields_index.json` and the figures (hollow markers). They are **not declared converged**: their numbers are not an estimate of a converged model. The plateau rule itself is an operational convention, not a physical certification.
- Technical errors, missing continuation, non-finite values, invalid checkpoints or failed gates are NOT accepted: they block the package (BLOCK_REPORT.md).

## Figures

- `figures/fig1_gain_matrix_best.png`
- `figures/fig1_gain_matrix_last.png`
- `figures/fig2_own_scenario.png`
- `figures/fig3_clean_cost.png`

## Tables

- `tables/per_condition.csv`
- `tables/macro_by_replica.csv`
- `tables/per_run_scenario.csv`
- `tables/seed_summary.csv`
- `tables/diagonal_own_scenario.csv`
- `tables/training_diagnostics.csv`

