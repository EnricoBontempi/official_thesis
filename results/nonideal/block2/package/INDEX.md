# Block II — frozen models on degraded velocity observations

Descriptive scenario study (synthetic degradations, not PIV validation). VAL123 was used for selection/development: not an independent confirmation. No tests, no thresholds.

Rows: 74088 nominal frame evaluations (expected 74088); 6174 technical GLU replay rows counted separately.

## Mean deterioration (RMSE_scenario - RMSE_R0, K) — mean over seeds [min, max]

### seen220

| family | p | N1 | G1 | N1G1 | G1-eq | N2 |
|---|---|---|---|---|---|---|
| sparse_unet | p_on | 0.30 [0.27, 0.35] | 5.42 [4.86, 6.46] | 5.82 [5.19, 6.93] | 1.48 [1.24, 1.90] | 2.59 [2.38, 2.84] |
| sparse_unet | p_off | 0.32 [0.30, 0.35] | 6.21 [5.87, 6.53] | 6.68 [6.38, 6.92] | 2.11 [1.75, 2.43] | 3.15 [2.99, 3.26] |
| adapted_glu | p_on | 0.26 [0.22, 0.28] | 7.53 [5.84, 8.49] | 7.78 [6.17, 8.72] | 2.29 [1.85, 2.53] | 3.30 [2.73, 4.05] |
| adapted_glu | p_off | 0.30 [0.20, 0.40] | 8.12 [7.69, 8.96] | 8.40 [7.89, 9.29] | 2.50 [2.31, 2.66] | 4.74 [4.43, 5.11] |

### VAL123

| family | p | N1 | G1 | N1G1 | G1-eq | N2 |
|---|---|---|---|---|---|---|
| sparse_unet | p_on | 0.17 [0.11, 0.23] | 4.12 [3.00, 5.70] | 4.36 [3.19, 6.03] | 2.02 [1.31, 2.92] | 1.34 [1.09, 1.58] |
| sparse_unet | p_off | 0.52 [0.44, 0.57] | 5.61 [5.57, 5.64] | 6.04 [5.96, 6.12] | 3.60 [2.98, 4.18] | 3.38 [3.00, 3.63] |
| adapted_glu | p_on | 0.18 [0.13, 0.25] | 4.34 [2.99, 5.32] | 4.57 [3.24, 5.61] | 1.29 [0.87, 1.73] | 2.94 [2.30, 3.64] |
| adapted_glu | p_off | 0.10 [0.02, 0.16] | 5.55 [4.49, 6.86] | 5.89 [4.77, 7.11] | 1.51 [1.25, 1.90] | 4.15 [3.83, 4.46] |

## Figures

- `figures/fig1_macro_rmse_by_cell.png`
- `figures/fig2_deterioration_vs_R0.png`
- `figures/fig3_val_by_condition.png`
- `figures/fig4_level_vs_structure.png`
- `figures/fig5_fields_O-C149-on-s0.png`
- `figures/fig5_fields_O-C149-off-s0.png`
- `figures/fig5_fields_S7-C149-glu-on-s0.png`
- `figures/fig5_fields_S7-C149-glu-off-s0.png`

## Tables

- `tables/per_condition.csv`
- `tables/macro_by_replica.csv`
- `tables/delta_vs_R0_by_replica.csv`
- `tables/delta_vs_R0_by_condition.csv`
- `tables/delta_vs_R0_per_model.csv`
- `tables/absolute_per_model.csv`
- `tables/delta_seed_summary.csv`
- `tables/absolute_seed_summary.csv`
- `tables/structure_contrasts.csv`
- `tables/pon_minus_poff.csv`
- `tables/family_contrast_by_seed_index.csv`
- `tables/regional_g1.csv`
- `tables/sites_and_area.csv`
- `tables/glu_batching_diagnostic.csv`
- `tables/per_frame_all.csv.gz` (sul cluster; hash nel manifest; assente da questo checkout)

## Lettura per la discussione

Figure e nota successive, senza nuova valutazione: [../reading/INDEX.md](../reading/INDEX.md) e [docs/nonideal/03_block2_reading.md](../../../../docs/nonideal/03_block2_reading.md). Il pacchetto sopra non è stato ricalcolato.

