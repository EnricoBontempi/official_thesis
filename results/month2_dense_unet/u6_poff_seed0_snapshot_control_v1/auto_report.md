# U6 auto-report — p_off eval + snapshot control (numeric)

- Commit: `815bb7a086d8780f743a1c909833b7b9bb1e093f`
- U6 checkpoint sha256: `8330730817dae5f674d981cdcd006e7cf97fa1d5f90a5365510c20b4099f8e01` (best epoch 30)
- Device: NVIDIA L40S (cuda)
- Aligned replay VAL/seen: 39.668897 / 51.029506 K (Δ vs U6 best 0 / 0; atol=0.05 K) → OK

## Best p_on vs p_off (VAL macro RMSE K)

- p_on best: 26.931 @ 160
- p_off best: 39.669 @ 30
- Δ (p_off − p_on): +12.737

## p_off snapshot control macro RMSE [K]

| Role | aligned | swapped | Δ | GlobalMean | GappyPOD p_off | GappyPOD p_on |
|---|---:|---:|---:|---:|---:|---:|
| val_eval | 39.669 | 102.405 | +62.736 | 139.700 | 93.663 | 70.463 |
| seen_train_eval | 51.030 | 153.101 | +102.071 | 173.685 | 112.175 | 108.103 |

## VAL per-condition p_off Δ RMSE (sw−al)

- 0NH3_1.05.h5: aligned 45.676 → swapped 199.476 (Δ +153.800); Δ centered +154.245; Δ |bias| +21.059
- 20NH3_1.1.h5: aligned 35.649 → swapped 54.192 (Δ +18.543); Δ centered +18.866; Δ |bias| +0.148
- 40NH3_0.9.h5: aligned 37.681 → swapped 53.546 (Δ +15.864); Δ centered +17.643; Δ |bias| +0.988

## Figures

- `figures/val_frame1000_0NH3_1.05_poff_panels.png`
- `figures/val_frame1000_0NH3_1.05_error_tight.png`
- `figures/val_frame1000_20NH3_1.1_poff_panels.png`
- `figures/val_frame1000_20NH3_1.1_error_tight.png`
- `figures/val_frame1000_40NH3_0.9_poff_panels.png`
- `figures/val_frame1000_40NH3_0.9_error_tight.png`
- `figures/val_mean_abs_error_poff_aligned_vs_swapped.png`
- `figures/u4_u6_curves_pon_vs_poff.png`
- `figures/u6_curves_train_seen_val_lr.png`
- `figures/val_per_condition_pon_poff_comparison.png`
