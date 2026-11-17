# U5 auto-report — snapshot control (numeric)

- Commit: `8549a9b900a038d41da7513831a90908c49b87aa`
- Checkpoint sha256: `6d456abcb5653c16a7a90df40eda377caf42ffb94fd829a65f4c06be58a03634` (epoch 160)
- Device: NVIDIA L40S (cuda)
- Aligned replay VAL/seen: 26.931457 / 37.675490 K (Δ vs U4 0 / 0; atol=0.05 K) → OK

## Macro RMSE [K]

| Role | UNet aligned | UNet swapped | Δ (sw−al) | ParamMean | GappyPOD dense | GlobalMean |
|---|---:|---:|---:|---:|---:|---:|
| val_eval | 26.931 | 96.549 | +69.618 | 88.008 | 70.463 | 139.700 |
| seen_train_eval | 37.675 | 150.305 | +112.629 | 129.348 | 108.103 | 173.685 |

## VAL per-condition Δ RMSE (sw−al)

- 0NH3_1.05.h5: aligned 32.015 → swapped 196.208 (Δ +164.193 K); Δ centered +161.733; Δ |bias| +26.658
- 20NH3_1.1.h5: aligned 27.610 → swapped 49.421 (Δ +21.812 K); Δ centered +21.978; Δ |bias| +0.370
- 40NH3_0.9.h5: aligned 21.170 → swapped 44.018 (Δ +22.848 K); Δ centered +25.323; Δ |bias| +0.000

## Figures

- `figures/val_frame1000_0NH3_1.05_panels.png`
- `figures/val_frame1000_20NH3_1.1_panels.png`
- `figures/val_frame1000_40NH3_0.9_panels.png`
- `figures/val_mean_abs_error_aligned_vs_swapped.png`
- `figures/u4_curves_train_seen_val_lr.png`
- `figures/val_per_condition_rmse_comparison.png`
