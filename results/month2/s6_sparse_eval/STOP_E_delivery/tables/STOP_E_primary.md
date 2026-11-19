# STOP E — primary tables

Execution SHA tip S5: `38ac7fe40178982ac1bd33c5d5602a5e7d928d3b`. Gappy job **29601906**; S6 last+swap **29601907–29601910**; best coverages reused from prior S6 attempt on same checkpoints.

## Best VAL @5% (mean over mask replicas 0/1/2)

| label | mean_condition_rmse_K | mean_abs_bias_K | mean_centered_rmse_K |
| --- | --- | --- | --- |
| U-Net p_on | 42.9640 | 12.8841 | 38.3647 |
| U-Net p_off | 43.5257 | 17.5176 | 38.0018 |
| GLU p_on | 45.8007 | 8.7737 | 44.6089 |
| GLU p_off | 47.5354 | 18.5755 | 42.7163 |

## Last VAL @5%

| label | mean_condition_rmse_K |
| --- | --- |
| GLU p_on | 46.8392 |
| U-Net p_off | 47.6724 |
| GLU p_off | 47.8706 |
| U-Net p_on | 49.4958 |

## Best VAL coverage 1/5/20%

| label | site_fraction | mean_condition_rmse_K | mean_abs_bias_K | mean_centered_rmse_K |
| --- | --- | --- | --- | --- |
| GLU p_off | 0.0100 | 67.2027 | 24.8241 | 60.7461 |
| GLU p_off | 0.0500 | 47.5354 | 18.5755 | 42.7163 |
| GLU p_off | 0.2000 | 48.5164 | 18.7045 | 43.8952 |
| GLU p_on | 0.0100 | 68.0847 | 11.1258 | 66.6320 |
| GLU p_on | 0.0500 | 45.8007 | 8.7737 | 44.6089 |
| GLU p_on | 0.2000 | 46.8442 | 9.0164 | 45.7009 |
| U-Net p_off | 0.0100 | 73.6963 | 39.5518 | 58.4432 |
| U-Net p_off | 0.0500 | 43.5257 | 17.5176 | 38.0018 |
| U-Net p_off | 0.2000 | 54.0759 | 30.1732 | 40.8665 |
| U-Net p_on | 0.0100 | 75.1988 | 21.5396 | 69.2363 |
| U-Net p_on | 0.0500 | 42.9640 | 12.8841 | 38.3647 |
| U-Net p_on | 0.2000 | 44.3211 | 16.9823 | 38.1527 |

## Last VAL coverage 1/5/20%

| label | site_fraction | mean_condition_rmse_K |
| --- | --- | --- |
| GLU p_off | 0.0100 | 67.4081 |
| GLU p_off | 0.0500 | 47.8706 |
| GLU p_off | 0.2000 | 48.8761 |
| GLU p_on | 0.0100 | 71.1579 |
| GLU p_on | 0.0500 | 46.8392 |
| GLU p_on | 0.2000 | 47.3628 |
| U-Net p_off | 0.0100 | 70.5999 |
| U-Net p_off | 0.0500 | 47.6724 |
| U-Net p_off | 0.2000 | 57.2329 |
| U-Net p_on | 0.0100 | 62.5102 |
| U-Net p_on | 0.0500 | 49.4958 |
| U-Net p_on | 0.2000 | 44.6334 |

## GappyPOD VAL (job 29601906, gate passed)

| label | site_fraction | rmse_K | abs_bias_K | centered_rmse_K |
| --- | --- | --- | --- | --- |
| GappyPOD p_off | 0.0100 | 98.4356 | 57.8244 | 75.7647 |
| GappyPOD p_off | 0.0500 | 94.7987 | 57.5788 | 71.6916 |
| GappyPOD p_off | 0.2000 | 93.8078 | 57.0014 | 71.0267 |
| GappyPOD p_on | 0.0100 | 71.3726 | 15.3057 | 69.0031 |
| GappyPOD p_on | 0.0500 | 70.5985 | 15.3824 | 68.1938 |
| GappyPOD p_on | 0.2000 | 70.4572 | 15.3637 | 68.0538 |

## Snapshot swap best@5% rep0

| label | role | aligned_rmse_K | swapped_rmse_K | delta_swap_minus_aligned | donor_T_ref_rmse_K |
| --- | --- | --- | --- | --- | --- |
| U-Net p_on | seen_train_eval | 51.4914 | 148.6559 | 97.1645 | 153.9429 |
| U-Net p_on | val_eval | 42.8150 | 100.8624 | 58.0474 | 97.3624 |
| U-Net p_off | seen_train_eval | 55.2759 | 149.1371 | 93.8612 | 153.9429 |
| U-Net p_off | val_eval | 43.6250 | 101.4273 | 57.8023 | 97.3624 |
| GLU p_on | seen_train_eval | 56.0591 | 146.7201 | 90.6610 | 153.9429 |
| GLU p_on | val_eval | 45.9865 | 99.5844 | 53.5979 | 97.3624 |
| GLU p_off | seen_train_eval | 60.2516 | 146.7542 | 86.5027 | 153.9429 |
| GLU p_off | val_eval | 48.0241 | 103.6355 | 55.6114 | 97.3624 |

## Ranking notes
```json
[
  {
    "contrast": "best_vs_last_val5",
    "best_rank": [
      "U-Net p_on",
      "U-Net p_off",
      "GLU p_on",
      "GLU p_off"
    ],
    "last_rank": [
      "GLU p_on",
      "U-Net p_off",
      "GLU p_off",
      "U-Net p_on"
    ],
    "changed": true
  },
  {
    "contrast": "best_val_cov_0.01",
    "rank": [
      "GLU p_off",
      "GLU p_on",
      "U-Net p_off",
      "U-Net p_on"
    ],
    "rmse": [
      67.20267043792262,
      68.08468715573008,
      73.69634787749483,
      75.19877039628466
    ]
  },
  {
    "contrast": "best_val_cov_0.05",
    "rank": [
      "U-Net p_on",
      "U-Net p_off",
      "GLU p_on",
      "GLU p_off"
    ],
    "rmse": [
      42.964016515760335,
      43.52570262224853,
      45.80072748522861,
      47.535399850964815
    ]
  },
  {
    "contrast": "best_val_cov_0.2",
    "rank": [
      "U-Net p_on",
      "GLU p_on",
      "GLU p_off",
      "U-Net p_off"
    ],
    "rmse": [
      44.32112413594962,
      46.84423564872471,
      48.516402514842014,
      54.075916690907
    ]
  }
]
```


## Gaps
```json
{
  "regional_thermal_maps_in_s6_csv": "not present as separate region columns in per_frame outputs; roadmap asks T-region metrics where supported \u2014 need separate computation if required",
  "common_scale_panel_montage": "per-run VAL frame1000 panels exist; no single cross-method common-scale montage file generated",
  "training_seed_dispersion": "seed 0 only; mask replicas 0/1/2 are not model seeds"
}
```

