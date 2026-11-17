# U4 auto-report — `u4_pon_seed0_engaging_v1`

**Decision:** Enrico 2026-11-17 (operational). No supervisor ratification claimed.  
**SHA executed:** `50bda7fe42323d2abd8ef46763be65ccbaf3f008`  
**Job:** Slurm `29465586` · account/QoS `mit_amf_advanced_gpu` · partition `mit_normal_gpu` · GPU **NVIDIA L40S** ×1 · node `node3008`  
**Env:** `/home/bontempi/envs/official_thesis` · torch `2.8.0+cu128` · TF32 off · fp32  
**Cache payload SHA256:** `d0b1d06805562cda49a86ea9ab1f289ce202562549482533b272efb19a69c725` (verified post-transfer)

## Timing (separated)

| Phase | Value |
|---|---|
| Queue wait | submit 17:03:50 → start 17:06:02 ET ≈ **132 s** |
| Compute (fit elapsed) | **1712.4 s ≈ 28.5 min** |
| Slurm elapsed | 00:28:43 |
| Smoke projection (cadence-10 est.) | ≈ **0.85 h** for 1000 ep (gate 4.5 h: pass) |
| Peak VRAM (smoke) | ≈ 0.57 GiB |

## Outcome

| Field | Value |
|---|---|
| Status | `stopped_early_plateau` |
| Stop reason | 15 VAL evals without ≥0.1% relative improvement |
| Epochs completed | **730** / 1000 max |
| Updates | 69350 (95/epoch) |
| Best VAL macro RMSE | **26.931 K** at epoch **160** |
| Last VAL macro RMSE | 28.055 K at epoch 730 |
| Best seen / train (at best epoch) | 37.675 K / 13.652 K |
| Last seen / train | 36.908 K / 3.734 K |
| Final LR | 1e-5 (plateau floor) |
| Clip frequency | ≈ 0.91% of updates |
| Still improving at 1000? | **No** (early-stopped; budget continuation not indicated) |

## Best VAL per condition (epoch 160)

| Condition | RMSE_K | \|bias\|_K | Centered-RMSE_K |
|---|---:|---:|---:|
| 0NH3_1.05 | 32.01 | 3.30 | 31.80 |
| 20NH3_1.1 | 27.61 | 5.08 | 27.05 |
| 40NH3_0.9 | 21.17 | 12.88 | 16.61 |
| **macro** | **26.93** | | |

## Checkpoints (persistent on Engaging; gitignored `.pt`)

| File | SHA256 | Bytes | Path |
|---|---|---:|---|
| best.pt | `6d456abcb5653c16a7a90df40eda377caf42ffb94fd829a65f4c06be58a03634` | 60028822 | `/home/bontempi/orcd/pool/official_thesis_runs/u4_pon_seed0_engaging_v1/checkpoints/best.pt` |
| last.pt | `b2824084052e797c0e864e38848eed4fd8a8a0479908e2fe7862a96623e72538` | 64032662 | `.../checkpoints/last.pt` |

Artifacts: `eval_history.json`, `curves.csv.gz`, `evals/epoch_*/{train,seen,val}_per_frame.csv.gz`, `selection.json`, `effective_config.json`.

## Matched linear reference (U3, same dense contract)

Dense GappyPOD p_on VAL selection from `results/month2_dense_linear/dense_linear_u3_v1` remains the linear comparator (see that run’s selection/summary). U-Net best VAL **26.9 K** is substantially below the dense GappyPOD p_on VAL figure recorded at STOP B (~70.5 K with selected λ). This is empirical for this seed/config only — not an information ceiling or causal claim.
