# U6 auto-report — p_off seed 0 training (numeric)

- Commit executed: `25112807963a86322e98c20a6f1f826806b8b2c5`
- Job: `29476814` · `mit_amf_advanced_gpu` · `mit_normal_gpu` · QoS AMF · **L40S** · `node3202`
- Run: `u6_poff_seed0_engaging_v1`
- `p_conditioning`: false (channels 4/5 zero; fresh init, no p_on weight reuse)
- Status: `stopped_early_plateau` at **600**/1000 epochs
- Stop reason: `early_stop_no_rel_improvement_0.001_for_15_evals`
- Updates: 57000 (95 / epoch)
- Wall compute: ≈1433 s (≈23.9 min); smoke gate ≪ 4.5 h
- Best VAL macro RMSE: **39.6689 K** @ epoch **30** (first absolute min)
- Last VAL: 44.1018 K @ 600
- Seen / TRAIN @ best: 51.0295 K / 38.9535 K
- Last LR: 1e-5
- Best checkpoint sha256: `8330730817dae5f674d981cdcd006e7cf97fa1d5f90a5365510c20b4099f8e01`
- Last checkpoint sha256: `8bb499743fdd91b4cd2ba114fae160477a2f065411a1029a43258c431d4887cd`

## Schedule differences vs U4 (same rules, different curve)

| | U4 p_on | U6 p_off |
|---|---:|---:|
| Best epoch | 160 | 30 |
| Best VAL K | 26.931 | 39.669 |
| Last epoch | 730 | 600 |
| Last VAL K | 28.055 | 44.102 |
| Updates | 69350 | 57000 |
| Compute s | ≈1712 | ≈1433 |

Checkpoints remain on Engaging pool (gitignored); hashes in `checkpoint_hashes.json`.
