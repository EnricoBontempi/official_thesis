# Delivery check of the 30-specialist numeric package

Read-only. No inference. Failed checks are listed first.

**Overall: OK**

| check | result | detail |
|---|---|---|
| freeze: 30 expected run ids | ok | 30 runs |
| freeze: n_runs/n_valid = 30, problems empty | ok | n_runs=30 n_valid=30 problems=[] |
| freeze: training SHA | ok | b6fc9fefda5a1d4776c87c9285d7187c0d89fc48 |
| freeze: post-processing SHA | ok | b1c095507b803e5fa27ad8ba8bdad2febd8d2541 |
| freeze: 30 plateau, 0 budget_exhausted | ok | {'plateau': 30} |
| 60/60 gate.json all_pass (5/5 comparisons within 0.05 K) | ok | 60/60, max |delta| 0.0024 K |
| 60/60 COMPLETE.json ok, 6174 rows, gate_all_pass | ok | 60/60 |
| 60/60 provenance = freeze (weights, file, epoch, regime, spec, eval SHA, weights unchanged) | ok | 60/60 |
| one observation-matrix hash, one cache identity, one spec hash across 60 evaluations | ok | 1 matrix, 1 cache, 1 spec |
| per_run_scenario.csv has 720 rows | ok | 720 |
| keys unique | ok | 720 unique |
| keys = 30 runs x 2 checkpoints x 6 cells x 2 roles | ok |  |
| finite values | ok |  |
| gain = base - spec, delta = spec - base, clean_cost = spec_R0 - base_R0 (<1e-9 K) | ok | max 0.00e+00 |
| baseline RMSE identical across regimes and checkpoints for each (family, seed, cell, role) | ok | 72 baseline groups (expect 2 families x 3 seeds x 6 cells x 2 roles = 72) |
| 72 baseline groups | ok | 72 |
| package_manifest hashes match files | ok | 12 files; mismatch: [] |
| package_manifest freeze hash = sha256(freeze.json) | ok |  |
| baseline per-frame CSV sha256 = expected = package manifest | ok | cb96687cf3614c5393bc9658ec3469980313413f495f32930fd2e810ffc070b3 |
| fields_index.json readable | ok | keys ['format', 'n_runs', 'eval_root', 'per_frame_csvs'] |
| 720 aggregated RMSE = re-reduction of per-frame CSVs (frames -> conditions -> replicas) | ok | 720 rows, max |diff| 2.13e-14 K (tol 1e-06) |

The 30 per-run `valid/terminal state` checks are included in the JSON (30 entries).
