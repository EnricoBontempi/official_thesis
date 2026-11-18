# U7 derived VAL level/structure table

**File:** `u7_closure_val_level_structure.csv`  
**Created:** 2026-11-17 (U7 documentary closure)  
**Inference / training:** none.

**Origin:** per-condition then macro means of `rmse_K`, `abs_bias_K`, `centered_rmse_K` from existing frame tables:

- `u4_pon_seed0_engaging_v1/evals/epoch_0160/val_eval_per_frame.csv.gz`
- `u4_pon_seed0_engaging_v1/evals/epoch_0730/val_eval_per_frame.csv.gz`
- `u6_poff_seed0_engaging_v1/evals/epoch_0030/val_eval_per_frame.csv.gz`
- `u6_poff_seed0_engaging_v1/evals/epoch_0600/val_eval_per_frame.csv.gz`

**Command (local tip):** aggregate with pandas `groupby(sim_id).mean` then mean over the three VAL conditions.  
Do not recombine averaged |bias| and Centered-RMSE into RMSE.
