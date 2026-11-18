# S4 scientific freeze candidate (not an S5 authorization)

**Branch:** `mission/dense-unet`  
**Profiling code SHA:** `740805f7e5537090a0f08478cdc1587ed6c9c5e3`  
**Jobs:** U-Net `29554165`, GLU `29554166` — both `COMPLETED` `0:0`, status `completed_diagnostic`.  
**This file does not authorize S5 submission.** `scientific_training_authorized` / `s5_submission_authorized` remain false until Enrico explicitly enables them.

---

## 1. Job outcomes (read from reports, not ExitCode alone)

| Job | Family | State / Exit | Node / GPU | Queue≈ | Compute | SHA | Report status |
|---|---|---|---|---|---|---|---|
| 29554165 | sparse_unet | COMPLETED 0:0 | node3008 / L40S | ~189 s | 68 s | 740805f | `completed_diagnostic` |
| 29554166 | adapted_glu | COMPLETED 0:0 | node3408 / L40S | ~189 s | 79 s | 740805f | `completed_diagnostic` |

Phases present for both: dep preflight ok; cache hash match; smoke finite loss/grad; 2 measured TRAIN epochs (batch 8, 95 upd/epoch); VAL 5% × replicas 0/1/2 inside fit; inference 1/5/20%; checkpoint write; CPU/GPU compare within atol/rtol `1e-4`.  
Weights: `weights_reusable_for_science: false`. Attempt-1 FAILED dirs preserved.

Env: `/home/bontempi/envs/official_thesis`, Python 3.11.18, torch `2.8.0+cu128`, CUDA 12.8, einops `0.8.2`.

---

## 2. Measured technical costs (L40S)

### Sparse U-Net (`29554165`)

| Quantity | Value | Kind |
|---|---|---|
| Measured loop (2 TRAIN ep + VAL@each ep, VAL-only roles) | 31.60 s | **measure** |
| Checkpoint write | 0.094 s (~56.2 MiB report / ~56.2 MiB best+last each) | **measure** |
| Peak VRAM / RSS | 0.605 GB / 1.90 GB (Slurm MaxRSS ~2.15 GB) | **measure** |
| Batch / microbatch / queries | 8 / null (no accum) / full-field U-Net grid | **measure** |
| Inference full-role val_eval (repl 0) 1/5/20% | 4.50 / 4.50 / 4.58 s | **measure** |
| Latency batch=1 1/5/20% | 17.9 / 1.88 / 2.24 ms | **measure** (1% includes first-call overhead) |
| CPU↔GPU max_abs | 4.84e-8 (atol/rtol 1e-4; allclose true) | **measure** |
| Finite loss/grad smoke | passed (else job would raise) | **measure** |
| Separate TRAIN-only vs VAL-only wall split | **not recorded** | gap |
| Encoder/decoder breakdown | N/A | — |

### Adapted GLU (`29554166`)

| Quantity | Value | Kind |
|---|---|---|
| Measured loop (2 TRAIN ep + VAL@each ep) | 40.71 s | **measure** |
| Checkpoint write | 0.034 s (~13.3 MiB) | **measure** |
| Peak VRAM / RSS | 4.33 GB / 2.03 GB (Slurm MaxRSS ~2.43 GB) | **measure** |
| Batch / microbatch / queries | 8 / null / **16384** full-field queries | **measure** |
| Inference full-role 1/5/20% | 4.66 / 4.71 / 6.38 s | **measure** |
| Latency batch=1 1/5/20% | 32.6 / 32.8 / 33.9 ms | **measure** |
| Coarse forward full / chunked | 0.019 / 0.031 s | **measure** (not isolated enc/dec) |
| Neighbor sort | `topk_distance_then_site_index` (code path) | reported |
| Geometry buffers | center/L from cache `geometry.npz`; persistent | **measure** in report |
| CPU↔GPU max_abs | 5.87e-5 (atol/rtol 1e-4; allclose true) | **measure** |
| Finite loss/grad smoke | passed | **measure** |

Diagnostic VAL RMSE (not for architecture/LR/variant choice): U-Net ~115.6 K @ep2; GLU ~130.2 K @ep2.

### Tranche-1000 estimate (scientific schedule)

Assumptions / uncertainties:

- Profiling used `eval_every_epochs=1`, `eval_train_fit=False`, warmup technical 4 updates.  
- S5 uses `eval_every_epochs=10`, roles train_fit+seen+VAL×3, warmup 950.  
- TRAIN vs VAL walls not split in the report → VAL cost proxied by `inference.full_role_s` @5% ×3 replicas; train_fit/seen scaled by frame counts (760/220 vs 123). **Estimate**, not measure.

| Model | Est. train 1000 ep | Est. ~100 multi-role evals | Est. total compute | Uncertainty |
|---|---|---|---|---|
| U-Net | ~0.6–1.0 h | ~1.0–1.7 h | **~2–3 h** | high (±40%): role scaling + cold vs steady) |
| GLU | ~1.5–2.0 h | ~1.2–1.9 h | **~3–4.5 h** | high (±40%) |

No arbitrary scientific hour cap. Scheduler wall recommendation below is operational (partition MaxTime 6 h) with **checkpoint/resume** for extension beyond a single allocation.

---

## 3. Freeze prerequisites — verified vs code-only

| Requirement | Evidence | Status |
|---|---|---|
| Resume continuous vs resumed (LR, warmup/plateau counters, best, early-stop, history, RNG/shuffle/`set_epoch`) | Local synthetic `tests/test_sparse_scientific_trainer.py` (passed in prior S4 work) | **verified locally**; **not** re-proven by Engaging profiling resume of a long run |
| Complete vs partial checkpoints; partial does not overwrite `last` | Runner + tests | **verified in tests** |
| Reject incompatible config/data | `test_resume_rejects_incompatible_and_partial` | **verified in tests** |
| Cache/stats/geometry/mask identity | Job reports: cache aggregate match; stats/mask hashes in `run_identity` | **verified on Engaging** |
| GLU persistent geometry + tie-break | Report geometry block + S3 tests/`topk_distance_then_site_index` | geometry **on job**; tie-break **prior tests** |
| Full-field queries + effective batch 8 | 16384 queries; 95 updates/epoch = 760/8 | **verified** |
| Microbatch / accum / chunked backward | `microbatch_size: null`; not used | **N/A — not required** |
| Chunked decode eval | enabled; GLU chunked forward timed | **used for eval**, not training backward |
| Engaging deps | preflight + einops 0.8.2 in thesis env | **verified** |

**Not declared ready if “necessary” means GappyPOD 1/20% matched baselines:** that remains blocked (below).  
**Neural S5 screening:** operational evidence from profiling + prior resume tests is **sufficient for a candidate freeze**; Enrico still must authorize submission.

---

## 4. GappyPOD 1%/20% block

- **Missing object:** persisted fitted POD archive with `pod_p_off` / `pod_p_on` (never written by `scripts/06_run_month2_baselines.py`; fit lived in-process only).  
- **Searched:** Engaging `…/official_thesis_data`, `…/official_thesis/results`, `…/data`; local `results/**` — no POD `.pt`/`.npz`/`joblib` basis found. Script `15_gappypod_nested_coverage_eval.py` returns `blocked_missing_pod_artifacts`.  
- **Reusable at 5%:** frozen λ `p_off=1e-3`, `p_on=1e-2`, R=32 from `results/meta/month2_baseline_run_manifest.json`; metrics in `results/tables/month2_baseline_summary.csv` (e.g. VAL free p_on RMSE ≈70.60 K, p_off ≈94.80 K). Historical 5% layouts/join intact.  
- **To unblock later:** recover a provenanced artifact **or** Enrico-authorized deterministic refit of train_fit POD only (no λ retune). Nested eval runner still needs completion after artifacts exist.  
- **Consequence:** blocks matched **baseline** comparison at 1/20%, **not** neural U-Net/GLU S5 training at 5%.

---

## 5. Candidate operational freeze (S5 configs)

Protocol-aligned fields already match `month2_sparse_scientific_candidate.yaml` / `month2_sparse_protocol.yaml` (5%, K=0, batch 8, AdamW 1e-3/1e-4, clip 1, fp32, warmup 950, plateau, early-stop, eval/10, fresh init, no S4 weight reuse).

Resolved from profiling (still **not** submission authorization):

| Field | Candidate value | Basis |
|---|---|---|
| `microbatch_size` | `null` | Peak VRAM U-Net 0.6 GB / GLU 4.3 GB ≪ L40S; batch 8 OK |
| `sbatch_gres` | `gpu:l40s:1` | measured |
| `sbatch_mem` | `64G` | measured RSS ≪64G; keep profiling-proven envelope |
| `sbatch_time` | `06:00:00` | partition MaxTime; covers est. 2–4.5 h + margin; **not** a scientific hour ceiling |
| Checkpoint/resume plan | complete-boundary resume if allocation ends mid-tranche; extend `n_epochs` only with explicit flag | contract already in configs |

`wall_time_cap_hours: 6.0` remains an **operational** guard aligned to the allocation, configurable, not a physics/training-quality cap.

---

## Absolute paths (Engaging)

- U-Net run: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29554165/`  
- GLU run: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29554166/`  
- Slurm: `…/official_thesis/slurm-29554165.out`, `…/slurm-29554166.out`  
- Compact Git copy: `results/month2/s4_profiling/` (JSON/logs only; `.pt` out of Git)
