# S4 profiling — submission record (not a freeze authorization for S5)

**Branch:** `mission/dense-unet`  
**Profiling code SHA:** `740805f7e5537090a0f08478cdc1587ed6c9c5e3`  
**S4 status:** profiling attempt-2 **COMPLETED** (`completed_diagnostic`); scientific freeze **candidate** documented — **S5 not authorized/submitted**.  
**HONF:** untouched. Freeze detail: [s4_scientific_freeze_candidate.md](s4_scientific_freeze_candidate.md).

---

## Attempt 1 — FAILED (preserved)

| Job ID | Model | Node / GPU | SHA | State | Cause |
|---|---|---|---|---|---|
| **29552531** | Sparse U-Net | `node3504` / L40S | `b844944…` | **FAILED** `1:0` (~14 s) | `ModuleNotFoundError: einops` (eager GLU import) |
| **29552532** | Adapted GLU | `node3507` / L40S | `b844944…` | **FAILED** `1:0` (~14 s) | same |

Dirs: `…/s4_profile_sparse_unet_29552531/`, `…/s4_profile_adapted_glu_29552532/`.

---

## Attempt 2 — COMPLETED diagnostic

Submitted `2026-11-18T12:25:12-05:00`. Env: `/home/bontempi/envs/official_thesis` (`einops==0.8.2`, torch `2.8.0+cu128`).

| Job ID | Model | Node / GPU | Queue≈ | Elapsed | Exit | Report status |
|---|---|---|---|---|---|---|
| **29554165** | Sparse U-Net | `node3008` / **L40S** | ~189 s | 00:01:08 | **0:0** | `completed_diagnostic` |
| **29554166** | Adapted GLU | `node3408` / **L40S** | ~189 s | 00:01:19 | **0:0** | `completed_diagnostic` |

Resources requested: `mit_normal_gpu` / `mit_amf_advanced_gpu` / QoS same / `gpu:l40s:1` / 64G / 8 CPU / wall 2:00:00.

### Absolute paths

**U-Net `29554165`**

- Run: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29554165/`
- Report: `…/profile_report.json` (sha256 `3e890902…`)
- Preflight: `…/dep_preflight.json`
- Short loop: `…/measured_short_loop/` (`summary.json`, `eval_history.json`, `run_identity.json`)
- Checkpoints (out of Git): `…/measured_short_loop/checkpoints/{best,last}.pt`, `…/diagnostic_last.pt` (sha256 `73db6235…`)
- Slurm: `/home/bontempi/orcd/pool/official_thesis/slurm-29554165.out`

**GLU `29554166`**

- Run: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29554166/`
- Report: `…/profile_report.json` (sha256 `123ed8e7…`)
- Checkpoints: `diagnostic_last.pt` sha256 `66c914aa…`
- Slurm: `/home/bontempi/orcd/pool/official_thesis/slurm-29554166.out`

Git compact copies: `results/month2/s4_profiling/` + `delivery_compact.json`.

### Provenance hashes (both jobs)

- Cache aggregate: `d0b1d06805562cda49a86ea9ab1f289ce202562549482533b272efb19a69c725`
- Geometry: `97e2e9df79c87a3fbf019fa1d9ac270aafe42319f83425afb5f3fb3be8f36ce7`
- Stats: `678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e`
- Mask manifest file: `2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e`

ExitCode 0 ≠ automatic scientific freeze: measures and checklist are in the freeze-candidate doc. VAL/inference RMSE diagnostic only. No auto-requeue. S5 **not** submitted.
