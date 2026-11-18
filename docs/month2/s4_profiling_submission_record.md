# S4 profiling — submission record (not a freeze)

**Branch:** `mission/dense-unet`  
**SHA executed by attempt-2 jobs:** `740805f7e5537090a0f08478cdc1587ed6c9c5e3`  
**S4 status:** **IN CORSO** — attempt-2 **submitted**; **not** concluded; scientific configs **not** frozen.  
**HONF jobs 29547931 / 29547932:** untouched. S5: **not submitted**.

---

## Attempt 1 — FAILED (preserved; do not overwrite)

| Job ID | Model | Node / GPU | SHA | State | Cause |
|---|---|---|---|---|---|
| **29552531** | Sparse U-Net | `node3504` / L40S | `b844944…` | **FAILED** `1:0` (~14 s) | `ModuleNotFoundError: einops` via eager `models/__init__` → `glu_reconstructor` → vendored `external/GLU_Reconstruction/src/models.py:21` (`from einops import repeat`). No useful timing/measure. |
| **29552532** | Adapted GLU | `node3507` / L40S | `b844944…` | **FAILED** `1:0` (~14 s) | Same import failure before measure. |

Interpreter on both: `/home/bontempi/envs/official_thesis/bin/python` (torch `2.8.0+cu128`, CUDA 12.8). Run dirs kept:

- `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29552531/`
- `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29552532/`
- Slurm: `/home/bontempi/orcd/pool/official_thesis/slurm-29552531.out`, `…/slurm-29552532.out`

---

## Fix (env + import isolation; scientific contract unchanged)

- Install **only** in `~/envs/official_thesis` (HONF env not modified):  
  `/home/bontempi/envs/official_thesis/bin/pip install --disable-pip-version-check 'einops==0.8.2'`  
  → `einops 0.8.2` (torch left at `2.8.0+cu128`).
- Lazy GLU exports in `src/combustion_recon/models/__init__.py`; family-conditional GLU import in `s4_sparse_profile.py`; lazy GLU class check in `sparse_scientific_loop.py`.
- Launcher preflight: `scripts/engaging/s4_dep_preflight.py` (import/interpreter only; no CUDA kernels on login). Login-node preflight `ok` for both families before submit.
- Local tests: `30 passed` (`test_sparse_scientific_trainer`, `test_glu_adapted`, `test_sparse_unet`).

Provenance (unchanged payloads):

- Cache aggregate SHA256: `d0b1d06805562cda49a86ea9ab1f289ce202562549482533b272efb19a69c725`
- Geometry: `97e2e9df79c87a3fbf019fa1d9ac270aafe42319f83425afb5f3fb3be8f36ce7`
- Masks payload root: `/home/bontempi/orcd/pool/official_thesis_data/month2_sparse_eval_masks`
- Masks aggregate (Git manifest): `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb`

---

## Attempt 2 — submitted (await Enrico; no polling)

**SHA:** `740805f7e5537090a0f08478cdc1587ed6c9c5e3`  
**Env:** `/home/bontempi/envs/official_thesis` — `einops==0.8.2`; torch `2.8.0+cu128` / CUDA 12.8 (unchanged).  
**Submitted at:** `2026-11-18T12:25:12-05:00`

| Job ID | Model | JobName | Partition / Account / QoS | GRES | Wall | Mem / CPUs | State |
|---|---|---|---|---|---|---|---|
| **29554165** | Sparse U-Net | `s4_prof_unet` | `mit_normal_gpu` / `mit_amf_advanced_gpu` / `mit_amf_advanced_gpu` | `gpu:l40s:1` | 2:00:00 | 64G / 8 | **submitted** |
| **29554166** | Adapted GLU | `s4_prof_glu` | same | `gpu:l40s:1` | 2:00:00 | 64G / 8 | **submitted** |

New run dirs (attempt-1 not reused):

- U-Net: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29554165/` (`PREV_FAILED_REF=…/s4_profile_sparse_unet_29552531`)
- GLU: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29554166/` (`PREV_FAILED_REF=…/s4_profile_adapted_glu_29552532`)
- Expected artifacts: `dep_preflight.json`, `slurm_context.txt`, `stdout.log`, `profile_report.json` (if success — **not** claimed here)
- Slurm stdout: `/home/bontempi/orcd/pool/official_thesis/slurm-29554165.out`, `…/slurm-29554166.out`

Submit record: `/home/bontempi/orcd/pool/official_thesis_runs/launcher/s4_resubmit_740805f7e5537090a0f08478cdc1587ed6c9c5e3.txt`

One-shot queue note at submit: only unrelated HONF preemptable pending (`29547932`); new jobs PENDING on `mit_normal_gpu` with `gres/gpu:l40s:1`. Four “normal slots” are not guaranteed physical GPUs.

S5 screening remains **prepared only**; **not submitted**, no `afterok`, no auto-relaunch on error/OOM.

### Manual checks (Enrico)

```bash
sacct -j 29554165,29554166 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,AllocTRES%40
sed -n '1,80p' /home/bontempi/orcd/pool/official_thesis/slurm-29554165.out
sed -n '1,80p' /home/bontempi/orcd/pool/official_thesis/slurm-29554166.out
cat /home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29554165/slurm_context.txt
cat /home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29554166/slurm_context.txt
```
