# S4 profiling — submission record (not a freeze)

**Branch:** `mission/dense-unet`  
**SHA executed by jobs:** `b844944abc17ecbcdf83d044fc4b8f5ccf8db520`  
**S4 status:** **IN CORSO** — jobs submitted; **not** concluded; scientific configs **not** frozen.  
**HONF jobs 29547931 / 29547932:** untouched.

One-shot descriptive `scontrol show job` (no monitoring loop). Observed after Enrico’s PD check: both jobs reached compute nodes then **FAILED** (`ExitCode=1:0`, `Reason=NonZeroExitCode`). Failure signature in Slurm stdout: `ModuleNotFoundError: No module named 'einops'` during import of vendored GLU (also pulled via `models/__init__.py` for the U-Net job). **No automatic relaunch in this turn.**

---

## Job ↔ model association

| Job ID | Model | JobName | Partition / Account / QoS | GRES requested | Allocated node / GPU | Wall | Mem / CPUs | State (one-shot) |
|---|---|---|---|---|---|---|---|---|
| **29552531** | Sparse U-Net | `s4_prof_unet` | `mit_normal_gpu` / `mit_amf_advanced_gpu` / `mit_amf_advanced_gpu` | `gpu:l40s:1` | `node3504` / **NVIDIA L40S** | 2:00:00 | 64G / 8 | **FAILED** |
| **29552532** | Adapted GLU | `s4_prof_glu` | same | `gpu:l40s:1` | `node3507` / **NVIDIA L40S** | 2:00:00 | 64G / 8 | **FAILED** |

Submit record file: `/home/bontempi/orcd/pool/official_thesis_runs/launcher/s4_submission_b844944abc17ecbcdf83d044fc4b8f5ccf8db520.txt`

---

## Absolute paths

### U-Net `29552531`
- Run dir: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29552531/`
- Context: `.../slurm_context.txt`
- Script tee log: `.../stdout.log` (empty — failed before profile body)
- Slurm stdout: `/home/bontempi/orcd/pool/official_thesis/slurm-29552531.out`
- Expected report if success: `.../profile_report.json` (**absent**)

### GLU `29552532`
- Run dir: `/home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29552532/`
- Context: `.../slurm_context.txt`
- Script tee log: `.../stdout.log` (empty)
- Slurm stdout: `/home/bontempi/orcd/pool/official_thesis/slurm-29552532.out`
- Expected report if success: `.../profile_report.json` (**absent**)

### Checkout / env
- Code: `/home/bontempi/orcd/pool/official_thesis` (`.git_commit` = `b844944…`)
- Env: `/home/bontempi/envs/official_thesis`
- Cache: `/home/bontempi/orcd/pool/official_thesis_data/month2_dense_unet_cache`
- Masks: `/home/bontempi/orcd/pool/official_thesis_data/month2_sparse_eval_masks`

---

## Manual checks (Enrico)

```bash
sacct -j 29552531,29552532 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList,AllocTRES%40
sed -n '1,80p' /home/bontempi/orcd/pool/official_thesis/slurm-29552531.out
sed -n '1,80p' /home/bontempi/orcd/pool/official_thesis/slurm-29552532.out
cat /home/bontempi/orcd/pool/official_thesis_runs/s4_profile_sparse_unet_29552531/slurm_context.txt
cat /home/bontempi/orcd/pool/official_thesis_runs/s4_profile_adapted_glu_29552532/slurm_context.txt
```

S5 screening is **prepared only** in-repo; **not submitted**, no `afterok`, no auto-launch.
