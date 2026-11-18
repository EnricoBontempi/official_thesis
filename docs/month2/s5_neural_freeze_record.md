# S5 neural protocol freeze + submission record

**Branch:** `mission/dense-unet`  
**Freeze / execution SHA:** `797134348e78feb9c4d26ada1157fc03cbbf9f9d`  
**Profiling SHA (reference):** `740805f7e5537090a0f08478cdc1587ed6c9c5e3` (jobs 29554165 / 29554166)  
**S4:** neural feasibility verified; GappyPOD 1/20% **still open** — S4 not declared fully complete.  
**S5 status:** **submitted** (`2026-11-18T12:47:33-05:00`). Runs **not** declared completed here.

## Checkpoint correction (pre-launch)

- `best.pt` now written **after** plateau / early-stop counters / history / stop-status updates for the eval epoch.
- Absolute-best **weights** remain the selection snapshot; state is epoch-boundary coherent.
- Atomic `torch.save` via temp + `os.replace`.
- Partial checkpoints remain separate; never overwrite complete `last.pt`.
- Tests: `tests/test_sparse_scientific_trainer.py` (7 passed) — warmup crossed, LR reduce asserted, early-stop reset, continuous vs resume, resume-from-best + extend, partial/incompat reject.

## Frozen runs (seed 0, fresh init)

| Job ID | run_id | Family / p | Config | config sha256 |
|---|---|---|---|---|
| **29555700** | `s5_sparse_unet_pon_seed0` | U-Net p_on | `configs/experiments/s5_sparse_unet_pon_seed0.yaml` | `be71eafd740cd40e3f757d8cd1bcd38cfe81a416914f5129d1d656fe0c873772` |
| **29555701** | `s5_sparse_unet_poff_seed0` | U-Net p_off | `configs/experiments/s5_sparse_unet_poff_seed0.yaml` | `e6a56bbfec29e674c3f89f86ef29459bbe345d7d0ed05477959d34e239013859` |
| **29555702** | `s5_adapted_glu_pon_seed0` | GLU p_on | `configs/experiments/s5_adapted_glu_pon_seed0.yaml` | `cd72809489ab8d32662abdbd82eac9114c646c12bc1b275f8fceb79a6c46886b` |
| **29555703** | `s5_adapted_glu_poff_seed0` | GLU p_off | `configs/experiments/s5_adapted_glu_poff_seed0.yaml` | `fbe862fc82982681d90b350bf068804efc4e03ee7bd98330d2f557983003e977` |

Contract: 5% only, K=0, batch 8, microbatch null, 16384 GLU queries, fp32, AdamW 1e-3/1e-4, clip 1, warmup 950, plateau/early-stop per roadmap, VAL 5%×3 every 10 epochs, tranche 1000, no S4 weight reuse. `p_off` zeros parameter channels / GLU `p` per batch.

## Operational resources (each job)

- Partition/account/QoS: `mit_normal_gpu` / `mit_amf_advanced_gpu` / `mit_amf_advanced_gpu`
- GRES: `gpu:l40s:1` (non-preemptable); mem 64G; 8 CPU; Slurm wall `06:00:00`
- Internal `wall_time_cap_hours: 5.5` (margin under Slurm; incomplete epoch → partial interrupt, not “converged”)
- No automatic requeue; no preemptable fallback; HONF untouched
- One-shot queue at submit: empty personal queue before submit; all four PENDING on normal L40S after submit

## Absolute paths (Engaging)

| run_id | Run dir | Expected artifacts |
|---|---|---|
| s5_sparse_unet_pon_seed0 | `/home/bontempi/orcd/pool/official_thesis_runs/s5_sparse_unet_pon_seed0/` | `progress.json`, `checkpoints/{best,last}.pt`, `stdout.log`, `slurm_context.txt` |
| s5_sparse_unet_poff_seed0 | `/home/bontempi/orcd/pool/official_thesis_runs/s5_sparse_unet_poff_seed0/` | same |
| s5_adapted_glu_pon_seed0 | `/home/bontempi/orcd/pool/official_thesis_runs/s5_adapted_glu_pon_seed0/` | same |
| s5_adapted_glu_poff_seed0 | `/home/bontempi/orcd/pool/official_thesis_runs/s5_adapted_glu_poff_seed0/` | same |

Slurm stdout: `/home/bontempi/orcd/pool/official_thesis/slurm-29555700.out` … `slurm-29555703.out`  
Launcher record: `/home/bontempi/orcd/pool/official_thesis_runs/launcher/s5_submit_797134348e78feb9c4d26ada1157fc03cbbf9f9d.txt`

## Provenance (Engaging payloads)

- Cache aggregate: `d0b1d06805562cda49a86ea9ab1f289ce202562549482533b272efb19a69c725`
- Geometry: `97e2e9df79c87a3fbf019fa1d9ac270aafe42319f83425afb5f3fb3be8f36ce7`
- Stats: `678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e`
- Masks aggregate (Git manifest): `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb`
- Mask manifest file sha: `2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e`
- Env: `/home/bontempi/envs/official_thesis` (torch `2.8.0+cu128`, einops `0.8.2`)

## Duration note

S4 **estimates** (not measured S5 runtime): U-Net ~2–3 h/run, GLU ~3–4.5 h/run, high uncertainty, possible early stop. Actual duration unknown until Enrico inspects.

## Failed first attempt and launcher fix (2026-11-18)

**Correction to the status above:** jobs 29555700–29555703 were reported FAILED (ExitCode 1:0, 0 s); no S5 training ran. They are preserved as failed attempts, with no scientific result.

- **Verified bug (local reproduction):** launchers did `source "$(dirname "$0")/_s5_train_common.inc.sh"`. Under `sbatch`, `$0` is the Slurm spool copy, so the include does not exist. A copy of the old launcher run outside the repo directory exits rc=1 at that line, matching `1:0` / 0 s.
- **Not confirmed:** the Slurm logs (`slurm-2955570x.out`) were not readable, so the historical cause is inferred, not read from logs. The inference is consistent but unproven.
- **Fix:** the common env is inlined in each `scripts/engaging/s5_*.sbatch`; `_s5_train_common.inc.sh` removed. The S5 config, data, masks and protocol are unchanged.
- **Checks:** `bash -n` on all four launchers; the fixed launcher, run from a spool-like copy with a fake HOME and a stub python, reaches the training entrypoint with the expected arguments. The gate exits 78 without `AUTHORIZE_S5_SUBMIT=1`. No CUDA or training was run.
- **Submission blocked (2026-11-18 ~14:00 EST):** on login008 (via a new channel on the existing `engaging` mux master, no session closed), `/home` and `~/envs/official_thesis` respond, but `ls -d /orcd/pool/005/bontempi` timed out (`timeout 6`, rc=124). The code checkout and data are in that pool, so the remote checkout could not be updated or verified and **no new jobs were submitted**.

## Pool unavailable: home-copy relaunch preparation (2026-11-18)

`/orcd/pool/005/bontempi` (autofs, host login008) does not complete its mount (`ls` rc=124), while other pools and scratch respond; quota POOL 1.4/1024 GB. Pool left untouched. The S5 runs need only compact inputs, so they use a separate checkout in HOME:

- Checkout: `/home/bontempi/projects/official_thesis_s5` (git archive of the commit + `.git_commit`); outputs: `/home/bontempi/official_thesis_s5_runs/<run_id>/` (new root; failed attempts 2955570x preserved untouched).
- Inputs transferred (ignored by Git): `data/processed/month2_dense_unet_cache` (277 MB), `data/processed/month2_sparse_eval_masks` (25 MB). Stats and hashes come from Git.
- **Environment:** the conda env's editable-install `.pth` pointed at `/orcd/pool/005/bontempi/official_thesis/src`, so *every* Python start hung (user CPU 0, `-S` ok, `-I` hang). Repointed to the new checkout's `src`; backup at `/home/bontempi/envs/__editable__.combustion_recon-0.1.0.pth.pool_backup`. `import torch,einops,yaml,numpy` then works (7.8 s: slow cold import, not an error; torch 2.8.0+cu128, einops 0.8.2).
- **Cache manifest difference:** the local `manifest.json` has sha256 `54f16da8890a43ba7f9a7149d6802607e73a083edb5ebd217319e3311140d350`; the frozen pool copy was `56e988e8b89b436c0e96659e1103359b658ae08a60160a635ec39b1ef2aa36f0`. The 19 payload files (geometry, 3 roles × frames/p/targets/vi/vj) are hash-identical to the frozen set.
  - Evidence of provenance-only change: local manifest was re-refreshed 2026-11-18 (`metadata_refreshed_commit` 9185807, vs 3643ce5 in the Nov-17 tracked copy, `data/manifests/month2_dense_unet_cache_manifest.json`). All shared fields of that tracked copy are equal (counts, data_fingerprints, model_fingerprints, moments_train_fit, bytes_total, cache_id, schema version, channel_order, experiment_id, shard policy) except `metadata_refreshed_commit`; payload bytes sum equals `bytes_total`. Independently: roles 220/760/123 frames, arrays float32 (N,128,128)/(N,2), geometry and stats sha match manifest fingerprints, `validate_dense_cache_manifest` passes.
  - **Limit:** the original pool `manifest.json` could not be recovered, so keys absent from the tracked copy (shard_provenance, created_unix, build_wall_s, git_commit, etc.) were not diffed against it.
- **Hash files:** frozen `data/manifests/month2_dense_unet_cache_payload_hashes.json` (aggregate `d0b1d068…`) preserved. New `data/manifests/month2_dense_unet_cache_payload_hashes_s5_homecopy.json` (aggregate `710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d`; only `manifest.json` differs). Integrity check on the manifest stays active; launchers point to the new file.
- Launcher defaults changed (operational paths only): TIP, CACHE, HASHES, MASKS, OUT_ROOT. Mask manifest sha `2e2d8ff6…` and stats sha `678263fb…` unchanged. Protocol/configs unchanged.
