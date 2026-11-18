# S5 neural protocol freeze + submission record

**Branch:** `mission/dense-unet`  
**Freeze / execution SHA:** _(filled at commit/submit)_  
**Profiling SHA (reference):** `740805f7e5537090a0f08478cdc1587ed6c9c5e3` (jobs 29554165 / 29554166)  
**S4:** neural feasibility verified; GappyPOD 1/20% **still open** — S4 not declared fully complete.  
**S5 status:** **submitted** (this record). Runs **not** declared completed here.

## Checkpoint correction (pre-launch)

- `best.pt` now written **after** plateau / early-stop counters / history / stop-status updates for the eval epoch.
- Absolute-best **weights** remain the selection snapshot; state is epoch-boundary coherent.
- Atomic `torch.save` via temp + `os.replace`.
- Partial checkpoints remain separate; never overwrite complete `last.pt`.

## Frozen runs (seed 0, fresh init)

| run_id | Family | p | Config |
|---|---|---|---|
| `s5_sparse_unet_pon_seed0` | sparse_unet | on | `configs/experiments/s5_sparse_unet_pon_seed0.yaml` |
| `s5_sparse_unet_poff_seed0` | sparse_unet | off | `configs/experiments/s5_sparse_unet_poff_seed0.yaml` |
| `s5_adapted_glu_pon_seed0` | adapted_glu | on | `configs/experiments/s5_adapted_glu_pon_seed0.yaml` |
| `s5_adapted_glu_poff_seed0` | adapted_glu | off | `configs/experiments/s5_adapted_glu_poff_seed0.yaml` |

Contract: 5% only, K=0, batch 8, microbatch null, 16384 GLU queries, fp32, AdamW 1e-3/1e-4, clip 1, warmup 950, plateau/early-stop per roadmap, VAL 5%×3 every 10 epochs, tranche 1000, no S4 weight reuse. `p_off` zeros parameter channels / GLU `p` per batch.

## Operational resources

- Partition/account/QoS: `mit_normal_gpu` / `mit_amf_advanced_gpu` / `mit_amf_advanced_gpu`
- GRES: `gpu:l40s:1` (non-preemptable); mem 64G; 8 CPU; Slurm wall `06:00:00`
- Internal `wall_time_cap_hours: 5.5` (margin under Slurm; incomplete epoch → partial interrupt, not “converged”)
- No automatic requeue; no preemptable fallback; HONF untouched

## Provenance (Engaging payloads)

- Cache aggregate: `d0b1d06805562cda49a86ea9ab1f289ce202562549482533b272efb19a69c725`
- Geometry: `97e2e9df79c87a3fbf019fa1d9ac270aafe42319f83425afb5f3fb3be8f36ce7`
- Stats: `678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e`
- Masks aggregate (Git manifest): `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb`
- Env: `/home/bontempi/envs/official_thesis` (torch `2.8.0+cu128`, einops `0.8.2`)

## Job IDs (filled at submit)

| Job ID | run_id | Model / p |
|---|---|---|
| TBD | s5_sparse_unet_pon_seed0 | U-Net p_on |
| TBD | s5_sparse_unet_poff_seed0 | U-Net p_off |
| TBD | s5_adapted_glu_pon_seed0 | GLU p_on |
| TBD | s5_adapted_glu_poff_seed0 | GLU p_off |

Run roots: `/home/bontempi/orcd/pool/official_thesis_runs/<run_id>/`  
Slurm stdout: `/home/bontempi/orcd/pool/official_thesis/slurm-<jobid>.out`

## Duration note

S4 **estimates** (not measured S5 runtime): U-Net ~2–3 h/run, GLU ~3–4.5 h/run, high uncertainty, possible early stop. Actual duration unknown until Enrico inspects.
