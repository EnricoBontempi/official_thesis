# Opus overnight handoff — wave-1 submission

**Date:** 2026-11-19 (America/New_York).  
**Branch:** `mission/dense-unet`.  
**Prep SHA:** `969b6c78a84465eb89b63bb5b7a0187c908652e5`.  
**Organization tomorrow:** Cursor follows Opus; a new Claude Code/Sonnet chat follows Astra; coordinated by a new Sol chat. Astra may use the four original S5 best checkpoints for its primary diagnosis — it does **not** wait for C149.

## Submitted wave-1 runs (authorized)

| Run ID | Pool | p | seeds (model=shuffle) | Purpose |
| --- | --- | --- | ---: | --- |
| `O-A38-on-s1` | A38 (760) | on | 1 | Training variability vs S5 seed 0 |
| `O-A38-off-s1` | A38 (760) | off | 1 | Same, p_off |
| `O-B19-on-s0` | B19 (380) | on | 0 | Fewer supervised frames, fixed update budget |
| `O-B19-off-s0` | B19 (380) | off | 0 | Same, p_off |

All: SparseUNet, TRAIN coverage 5%, `train_mask_seed=20261118`, `eval_mask_seed=20261116`, frozen fit-only 760-frame normalizer, scientific epoch = **760 presentations / 95 updates**, fresh init, no S5 warm start. **Not launched:** C149, GLU, Astra, S7, TEST.

**Engaging paths**

| Item | Absolute path |
| --- | --- |
| Code tip | `/home/bontempi/projects/official_thesis_opus` |
| Cache / masks / hashes (reuse S5 HOME) | `/home/bontempi/projects/official_thesis_s5/data/processed/month2_dense_unet_cache`, `.../month2_sparse_eval_masks`, `.../month2_dense_unet_cache_payload_hashes_s5_homecopy.json` |
| Outputs | `/home/bontempi/official_thesis_opus_runs/<RUN_ID>/` |
| Checkpoints | `.../checkpoints/{best,last,best_seen_diagnostic,partial_interrupt}.pt` |
| Logs | `.../stdout.log`, `.../slurm_context.txt`; Slurm `~/slurm-<jobid>.out` if default |

**Resources requested:** account/QoS `mit_amf_advanced_gpu`, partition `mit_normal_gpu`, `gres=gpu:l40s:1`, mem `64G`, time `06:00:00`, internal wall cap 5.5 h (checkpoint margin ≠ plateau).

## Common corrections reusable by Astra (Sonnet)

Do **not** reimplement from scratch; reuse these:

1. **YAML protocol parse** — `authorized_s6_offline` is a mapping with `items` + `note` (bare list + sibling key was invalid YAML). Loader allows `authorized_s6_offline` and `opus_wave1_prepared`.
2. **Eval mask integrity** — `SparseUNetCacheDataset(require_eval_mask_payload=True)` refuses silent seed regeneration when payloads are missing.
3. **Last eval persistence** — early-stop path now writes `eval_history.json` including the final eval (also still inside `last.pt`).
4. **Explicit precision** — Opus configs set `precision: fp32`, `amp: false`, `torch_compile: false`, `allow_tf32: false`; runner refuses amp/compile.
5. **Gappy gate empty reference** — `_gate_vs_historical` fails closed on empty/missing historical rows and requires `expected_n_checks=6`.
6. **Init manifest** — `init_manifest.json` with `init_state_dict_sha256` before any step (same seed ⇒ same hash for on/off).
7. **Scientific presentation stream** — `scientific_epoch_sampler.py`; B19 keeps 95 updates/epoch without shortening warmup. Sampler state in checkpoints for resume.

**Not done for Astra tonight:** no Astra code, no AO/oracle-offset, no T-anchor pipeline, no S5 best packaging beyond existing Engaging S5 runs.

## Files touched in this preparation (local)

- `configs/experiments/month2_sparse_protocol.yaml`
- `configs/experiments/opus_O-*.yaml` (four)
- `src/combustion_recon/data/sparse_protocol.py`
- `src/combustion_recon/training/scientific_epoch_sampler.py` *(new)*
- `src/combustion_recon/training/sparse_scientific_loop.py`
- `src/combustion_recon/training/sparse_dataset.py`
- `scripts/16_train_sparse_scientific.py`
- `scripts/15_gappypod_nested_coverage_eval.py` (gate empty-ref)
- `scripts/engaging/opus_O-*.sbatch` (four)
- `tests/test_scientific_epoch_sampler.py` *(new)*
- `docs/RESEARCH_LOG.md`, this handoff

## Opus work still open for tomorrow

| Step | Status |
| --- | --- |
| O0 remainder | Provenance ledger for S6 scripts/patches; regional aggregate tables; common-scale figures; Gappy swap; roadmapsparse status prose — **not** all closed tonight |
| O1 R0 selection reanalysis | **Deferred** (explicitly allowed to wait) |
| O2 C149 | Raw/cache availability **blocked** for increase-data arm; A/B proceeded |
| O2 bridge | A38 stream **order** matches S5 DataLoader seed formula; **full scientific equivalence with Engaging S5 run not claimed** |
| O3 C149 configs | Not prepared |
| O4 wave-2 C149 | Not authorized tonight |
| O5/O6 | After jobs finish — recover once when Enrico signals; no polling by agents |

## Hard rules for follow-on chats

- **Do not duplicate** these four runs or overwrite `official_thesis_opus_runs/<RUN_ID>/` if artifacts exist.
- **Do not** launch C149, GLU training, Astra jobs, S7, or TEST from this handoff alone.
- **Do not** touch HONF / ModularDT.
- Preserve concurrent untracked results (e.g. local `results/month2/sparse_gappypod_nested_1_20/` if present).
- Astra primary diagnosis uses **original S5 best** checkpoints; optional later Opus-checkpoint extensions are separate.

## Manual commands for Enrico

```bash
ssh engaging
squeue -u $USER -o '%.18i %.12P %.20j %.8u %.2t %.10M %R %b'
# per run
ls -lt ~/official_thesis_opus_runs/O-A38-on-s1/
tail -n 50 ~/official_thesis_opus_runs/O-A38-on-s1/stdout.log
# cancel only if needed (do not cancel HONF)
# scancel <jobid>
```

## Submission record

Submitted 2026-11-19T00:20:36-05:00 on Engaging (`mit_normal_gpu` / `mit_amf_advanced_gpu`, L40S×1, 64G, 6h). Remote note: `~/official_thesis_opus_runs/wave1_submission.txt`.

| Run ID | Job ID | State at submit |
| --- | --- | --- |
| O-A38-on-s1 | **29612010** | PD (submitted) |
| O-A38-off-s1 | **29612011** | PD (submitted) |
| O-B19-on-s0 | **29612012** | PD (submitted) |
| O-B19-off-s0 | **29612013** | PD (submitted) |

Prep SHA: `969b6c78a84465eb89b63bb5b7a0187c908652e5`  
No Astra activity executed in this session. No polling/relaunch after submit.
