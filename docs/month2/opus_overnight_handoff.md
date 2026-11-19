# Opus overnight handoff — wave-1 + C149 wave-2

**Branch:** `mission/dense-unet`.  
**Wave-1 prep SHA:** `969b6c78a84465eb89b63bb5b7a0187c908652e5`.  
**C149 prep SHA:** *(filled at C149 commit; see submission record)*.  
**Organization tomorrow:** Cursor → Opus; Sonnet → Astra (original S5 best; does **not** wait for C149); Sol coordinates.

## Six new Opus trainings (matrix)

| Run ID | Pool | p | seeds | Job / dependency | Tip |
| --- | --- | --- | ---: | --- | --- |
| `O-A38-on-s1` | A38 (760) | on | 1 | **29612010** | `official_thesis_opus` |
| `O-A38-off-s1` | A38 (760) | off | 1 | **29612011** | same |
| `O-B19-on-s0` | B19 (380) | on | 0 | **29612012** | same |
| `O-B19-off-s0` | B19 (380) | off | 0 | **29612013** | same |
| `O-C149-on-s0` | C149 (2980) | on | 0 | afterok ← cache prepare | `official_thesis_opus_c149` |
| `O-C149-off-s0` | C149 (2980) | off | 0 | afterok ← cache prepare | same |

All: SparseUNet, TRAIN 5%, `train_mask_seed=20261118`, `eval_mask_seed=20261116`, historical fit-only **760-frame** normalizer, scientific epoch **760/95**, fresh init. **Not launched:** GLU, Astra, S7, TEST.

## C149 data path (facts)

- **Raws verified locally (WSL):** `data/raw/combustion/hf_dataset/numerical/` (30 Arrow shards, ~59 GB). TRAIN+VAL = 23 shards (~48 GB).
- **Engaging:** no raw Arrow tree under pool/HOME at check time; pool held only historical dense cache + masks.
- **Construction:** densified cache built with streaming `ShardStore` (one condition prefetch) via `scripts/18_build_opus_c149_cache.py` from local raws; **not** a Hub re-download of 70 GB.
- **Gates passed locally:** counts 2980/220/123; B19⊂A38⊂C149; train∩seen=∅; **760 shared A38 frames bit-identical** vs historical cache; geometry identical; `40NH3_1.2` anomaly retained (warning only).
- **Staged on Engaging:** `/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149/` (~873 MB) + `payload_hashes.json` + `c149_build_verify_report.json`.
- **CPU prepare job:** verify-only re-check against S5 historical cache; **fails closed** before GPU `afterok` dependents start. Does **not** claim Engaging raw rebuild.

**Still open / declared limits:** Engaging raw rebuild not executed (raws not on cluster); full scientific equivalence of Opus runner vs historical Engaging S5 **not** claimed; O1 R0 deferred; Astra untouched.

## Paths

| Item | Absolute path |
| --- | --- |
| Wave-1 tip (DO NOT update while jobs run) | `/home/bontempi/projects/official_thesis_opus` |
| C149 tip (separate frozen checkout) | `/home/bontempi/projects/official_thesis_opus_c149` |
| Astra tip (DO NOT modify) | `/home/bontempi/projects/official_thesis_astra_frozen` |
| C149 cache | `/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149` |
| Historical cache / masks (S5 HOME) | `/home/bontempi/projects/official_thesis_s5/data/processed/...` |
| Wave-1 outs | `/home/bontempi/official_thesis_opus_runs/<RUN_ID>/` |
| C149 outs | `/home/bontempi/official_thesis_opus_c149_runs/<RUN_ID>/` |

## Resources

- GPU trainings: `mit_normal_gpu` + account/QoS `mit_amf_advanced_gpu`, L40S×1, 64G, 6h (internal 5.5h).
- CPU prepare: `mit_normal` + account/QoS `mit_amf_advanced_cpu`, 8 CPU, 64G, 4h.
- No general/preemptable fallback. HONF untouched.

## Common corrections (Astra may reuse)

Same as wave-1 handoff (YAML parse, eval mask payload gate, eval_history on early-stop, fp32/amp/compile, Gappy empty-ref, init manifest, scientific stream) **plus**:

8. **C149 densified cache** — `month2_frame_protocol_opus_c149.yaml`, `OPUS_C149_CACHE_ID`, shared-A38 bit-identity gate vs historical, `scripts/18_build_opus_c149_cache.py`.
9. **reference760 eval** — even on C149/B19, TRAIN diagnostic uses A38 indices only.

## Hard rules

- Do **not** duplicate/overwrite wave-1 run dirs or update `official_thesis_opus` tip while 29612010–13 run.
- Do **not** modify `official_thesis_astra_frozen` or include Astra edits in Opus commits.
- Do **not** launch GLU / S7 / TEST / Neon from this handoff.
- No polling/relaunch by agents after submit.

## Manual commands

```bash
ssh engaging
squeue -u $USER -o '%.18i %.12P %.20j %.2t %.10M %R %b'
tail -n 40 ~/official_thesis_opus_c149_runs/c149_cache_prepare/stdout.log
tail -n 40 ~/official_thesis_opus_c149_runs/O-C149-on-s0/stdout.log
```

## Submission record

### Wave-1 (already submitted)

| Run ID | Job ID | State at submit |
| --- | --- | --- |
| O-A38-on-s1 | **29612010** | submitted 2026-11-19T00:20:36-05:00 |
| O-A38-off-s1 | **29612011** | submitted |
| O-B19-on-s0 | **29612012** | submitted |
| O-B19-off-s0 | **29612013** | submitted |

### Wave-2 C149 *(filled at submit)*

| Step | Job ID | Notes |
| --- | --- | --- |
| `opus_c149_cache` prepare/verify | TBD | CPU Advanced; verify-only |
| `O-C149-on-s0` | TBD | `--dependency=afterok:<cache>` |
| `O-C149-off-s0` | TBD | `--dependency=afterok:<cache>` |

C149 prep SHA: TBD  
No Astra activity in this Opus session.
