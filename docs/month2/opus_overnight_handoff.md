# Opus overnight handoff — wave-1 + C149 wave-2

**Branch:** `mission/dense-unet`.  
**Wave-1 prep SHA:** `969b6c78a84465eb89b63bb5b7a0187c908652e5`.  
**C149 prep SHA:** `07646e884d89da9b04383b77b6c71802f8870a81`.  
**Organization tomorrow:** Cursor → Opus; Sonnet → Astra (original S5 best; does **not** wait for C149); Sol coordinates.

## Six new Opus trainings (matrix)

| Run ID | Pool | p | seeds | Job / dependency | Tip |
| --- | --- | --- | ---: | --- | --- |
| `O-A38-on-s1` | A38 (760) | on | 1 | **29612010** | `official_thesis_opus` |
| `O-A38-off-s1` | A38 (760) | off | 1 | **29612011** | same |
| `O-B19-on-s0` | B19 (380) | on | 0 | **29612012** | same |
| `O-B19-off-s0` | B19 (380) | off | 0 | **29612013** | same |
| `O-C149-on-s0` | C149 (2980) | on | 0 | **29613545** afterok←**29613544** | `official_thesis_opus_c149` |
| `O-C149-off-s0` | C149 (2980) | off | 0 | **29613546** afterok←**29613544** | same |

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

### Wave-2 C149 (submitted 2026-11-19T00:46:52-05:00)

| Step | Job ID | Notes |
| --- | --- | --- |
| `opus_c149_cache` prepare/verify | **29613544** | CPU Advanced; verify-only on staged cache |
| `O-C149-on-s0` | **29613545** | `--dependency=afterok:29613544` |
| `O-C149-off-s0` | **29613546** | `--dependency=afterok:29613544` |

Remote note: `~/official_thesis_opus_c149_runs/wave2_c149_submission.txt`.  
C149 prep SHA: `07646e884d89da9b04383b77b6c71802f8870a81`.  
Wave-1 tip left at `969b6c7`. Astra tip not modified (Astra GPU jobs observed pending separately). No polling after submit.

## Morning read and C149 attempt2 (submitted, not finished)

Wave-1 jobs **29612010–29612013** reached early-stop with complete checkpoints. Verify **29613544** completed; its remote report is `status=ok`. GPU jobs **29613545** and **29613546** exited 1 in seconds on L40S. `SparseUNetCacheDataset` called `validate_dense_cache_manifest` with the historical allowlist only, so cache id `month2_dense_unet_cache_opus_c149_v1` was refused before epoch 0. Those run directories and `slurm-29613545.out` / `slurm-29613546.out` stay in place.

Attempt2 does not resume and does not reuse those directories. Executed code is `d6332000491ba0e8983b9e7fa09b34badb191ea4` on `/home/bontempi/projects/official_thesis_opus_c149_attempt2` (detached, clean except `.git_commit`). The previous C149 checkout remains `f061df1`. Output root: `/home/bontempi/official_thesis_opus_c149_attempt2_runs/`. Cache and verify report of 29613544 were reused after the login gate matched id, counts 2980/220/123, shared-A38 760 frames at max abs 0, and payload aggregate `fa1c9358deb3bda0313cd74f9ab6855af87bc969c4069a463073d4c4120f03b4`.

| Scientific run | Attempt2 job | State at submit | Preserved failure |
| --- | --- | --- | --- |
| O-C149-on-s0 | **29645053** | PENDING 2026-11-19T09:38:08-05:00 | 29613545 |
| O-C149-off-s0 | **29645054** | PENDING 2026-11-19T09:38:08-05:00 | 29613546 |

Resources: `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, 6h. No general/preemptable fallback. The table above is the state at submit. The outcome is the next section.

## 2026-11-19 outcome — six early-stops, bridge, R0

Slurm `COMPLETED` with exit 0 is the early-stop boundary. `summary.completed` stays false.

| Run | Job | Epochs | Best VAL (K) | Best epoch | Output |
| --- | --- | ---: | ---: | ---: | --- |
| O-A38-on-s1 | 29612010 | 630 | 48.219 | 60 | `~/official_thesis_opus_runs/O-A38-on-s1` |
| O-A38-off-s1 | 29612011 | 610 | 43.056 | 40 | `~/official_thesis_opus_runs/O-A38-off-s1` |
| O-B19-on-s0 | 29612012 | 620 | 43.929 | 50 | `~/official_thesis_opus_runs/O-B19-on-s0` |
| O-B19-off-s0 | 29612013 | 630 | 46.737 | 60 | `~/official_thesis_opus_runs/O-B19-off-s0` |
| O-C149-on-s0 attempt2 | 29645053 | 660 | 38.712 | 90 | `~/official_thesis_opus_c149_attempt2_runs/O-C149-on-s0` |
| O-C149-off-s0 attempt2 | 29645054 | 710 | 39.824 | 260 | `~/official_thesis_opus_c149_attempt2_runs/O-C149-off-s0` |

Attempt1 C149 directories and the attempt2 checkout at `d633200` stay untouched. Wave-1 tip stays at `969b6c7`.

**Bridge.** Static diff plus a synthetic check on torch 2.8: the new A38 stream is a raw `randperm` on `shuffle_seed + 100003*(epoch+1)`. Executed S5 used `DataLoader(shuffle=True)`, which draws one integer from that generator before `randperm`. Epoch 0 starts `[26, 184, 591, …]` versus `[193, 743, 248, …]`. Same 760 indices, all 95 batches differ. Optimizer, warmup, plateau and TF32-off match. S5 saved no init hash. This blocks using historical S5 as the seed-0 A38 arm of the new runner. Ledger: `results/month2/opus_data_study/bridge_s5/LEDGER.md`.

**Inference replay, separate from that order check.** Job **29670888** submitted, not read. L40S, Advanced+normal, 1h. Checkout `/home/bontempi/projects/official_thesis_opus_bridge` detached at `bc7ab22a54c1bfb2a97c9a19fa96745a7aee1493`. Report path: `/home/bontempi/official_thesis_opus_bridge_runs/replay_unet_best_5pct.json`. It replays the two U-Net `best.pt` at 5% against the recorded VAL macro, tolerance 0.05 K. It does not prove training equivalence.

**R0 done** from recorded curves only: `results/month2/opus_data_study/r0_selection/`. Original best and last kept. Not an independent validation.

**O5 not prepared and not submitted.** Two new A38 seed-0 runs on the scientific stream would be required before a seed-0 data comparison. They were not launched. Opus remains open. No S7, A4, or TEST.

## A38 seed-0 references (submitted 2026-11-19T14:19:40-05:00)

S5 cannot be the seed-0 A38 arm of the Opus runner: its DataLoader order differs from the stream used by the six trainings. Enrico authorized two fresh SparseUNet references on that same stream. Only `seed` and `shuffle_seed` change from the A38 seed-1 contract, both to 0. The stream was not edited to imitate S5. No GLU and no O5.

Replay **29670888** was RUNNING on an L40S at 14:13 (startup lines only, empty stderr, no report) and still running at this submission. No identity error was visible. It was not awaited. Inference replay stays separate from the shuffle finding.

| Run | Job | State at submit | Checkout |
| --- | --- | --- | --- |
| O-A38-on-s0 | **29671719** | PENDING | `/home/bontempi/projects/official_thesis_opus_a38s0` @ `32a59cab21884d8c052f941f526c7cf09367b1f3` |
| O-A38-off-s0 | **29671721** | PENDING | same detached checkout |

Output root: `/home/bontempi/official_thesis_opus_a38s0_runs/`. Logs: `slurm-29671719.out/.err` and `slurm-29671721.out/.err`. Consumed cache, mask, and stats hashes are written to `slurm_context_O-A38-on-s0.txt` and `slurm_context_O-A38-off-s0.txt` when each job passes its gate. Expected cache id `month2_dense_unet_cache_v1`, aggregate `710bab909a0efb6cc1b4b4a0d7f7dcc4889472b7adc8076abb94fd529576ab5d`. C149 cache id is refused.

Resources, same as the A38 seed-1 freeze: `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, Slurm 6h, internal cap 5.5h. QoS MaxTRESPU is cpu=64, gres/gpu=4, mem=1T. The replay held one GPU, so both references fit under the user cap and were pending for partition resources. Prior run directories, Astra, and HONF were not modified.

```bash
ssh engaging 'squeue -j 29671719,29671721 -o "%.18i %.24j %.20P %.24a %.24q %.2t %.10M %R"'
ssh engaging 'sacct -j 29671719,29671721 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'tail -n 40 ~/official_thesis_opus_a38s0_runs/O-A38-on-s0/stdout.log'
```

```bash
ssh engaging 'sacct -j 29670888 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'cat ~/official_thesis_opus_bridge_runs/replay_unet_best_5pct.json'
```

## 2026-11-19 verified outcomes and O5 preparation

One read after the queue was empty. No forward on the login. The earlier PENDING lines above stay as the submission record.

| Job | State | Exit | Elapsed | Node | Stop |
| --- | --- | --- | ---: | --- | --- |
| 29671719 O-A38-on-s0 | COMPLETED | 0:0 | 03:42:22 | node2804 | early-stop, 630 epochs, best VAL 42.389 K at epoch 120 |
| 29671721 O-A38-off-s0 | COMPLETED | 0:0 | 03:55:50 | node3008 | early-stop, 660 epochs, best VAL 42.915 K at epoch 90; seen diagnostic epoch 130 |
| 29670888 replay | COMPLETED | 0:0 | 00:06:26 | node2804 | inference only, both U-Net deltas 0.0 K, tolerance 0.05 K |

`summary.completed` is false for both trainings. Histories contain the stop epoch and match the summary minima. Stderr is empty. No partial, NaN, OOM, or timeout. Executed SHA remains `32a59ca` on the frozen checkout. The replay report is `passed: true` with `training_equivalence: not_tested`. It does not cancel the S5/Opus shuffle difference.

**O5 is prepared and not submitted.** Matrix: `configs/experiments/opus_o5_matrix.json`. Eight Opus-runner runs only. Index for Neon: `results/month2/opus_data_study/o5_prep/INDEX.md`. Loop curves in that directory are labeled `training_loop_not_o5`. R0 is unchanged. Launcher: `scripts/engaging/opus_o5_eval.sbatch`, array `0-7%4`, `mit_normal_gpu` / `mit_amf_advanced_gpu`, L40S×1, 8 CPU, 64G, wall 4h. It refuses without `AUTHORIZE_OPUS_O5=1` and `EXPECT_SHA`.

Proposed paths, to be created only at submission:

| Item | Path |
| --- | --- |
| Detached checkout | `/home/bontempi/projects/official_thesis_opus_o5` |
| Output, one directory per run | `/home/bontempi/official_thesis_opus_o5_runs/<run_id>/` |
| Logs | `/home/bontempi/official_thesis_opus_o5_runs/slurm-%A_%a.out` |

Forward estimate from the measured replay: about 26 minutes per distinct checkpoint, about 9.2 GPU-hours for the 21 distinct weight sets, about 12 GPU-hours if every frame costs the reference760 rate. The 4h wall covers that slower estimate plus the cache rehash. Three pairs may share a forward because the weight hashes match: `O-B19-on-s0`, `O-A38-on-s0`, `O-A38-on-s1`. Their checkpoint files stay distinct.

```bash
# Not run. Create the log directory and a detached checkout of the O5 commit first.
AUTHORIZE_OPUS_O5=1 EXPECT_SHA=<o5-commit> sbatch scripts/engaging/opus_o5_eval.sbatch
```

**Still open.** O5 execution, the post-job aggregation and common-scale figures, and the scientific report. O6 is not started. S7 is not chosen. A4 is not authorized. Opus is not closed.

## 2026-11-19 O5 submission refused before sbatch

Enrico authorized `64d0cc321dc5bc77ae48e44b063ee7a05cd87e48`. No array was submitted. The queue stayed empty.

The detached checkout `/home/bontempi/projects/official_thesis_opus_o5` is at that SHA, with `.git_commit` matching and no other porcelain. Training checkouts, Astra, and the bridge tip were read and left at their previous SHAs. `/home/bontempi/official_thesis_opus_o5_runs` was not created.

Checks that passed: QoS `mit_amf_advanced_gpu` MaxTRESPU cpu=64, gres/gpu=4, mem=1T; partition `mit_normal_gpu` MaxTime 6h; both historical and C149 cache ids and recorded payload aggregates; normalizer file sha `678263fb…`; all 24 checkpoint file hashes; training `run_identity` git and pool. Mask payload files match `manifest.json` (`verify_payload_against_manifest` ok, replicas 0/1/2, 3309 instances). `require_eval_mask_payload` refuses seed regeneration.

The blocking contrast is two different mask hashes:

| Object | SHA-256 |
| --- | --- |
| `manifest.json` file, recorded by training as `MASK_MANIFEST_SHA256` and `mask_payload_sha256` | `2e2d8ff6edabf62ad56702a155140d064c0b4014d06ee96debf52811efa5c75e` |
| Field `aggregate_payload_sha256` inside that manifest | `7be78f35ffea2ebc598adce8cec39dc719339aac17ecadde3137f2064891ffeb` |

`configs/experiments/opus_o5_matrix.json` stores the file hash. `scripts/26_opus_o5_eval.py` compares it with `verify_payload_against_manifest()["aggregate"]`, which is the inner field. Each task would exit with `eval-mask payload mismatch` before creating its run directory. The matrix was not edited and the authorized SHA was not replaced.
