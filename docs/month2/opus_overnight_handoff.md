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

## 2026-11-19 O5 submitted after the mask-hash correction

The preparatory checkout `/home/bontempi/projects/official_thesis_opus_o5` stays at `64d0cc321dc5bc77ae48e44b063ee7a05cd87e48`. No job was launched from it.

Cause: `mask_aggregate_sha256` in that matrix held the sha256 of `manifest.json` (`2e2d8ff6…`), while the evaluator compared it with `aggregate_payload_sha256` (`7be78f35…`). Historical `run_identity.mask_payload_sha256` and `MASK_MANIFEST_SHA256` keep the file hash. They were not rewritten.

Correction in `c70565978b27ebdae7ad11a84dcefecf3845dde2`: the matrix and evaluator now name `eval_mask_manifest_file_sha256` and `eval_mask_aggregate_payload_sha256`. The job checks the manifest file, then the payload files against the manifest, then the aggregate recomputed from those file hashes. Tests `tests/test_opus_o5_mask_contract.py` plus the O5 and mask suites: 14 passed. The ambiguous key is no longer a gate.

Submission after that preflight passed on the new checkout. Array **29716019**, `0-7%4`, submitted 2026-11-19T21:02:35-05:00, PENDING. Account and QoS `mit_amf_advanced_gpu`, partition `mit_normal_gpu`, `gpu:l40s:1`, 8 CPU, 64G, wall 4h. QoS cap remains cpu=64, gres/gpu=4, mem=1T.

| Task | Run |
| --- | --- |
| 29716019_0 | O-B19-on-s0 |
| 29716019_1 | O-B19-off-s0 |
| 29716019_2 | O-A38-on-s0 |
| 29716019_3 | O-A38-off-s0 |
| 29716019_4 | O-A38-on-s1 |
| 29716019_5 | O-A38-off-s1 |
| 29716019_6 | O-C149-on-s0 |
| 29716019_7 | O-C149-off-s0 |

| Path | Role |
| --- | --- |
| `/home/bontempi/projects/official_thesis_opus_o5_v2` | executed checkout, detached at `c70565978b27ebdae7ad11a84dcefecf3845dde2` |
| `/home/bontempi/official_thesis_opus_o5_runs/<run_id>/` | per-run output, created by the task, refused if it already exists |
| `/home/bontempi/official_thesis_opus_o5_runs/slurm-29716019_<task>.out` | stdout |
| `/home/bontempi/official_thesis_opus_o5_runs/slurm-29716019_<task>.err` | stderr |

```bash
ssh engaging 'squeue -j 29716019 -r -o "%.18i %.20P %.24a %.24q %.2t %.10M %b %R"'
ssh engaging 'sacct -j 29716019 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'tail -n 40 ~/official_thesis_opus_o5_runs/slurm-29716019_0.out'
```

Aggregation, figures, and the Neon package wait for the outcome check. No polling from this record.

## 2026-11-19 O5 array 29716019 failed before evaluation

One read after Enrico reported the array finished. All eight tasks are FAILED, ExitCode 1:0. No scientific output was written, so aggregation and figures were not run and no result was relaunched.

| Task | Run | State | Exit | Elapsed | Node | End |
| --- | --- | --- | --- | ---: | --- | --- |
| 29716019_0 | O-B19-on-s0 | FAILED | 1:0 | 00:00:11 | node4108 | 2026-11-19T21:02:58 |
| 29716019_1 | O-B19-off-s0 | FAILED | 1:0 | 00:00:11 | node4203 | 2026-11-19T21:02:58 |
| 29716019_2 | O-A38-on-s0 | FAILED | 1:0 | 00:00:11 | node4206 | 2026-11-19T21:02:58 |
| 29716019_3 | O-A38-off-s0 | FAILED | 1:0 | 00:00:11 | node3007 | 2026-11-19T21:02:58 |
| 29716019_4 | O-A38-on-s1 | FAILED | 1:0 | 00:00:12 | node4106 | 2026-11-19T21:08:23 |
| 29716019_5 | O-A38-off-s1 | FAILED | 1:0 | 00:00:08 | node4108 | 2026-11-19T21:08:19 |
| 29716019_6 | O-C149-on-s0 | FAILED | 1:0 | 00:00:11 | node4203 | 2026-11-19T21:08:22 |
| 29716019_7 | O-C149-off-s0 | FAILED | 1:0 | 00:00:08 | node3204 | 2026-11-19T21:08:20 |

Stdout is empty. All eight stderr files share sha256 `5e1f9836f16cc0881ed234995b67e6f6e45efba278b975c35ad645da81f31e11`. The traceback is `ModuleNotFoundError: No module named 'combustion_recon.training.normalizer'` at `scripts/26_opus_o5_eval.py` line 242. `Month2FitOnlyNormalizer` is defined in `combustion_recon.data.month2_normalizer`, which `scripts/17_s6_sparse_eval_and_swap.py` already imports. The failed import is after `out_run.mkdir`, so the eight run directories exist and are empty. The launcher refuses an existing run directory. Checkout `/home/bontempi/projects/official_thesis_opus_o5_v2` is still `c70565978b27ebdae7ad11a84dcefecf3845dde2`. No `COMPLETE.json`, no per-frame rows, and no 5% comparison with the loop.

## 2026-11-19 O5 attempt2: import fixed, smoke submitted (campaign not submitted)

Responsibility passed from Cursor to Claude (Enrico authorized). Work lives on branch `opus/o5-eval-attempt2` in worktree `/home/enric/projects/official_thesis_opus_o5_fix`; the main worktree and the presentation work were not touched. Astra, HONF and the training checkouts were not modified.

**Diagnosis.** Attempt1 (array 29716019, code `c705659…`) failed at `import combustion_recon.training.normalizer`, a module that never existed. The consolidated class is `combustion_recon.data.month2_normalizer.Month2FitOnlyNormalizer`, the one `scripts/16_train_sparse_scientific.py` and `scripts/17_s6_sparse_eval_and_swap.py` already use. Everything before that line (checkpoint hashes, cache and eval-mask gates, training identity) had passed on the real data. The failed run directories and logs under `/home/bontempi/official_thesis_opus_o5_runs/` and the checkout `official_thesis_opus_o5_v2` are kept as they are.

**Fix, no change to formulas, data, selection, precision or protocol** (`a4aceafe6a51b338475f0d7c2f95318983079804`):

- Normalizer imported from `data.month2_normalizer`; project imports are now at module top.
- Normalizer, datasets, and model are built before the run directory is created, so an import or API error can no longer leave an empty directory.
- `verify_outputs` reads every CSV and field file back (row counts, finite RMSE, checkpoint and weight hashes against the matrix, shared-forward rows equal to their source). `COMPLETE.json` is written only after it passes.
- `--smoke` runs a non-scientific spread subset of frames (VAL keeps its three frame-1000 rows) through the same entrypoint. It requires an out-root whose final name contains `smoke`, a scientific out-root must not, and it writes `SMOKE_OK.json`, never `COMPLETE.json`.
- Launcher defaults moved to `official_thesis_opus_o5_v3` and `official_thesis_opus_o5_attempt2_runs`. It refuses a smoke root or the attempt1 root.

**Tests** (`tests/test_opus_o5_eval_calls.py`): every `combustion_recon` import in the script resolves; the attempt1 module is shown not to exist; every keyword the script passes to the dataset, normalizer, mask contract, cache hash check, and S6 functions binds to the real signature; read-back accepts a consistent run and rejects NaN, a wrong weight hash, and a shared-forward mismatch. With the old import restored the suite fails. On the cluster with the thesis env: 31 passed (new file plus O5 prep, mask contract, C149 cache, A38 seed-0 suites). No forward on the login.

**Smoke submitted** 2026-11-19T21:20:02-05:00: job **29717341**, array tasks `0` and `6`.

| Task | Run | Why |
| --- | --- | --- |
| 29717341_0 | O-B19-on-s0 | historical cache, `best_seen_diagnostic` shares weights with `best` |
| 29717341_6 | O-C149-on-s0 | C149 cache and its manifest gate |

`mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, wall 1h. QoS cap cpu=64, gres/gpu=4, mem=1T, partition max 6h. Same entrypoint `scripts/26_opus_o5_eval.py`, 6 frames per role, all three checkpoints, coverages 1/5/20%, replicas 0/1/2, persisted masks required. Executed SHA `a4aceafe…` on the frozen detached checkout `/home/bontempi/projects/official_thesis_opus_o5_v3` (clean except `.git_commit`). Not chained to the campaign.

| Path | Role |
| --- | --- |
| `/home/bontempi/official_thesis_opus_o5_smoke_runs/<run_id>/` | smoke output, `SMOKE_OK.json` on success |
| `/home/bontempi/official_thesis_opus_o5_smoke_runs/slurm-29717341_<task>.out/.err` | logs |
| `/home/bontempi/official_thesis_opus_o5_attempt2_runs/` | campaign output root, created empty, not used yet |

```bash
ssh engaging 'sacct -j 29717341 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'ls -R ~/official_thesis_opus_o5_smoke_runs | head -60'
ssh engaging 'cat ~/official_thesis_opus_o5_smoke_runs/O-B19-on-s0/SMOKE_OK.json'
ssh engaging 'tail -n 40 ~/official_thesis_opus_o5_smoke_runs/slurm-29717341_0.err'
```

The campaign (array 0-7%4, attempt2 root, same SHA) is submitted only after Enrico reports the smoke end and its artifacts are checked. No polling from this record.

## 2026-11-19 smoke 29717341 outcome and C149 smoke retry

One read of accounting, logs, artifacts. Task 0 (O-B19-on-s0) COMPLETED 0:0 in 45 s, `SMOKE_OK.json` 81/81, read-back passed, sampled CSV and fields match the smoke contract. Task 6 (O-C149-on-s0) **TIMEOUT** at 1:00:21 on node4108 with no run directory and no output. Its stderr holds one line (cancelled at the time limit); stdout is empty. At the one earlier read the process was in state `D`, wchan `autofs_mount_wait`, 4 s CPU, no cache file open. The cause is a wait for an automount on that node; which mount was not readable. The C149 manifest and hash file answered from the login in 0.01 s and the C149 trainings had run on other nodes, so the evidence does not point at the pool. Logs and artifacts of 29717341 are preserved in `/home/bontempi/official_thesis_opus_o5_smoke_runs/`.

**Operational change only** (`918286809d3cbb33d155b70a6e03a9b898bcab9e`, diff of `scripts/26_opus_o5_eval.py` against `a4aceafe…` is a `_log` helper, `import time`, and flushed timestamped lines at the phases checkpoints, cache, dataset, forward per coverage, readback). Both launchers now `stat` the C149 and historical cache roots and the mask root under `timeout 120` before starting and exit 6 if one does not answer (the timeout bounds the `stat`; it does not guarantee that a process in D state can be stopped). Both exclude `node4108`. Data, formulas, checkpoints, selection, precision, and protocol are unchanged, so the task-0 PASS at `a4aceafe…` is carried over. Tests on the cluster: 27 passed.

**Smoke retry, task 6 only:** job **29725163**, array `6`, submitted 2026-11-19T22:55:13-05:00. `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, wall 30 min, `--exclude=node4108`. Frozen detached checkout `/home/bontempi/projects/official_thesis_opus_o5_v4` at `91828680…`, clean except `.git_commit`. `TIP` was passed explicitly because the launcher default still names `official_thesis_opus_o5_v3`; the campaign must pass its own `TIP`.

| Path | Role |
| --- | --- |
| `/home/bontempi/official_thesis_opus_o5_smoke_attempt2_runs/O-C149-on-s0/` | smoke output |
| `/home/bontempi/official_thesis_opus_o5_smoke_attempt2_runs/slurm-29725163_6.out/.err` | logs with phase timestamps |

```bash
ssh engaging 'sacct -j 29725163 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'cat ~/official_thesis_opus_o5_smoke_attempt2_runs/slurm-29725163_6.out ~/official_thesis_opus_o5_smoke_attempt2_runs/slurm-29725163_6.err'
ssh engaging 'cat ~/official_thesis_opus_o5_smoke_attempt2_runs/O-C149-on-s0/SMOKE_OK.json'
```

The campaign stays authorized only after this smoke passes. If the new node also stalls, evidence goes to Enrico before any further attempt. No polling from this record.

## 2026-11-19 C149 smoke passed, O5 campaign attempt2 submitted

Smoke **29725163_6** (O-C149-on-s0, `918286809d3cbb33d155b70a6e03a9b898bcab9e`): COMPLETED 0:0 in 1:07 on node3302. Launcher access check ok for the C149 cache, historical cache, and mask roots. Phase log: checkpoints 1 s, cache 3 s, dataset 1 s, nine coverage forwards about 6 s each, read-back, `SMOKE_OK.json`. 81/81 settings, 81 CSV, 3 field files. All 81 CSV have finite RMSE, pool C149, model seed 0, the executed eval SHA, and 64-character mask hashes. Training SHA `d633200…`, job 29645053, last epoch 660. Fields are (3,128,128), finite, frame 1000. Provenance lists cache id `month2_dense_unet_cache_opus_c149_v1`, aggregate `fa1c9358…`, stats `678263fb…`, mask file hash `2e2d8ff6…`, mask aggregate `7be78f35…`. `scientific: false`, no `COMPLETE.json`. With the B19 smoke (job 29717341 task 0) both caches and the shared-forward path are covered. Outputs in `/home/bontempi/official_thesis_opus_o5_smoke_attempt2_runs/`.

**Campaign submitted** 2026-11-19T23:28:19-05:00: array **29728465**, `0-7%4`. Executed SHA `918286809d3cbb33d155b70a6e03a9b898bcab9e`, frozen detached checkout `/home/bontempi/projects/official_thesis_opus_o5_v5` (clean except `.git_commit`, 27 tests passed). Preflight: queue empty, QoS cap cpu=64 gres/gpu=4 mem=1T, partition max 6h, attempt2 root empty. `TIP`, `EXPECT_SHA`, `AUTHORIZE_OPUS_O5`, `OUT_ROOT`, `PY` passed explicitly. `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, wall 4h, `--exclude=node4108` (documented above). No general or preemptable fallback.

Tasks 0-7 follow the matrix order: O-B19-on-s0, O-B19-off-s0, O-A38-on-s0, O-A38-off-s0, O-A38-on-s1, O-A38-off-s1, O-C149-on-s0, O-C149-off-s0. Output `/home/bontempi/official_thesis_opus_o5_attempt2_runs/<run_id>/`, logs `slurm-29728465_<task>.out/.err` in the same root. A scientific COMPLETE is `COMPLETE.json` per run after the in-script read-back; aggregation and Neon package wait for the outcome. Attempt1 directories `/home/bontempi/official_thesis_opus_o5_runs/` and the first smoke root are preserved.

```bash
ssh engaging 'squeue -j 29728465 -r -o "%.18i %.2t %.10M %R"'
ssh engaging 'sacct -j 29728465 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'tail -n 20 ~/official_thesis_opus_o5_attempt2_runs/slurm-29728465_0.out'
```

## 2026-11-20 O5 attempt2: task 1 replaced

**What failed.** Array 29728465 task 1 (O-B19-off-s0) FAILED 6:0 after 2:01 on node4204. The launcher preflight gave a 120 s timeout on `/orcd/pool/005/bontempi/official_thesis_data/month2_dense_unet_cache_c149`, then refused to start. The Python evaluator never ran and `O-B19-off-s0` was never created. A B19 run does not read the C149 cache, so the preflight was stricter than the run. Log `slurm-29728465_1.out/.err` is preserved in `/home/bontempi/official_thesis_opus_o5_attempt2_runs/`.

**Correction, operational only** (`29b94345f0cac3f725662ad687a15ff841c7c730`). `scripts/engaging/o5_preflight_paths.py` builds the list of inputs from the matrix for the task's own run: cache manifest and hash file of its pool (C149 for C149 runs, historical for B19/A38), stats, mask manifest, training `run_identity.json`, and the three checkpoints. Both launchers stat that list under `timeout 120` and fail closed on an empty list. `tests/test_opus_o5_preflight_paths.py`: a B19/A38 run never calls stat on a C149 path (an unreachable C149 path would raise and the test records zero hits), C149 runs never list the historical cache, an unreachable required path is reported, and the listed paths equal what the evaluator's `_verify_inputs` reads for both pools (fakes record the reads). `scripts/26_opus_o5_eval.py`, `src/`, and `configs/` are byte-identical to `918286809d3cbb33d155b70a6e03a9b898bcab9e`. Cluster tests: 37 passed.

**Replacement submission:** job **29732630**, array task `1` only, submitted 2026-11-20T00:10:04-05:00. It replaces `29728465_1`. `mit_normal_gpu`, account/QoS `mit_amf_advanced_gpu`, `gpu:l40s:1`, 8 CPU, 64G, wall 4h, `--exclude=node4108,node4204` (not assumed to solve filesystem problems). Frozen detached checkout `/home/bontempi/projects/official_thesis_opus_o5_v6`, clean except `.git_commit`. `/home/bontempi/official_thesis_opus_o5_attempt2_runs/O-B19-off-s0` was absent before submission; no overwrite. Output `/home/bontempi/official_thesis_opus_o5_attempt2_runs/O-B19-off-s0/`, logs `slurm-29732630_1.out/.err` in the same root. The rest of array 29728465, including its pending tasks, was not touched and keeps its frozen checkout `/home/bontempi/projects/official_thesis_opus_o5_v5` at `91828680…`. The new job shares the QoS cap of 4 GPUs with that array and may wait for a free slot.

**For the final package, two SHAs.** Evaluator code is identical in both: `918286809d3cbb33d155b70a6e03a9b898bcab9e` (tasks 0 and 2-7) and `29b94345…` (O-B19-off-s0, launcher preflight differs). The run's `provenance.json` records `eval_git_sha` for each executed checkout; the package must list both and not describe the campaign as a single execution.

```bash
ssh engaging 'squeue -j 29732630 -r -o "%.18i %.2t %.10M %R"'
ssh engaging 'sacct -j 29732630 -X -o JobID,State,ExitCode,Elapsed,NodeList'
ssh engaging 'tail -n 20 ~/official_thesis_opus_o5_attempt2_runs/slurm-29732630_1.out'
```

## 2026-11-20 unattended final gate submitted

State at 00:34 (one read): 29728465 tasks 0 and 2 COMPLETED with `COMPLETE.json`; tasks 3, 4, 5, 6 RUNNING (5 and 6, the A38-off-s1 and C149-on-s0 runs, passed the preflight on node3302 and node3404 and reached the forward); task 7 (O-C149-off-s0) and the replacement 29732630_1 (O-B19-off-s0) PENDING behind the four-GPU QoS cap, an expected wait. Task 1 of the array stays FAILED (preflight on node4204) and is replaced by 29732630_1. An attempt to add `node4204` to the excluded nodes of pending task 7 was refused by Slurm (`ExcludeNodes` update not supported); task 7 keeps `node4108` excluded and relies on the preflight, which exits in about two minutes on a bad node. **No automatic recovery is implemented.** A failed task leaves a block report and waits for a diagnosis.

**Final gate:** job **29735692**, CPU `mit_normal`, account/QoS `mit_amf_advanced_cpu`, 4 CPU, 32G, wall 1:30, `--dependency=afterany:29728465:29732630` (the failed task 1 does not block it). Frozen checkout `/home/bontempi/projects/official_thesis_opus_o5_v7` at `6cdb25d1b5e1190f0fa1c683e6abc8d97bac8518`, clean except `.git_commit`, 36 tests passed. It runs `scripts/30_opus_o5_finalize.py`: checks the eight runs (`COMPLETE.json`, 81 CSV and 3 fields each, provenance and hashes against the matrix, frame sets and mask hashes identical across runs, shared forwards equal to their source, evaluator SHA in `{918286809d…, 29b94345f0…}`). If every check passes it builds the package (`scripts/29_opus_o5_package.py`) and the VAL frame-1000 field figures (`scripts/28_opus_o5_figures.py`) and writes the index, ledger, and summary. Otherwise it writes `BLOCK_REPORT.md` and exits 3. Both paths were exercised on a synthetic root. It does not use Git, the IDE, or anything local.

| Path | Content |
| --- | --- |
| `/home/bontempi/official_thesis_opus_o5_final/package_29735692/FINAL_SUMMARY.md` | readable summary, or absent if blocked |
| `/home/bontempi/official_thesis_opus_o5_final/package_29735692/BLOCK_REPORT.md` | present only if blocked, with exact missing items |
| `/home/bontempi/official_thesis_opus_o5_final/package_29735692/INDEX_FOR_NEON.md`, `LEDGER_O0_O6.md` | index and ledger |
| `/home/bontempi/official_thesis_opus_o5_final/package_29735692/package/` | `tables/`, `figures/`, `TABLES.md`, `package_manifest.json` |
| `/home/bontempi/official_thesis_opus_o5_final/slurm-29735692.out/.err` | job logs |

Not done by the gate: visual check of the figures, scientific report, O6, copy of the artifacts into the repository and push (done from the local machine afterwards). Opus is not scientifically closed. The ledger source is `docs/month2/opus_ledger_o0_o6.md`; its two open O0 items (S6 figure/region check, stale roadmap summaries) are unresolved.

## 2026-11-20 O5 complete, technical package for Neon delivered

**Outcome of the night.** Array 29728465 tasks 0 and 2-7 COMPLETED, replacement 29732630_1 COMPLETED. Task 1 of the array stays FAILED (preflight on node4204) and is replaced. The unattended gate 29735692 ran at 02:11 and wrote a block report (219 issues) even though all eight runs were valid: the checker treated `peak_rmse_K` = NaN (three reference frames of O-B19-on-s0 `best`, and similar) as a failure, and NaN != NaN broke the shared-forward comparison. This was a post-processing defect. No evaluation was repeated or changed. The checker now requires finite core metrics (RMSE, MAE, signed bias, absolute bias, centered RMSE), counts undefined regional values in `undefined_regional_metric_frames.csv`, and compares shared forwards with NaN-equal. Tests: 13 in `tests/test_opus_o5_package.py`, 36 passed with the other O5 suites.

**Package.** Job 29771069 built it, 29771334 (checkout `258106bc6e0e6dc7a721be2645b5ba66472acc8b`, after the legend and colorbar fixes found by looking at the figures) is the reference copy. Cluster path `/home/bontempi/official_thesis_opus_o5_final/package_29771334/`; copy in the repository under `results/month2/opus_data_study/o5/`. Checks passed: eight `COMPLETE.json`, 81 CSV and 3 field files per run, provenance and cache, stats and mask hashes against the matrix, training SHA, checkpoint identity, identical frame sets and mask hashes across runs, shared forwards equal to their source. O5 against the training loop at 5%, same epoch, role and replicas: 72 of 72 within 0.05 K, largest gap 0.0011 K. Evaluator and launcher SHAs declared: `918286809d3cbb33d155b70a6e03a9b898bcab9e` (seven runs) and `29b94345f0cac3f725662ad687a15ff841c7c730` (O-B19-off-s0).

**For Neon.** Start from `results/month2/opus_data_study/o5/INDEX_FOR_NEON.md` (attempts and SHAs, tables, figures, limits), then `FINAL_SUMMARY.md`, `LEDGER_O0_O6.md`, `package/TABLES.md`. R0 and the S5 bridge are referenced with their limits. O0 items closed in `docs/month2/opus_ledger_o0_o6.md` and `results/month2/opus_data_study/o0_checks/S6_DELIVERY_CHECK.md`. `roadmapsparse.md` and `roadmapopus.md` carry a current status line; their 2026-11-18 status is kept as history.

**Not done, by mandate.** `docs/month2/06_opus_data_and_training_variability.md`, the O6 decision, S7, A4, TEST, new trainings. Opus is not scientifically closed. Figures for `best_seen_diagnostic` regional and field panels were not opened one by one. S6 figures were not regenerated.
