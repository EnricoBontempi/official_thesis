# Month 2 — Pipeline validation (Step 2.2)

**Date:** 2026-11-16 (America/New_York)  
**Host:** local WSL workspace. Engaging GPU/allocation **not verified**.  
**Base commit before this step:** `5bce7142d7dbaf9e5ea6bc1e3c8dffe75e066ce9`

## How to use

```bash
# Rebuild manifests + fit-only stats and validate (CPU, uses local shards)
python scripts/05_validate_month2_pipeline.py --write-artifacts

# Or validate against already written artifacts
python scripts/05_validate_month2_pipeline.py --validate-only

# Unit tests
python -m pytest tests/ -q
```

Minimal programmatic usage:

```python
from combustion_recon.data.month2_dataset import Month2ReconDataset
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.collate import collate_month2_batch
from torch.utils.data import DataLoader

norm = Month2FitOnlyNormalizer("data/manifests/train_norm_stats_month2_fit.json")
ds = Month2ReconDataset("train_fit", task_mode="temperature_free", K=0, p_conditioning=False, normalizer=norm)
loader = DataLoader(ds, batch_size=2, collate_fn=lambda xs: collate_month2_batch(xs))
batch = next(iter(loader))  # batch["model_inputs"], batch["targets"], batch["metadata"]
```

Pass **only** `batch["model_inputs"]` into future models. TEST roles require a selection freeze (`allow_test_payloads=True` + valid freeze JSON).

## Environment (measured)

| Quantity | Value |
|---|---|
| CPUs | 16 |
| RAM available | ~24 GiB (`MemAvailable`) |
| Disk free on workspace | ~819–879 GiB |
| Raw shards | `data/raw/combustion/hf_dataset/numerical`, 30× `.arrow`, ~59 GiB |
| Engaging / GPU | **not verified** |

## Artifacts

- `data/manifests/train_norm_stats_month2_fit.json` — fit-only on 20×38 `train_fit` frames
- `data/manifests/month2_frames.json` — role recipes + counts (TEST indices metadata-only)
- `data/manifests/month2_observations.json` — numeric mask seeds + verified hashes
- `data/manifests/month2_pipeline_validation_summary.json` — machine-readable PASS summary
- Month-1 `data/manifests/train_norm_stats.json` **unchanged** (sha256 `37bf7e8579812930d37e816d1689b89a3a91666e56af66d24b08fd958cfa9b76`)

## Frame counts

| Role | Conditions | Frames/cond | Samples | Payload |
|---|---:|---:|---:|---|
| train_fit | 20 | 38 | 760 | allowed |
| seen_train_eval | 20 | 11 | 220 | allowed (diagnostics) |
| val_eval | 3 | 41 | 123 | allowed |
| test_interp_eval | 3 | 41 | 123 | freeze required |
| test_extrap_eval | 4 | 41 | 164 | freeze required |

`train_fit` ∩ `seen_train_eval` = ∅; gap 1480 → 1600 = 120 frames.

## Fit-only statistics

- Temperature mean/std (K): **2039.508661 / 331.737843** (unclipped)
- Velocity_i / Velocity_j included; vk not used as 2C input
- Independent subset recomputation abs diff mean/std: ~1e-13 / ~1e-14
- Fold-capable: normalizer accepts reduced `sim_ids` (synthetic test); no fold fitting performed

## Real-data checks (PASS)

- Sample0 `0NH3_0.85.h5` frame 0: coordinates, T, and observed velocity sites match the Arrow shard
- Denormalization round-trip OK
- DataLoader `batch_size=2`, `num_workers=2`, collate validity masks OK (~16 s first batch including worker start/decode)
- `p_off` omits `op_params`; no `velocity_k` in `model_inputs`
- TEST payload access denied without freeze
- Evaluator smoke (constant mean predictor) + frame→condition→split aggregation OK
- Isolation tests (free T change, assisted off-anchor T, hidden velocity, vk, p_off): PASS

## Memory / I/O

| Layer | Observation |
|---|---|
| Arrow decode | ~2.1 GiB peak per full numerical blob (tracemalloc earlier probe) |
| Compact LRU cache | only extracted frames (4×128×128 float32 ≈ 256 KiB/frame); after validation ~1 frame cached |
| Script peak RSS | ~**6483 MiB** during validation with DataLoader workers (host-level; includes decode + worker copies) |
| Persistent extras | **none** — no 59 GiB conversion written |

Implementation keeps at most one shard decode transiently and discards the full array after compact extract. Host RAM must still tolerate ~2 GiB decode spikes (more with multiple workers).

## Tests executed

- Full suite: **43 passed** (`pytest tests/ -q`)
- Includes Month-1 regression tests + new Month-2 pipeline/isolation tests
- No skipped real-data tests once stats artifacts exist

## Limits / not claimed

- Engaging access, GPU allocation, and the 24 GPU-hour ceiling remain unverified/unallocated
- No baseline/POD/ParamMean refit, no neural training
- `InputIsolationProbe` is **not** a U-Net/GLU forward validation
- Pseudo-extrap folds not fitted; scientific fold lock still open
- Month-1 `CombustionReconDataset` left in place for historical scripts; new work should use the Month-2 path

## Overall status: **PASS (local Step 2.2)**
