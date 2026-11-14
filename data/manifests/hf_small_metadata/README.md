---
license: cc-by-nc-4.0
pretty_name: RealPDEBench
tags:
  - scientific-ml
  - physics
  - pde
  - sim-to-real
  - fluid-dynamics
  - combustion
  - spatiotemporal
task_categories:
  - time-series-forecasting
---

<p align="center">
  <img src="assets/logo.png" alt="RealPDEBench logo" width="700" />
</p>

# RealPDEBench

[![HF Dataset](https://img.shields.io/badge/HF%20Dataset-RealPDEBench-FFD21E?logo=huggingface)](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01829-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.01829)
[![Website & Docs](https://img.shields.io/badge/Website%20%26%20Docs-realpdebench.github.io-1f6feb?logo=google-chrome)](https://realpdebench.github.io/)
[![Codebase](https://img.shields.io/badge/Codebase-GitHub-181717?logo=github)](https://github.com/AI4Science-WestlakeU/RealPDEBench)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-9cf?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc/4.0/)

RealPDEBench is a benchmark of **paired real-world measurements and matched numerical simulations** for complex physical systems. It is designed for **spatiotemporal forecasting** and **sim-to-real transfer** evaluation on *real data*.

This Hub repository (`AI4Science-WestlakeU/RealPDEBench`) is the **release repo** for RealPDEBench.

- **Website & documentation**: [realpdebench.github.io](https://realpdebench.github.io/)
- **Raw HDF5 distribution**: [realpdebench.westlake.edu.cn](https://realpdebench.westlake.edu.cn/)
- **Benchmark codebase**: [AI4Science-WestlakeU/RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench)

<p align="center">
  <img src="assets/figure1.png" alt="RealPDEBench overview figure" width="900" style="max-width: 100%; height: auto;" />
</p>
<p align="center">
  <em>Figure 1. RealPDEBench provides paired real-world measurements and matched numerical simulations for sim-to-real evaluation.</em>
</p>

## What makes RealPDEBench different?

- **Paired real + simulated data**: each scenario provides experimental measurements and corresponding CFD/LES simulations.
- **Real-world evaluation**: models are evaluated on real trajectories to quantify the sim-to-real gap.
- **Multi-modal mismatch**: simulations include additional unmeasured modalities (e.g., pressure, species fields), enabling modality-masking and transfer strategies.

## Data sources (high level)

- **Fluid systems** (`cylinder`, `controlled_cylinder`, `fsi`, `foil`):
  - **Real**: Particle Image Velocimetry (PIV) in a circulating water tunnel
  - **Sim**: CFD (2D finite-volume + immersed-boundary; 3D GPU solvers depending on scenario)
- **Combustion** (`combustion`):
  - **Real**: OH* chemiluminescence imaging (high-speed)
  - **Sim**: Large Eddy Simulation (LES) with detailed chemistry (NH3/CH4/air co-firing)

## Scenarios (5)

| Scenario | Real data (measured) | Numerical data (simulated) | Frames / trajectory | Spatial grid (full resolution) | Trajectories (real / numerical) |
|---|---|---|---:|---:|---:|
| cylinder | velocity \(u,v\) | \(u,v,p\) | 3990 | 128×256 | 92 / 92 |
| controlled_cylinder | \(u,v\) | \(u,v,p\) (+ control params in filenames) | 3990 | 128×256 | 96 / 96 |
| fsi | \(u,v\) | \(u,v,p\) | 2173 | 128×128 | 51 / 51 |
| foil | \(u,v\) | \(u,v,p\) | 3990 | 128×256 | 98 / 99 |
| combustion | OH* chemiluminescence intensity (1 channel) | intensity surrogate (1) + 15 simulated fields | 2001 | 128×128 | 30 / 30 |

**Total trajectories** (HDF5 files): **~735** (≈364 real + ≈368 numerical).

### Physical parameter ranges (real experiments)

| Scenario | Key parameters (real) |
|---|---|
| cylinder | Reynolds number \(Re\): 1800–12000 |
| controlled_cylinder | \(Re\): 1781–9843; control frequency \(f\): 0.5–1.4 Hz |
| fsi | \(Re\): 3272–9068; mass ratio \(m^*\): 18.2–20.8 |
| foil | angle of attack \(\alpha\): 0°–20°; \(Re\): 2968–17031 |
| combustion | CH4 ratio: 20–100%; equivalence ratio \(\phi\): 0.75–1.3 |

## Data format on the Hub

RealPDEBench stores **complete trajectories** in HuggingFace Arrow format, with separate JSON index files for train/val/test splits. This enables dynamic `N_autoregressive` support at runtime.

Each scenario contains:
- **Trajectory data**: `hf_dataset/{real,numerical}/` — Arrow files with complete time series
- **Index files**: `hf_dataset/{split}_index_{type}.json` — maps sample indices to `(sim_id, time_id)`
- **test_mode metadata**: `{in_dist,out_dist,remain}_params_{type}.json`
- **Normalization statistics**: `mean_std.pt` — per-channel mean/std used by `GaussianNormalizer` (see [Normalization statistics](#normalization-statistics-mean_stdpt))

### Arrow shard files vs. trajectory counts

> **Important clarification**: The number of `.arrow` shard files does **not** equal the number of trajectories. HuggingFace Arrow format packs multiple rows into one shard up to a size limit (~500 MB by default), but never splits a single row across shards. Real trajectories are smaller (fewer channels, ~130–260 MB each), so 2–4 trajectories are packed per shard, resulting in fewer shards than trajectories. Numerical trajectories are larger (extra channels such as pressure or 15 simulated fields, ~1.5–2.1 GB each), so each one already exceeds the shard limit on its own, resulting in a 1:1 mapping between shards and trajectories.

Dataset version: **`2.0.1`** (see `version.json` at repo root; released 2026-09-18, format `lazy_slicing_v2`).

Updated in 2.0.1: real data shards for `cylinder`, `controlled_cylinder`, and `foil`.

| Scenario | Trajectories (real / numerical) | Arrow shards (real / numerical) |
|---|---:|---:|
| cylinder | 92 / 92 | 73 / 92 |
| controlled_cylinder | 96 / 96 | 51 / 96 |
| fsi | 51 / 51 | 51 / 51 |
| foil | 98 / 99 | 98 / 99 |
| combustion | 30 / 30 | 8 / 30 |

**Notes:**
- The **Trajectories** column is the ground-truth count (each row in the Arrow dataset = one complete trajectory). The **Arrow shards** column is the number of `.arrow` files on disk — a storage-level detail that depends on per-trajectory size.
- The `{remain,in_dist_test,out_dist_test}_params_{real,numerical}.json` files partition trajectories by **physical parameter regime**. Each entry is keyed by HDF5 filename and maps to its parameter tuple (e.g., `(Re, control_freq)`). The three groups sum to the total trajectory count:
  - `in_dist_test_params`: trajectories with in-distribution parameters, **entirely** reserved for testing.
  - `out_dist_test_params`: trajectories with out-of-distribution (edge/extreme) parameters, **entirely** reserved for testing.
  - `remain_params`: all other trajectories — **part** of each trajectory's time axis is used for training, the rest for validation/testing.

  At evaluation time, `test_mode` can be set to `"seen"` (remain), `"in_dist"`, `"out_dist"`, `"unseen"` (in\_dist + out\_dist), or `"all"`.

  <details>
  <summary>Per-scenario split counts</summary>

  | Scenario | Type | remain | in\_dist\_test | out\_dist\_test | Total |
  |---|---|---:|---:|---:|---:|
  | cylinder | real | 72 | 10 | 10 | 92 |
  | cylinder | numerical | 92 | 0 | 0 | 92 |
  | controlled\_cylinder | real | 76 | 10 | 10 | 96 |
  | controlled\_cylinder | numerical | 96 | 0 | 0 | 96 |
  | fsi | real | 39 | 0 | 12 | 51 |
  | fsi | numerical | 51 | 0 | 0 | 51 |
  | foil | real | 78 | 10 | 10 | 98 |
  | foil | numerical | 99 | 0 | 0 | 99 |
  | combustion | real | 30 | 0 | 0 | 30 |
  | combustion | numerical | 30 | 0 | 0 | 30 |

  </details>

**Repository layout**:

```
{repo_root}/
  cylinder/
    channels.json                     # Field / channel schema for this scenario
    mean_std.pt                       # Per-channel normalization statistics (see below)
    in_dist_test_params_real.json
    out_dist_test_params_real.json
    remain_params_real.json
    in_dist_test_params_numerical.json
    out_dist_test_params_numerical.json
    remain_params_numerical.json
    hf_dataset/
      real/                           # Arrow: complete trajectories (92 files)
        data-*.arrow
        dataset_info.json
        state.json
      numerical/                      # Arrow: complete trajectories
        data-*.arrow
        dataset_info.json
        state.json
      train_index_real.json           # Index: [{"sim_id": "xxx.h5", "time_id": 0}, ...]
      val_index_real.json
      test_index_real.json
      train_index_numerical.json
      val_index_numerical.json
      test_index_numerical.json
  fsi/
    ...  (same structure)
  controlled_cylinder/
    ...  (same structure)
  foil/
    ...  (same structure)
  combustion/
    ...  (same structure)
```

### How to download only what you need

For large data, use `snapshot_download(..., allow_patterns=...)` to avoid pulling the full repository.

```python
import os
from huggingface_hub import snapshot_download
from datasets import load_from_disk

repo_id = "AI4Science-WestlakeU/RealPDEBench"
os.environ["HF_HUB_DISABLE_XET"] = "1"
local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["fsi/**"],  # example: download only the FSI folder
    endpoint="https://hf-mirror.com",
)

# Load trajectory data
trajectories = load_from_disk(os.path.join(local_dir, "fsi", "hf_dataset", "real"))
print(f"Loaded {len(trajectories)} trajectories")
print(trajectories[0].keys())  # sim_id, u, v, vo, x, y, t, shape_t, shape_h, shape_w, ...
```

### Using the RealPDEBench loaders (recommended)

For automatic train/val/test splitting and dynamic `N_autoregressive` support, use the provided dataset loaders:

```python
from realpdebench.data.fluid_hf_dataset import FSIHFDataset

dataset = FSIHFDataset(
    dataset_name="fsi",
    dataset_root="/path/to/data",
    dataset_type="real",
    mode="test",
    N_autoregressive=10,  # Dynamic! Works with any value
)

input_tensor, output_tensor = dataset[0]
print(f"Input shape: {input_tensor.shape}")   # (20, H, W, 2)
print(f"Output shape: {output_tensor.shape}") # (200, H, W, 2) = 20 × 10
```

### Normalization statistics (`mean_std.pt`)

Each scenario root contains a `mean_std.pt` file with per-channel normalization statistics:

```python
import torch

mean_inputs, mean_targets, std_inputs, std_targets = torch.load("cylinder/mean_std.pt")
```

- Each element is a 1-D float32 tensor indexed by the **last (channel) axis** of the model inputs / targets.
- Statistics are computed on the **numerical training split** — the same normalizer dataset that `realpdebench/train.py` uses.
- Input and target channel counts can differ: for `controlled_cylinder`, inputs carry two extra conditioning channels (Reynolds number and control frequency) after the field channels.
- `realpdebench.data.data_normalizer.GaussianNormalizer` loads this file automatically from the scenario directory. If the file is missing, the normalizer recomputes the statistics with a full pass over the training data and writes the file back. Shipping the file only skips that pass; the values are reproducible from the data.

## Schema (columns)

### Fluid datasets (`cylinder`, `controlled_cylinder`, `fsi`, `foil`)

- **Keys** (each row = one complete trajectory):
  - `sim_id` (string): trajectory file name (e.g., `10031.h5`)
  - `u`, `v` (bytes): float32 arrays of shape `(T_full, H, W)` — **complete time series**
  - `p` (bytes): float32 array `(T_full, H, W)` *(numerical splits only)*
  - `vo` (bytes): float32 array `(T_full, H, W)` — vorticity
  - `x` (bytes): float32 array `(H, W)` — spatial x-coordinate grid *(time-invariant)*
  - `y` (bytes): float32 array `(H, W)` — spatial y-coordinate grid *(time-invariant)*
  - `t` (bytes): float32 array `(T_full,)` — time stamps
  - `shape_t` (int): **complete trajectory length** (e.g., 3990, 2173)
  - `shape_h`, `shape_w` (int): spatial dimensions
- **Field names** for each scenario are also documented machine-readably in [`{scenario}/channels.json`](#repository-layout) (with their Arrow keys and shapes).

> **Note on spatial grids:** `x` and `y` are identical across all time frames, so they are stored once as `(H, W)` instead of `(T, H, W)`. For methods that require per-frame coordinate grids (e.g., PINNs), broadcast at runtime: `x_grid = np.broadcast_to(x[np.newaxis, :, :], (T, H, W))`. This is a zero-copy view with no memory overhead.

### Combustion dataset (`combustion`)

- **Keys** (each row = one complete trajectory):
  - `sim_id` (string): e.g., `40NH3_1.1.h5`
  - `observed` (bytes): float32 array `(T_full, H, W)` — **complete time series**
  - `numerical` (bytes): float32 array `(T_full, H, W, 15)` *(numerical splits only)*
  - `numerical_channels` (int): number of numerical channels (15)
  - `x` (bytes): float32 array `(H, W)` — spatial x-coordinate grid *(time-invariant)*
  - `y` (bytes): float32 array `(H, W)` — spatial y-coordinate grid *(time-invariant)*
  - `t` (bytes): float32 array `(T_full,)` — time stamps
  - `shape_t` (int): **complete trajectory length** (e.g., 2001)
  - `shape_h`, `shape_w` (int): spatial dimensions
- **Channel order** of the 15 fields packed in `numerical` (along the last axis) is listed in [`combustion/channels.json`](combustion/channels.json) under `numerical.numerical_axis_names` (`index → name`).

### Index files (JSON)

Each split has an index file mapping sample indices to trajectory positions:

```json
[
  {"sim_id": "10031.h5", "time_id": 0},
  {"sim_id": "10031.h5", "time_id": 20},
  {"sim_id": "10031.h5", "time_id": 40},
  ...
]
```

## Data size

- **Total**: ~**783GB** across all scenarios (full resolution, all fields)
- **Largest shard file**: ~**2.1GB** (well below the Hub's recommended **<50GB per file**)
- **Total Arrow file count**: **649 files** (well below the Hub's recommended **<100k files per repo**)

Per-scenario totals:

| Scenario | real | numerical | Total |
|---|---:|---:|---:|
| cylinder | 36GB | 144GB | 180GB |
| controlled_cylinder | 25GB | 151GB | 176GB |
| fsi | 44GB | 58GB | 102GB |
| foil | 103GB | 155GB | 258GB |
| combustion | 4GB | 63GB | 67GB |
| **Total** | **212GB** | **571GB** | **~783GB** |

## Recommended benchmark protocols

RealPDEBench supports three standard training paradigms (all evaluated on **real-world** data):
- **Simulated training** (numerical only)
- **Real-world training** (real only)
- **Simulated pretraining + real finetuning**

## License

This dataset is released under **CC BY‑NC 4.0** (non‑commercial). Please credit the authors and the benchmark paper when using the dataset.

## Citation

If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{hu2026realpdebench,
      title={RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data},
      author={Peiyan Hu and Haodong Feng and Hongyuan Liu and Tongtong Yan and Wenhao Deng and Tianrun Gao and Rong Zheng and Haoren Zheng and Chenglei Yu and Chuanrui Wang and Kaiwen Li and Zhi-Ming Ma and Dezhi Zhou and Xingcai Lu and Dixia Fan and Tailin Wu},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://openreview.net/forum?id=y3oHMcoItR},
      note={Oral Presentation}
}
```

## Contact

AI for Scientific Simulation and Discovery Lab, Westlake University
Maintainer: `westlake-ai4s` (Hugging Face)
Org: `AI4Science-WestlakeU`
