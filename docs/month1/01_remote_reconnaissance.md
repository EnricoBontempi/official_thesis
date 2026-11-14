# Step 1.2 — Remote Reconnaissance of RealPDEBench Combustion

**Generated:** 2026-11-13  
**Repo inspected:** [`AI4Science-WestlakeU/RealPDEBench`](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench) (dataset, Hub format `lazy_slicing_v2`, `data_version` **2.0.1**)  
**Code inspected:** shallow clone of [`AI4Science-WestlakeU/RealPDEBench`](https://github.com/AI4Science-WestlakeU/RealPDEBench) at `/tmp/RealPDEBench_code`  
**Docs:** [combustion scenario page](https://realpdebench.github.io/datasets/combustion/)  
**Inventory artifact:** [`data/manifests/remote_file_inventory.json`](../../data/manifests/remote_file_inventory.json)  
**Small metadata copies:** [`data/manifests/hf_small_metadata/`](../../data/manifests/hf_small_metadata/)

No large tensors (`.arrow`, `.h5`, `.npy`, …) were downloaded in this step.

---

## 1. Local hardware & storage summary

| Item | Value |
|------|-------|
| Root filesystem | `/dev/sdd`, **1007 GB** total, **~880 GB free** (~8% used after env install) |
| Python | 3.13.13 (`miniconda3`); project venv at `.venv/` |
| Package install | `pip install -e .` into `.venv` succeeded (`combustion_recon` 0.1.0) |
| PyTorch | 2.15.0+cu132 |
| CUDA / GPU | **`torch.cuda.is_available() == False`**; `nvidia-smi` not available in this WSL environment (CPU-only for now) |
| Full RealPDEBench Hub repo | **~729 GB** (all scenarios) |
| Combustion subset (this thesis) | **~62.3 GB** total on Hub |
| Numerical combustion only | **~58.6 GB** |
| Experimental / real OH* only | **~3.7 GB** |

**Disk verdict:** Local free space (**~880 GB**) is ample for the full numerical combustion set (~59 GB) plus headroom for interim/processed copies. GPU training will require a machine/session with a working NVIDIA stack; Month 1 feasibility checks can proceed on CPU.

---

## 2. Hugging Face remote file breakdown (Combustion)

### 2.1 Totals

| Category | Files | Size |
|----------|------:|-----:|
| All combustion paths | 56 | **62.292 GB** (`66 886 027 710` B) |
| Numerical (`hf_dataset/numerical/` + `*_numerical.json`) | 38 | **58.626 GB** |
| Experimental / real (`hf_dataset/real/` + `*_real.json`) | 16 | **3.666 GB** |
| Other combustion meta (`channels.json`, `mean_std.pt`) | 2 | 3.4 KB |

By extension (combustion only): **38× `.arrow` (62.288 GB)**, **17× `.json` (4.9 MB)**, **1× `.pt` (2.1 KB)**.

Official Hub README quotes combustion as ~4 GB real + ~63 GB numerical ≈ **67 GB**; measured LFS sizes sum to **~62.3 GB** (same order; difference is rounding / packaging). Docs page “Memory 110.12 GB” is an in-memory / uncompressed estimate, not the download footprint.

### 2.2 Directory layout on the Hub

```
combustion/
  channels.json
  mean_std.pt
  in_dist_test_params_{real,numerical}.json   # empty {} for combustion
  out_dist_test_params_{real,numerical}.json  # empty {} for combustion
  remain_params_{real,numerical}.json         # all 30 trajectories
  hf_dataset/
    numerical/   # 30 Arrow shards (1 traj / shard) + dataset_info.json + state.json
    real/        # 8 Arrow shards (multiple traj / shard) + dataset_info.json + state.json
    {train,val,test}_index_{real,numerical}.json
```

Legacy per-trajectory HDF5 names still appear as `sim_id` strings (e.g. `40NH3_1.1.h5`). Raw HDF5 is also advertised at [realpdebench.westlake.edu.cn](https://realpdebench.westlake.edu.cn/); Hub release is Arrow V2.

### 2.3 Numerical combustion Arrow shards

Each of the **30** files is ~**1.954 GB** (`2 098 221 120` B; a few shards differ by ≤104 B). LFS SHA256 hashes are recorded in `remote_file_inventory.json`.

| Path | Size | Role |
|------|------|------|
| `combustion/hf_dataset/numerical/data-00000-of-00030.arrow` … `data-00029-of-00030.arrow` | ~1.954 GB each | One complete numerical trajectory per shard |
| `…/numerical/dataset_info.json` | 1.4 KB | Arrow schema |
| `…/numerical/state.json` | 1.9 KB | Shard map |

**Full numerical combustion download requirement ≈ 58.6 GB** (shards + tiny JSON).

### 2.4 Experimental / real OH* shards

| Path | Size |
|------|------|
| `combustion/hf_dataset/real/data-00000-of-00008.arrow` … `data-00005` | ~500.3 MB each |
| `…/real/data-00006-of-00008.arrow`, `data-00007` | ~375.2 MB each |
| Index / `dataset_info` / `state` | small |

**Total real OH* ≈ 3.67 GB.** These are chemiluminescence intensity fields only — **not** experimental temperature maps (thesis guardrail).

### 2.5 Index & parameter JSON (official RealPDEBench splits)

| File | Size / content |
|------|----------------|
| `train_index_numerical.json` | 2.49 MB — **58 860** `(sim_id, time_id)` windows over **all 30** sims |
| `val_index_numerical.json` | `[]` (empty) |
| `test_index_numerical.json` | `[]` (empty) |
| `train_index_real.json` | 1.96 MB — 46 223 windows / 30 sims |
| `val_index_real.json` | 248 KB — 5 718 windows / 30 sims |
| `test_index_real.json` | 248 KB — 5 719 windows / 30 sims |
| `remain_params_{real,numerical}.json` | **30/30** trajectories |
| `in_dist_test_params_*.json` | **empty** |
| `out_dist_test_params_*.json` | **empty** |

**Implication for this thesis:** RealPDEBench’s released combustion indices do **not** withhold entire operating conditions for numerical data (numerical val/test indices are empty; all sims sit in `remain_params`). Our Month 1 protocol must define **our own** Train / Val / Test-Interp / Test-Extrap **operating-condition** splits (Step 1.4). Their time-axis trunk split on real data is a forecasting convention, not our leakage policy.

---

## 3. Findings from official docs & codebase

### 3.1 Physics & sampling (docs)

- Swirl-stabilized **NH₃/CH₄/air** flame; paired real OH* and LES numerical fields.
- `n_traj = 30 × 2` (30 real + 30 numerical), `n_frame = 2001`, `Δt = 2.5×10⁻⁴` s (4000 Hz for 1 s).
- Spatial resolution (full): **128 × 128** for both real and sim.
- Fuel composition (docs): CH₄ ratios {100%, 80%, 60%, 40%, 20%} ↔ NH₃ {0%, 20%, 40%, 60%, 80%}.
- Equivalence ratios (docs list): {0.75, 0.85, 0.9, 1.0, 1.05, 1.1, 1.2, 1.25, 1.3} — **released files also include φ = 0.95 and 1.15** (see §3.2).

### 3.2 Operating-condition naming (code + `remain_params`)

Filenames / `sim_id`s follow:

```text
{NH3_percent}NH3_{phi}.h5
```

Parsed in `combustion_dataset.py` via `r'(\d+)NH3_(\d+\.?\d*)\.h5'` → `(gas_ratio, equivalence_ratio)`.

**Released 30 numerical/real conditions** (`remain_params_*.json`):

| NH₃ % | CH₄ % | φ values present |
|------:|------:|------------------|
| 0 | 100 | 0.85, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30 |
| 20 | 80 | 0.75, 0.85, 0.90, 1.00, 1.05, 1.10, 1.20, 1.25, 1.30 |
| 40 | 60 | 0.85, 0.90, 1.00, 1.10, 1.15, 1.20 |
| 60 | 40 | 0.95 only |
| 80 | 20 | 0.90, 1.05, 1.15, 1.20, 1.25 |

Coverage is **not** a full Cartesian product of the doc lists — sparse at high NH₃. Parameter space for interp/extrap splits must use this actual set.

### 3.3 Tensor shapes, channels, coordinates

From docs + `combustion/channels.json` + `CombustionHFDataset` / `convert_hdf5_to_hf.py`:

| Field | Shape / type | Meaning |
|-------|--------------|---------|
| `sim_id` | string | e.g. `40NH3_1.1.h5` |
| `observed` | `float32 (2001, 128, 128)` | Real: OH* intensity \(I\); numerical: surrogate-mapped intensity |
| `numerical` | `float32 (2001, 128, 128, 15)` | LES fields (numerical only) |
| `numerical_channels` | `int = 15` | |
| `x`, `y` | spatial grids (time-invariant) | Docs/README: `(H, W)` float32; converter comments also mention 1D `(H,)` from H5 — **verify in pilot** |
| `t` | `(2001,)` | time stamps |
| `shape_t/h/w` | ints | full trajectory dimensions |

**Channel order of `numerical[..., c]`** (`channels.json` / docs):

| Idx | Name | Thesis role |
|----:|------|-------------|
| 0 | `Absolute_Pressure` | |
| 1 | `Chemistry_Heat_Release_Rate` | |
| 2 | `Mole_Fraction_of_CH4` | |
| 3 | `Mole_Fraction_of_CO` | |
| 4 | `Mole_Fraction_of_CO2` | |
| 5 | `Mole_Fraction_of_H2O` | |
| 6 | `Mole_Fraction_of_NH2` | |
| 7 | `Mole_Fraction_of_NH3` | fuel / conditioning |
| 8 | `Mole_Fraction_of_OH` | **≠ OH*** (guardrail) |
| 9 | `Pressure` | |
| 10 | **`Temperature`** | **reconstruction target \(T\)** |
| 11 | **`Velocity[i]`** | observation (with j) |
| 12 | **`Velocity[j]`** | observation |
| 13 | `Velocity[k]` | 3rd component present in 2D grid — inspect orientation in pilot |
| 14 | `Velocity_Magnitude` | derived |

Units are **not** stated in the Hub card or loader; Step 1.3 must confirm (expect K for \(T\), m/s for velocity) from attributes or physical ranges.

**Loader output packing:** `CombustionHFDataset.__getitem__` returns `(input, output)` with shape `(T, H', W', C)` where `C = 1 + 15 = 16` after concatenating `observed` and the 15 numerical channels (zeros for unobserved numerical channels when `dataset_type=="real"` or when `mask_prob` triggers masking). Default `sub_s_real = sub_s_numerical = 2` → **64×64** unless overridden to `1` for full 128×128 (thesis default intent).

Legacy H5 paths (if used later):

- Real: `{root}/combustion/real/{sim_id}` key `trajectory`
- Numerical: `{root}/combustion/numerical/{sim_id}` key `measured_data` shape `(T,H,W,15)`
- Surrogate intensity: `{root}/combustion/surrogate/{sim_id}` key `measured_data`

### 3.4 Official preprocessing / splits (benchmark paper codebase)

- **Train/val/test** are sliding windows of length `in_step + out_step` (defaults 20+20) over time, with trunk-based separation for remain trajectories (`trunk_length=101`).
- **test_mode** filters: `in_dist` / `out_dist` / `seen` / `unseen` / `all` — for combustion, in/out-dist JSON maps are empty, so only `seen`/`all` are meaningful on the released indices.
- Numerical data is treated as **train-only** in the base `RealDataset` logic when `split_numerical=False` (matches empty numerical val/test index files).
- Normalization: `combustion/mean_std.pt` (~2 KB) — Gaussian stats from numerical training; loaded by `GaussianNormalizer`.
- License: **CC BY-NC 4.0** (code `LICENSE` + Hub `license: cc-by-nc-4.0`). Citation: Hu et al., RealPDEBench, ICLR 2026 Oral (`arXiv:2601.01829`).

### 3.5 Key dataloaders / configs to reuse as reference

| Path (in official repo) | Role |
|-------------------------|------|
| `realpdebench/data/combustion_hf_dataset.py` | Preferred Arrow V2 loader |
| `realpdebench/data/combustion_dataset.py` | Legacy H5 loader + param parsing / split logic |
| `realpdebench/hf_download.py` | Patterned `snapshot_download` (`metadata` vs `hf_dataset`) |
| `realpdebench/utils/convert_hdf5_to_hf.py` | Schema + H5→Arrow conversion |
| `realpdebench/configs/combustion/*.yaml` | Benchmark training configs (forecasting, not our recon task) |

---

## 4. Recommended plan for Step 1.3 (pilot → full numerical)

1. **Pilot (≈ 2 GB):** Download **one** numerical Arrow shard, e.g.  
   `combustion/hf_dataset/numerical/data-00000-of-00030.arrow`  
   plus already-local `dataset_info.json`, `state.json`, `channels.json`, `remain_params_numerical.json`.  
   Prefer a mid-blend ID once `state.json` / shard→`sim_id` mapping is confirmed (e.g. `40NH3_1.1.h5` if that shard maps there).
2. **Pilot checks:** decode row → verify `shape_t,h,w == (2001,128,128)`, channel 10 = Temperature, 11/12 = Velocity[i]/[j], `x`/`y` rank, NaN/Inf scan, instantaneous + mean \(T,u,v\) plots.
3. **Do not** treat real OH* as temperature; optional later download of real shards only for qualitative pairing, not core targets.
4. **Full numerical staging:** after pilot OK, download all  
   `combustion/hf_dataset/numerical/**` + numerical index/param JSONs  
   → **≈ 58.6 GB** under `data/raw/` (fits easily in ~880 GB free). Record SHA256 from inventory.
5. **Our splits:** ignore empty official in/out-dist maps; build `configs/splits/splits_interp_extrap.yaml` from the 30 `(NH₃%, φ)` points in Step 1.4.
6. **Loader for thesis:** implement a thin reader in `src/combustion_recon/data/` that loads Arrow/H5 at **full 128×128** (`sub_s=1`) and exposes tokens `(x,y,var_id,value)` without zero-filling unobserved locations as physical zeros.

### Disk budget checklist

| Artifact | Approx. size |
|----------|-------------:|
| Pilot 1 numerical trajectory | ~2.0 GB |
| Full numerical combustion | ~58.6 GB |
| Optional full real OH* | ~3.7 GB |
| Interim/processed duplicates (budget) | +20–60 GB |
| **Comfortable local free space** | **~880 GB** |

---

## 5. Deliverables status

| Deliverable | Status |
|-------------|--------|
| `scripts/01_remote_recon.py` | Done |
| `data/manifests/remote_file_inventory.json` | Done |
| `docs/month1/01_remote_reconnaissance.md` | Done (this file) |
| Small HF metadata under `data/manifests/hf_small_metadata/` | Done (text/JSON only) |
