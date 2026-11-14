# Step 1.3 — Pilot Data Transfer & Trajectory Deep Audit

**Date:** 2026-11-13  
**Scope:** Three representative **numerical** Arrow shards (~5.9 GB), full **128×128**, before committing to the full ~58.6 GB numerical download.  
**Artifacts:** `scripts/02_audit_pilot_trajectories.py`, `data/manifests/operating_conditions.yaml`, `data/manifests/pilot_audit_summary.json`, `data/manifests/pilot_shard_map.json`, `results/figures/month1_audit/fig01–fig04_*.png`.

---

## 1. Shard → `sim_id` mapping and downloads

HuggingFace `state.json` does not embed `sim_id`s. Conversion order matches **lexicographic `sorted(sim_id)`**, verified by reading each pilot shard:

| Pilot role | `sim_id` | NH₃% | φ | Shard | Size | SHA256 |
|------------|----------|-----:|--:|-------|-----:|--------|
| Pure CH₄, near-stoich | `0NH3_1.h5` | 0 | 1.00 | `data-00008-of-00030.arrow` | 2 098 221 120 B | `df315dfb…0ed89` |
| Intermediate blend | `40NH3_1.h5` | 40 | 1.00 | `data-00023-of-00030.arrow` | 2 098 221 120 B | `335fff36…02670` |
| High NH₃ | `80NH3_1.05.h5` | 80 | 1.05 | `data-00026-of-00030.arrow` | 2 098 221 112 B | `569a668c…01aa5` |

All sizes/SHA256 match `data/manifests/remote_file_inventory.json`. Files live under `data/raw/combustion/hf_dataset/numerical/` (plus `data/raw/combustion/mean_std.pt`).

**Lexicographic hypothesis: confirmed** for these three shards (`pilot_shard_map.json`). Non-pilot shard names in `operating_conditions.yaml` use the same hypothesis pending full download verification.

Also downloaded: `combustion/mean_std.pt`, `dataset_info.json`, `state.json`.

---

## 2. Arrow schema and coordinates

Each numerical shard is an **Arrow IPC stream** with **one row** (one complete trajectory).

| Key | Storage | Decoded shape / dtype |
|-----|---------|------------------------|
| `sim_id` | string | e.g. `0NH3_1.h5` |
| `observed` | bytes | `float32 (2001, 128, 128)` — surrogate OH* intensity |
| `numerical` | bytes | `float32 (2001, 128, 128, 15)` |
| `numerical_channels` | int | 15 |
| `shape_t/h/w` | int | 2001, 128, 128 |
| `x`, `y` | bytes + `x_shape`/`y_shape` | **1D `float64 (128,)`** each (not 2D) |
| `t` | bytes + `t_shape` | **1D `float64 (2001,)`** |

### Coordinate / time conventions

- **Units:** metres for \(x,y\) (domain width ~7.4 cm, height span ~9.6 cm). Time in **seconds**.
- \(x \in [-0.0369,\ 0.0369]\) m (≈ symmetric about 0); increases with array axis 1 (columns).
- \(y \in [0.0629,\ 0.1591]\) m; increases with array axis 0 (rows). Burner nozzle lies **below** the cropped window (\(y\) does not start at 0).
- Uniform spacing: \(\Delta x \approx 5.81\times 10^{-4}\) m, \(\Delta y \approx 7.58\times 10^{-4}\) m.
- \(\Delta t = 2.5\times 10^{-4}\) s exactly (4000 Hz). Each trajectory stores **0.5 s** (2001 frames), but **absolute time windows differ** by case (`0NH3_1`: \(t\in[3.5,4.0]\) s; `40NH3_1` / `80NH3_1.05`: \(t\in[1.5,2.0]\) s).

Mesh for plotting: `np.meshgrid(x, y, indexing="xy")` → arrays shaped `(128, 128)`.

---

## 3. Channel physics, units, and velocity orientation

Channel order matches `combustion/channels.json` (Temperature = 10, Velocity[i/j/k] = 11/12/13, Magnitude = 14).

### Temperature (ch 10) — **Kelvin**

| Condition | min | mean | max | Notes |
|-----------|----:|-----:|----:|-------|
| `0NH3_1.h5` | **204.8** | **2186.5** | **2382.9** | Clean; unburnt pocket ~205 K; peak consistent with CH₄–air |
| `40NH3_1.h5` | −160.9* | **2087.3** | **3001.7** | Rare undershoot; mean cooler than pure CH₄ |
| `80NH3_1.05.h5` | −144.6* | **1957.2** | **2977.9** | Rare undershoot; coolest mean (NH₃ dilution) |

\*Negative temperatures are **numerical artifacts** (85 and 36 voxels respectively over \(2001\times128\times128\)); not physical. Peak values ≳3000 K on NH₃ blends also look like LES overshoot. Domain-mean / q50 (~2120–2260 K) and `mean_std.pt` Temperature mean (~1019 K over the **training** distribution with observed channel prepended — see below) support **Kelvin**, not °C.

Unburnt fraction is small (e.g. pure CH₄: only ~0.3% of samples with \(T<400\) K; ~95% with \(T>1800\) K). Histograms are strongly **burnt-product dominated** with a weak cold shoulder — important for the Month‑1 high-\(T\) region threshold.

### Velocity components — orientation conclusion

Evidence from mean fields and edge statistics (all three pilots):

1. **Velocity[j] (ch 12)** has the largest \(|\mathrm{mean}|\) and peak magnitudes (~40 m/s), strongly positive near the bottom and top edges, with a reversed (negative) core — consistent with a **swirl-stabilized jet + central recirculation** in an axial–radial plane.
2. **Velocity[k] (ch 13)** flips sign from bottom (+) to top (−) with near-zero domain mean — classic **out-of-plane / azimuthal swirl** signature on a diametral cut.
3. **Velocity[i] (ch 11)** is weaker, roughly antisymmetric about \(x=0\) — **in-plane horizontal / radial** component.
4. Speeds are \(O(1\text{–}40)\) m/s → **m/s**.

**Working convention for the thesis (to lock after supervisor review):**

| Symbol | Channel | Physical meaning |
|--------|---------|------------------|
| \(x\) | grid axis 1 | Horizontal / radial in diametral plane (m), ≈0 on centreline |
| \(y\) | grid axis 0 | Axial height above burner (m); **burner below** domain |
| Velocity[i] | 11 | In-plane horizontal / radial |
| Velocity[j] | 12 | **Axial** (primary jet direction, \(+j\) roughly \(+y\)) |
| Velocity[k] | 13 | **Azimuthal / swirl** (out-of-plane) |

### Velocity_Magnitude (ch 14) caveat

\(\sqrt{v_i^2+v_j^2+v_k^2}\) is **highly correlated** with ch 14 (\(r\approx0.997\)) but **not identical** (mean abs err ~0.27–0.30 m/s; max err ~20 m/s). Channel 14 even takes **negative** values (min ≈ −8 m/s). Treat ch 14 as a CFD-exported related field, **not** a strict Euclidean magnitude; for physics use recompute \(\lvert\mathbf{u}\rvert=\sqrt{v_i^2+v_j^2+v_k^2}\) from ch 11–13.

### `mean_std.pt`

Tuple `(mean_inputs, mean_targets, std_inputs, std_targets)`, each length **16** = `[observed] + 15 numerical` (RealPDEBench forecasting packing). Temperature sits at index **11** (mean ≈ 1019 K, std ≈ 1042 K). Useful as a reference, but **thesis normalizers must be refit on our Train operating conditions only**.

---

## 4. Data integrity

| Check | `0NH3_1` | `40NH3_1` | `80NH3_1.05` |
|-------|----------|-----------|--------------|
| NaN / Inf (numerical & observed) | **0** | **0** | **0** |
| Identical consecutive frames | 0 | 0 | 0 |
| Negative \(T\) voxels | 0 | 85 | 36 |
| Negative mole-fraction voxels (chs 2–8) | ~4.0e6 | ~1.0e7 | ~1.5e7 |
| Mole fraction > 1 | 0 | 0 | 0 |

No NaNs/Infs or stuck duplicate frames. Species fields show frequent tiny negative values (LES clipping noise), worse at higher NH₃. Heat-release rate also takes large negative values. For reconstruction targets we will use **Temperature** and velocities; apply non-negativity clips only if needed for diagnostics, not by silently zeroing “missing” observations.

---

## 5. Temporal correlation and leakage implication

\(\Delta t=0.25\) ms. Autocorrelation \(R(\tau)\) of **spatially averaged** \(T\) and \(|U|\) decays to \(\lesssim0.1\) within ~**7–21 frames** (~2–5 ms); integral timescales \(\tau_c\sim1.4\text{–}2.1\) ms for spatial-mean \(T\).

At **high-variance flame-brush points**, integral timescales are longer (\(\tau_c\sim14\) ms ≈ **55–60 frames**; \(R<0.1\) near ~100 frames).

**Conclusion:** Neighbouring snapshots on the same trajectory are strongly correlated. Any train/test split that mixes time indices from the **same** `sim_id` (as RealPDEBench’s forecasting trunk split does) would leak. Thesis protocol remains: **withhold entire operating conditions**, and within a kept condition treat temporal subsampling with lag \(\gtrsim O(50\text{–}100)\) frames if multiple snapshots are used.

---

## 6. Figures

| File | Content |
|------|---------|
| `fig01_parameter_space_30_conditions.png` | All 30 \((\mathrm{NH}_3\%,\phi)\) points; pilots starred |
| `fig02_pilot_instantaneous_fields.png` | Frame 1000: \(T\), \(v_i\), \(v_j\), \(v_k\), mag, heat release |
| `fig03_pilot_mean_and_std_fields.png` | Time mean/std of \(T\), \(|U|\), \(v_i\), \(v_j\) |
| `fig04_temperature_histograms_and_autocorr.png` | \(T\) PDFs + \(R(\tau)\) for \(T\) and velocity |

---

## 7. Recommendation for full numerical staging

Pilot checks are sufficient to proceed:

1. Download remaining **27** numerical shards (~52.7 GB) into the same `data/raw/combustion/hf_dataset/numerical/` tree; verify SHA256 against the inventory.
2. Confirm lexicographic shard↔`sim_id` map for all 30 rows; refresh `operating_conditions.yaml`.
3. Do **not** require real OH* data for core \(u,v\to T\) experiments.
4. Implement thesis loaders at **`sub_s=1` (128×128)**; recompute \(\lvert\mathbf{u}\rvert\) from ch 11–13; fit normalizers on Train conditions only.
5. Proceed to Step 1.4 splits using the 30 verified \((\mathrm{NH}_3\%,\phi)\) points (official in/out-dist maps remain empty).

**Disk:** ~58.6 GB numerical total vs ~880 GB free — cleared.
