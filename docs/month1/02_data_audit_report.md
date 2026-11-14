# Step 1.3 — Full Numerical Data Audit (30 Operating Conditions)

**Date:** 2026-11-14  
**Supersedes:** [`02_pilot_data_audit_report.md`](02_pilot_data_audit_report.md) (pilot of 3 shards; geometry below corrects the earlier burner-orientation guess)  
**Data root:** `data/raw/combustion/hf_dataset/numerical/` (30 Arrow shards, ~59 GB)  
**Primary artifacts:**  
- `scripts/02_audit_trajectories.py`  
- `data/manifests/operating_conditions.yaml`  
- `data/manifests/full30_audit_summary.json`  
- `results/tables/month1_30_conditions_inventory.csv`  
- `results/figures/month1_audit/fig01–fig05_*.png`

---

## 1. Staging status

| Item | Status |
|------|--------|
| All 30 numerical shards present | **Yes** (`data-00000` … `data-00029`) |
| Byte size + SHA256 vs `remote_file_inventory.json` | **Verified** (pilots earlier; remaining 27/13 resumes likewise) |
| Lexicographic shard ↔ `sim_id` map | **Verified for all 30** (`mapping_ok=True`) |
| NaN / Inf anywhere in numerical tensors | **None** |

Each shard is one trajectory: `float32` fields `observed (2001,128,128)` and `numerical (2001,128,128,15)`, with 1D `float64` grids `x,y` (length 128) and `t` (length 2001), \(\Delta t = 2.5\times 10^{-4}\) s.

Absolute time windows differ by case (0.5 s spans): mostly \(t\in[1.5,2.0]\) (20 traj.), \(t\in[3.5,4.0]\) (8), plus two outliers \([1.7,2.2]\) and \([3.8,4.3]\).

---

## 2. Corrected geometric & velocity orientation (from fig02–fig04)

Pilot figures show a **diametral cut of an annular swirl flame**, not a bottom-fired jet into \(+y\).

### Array / physical axes

| Array index | Physical coordinate | Range |
|-------------|---------------------|-------|
| **axis 0 (row)** | \(y\) [m] | \(\approx[0.063,\ 0.159]\) m; increases with row |
| **axis 1 (col)** | \(x\) [m] | \(\approx[-0.037,\ +0.037]\) m; increases with col |

- **Symmetry axis:** horizontal midline **row ≈ 64** (\(y\approx 0.111\) m).  
  fig04 panel (a2): mean \(T\) vs row has **two symmetric cold dips** near rows ~40 and ~88 — the two branches of the annular swirl jet.
- **Burner / inlet:** cold unburnt reactants (\(T\sim 200\text{–}400\) K) enter from the **right boundary** (**col ≈ 127**, \(x\approx +0.037\) m) and diverge in a **V** toward the **left** (**col ≈ 0**, \(x\approx -0.037\) m). The nozzle itself lies **outside** the cropped window (to the right of \(x_{\max}\)).

### Velocity channels

| Channel | Name | Role in this cut |
|--------:|------|------------------|
| 11 | `Velocity[i]` | **In-plane radial / transverse:** \(+\) in the upper branch (row > 64), \(-\) in the lower branch (row < 64) as the annular jet expands away from the centerline |
| 12 | `Velocity[j]` | **Axial (main jet direction):** strongly positive (\(\sim +25\) m/s) inside the two V-branches; negative (\(\sim -10\) m/s) in the **central Inner Recirculation Zone (IRZ)** between branches |
| 13 | `Velocity[k]` | **Out-of-plane azimuthal swirl:** opposite signs above/below row 64 |
| 14 | `Velocity_Magnitude` (exported) | Correlated with \(\lvert\mathbf{u}\rvert=\sqrt{v_i^2+v_j^2+v_k^2}\) but **not identical** and can be **negative** — **always recompute** \(\lvert\mathbf{u}\rvert\) from ch 11–13 for physics |

**Temperature (ch 10)** is in **Kelvin** (domain means ~1860–2190 K; burnt products dominate).

---

## 3. Inventory of all 30 operating conditions

Naming: `{NH3%}NH3_{φ}.h5`. Coverage is **sparse** (not a full Cartesian grid), especially at 60–80% NH₃.

| NH₃% | CH₄% | #traj | φ values |
|-----:|-----:|------:|----------|
| 0 | 100 | 9 | 0.85, 0.90, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30 |
| 20 | 80 | 9 | 0.75, 0.85, 0.90, 1.00, 1.05, 1.10, 1.20, 1.25, 1.30 |
| 40 | 60 | 6 | 0.85, 0.90, 1.00, 1.10, 1.15, 1.20 |
| 60 | 40 | 1 | 0.95 |
| 80 | 20 | 5 | 0.90, 1.05, 1.15, 1.20, 1.25 |

Full per-condition tables (all percentiles, velocity stats, cold fractions, anomaly counts):  
`results/tables/month1_30_conditions_inventory.csv` and `data/manifests/operating_conditions.yaml`.

---

## 4. Temperature & velocity trends across the 30 conditions

From fig05 and the inventory CSV:

### Temperature

- **Mean \(T\)** generally **decreases with NH₃ fraction** (fuel dilution / lower adiabatic flame temperature):  
  - 0% NH₃: mean \(T\) ≈ **2000–2187 K**  
  - 20%: ≈ **1861–2122 K**  
  - 40%: ≈ **1966–2087 K**  
  - 60%: ≈ **2020 K** (single point)  
  - 80%: ≈ **1864–1957 K**
- Versus \(\phi\): within a fixed NH₃ band, mean \(T\) often **peaks near stoichiometric / slightly rich** (\(\phi\sim 1.0\text{–}1.05\)) and falls toward lean and very rich edges.
- **Cold fraction** (\(T<1200\) K) **rises with NH₃** (avg ~3.5% → ~8.5% from 0% to 80% NH₃): larger / more persistent unburnt jet pockets.
- Burnt gas still dominates: typically \(\gtrsim 90\%\) of voxels with \(T>1800\) K.

### Velocity

- Recomputed mean \(\lvert\mathbf{u}\rvert\) is typically **~10–15 m/s** (one lean pure-CH₄ case higher, ~24 m/s).
- Mean \(\lvert\mathbf{u}\rvert\) tends to be **slightly lower at high NH₃** and varies with \(\phi\) (stronger jets on some lean/pure-CH₄ cases).

---

## 5. Data-quality anomalies

| Issue | Finding |
|-------|---------|
| NaN / Inf | **None** (all 30) |
| Duplicate consecutive frames | Not observed in pilots; full audit focused on finite-value scans |
| \(T < 0\) | **15/30** conditions; **rare** (typically \(10\text{–}600\) voxels of \(2001\times128\times128\approx 3.3\times 10^7\), i.e. \(\ll 0.01\%\)). Worst: `60NH3_0.95.h5` (600 voxels). LES undershoot. |
| \(T > 2600\) K | **24/30** conditions; still rare as a fraction (often \(\lt 0.03\%\)). Peaks up to ~3000 K — numerical overshoot, not to be trusted as physical \(T_{\mathrm{ad}}\). |
| Species mole fractions | Frequent tiny negatives (known from pilot); do not equate OH (ch 8) with OH*. |
| Exported magnitude (ch 14) | Unreliable as Euclidean speed; recompute from \(v_i,v_j,v_k\). |

**Thesis handling:** keep Temperature as supervised target in Kelvin; optionally clip only for visualization; never treat missing observations as physical zero; report metrics per operating condition.

---

## 6. Leakage / temporal structure (carry-over from pilot)

\(\Delta t=0.25\) ms. Spatial-mean \(T\) autocorrelation drops within \(O(10)\) frames; flame-brush points need \(O(50\text{–}100+)\) frames. **Entire operating conditions must be withheld** for val/test (Step 1.4). Do not reuse RealPDEBench’s empty in/out-dist maps or time-trunk splits for our protocol.

---

## 7. Figures

| Figure | Content |
|--------|---------|
| fig01 | 30 conditions in \((\mathrm{NH}_3\%,\phi)\) space (pilots highlighted) |
| fig02 | Instantaneous \(T,v_i,v_j,v_k,\lvert U\rvert_{\mathrm{export}}\), heat release (3 pilots) |
| fig03 | Time mean / std fields (3 pilots) |
| fig04 | \(T\) histograms + autocorrelation (3 pilots) |
| **fig05** | **All 30:** mean \(T\), \(T_{p95}\), cold fraction \(T<1200\) K, mean recomputed \(\lvert\mathbf{u}\rvert\) vs NH₃% & \(\phi\) |

---

## 8. Readiness for Step 1.4

Step 1.3 is **complete**. Next:

1. Lock Train / Val / Test-Interp / Test-Extrap **by operating condition** on the 30 \((\mathrm{NH}_3\%,\phi)\) points.  
2. Fix high-temperature region threshold from training-only \(T\) histograms (burnt-dominated; cold shoulder grows with NH₃).  
3. Implement loaders at full **128×128** with observation masks that do not zero-fill unobserved locations as physical zeros.  
4. Use channels **11–12** (and optionally 13) as velocity observations → channel **10** Temperature target; recompute \(\lvert\mathbf{u}\rvert\) if needed.

**Disk:** numerical combustion staged (~59 GB) with ~845 GB free remaining at last check.
