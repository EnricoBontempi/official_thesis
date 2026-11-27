"""F1.2 (TRAIN only): spectra, structure functions, filter effects at PIV scales.

Reads the C149 ``train_fit`` role (20 TRAIN conditions x 149 frames) from the dense cache,
reconstructs velocity in m/s with the frozen fit-only statistics, and computes:
  * 1D spectra along x and y (Blackman-Harris window, linear detrend), per condition;
  * second-order structure functions S2(r) along x and y;
  * |u - u_filtered| (RMS, p95) for W in {1.0,1.3,1.5,1.9} mm and kernels MHG / top-hat,
    global and by velocity-only regions (high gradient, high speed, rest);
  * RMS(MHG - top-hat) in high-gradient regions versus 0.25 sigma_u (D-kern criterion).
T is used only for one declared diagnostic split (T<1800 K) and never as a region definition.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal.windows import blackmanharris

sys.path.insert(0, str(Path(__file__).parent))
import common as C
import piv_kernels as K

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(C.REPO / "results/piv/phase1/f12"))
ap.add_argument("--registry", default=str(C.REPO / "results/piv/phase1/f12/kernel_registry.json"))
ap.add_argument("--swap-cells", action="store_true", help="sensitivity: exchange the cell sizes of the two array axes")
ap.add_argument("--sigma-px", type=float, default=0.10)
ap.add_argument("--dmax-px", type=float, default=6.0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

reg = json.load(open(a.registry))
g = C.load_geometry(); dx, dy = g["dx"], g["dy"]
if a.swap_cells: dx, dy = dy, dx
d = C.load_role("train_fit", want_T=True)
vi, vj, T = d["vi"], d["vj"], d["T"]
sims = d["sim_ids"]; conds = list(dict.fromkeys(sims))
assert len(conds) == 20 and vi.shape[0] == 2980
umag = np.hypot(vi, vj)

# ---------------------------------------------------------------- v_ref and sigma_u per condition (as in F1.3)
vref = {c: float(np.percentile(umag[sims == c], 99.9)) for c in conds}
sigma_u = {c: a.sigma_px * vref[c] / a.dmax_px for c in conds}

# ---------------------------------------------------------------- spectra
def psd_1d(field, axis, D):
    """field (n, Ny, Nx); mean PSD (cycles/mm) along axis, averaged over frames and lines."""
    f = np.moveaxis(field, axis + 1, -1)  # (n, lines, N)
    N = f.shape[-1]
    t = np.arange(N)
    # remove linear trend per line
    A = np.stack([np.ones(N), t - t.mean()], 1)
    coef = np.linalg.lstsq(A, f.reshape(-1, N).T, rcond=None)[0]
    f = f.reshape(-1, N) - (A @ coef).T
    w = blackmanharris(N, sym=False)
    F = np.fft.rfft(f * w, axis=-1)
    P = (np.abs(F) ** 2).mean(0) * D / (np.sum(w**2))
    P[1:-1] *= 2
    k = np.fft.rfftfreq(N, d=D)  # cycles/mm
    return k, P

spec = {}
for c in conds:
    m = sims == c
    for comp, arr in (("vi", vi), ("vj", vj)):
        for ax, D in ((1, dx), (0, dy)):
            k, P = psd_1d(arr[m], ax, D)
            spec[(c, comp, "x" if ax == 1 else "y")] = (k, P)

def tail_metrics(k, P, kN):
    """Spectral indicators near Nyquist (excluding k=0)."""
    var = np.trapezoid(P, k)
    sel_hi = k > 0.5 * kN                       # wavelengths shorter than 4 cells
    frac_hi = np.trapezoid(P[sel_hi], k[sel_hi]) / var
    sel_top = k > 0.75 * kN
    frac_top = np.trapezoid(P[sel_top], k[sel_top]) / var
    s = (k >= 0.5 * kN) & (k <= 0.95 * kN)
    slope = np.polyfit(np.log(k[s]), np.log(P[s]), 1)[0]
    s2 = (k >= 0.15 * kN) & (k <= 0.4 * kN)
    slope_mid = np.polyfit(np.log(k[s2]), np.log(P[s2]), 1)[0]
    # PSD at 0.9 kN relative to the PSD at 0.2 kN
    p_ratio = np.interp(0.9 * kN, k, P) / np.interp(0.2 * kN, k, P)
    return {"var": float(var), "frac_var_lambda_lt_4cells": float(frac_hi), "frac_var_top_quarter_band": float(frac_top),
            "slope_mid_0.15-0.4kN": float(slope_mid), "slope_high_0.5-0.95kN": float(slope), "psd_ratio_0.9kN_over_0.2kN": float(p_ratio)}

rows_spec = []
for (c, comp, ax), (k, P) in spec.items():
    D = dx if ax == "x" else dy
    kN = 0.5 / D
    r = {"condition": c, "component": comp, "axis": ax, "cell_mm": D, "nyquist_cyc_per_mm": kN}
    r.update(tail_metrics(k, P, kN)); rows_spec.append(r)
with open(out / "spectra_tail_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_spec[0].keys())); w.writeheader(); w.writerows(rows_spec)
np.savez_compressed(out / "spectra_1d.npz", **{f"{c}|{comp}|{ax}|k": v[0] for (c, comp, ax), v in spec.items()},
                    **{f"{c}|{comp}|{ax}|P": v[1] for (c, comp, ax), v in spec.items()})

# ---------------------------------------------------------------- structure functions
lags = np.arange(1, 17)
rows_sf = []
sf = {}
for c in conds:
    m = sims == c
    for comp, arr in (("vi", vi), ("vj", vj)):
        f = arr[m].astype(np.float64)
        for ax, D in ((1, dx), (0, dy)):
            s2 = []
            for r in lags:
                diff = np.take(f, np.arange(r, f.shape[ax + 1]), axis=ax + 1) - np.take(f, np.arange(0, f.shape[ax + 1] - r), axis=ax + 1)
                s2.append(float(np.mean(diff**2)))
            s2 = np.array(s2); var = float(f.var())
            sf[(c, comp, "x" if ax == 1 else "y")] = s2
            for r, v in zip(lags, s2):
                rows_sf.append({"condition": c, "component": comp, "axis": "x" if ax == 1 else "y", "lag_cells": int(r),
                                "lag_mm": float(r * D), "S2": v, "S2_over_2var": v / (2 * var)})
with open(out / "structure_functions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_sf[0].keys())); w.writeheader(); w.writerows(rows_sf)

# ---------------------------------------------------------------- gradient / regions (velocity only)
def grad_mag(u, v):
    du_dy, du_dx = np.gradient(u, dy, dx, axis=(1, 2))
    dv_dy, dv_dx = np.gradient(v, dy, dx, axis=(1, 2))
    return np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)  # (m/s)/mm

G = grad_mag(vi.astype(np.float64), vj.astype(np.float64)).astype(np.float32)
reg_masks = {}
thr = {}
for c in conds:
    m = sims == c
    gt90, gt95 = np.percentile(G[m], [90, 95]); ut90 = np.percentile(umag[m], 90)
    thr[c] = {"G_p90": float(gt90), "G_p95": float(gt95), "U_p90": float(ut90)}
hg = np.zeros(vi.shape, bool); hg95 = np.zeros(vi.shape, bool); jet = np.zeros(vi.shape, bool)
for c in conds:
    m = sims == c
    hg[m] = G[m] >= thr[c]["G_p90"]; hg95[m] = G[m] >= thr[c]["G_p95"]; jet[m] = umag[m] >= thr[c]["U_p90"]
rest = ~(hg | jet)
cold = T < 1800.0
regions = {"all": np.ones(vi.shape, bool), "HG_top10pct_gradient": hg, "HG_top5pct_gradient": hg95,
           "JET_top10pct_speed": jet, "REST": rest, "DIAG_T<1800K": cold, "DIAG_T>=1800K": ~cold}

# ---------------------------------------------------------------- filters
def build_axis_mats(W, kind):
    mats = {}
    for ax, D, N in (("x", dx, 128), ("y", dy, 128)):
        r = [q for q in reg["rows"] if q["W_mm"] == W and q["axis"] == ax][0]
        if kind == "mhg":
            s, w = K.mhg_weights(r["mhg_sigma_used_mm"], D)
        elif kind == "mhg_nominal":
            s, w = K.mhg_weights(r["mhg_sigma_nominal_mm"], D)
        else:
            s, w = K.tophat_weights(W, D)
        mats[ax] = K.axis_matrix(N, w, s, D, 0)
    return mats

def filt(arr, mats):
    return np.matmul(np.matmul(mats["y"], arr), mats["x"].T)

def valid_mask(W):
    my, mx = K.valid_margin(W, dy), K.valid_margin(W, dx)
    m = np.zeros((128, 128), bool); m[my:128 - my, mx:128 - mx] = True
    return m, my, mx

def stats(err, mask):
    e = err[mask]
    if e.size == 0: return (np.nan, np.nan, 0)
    return float(np.sqrt(np.mean(e.astype(np.float64) ** 2))), float(np.percentile(np.abs(e), 95)), int(e.size)

Ws = [1.0, 1.3, 1.5, 1.9]
rows_f = []; rows_dk = []
for W in Ws:
    vm, my, mx = valid_mask(W)
    fields = {}
    for kind in ("mhg", "mhg_nominal", "tophat"):
        M = build_axis_mats(W, kind)
        fields[kind] = (filt(vi.astype(np.float64), M).astype(np.float32), filt(vj.astype(np.float64), M).astype(np.float32))
    # nodes whose kernel support is entirely inside the field (SENS-EDGE-like), per kernel
    for kind in ("mhg", "mhg_nominal", "tophat"):
        fi, fj = fields[kind]
        for comp, ref, fl in (("vi", vi, fi), ("vj", vj, fj)):
            err = fl - ref
            for rname, rm in regions.items():
                for c in conds + ["POOLED"]:
                    sm = (sims == c) if c != "POOLED" else np.ones(len(sims), bool)
                    mask = rm[sm] & vm[None]
                    rms, p95, n = stats(err[sm], mask)
                    rows_f.append({"W_mm": W, "kernel": kind, "component": comp, "region": rname, "condition": c,
                                   "rms_m_s": rms, "p95_abs_m_s": p95, "n_nodes": n,
                                   "sigma_u_m_s": sigma_u[c] if c != "POOLED" else float(np.mean(list(sigma_u.values())))})
    # D-kern: MHG(calibrated) - top-hat
    for comp, ia in (("vi", 0), ("vj", 1)):
        diff = fields["mhg"][ia] - fields["tophat"][ia]
        for rname in ("all", "HG_top10pct_gradient", "HG_top5pct_gradient", "JET_top10pct_speed", "REST"):
            for c in conds + ["POOLED"]:
                sm = (sims == c) if c != "POOLED" else np.ones(len(sims), bool)
                mask = regions[rname][sm] & vm[None]
                rms, p95, n = stats(diff[sm], mask)
                su = sigma_u[c] if c != "POOLED" else float(np.mean(list(sigma_u.values())))
                rows_dk.append({"W_mm": W, "component": comp, "region": rname, "condition": c, "rms_mhg_minus_th": rms,
                                "p95_abs": p95, "sigma_u": su, "ratio_rms_over_sigma_u": rms / su, "n_nodes": n})
    print(f"W={W} done {time.time()-t0:.0f}s", flush=True)

for fn, rows in (("filter_effect.csv", rows_f), ("dkern_mhg_vs_tophat.csv", rows_dk)):
    with open(out / fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# per-region diagnostics: fraction of nodes and mean sigma
reg_frac = {r: float(m.mean()) for r, m in regions.items()}
json.dump({"v_ref_p99.9_train_fit_C149": vref, "sigma_u_sigma_px_%g_dmax_%g" % (a.sigma_px, a.dmax_px): sigma_u,
           "region_thresholds_per_condition": thr, "region_fraction_of_nodes": reg_frac,
           "n_frames": int(vi.shape[0])}, open(out / "f12_context.json", "w"), indent=1)
C.dump_provenance(out / "f12_grid_support_provenance.json", {"script": "f12_grid_support.py", "role": "train_fit (C149)",
                  "runtime_s": time.time() - t0, "sigma_px": a.sigma_px, "dmax_px": a.dmax_px})
print("done", time.time() - t0)
