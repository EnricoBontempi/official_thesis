"""F2.4 - metrological check on Neal et al. PIV-UQ "unsteady inviscid core" (non-reacting jet). OPTIONAL, qualification only.

Processes the MS images with OpenPIV (single-pass FFT cross-correlation, Gaussian sub-pixel; windows 16/32 px, overlap 50/75 %),
aligns with the HDR reference (5-pair sliding-average DaVis 8.1.6, 32 px, 75 %, outliers interpolated) and measures the
distribution and the spatial autocorrelation of the residuals, to compare with the noise-window relation of the model
(roadmap sec. 2.3(5): L_sr,noise = 1.2 W, rho(2d) ~ 0.11 at 50 % overlap).

Limits (reported, not hidden): OpenPIV single-pass != DaVis multi-pass; the reference has its own resolution, 5-pair averaging,
interpolated outliers and a resampling to MS pixels; this qualifies the error shape in a COLD, NON-REACTING jet only.
Run with the OpenPIV venv from a neutral cwd:  cd /tmp && ~/piv_qualification_data/venv_openpiv/bin/python -P f24_neal.py
Data are read from ~/piv_qualification_data (outside the repository) and never copied into it.
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import tifffile
from scipy.interpolate import RegularGridInterpolator
from openpiv import pyprocess

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/piv_qualification_data/neal_unsteady_core/extracted/Unsteady Inviscid core"))
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results/piv/phase2/f24"))
ap.add_argument("--n-pairs", type=int, default=140)
ap.add_argument("--workers", type=int, default=10)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
DATASETS = {"F001": ("medium-low seeding", "MS_images_F001"), "F005": ("medium-high seeding", "MS_Images_F005")}
CONFIGS = [(16, 0.5), (16, 0.75), (32, 0.5), (32, 0.75)]


def load_hdr(path):
    L = open(path).read().split("\n")
    I = int(L[2].split("I=")[1].split(",")[0]); J = int(L[2].split("J=")[1])
    arr = np.array([list(map(float, l.split())) for l in L[3:] if l.strip()])
    X = arr[:, 0].reshape(J, I); Y = arr[:, 1].reshape(J, I)
    U, V, F = arr[:, 2].reshape(J, I), arr[:, 3].reshape(J, I), arr[:, 5].reshape(J, I)
    U = np.where(F == 1, U, np.nan); V = np.where(F == 1, V, np.nan)
    return X[0], Y[:, 0], U, V


def ref_at(hdr, xs, ys):
    gx, gy, U, V = hdr
    iu = RegularGridInterpolator((gy, gx), U, bounds_error=False, fill_value=np.nan)
    iv = RegularGridInterpolator((gy, gx), V, bounds_error=False, fill_value=np.nan)
    pts = np.stack([ys.ravel(), xs.ravel()], axis=1)
    return iu(pts).reshape(xs.shape), iv(pts).reshape(xs.shape)


def read_img(root, folder, k):
    return tifffile.imread(os.path.join(root, folder, f"B{k:05d}.tif")).astype(np.float32)


def piv_pair(args):
    root, folder, hdr_folder, k, delta, W, ov = args
    a_, b_ = read_img(root, folder, k), read_img(root, folder, k + 1)
    ovp = int(round(W * ov))
    u, v, s2n = pyprocess.extended_search_area_piv(a_, b_, window_size=W, overlap=ovp, dt=1, search_area_size=W, sig2noise_method="peak2peak")
    x, y = pyprocess.get_coordinates(image_size=a_.shape, search_area_size=W, overlap=ovp)
    hdr = load_hdr(os.path.join(root, hdr_folder, f"B{k + delta:05d}.dat"))
    ru, rv = ref_at(hdr, x, y)
    return k, u - ru, v - rv, ru, rv, u, v, s2n


def rho_axis(R, axis, lags, robust_mask=None):
    """Pooled autocorrelation of residual stacks R (n, ny, nx) along axis (1 = y/rows, 2 = x/cols), NaN aware."""
    R = R - np.nanmean(R)
    out = []
    for m in lags:
        if axis == 2: r1, r2 = R[:, :, :-m], R[:, :, m:]
        else: r1, r2 = R[:, :-m, :], R[:, m:, :]
        ok = np.isfinite(r1) & np.isfinite(r2)
        out.append(float((r1[ok] * r2[ok]).sum() / (0.5 * ((r1[ok] ** 2).sum() + (r2[ok] ** 2).sum()))))
    return out


if __name__ == "__main__":
    root = a.root
    prov = {"zip_md5_declared": "eb6d86228f534d6285859ccc0f16c369", "zip_md5_computed": None, "openpiv": __import__("openpiv").__version__,
            "numpy": np.__version__, "datasets": {}}
    for z in ("../Unsteady Inviscid core.zip",):
        zp = os.path.abspath(os.path.join(root, "..", "..", "Unsteady Inviscid core.zip"))
        if os.path.exists(zp):
            h = hashlib.md5(); s = hashlib.sha256()
            with open(zp, "rb") as f:
                while b := f.read(1 << 23): h.update(b); s.update(b)
            prov["zip_md5_computed"], prov["zip_sha256"], prov["zip_bytes"] = h.hexdigest(), s.hexdigest(), os.path.getsize(zp)
    results = {}
    store = {}
    with Pool(a.workers) as pool:
        for ds, (desc, folder) in DATASETS.items():
            hdr_folder = f"{ds}/HDR_data_{ds}"
            folder = f"{ds}/{folder}"
            # ---- alignment HDR file <-> image pair, on a small sample at W=32, 75 %
            ks = list(range(40, 100, 6))
            al = {}
            for delta in range(-3, 4):
                rs = pool.map(piv_pair, [(root, folder, hdr_folder, k, delta, 32, 0.75) for k in ks])
                ru = np.concatenate([r[1][np.isfinite(r[1])] for r in rs]); rv = np.concatenate([r[2][np.isfinite(r[2])] for r in rs])
                q = lambda x: float(np.sqrt(np.mean((x - np.median(x)) ** 2)))
                al[delta] = {"n": int(ru.size), "rms_u": float(np.sqrt(np.mean(ru ** 2))), "robust_std_u": float(1.4826 * np.median(np.abs(ru - np.median(ru)))),
                             "robust_std_v": float(1.4826 * np.median(np.abs(rv - np.median(rv))))}
            best = min(al, key=lambda d_: al[d_]["robust_std_u"])
            print(ds, "alignment (delta: robust std u)", {d_: round(v["robust_std_u"], 4) for d_, v in al.items()}, "-> best", best, flush=True)
            pair_ks = list(range(10, 10 + 3 * a.n_pairs, 3))
            # v sign convention: corr of v with the reference v
            results[ds] = {"description": desc, "alignment": al, "best_delta": best, "configs": {}}
            for W, ov in CONFIGS:
                rs = pool.map(piv_pair, [(root, folder, hdr_folder, k, best, W, ov) for k in pair_ks])
                RU = np.stack([r[1] for r in rs]); RV = np.stack([r[2] for r in rs]); REFU = np.stack([r[3] for r in rs]); REFV = np.stack([r[4] for r in rs])
                MU = np.stack([r[5] for r in rs]); MV = np.stack([r[6] for r in rs])
                d_px = W - int(round(W * ov))
                ok = np.isfinite(RU) & np.isfinite(RV)
                # v sign check
                cv = float(np.corrcoef(MV[ok], REFV[ok])[0, 1]); cu = float(np.corrcoef(MU[ok], REFU[ok])[0, 1])
                entry = {"W_px": W, "overlap": ov, "step_px": d_px, "n_pairs": len(pair_ks), "n_vectors_with_reference": int(ok.sum()),
                         "corr_u_with_ref": cu, "corr_v_with_ref": cv, "ref_u_mean": float(np.nanmean(REFU)), "ref_u_std": float(np.nanstd(REFU)),
                         "ref_v_std": float(np.nanstd(REFV))}
                for name, R in (("u", RU), ("v", RV)):
                    x = R[np.isfinite(R)]
                    med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
                    keep = np.abs(R - med) <= 5 * mad
                    xr = R[np.isfinite(R) & keep]
                    from scipy.stats import kurtosis, skew
                    entry[name] = {"bias_px": float(x.mean()), "std_all_px": float(x.std()), "rms_all_px": float(np.sqrt(np.mean(x**2))),
                                   "robust_sigma_px": float(mad), "std_after_5MAD_px": float(xr.std()), "frac_outliers_gt_5MAD": float(1 - xr.size / x.size),
                                   "excess_kurtosis_all": float(kurtosis(x)), "excess_kurtosis_clean": float(kurtosis(xr)), "skew_clean": float(skew(xr))}
                lags = list(range(1, 9))
                for name, R in (("u", RU), ("v", RV)):
                    x = R[np.isfinite(R)]; med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
                    Rc = np.where(np.abs(R - med) <= 5 * mad, R, np.nan)
                    for tag, RR in (("all", R), ("clean", Rc)):
                        rx, ry = rho_axis(RR, 2, lags), rho_axis(RR, 1, lags)
                        entry.setdefault("rho", {})[f"{name}_{tag}"] = {"cols_x": rx, "rows_y": ry, "lag_px": [m * d_px for m in lags]}
                results[ds]["configs"][f"W{W}_ov{int(ov*100)}"] = entry
                store[f"{ds}|W{W}|ov{int(ov*100)}|RU"] = RU.astype(np.float32); store[f"{ds}|W{W}|ov{int(ov*100)}|RV"] = RV.astype(np.float32)
                store[f"{ds}|W{W}|ov{int(ov*100)}|REFU"] = REFU.astype(np.float32); store[f"{ds}|W{W}|ov{int(ov*100)}|MU"] = MU.astype(np.float32)
                print(ds, W, ov, "std_u clean %.4f robust %.4f px, rho_x(1,2)=%s" % (entry["u"]["std_after_5MAD_px"], entry["u"]["robust_sigma_px"], np.round(entry["rho"]["u_clean"]["cols_x"][:2], 3)),
                      f"{time.time()-t0:.0f}s", flush=True)
    json.dump(results, open(out / "neal_residual_statistics.json", "w"), indent=1)
    stack_dir = Path(os.path.expanduser("~/piv_qualification_data/neal_unsteady_core"))
    np.savez_compressed(stack_dir / "residual_stacks.npz", **store)   # outside the repository (derived from untrusted data, large)
    prov["runtime_s"] = time.time() - t0; prov["n_pairs"] = a.n_pairs
    json.dump(prov, open(out / "f24_provenance.json", "w"), indent=1)
    print("done", time.time() - t0)
