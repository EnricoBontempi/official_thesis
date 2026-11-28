"""F2.6 - statistics of the generated observations on ``train_fit`` (760 frames, 20 TRAIN conditions). Velocity only.

Every number requested by the roadmap (counts, realised response interior/edge, sigma_u, error autocorrelation,
ideal-vs-observed RMS by velocity region, SENS-REJ cluster sizes, SENS-TH, SENS-SWAP) is produced here and written to
CSV/JSON/NPZ in results/piv/phase2/f26/.  Temperature is not read.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
import common2 as C2
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(C.REPO / "results/piv/phase2/f26"))
ap.add_argument("--spec", default=str(C.REPO / "configs/observations/piv_observation_spec_v1.yaml"))
ap.add_argument("--max-frames-per-cond", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

sha = (Path(a.spec).with_suffix(".sha256")).read_text().split()[0]
spec = P.load_piv_spec(a.spec, expected_sha256=sha)
scale = load_velocity_scale(C.REPO / C.STATS_REL if (C.REPO / C.STATS_REL).exists() else C.DEFAULT_DATA_ROOT / C.STATS_REL, expected_sha256=C.STATS_SHA256_EXPECTED)
g = np.load(C.DEFAULT_DATA_ROOT / C.CACHE_A38 / "geometry.npz")
gen = P.PivObservationGenerator(spec, scale, x_vec_m=g["x"], y_vec_m=g["y"])
d = C.load_role("train_fit", cache=C.CACHE_A38)
VI, VJ, SIMS, FR = d["vi"].astype(np.float64), d["vj"].astype(np.float64), d["sim_ids"], d["frame_idx"]
conds = list(dict.fromkeys(SIMS))
assert VI.shape[0] == 760 and len(conds) == 20
steps = P._steps(spec.steps, "primary")

# ----------------------------------------------------------------------------- regions (R1 sec. 8): velocity only
def grad_mag(u, v):
    du_r, du_c = np.gradient(u, steps[0], steps[1], axis=(1, 2))
    dv_r, dv_c = np.gradient(v, steps[0], steps[1], axis=(1, 2))
    return np.sqrt(du_r**2 + du_c**2 + dv_r**2 + dv_c**2)
G = grad_mag(VI, VJ)
UM = np.hypot(VI, VJ)
HG = np.zeros(VI.shape, bool); JET = np.zeros(VI.shape, bool)
thr = {}
for c in conds:
    m = SIMS == c
    gt, ut = np.percentile(G[m], 90), np.percentile(UM[m], 90)
    HG[m] = G[m] >= gt; JET[m] = UM[m] >= ut
    thr[c] = {"G_p90": float(gt), "U_p90": float(ut)}
REST = ~(HG | JET)
REGIONS = {"all": np.ones(VI.shape, bool), "HG": HG, "JET": JET, "REST": REST}
json.dump({"definition": "decisione_R1 sec. 8: HG = |grad u| in top 10 % of the condition pool (r/c steps 0.581/0.758 mm); JET = |u| top 10 %; REST = other",
           "thresholds_per_condition": thr, "fraction_nodes": {k: float(v.mean()) for k, v in REGIONS.items()}}, open(out / "regions.json", "w"), indent=1)

# ----------------------------------------------------------------------------- accumulators
class Acc:
    def __init__(self): self.s = {}
    def add(self, key, x):
        x = np.asarray(x, dtype=np.float64)
        e = self.s.setdefault(key, [0.0, 0.0, 0])
        e[0] += float((x * x).sum()); e[1] += float(x.sum()); e[2] += x.size
acc = Acc()
ac_acc = {}   # autocorr: key (config, axis, lag) -> [sum z z', sum z^2 over pairs, n]
dist_acc = {}  # key (config, dist, kind) -> [sum, n]
zsamples = {}  # pooled standardised error samples for histogram / QQ
rej_stats = {}
sub_diff = {}
cluster_pool = {}
passes_pool = {}

CONFIGS = ["IDEAL-A", "FILT-A", "PIV-A", "FILT-B-in", "PIV-B-in", "PIV-B-down", "PIV-B-off", "SENS-TH-A", "SENS-TH-B-in",
           "SENS-SWAP-A", "SENS-SWAP-B-in", "SENS-W1.0-A", "SENS-W1.9-A", "SENS-SIGLO-A", "SENS-SIGHI-A", "SENS-RHO0-A", "SENS-RHO0-B-in",
           "SENS-EDGE-A", "SENS-REJ-iso-flag-A", "SENS-REJ-iso-sub-A", "SENS-REJ-clu-flag-A", "SENS-REJ-clu-sub-A", "IDEAL-B-in"]
LAGS = list(range(1, 9))
counts = {}
cache_obs = {}
dist_nodes = {}
def dist_map(rect):
    r0, r1, c0, c1 = rect
    ii, jj = np.mgrid[0:128, 0:128]
    return np.minimum(np.minimum(ii - r0, r1 - ii), np.minimum(jj - c0, c1 - jj))

n_done = 0
for k in range(VI.shape[0]):
    c, fr = SIMS[k], int(FR[k])
    if a.max_frames_per_cond and (SIMS[:k] == c).sum() >= a.max_frames_per_cond:
        continue
    vi, vj = VI[k], VJ[k]
    su = 0.10 * spec.vref(c) / 6.0
    reg = {r: REGIONS[r][k] for r in REGIONS}
    res = {}
    for cfg in CONFIGS:
        o = gen.observe(cfg, vi, vj, sim_id=c, frame_index=fr, role="train", replica=0, return_intermediates=True)
        res[cfg] = o
        it = o.intermediates
        cf = spec.config(cfg)
        rect = o.meta["field_rect_rows_cols"]
        conv = cf.get("steps", "primary")
        valid = it["valid"]
        counts[cfg] = (int(valid.sum()), o.meta["n_observed"], o.meta["field_rect_rows_cols"])
        if cfg.startswith("IDEAL"):
            continue
        which = "response_tophat" if cf["response"] == "tophat" else "response_mhg"
        edge = gen._edge_mask(tuple(rect), which, conv, float(cf["W_mm"]))
        for comp, (ide, fil, er, ob) in {"vi": (vi, it["filtered_vi"], it["err_vi"], it["vals_vi"]), "vj": (vj, it["filtered_vj"], it["err_vj"], it["vals_vj"])}.items():
            obsd = it["observed"]
            for rname, rm in list(reg.items()) + [("edge", np.ones_like(valid)), ("interior", np.ones_like(valid))]:
                if rname == "edge": m = obsd & edge
                elif rname == "interior": m = obsd & ~edge
                else: m = obsd & rm
                if not m.any(): continue
                if cf["response"] != "ideal":
                    acc.add((cfg, c, comp, rname, "filt_minus_ideal"), fil[m] - ide[m])
                acc.add((cfg, c, comp, rname, "obs_minus_ideal"), ob[m] - ide[m])
                if cf["noise"] != "none":
                    acc.add((cfg, c, comp, rname, "error"), er[m])
        # error statistics (pooled, standardised by sigma_u of the condition)
        if cf["noise"] != "none" and cfg in ("PIV-A", "PIV-B-in", "PIV-B-down", "PIV-B-off", "SENS-RHO0-A", "SENS-RHO0-B-in", "SENS-TH-A", "SENS-SWAP-A"):
            sig = cf["sigma_px"] * spec.vref(c) / 6.0
            for comp, er in (("vi", it["err_vi"]), ("vj", it["err_vj"])):
                z = er / sig
                zs = z[valid]
                zsamples.setdefault(cfg, []).append(zs[::7])
                for axis, ax_ in (("rows", 0), ("cols", 1)):
                    for lag in LAGS:
                        if ax_ == 0: z1, z2, v1, v2 = z[:-lag], z[lag:], valid[:-lag], valid[lag:]
                        else: z1, z2, v1, v2 = z[:, :-lag], z[:, lag:], valid[:, :-lag], valid[:, lag:]
                        mm = v1 & v2
                        e = ac_acc.setdefault((cfg, axis, lag), [0.0, 0.0, 0])
                        e[0] += float((z1[mm] * z2[mm]).sum()); e[1] += float(0.5 * ((z1[mm]**2).sum() + (z2[mm]**2).sum())); e[2] += int(mm.sum())
        # distance-to-edge profiles (stationarity of the error, filter effect near the edge)
        if cfg in ("PIV-A", "PIV-B-in"):
            key = tuple(rect)
            if key not in dist_nodes: dist_nodes[key] = dist_map(key)
            dm = dist_nodes[key]
            sig = 0.10 * spec.vref(c) / 6.0
            for dd in range(0, 12):
                mm = valid & (dm == dd) if cfg == "PIV-A" else valid & (dm == dd)
                if not mm.any(): continue
                for kind, arr in (("err2_over_sigma2", (it["err_vi"][mm] / sig) ** 2), ("err2_over_sigma2", (it["err_vj"][mm] / sig) ** 2),
                                  ("filt2_vi", (it["filtered_vi"][mm] - vi[mm]) ** 2), ("filt2_vj", (it["filtered_vj"][mm] - vj[mm]) ** 2)):
                    e = dist_acc.setdefault((cfg, dd, kind), [0.0, 0])
                    e[0] += float(arr.sum()); e[1] += arr.size
    # ---- SENS-TH vs MHG, SENS-SWAP vs primary, rejections
    for fld, cm, cf_ in (("A", "PIV-A", "FILT-A"), ("B-in", "PIV-B-in", "FILT-B-in")):
        th = res[f"SENS-TH-{fld}"].intermediates; base = res[cm].intermediates
        sw = res[f"SENS-SWAP-{fld}"].intermediates
        both = base["valid"] & th["valid"]
        for comp, kk in (("vi", "filtered_vi"), ("vj", "filtered_vj")):
            for rname, rm in reg.items():
                m = both & rm
                if m.any(): acc.add((f"TH-vs-MHG-{fld}", c, comp, rname, "filtered_diff"), th[kk][m] - base[kk][m])
            bs = base["valid"] & sw["valid"]
            for rname, rm in reg.items():
                m = bs & rm
                if m.any(): acc.add((f"SWAP-vs-primary-{fld}", c, comp, rname, "filtered_diff"), sw[kk][m] - base[kk][m])
    for rej in ("iso", "clu"):
        fl = res[f"SENS-REJ-{rej}-flag-A"].intermediates; sb = res[f"SENS-REJ-{rej}-sub-A"].intermediates
        pa = res["PIV-A"].intermediates
        rm = fl["rejected"]
        sizes = P.cluster_sizes(rm)
        cluster_pool.setdefault(rej, []).append(sizes)
        for comp, kk in (("vi", "vals_vi"), ("vj", "vals_vj")):
            for rname, rmask in reg.items():
                m = rm & rmask
                if m.any():
                    acc.add((f"REJ-{rej}-sub-vs-unrejected", c, comp, rname, "diff"), sb[kk][m] - pa[kk][m])
        passes_pool.setdefault(rej, []).append(sb.get("sub_passes"))
        e = rej_stats.setdefault(rej, [0, 0]); e[0] += int(rm.sum()); e[1] += int(fl["valid"].sum())
    n_done += 1
    if n_done % 100 == 0: print(n_done, f"{time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------- write
rows = []
for (cfg, c, comp, rname, kind), (s2, s1, n) in acc.s.items():
    rows.append({"config": cfg, "condition": c, "component": comp, "region": rname, "kind": kind, "n": n,
                 "rms": float(np.sqrt(s2 / n)), "mean": float(s1 / n), "sum_sq": s2, "sum": s1})
df = pd.DataFrame(rows); df.to_csv(out / "rms_by_config_condition_region.csv", index=False)

ac_rows = []
for (cfg, axis, lag), (s12, s11, n) in ac_acc.items():
    D = steps[0] if axis == "rows" else steps[1]
    if cfg == "SENS-SWAP-A": D = steps[1] if axis == "rows" else steps[0]
    ac_rows.append({"config": cfg, "axis": axis, "lag_nodes": lag, "lag_mm": lag * D, "rho": s12 / s11, "n_pairs": n})
pd.DataFrame(ac_rows).to_csv(out / "error_autocorrelation_empirical.csv", index=False)

zq = {}
for cfg, lst in zsamples.items():
    z = np.concatenate(lst)
    qs = np.linspace(0.001, 0.999, 999)
    from scipy.stats import norm, skew, kurtosis
    zq[cfg] = {"n": int(z.size), "mean": float(z.mean()), "std": float(z.std()), "skew": float(skew(z)), "excess_kurtosis": float(kurtosis(z)),
               "frac_abs_gt_3": float((np.abs(z) > 3).mean()), "gauss_frac_abs_gt_3": float(2 * norm.sf(3)), "quantile_levels": qs.tolist(),
               "quantiles": np.quantile(z, qs).tolist()}
json.dump(zq, open(out / "error_distribution.json", "w"))
np.savez_compressed(out / "z_samples_PIV-A_B-in.npz", **{k.replace("-", "_"): np.concatenate(v) for k, v in zsamples.items() if k in ("PIV-A", "PIV-B-in")})

dist_rows = [{"config": k[0], "dist_nodes": k[1], "kind": k[2], "mean": v[0] / v[1], "n": v[1]} for k, v in dist_acc.items()]
pd.DataFrame(dist_rows).to_csv(out / "distance_to_field_edge.csv", index=False)

rej_out = {}
for rej in ("iso", "clu"):
    sz = np.concatenate(cluster_pool[rej])
    rej_out[rej] = {"realised_fraction": rej_stats[rej][0] / rej_stats[rej][1], "n_clusters_per_frame": float(len(sz) / n_done),
                    "size_mean": float(sz.mean()), "size_median": float(np.median(sz)), "size_p90": float(np.percentile(sz, 90)),
                    "size_max": int(sz.max()), "share_of_rejected_in_clusters_ge3": float(sz[sz >= 3].sum() / sz.sum()),
                    "share_of_rejected_in_clusters_ge10": float(sz[sz >= 10].sum() / sz.sum()), "size_hist": np.bincount(np.minimum(sz, 60)).tolist(),
                    "substitution_passes_mean": float(np.mean([p for p in passes_pool[rej] if p is not None])),
                    "substitution_passes_max": int(np.max([p for p in passes_pool[rej] if p is not None]))}
json.dump(rej_out, open(out / "rejection_morphology.json", "w"), indent=1)
json.dump({"counts": {k: {"n_valid": v[0], "n_observed": v[1], "field_rect": v[2]} for k, v in counts.items()}, "frames": n_done, "runtime_s": time.time() - t0},
          open(out / "counts.json", "w"), indent=1)
C2.dump_provenance(out / "f26_stats_provenance.json", {"script": "f26_stats.py", "spec_sha256": sha, "cache": "month2_dense_unet_cache train_fit (760 frames)",
                  "role_for_observations": "train", "replica": 0, "runtime_s": time.time() - t0})
print("done", time.time() - t0, "frames", n_done)
