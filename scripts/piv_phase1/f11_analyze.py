"""F1.1 analysis of the auxiliary displacement check: regressions of measured displacement vs candidate velocities."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import common as C
out = C.REPO / "results/piv/phase1/f11"
d = pd.read_csv(out / "displacement_patches.csv")
rng = np.random.default_rng(20261127)

def boot_ci(x, y, n=400, block=None):
    # cluster bootstrap over (sim, frame) pairs
    keys = (d.loc[x.index, "sim"] + "|" + d.loc[x.index, "frame"].astype(str)).values
    uk = np.unique(keys); idx_by = {k: np.where(keys == k)[0] for k in uk}
    xs, ys = x.values, y.values; rs, ss = [], []
    for _ in range(n):
        pick = rng.choice(uk, len(uk))
        ii = np.concatenate([idx_by[k] for k in pick])
        rs.append(np.corrcoef(xs[ii], ys[ii])[0, 1]); ss.append(np.polyfit(xs[ii], ys[ii], 1)[0])
    return np.percentile(rs, [2.5, 97.5]), np.percentile(ss, [2.5, 97.5])

def quality(q, cmin=0.3, pmin=0.9, vmin=0.0):
    return q[(q.edge == 0) & (q.peak >= pmin) & (q.contrast >= cmin)]

res = {}
rows = []
for tracer in ("T", "OH"):
    for label, sel in (("all_structured", lambda q: q), ("fast_|vi,vj|>=8m/s", lambda q: q[np.hypot(q.vi, q.vj) >= 8]),
                       ("cold_Tmean<1800K", lambda q: q[q.Tmean < 1800])):
        q = sel(quality(d[d.tracer == tracer]))
        if len(q) < 30: continue
        for tgt in ("dx_mm", "dy_mm"):
            for k in ("vi", "vj", "vk"):
                x, y = q[f"{k}_ms"], q[tgt]
                r = np.corrcoef(x, y)[0, 1]; sl = np.polyfit(x, y, 1)[0]
                rci, sci = boot_ci(x, y)
                rows.append({"tracer": tracer, "subset": label, "n": len(q), "target": tgt, "predictor": f"Dt*{k}", "r": r, "r_lo": rci[0],
                             "r_hi": rci[1], "slope": sl, "slope_lo": sci[0], "slope_hi": sci[1]})
t = pd.DataFrame(rows)
t.to_csv(out / "displacement_regressions.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(t.round(3).to_string())
means = []
for tracer in ("T",):
    for label, sel in (("all_structured", lambda q: q), ("cold_Tmean<1800K", lambda q: q[q.Tmean < 1800])):
        q = sel(quality(d[d.tracer == tracer]))
        means.append({"tracer": tracer, "subset": label, "n": len(q), "mean_dx_mm": q.dx_mm.mean(), "mean_dy_mm": q.dy_mm.mean(),
                      "mean_vi_ms": q.vi.mean(), "mean_vj_ms": q.vj.mean(), "mean_vk_ms": q.vk.mean(),
                      "mean_Dt_vj_mm": q.vj_ms.mean(), "corr_vi_vk": float(np.corrcoef(q.vi, q.vk)[0, 1])})
pd.DataFrame(means).to_csv(out / "displacement_subset_means.csv", index=False)
print(pd.DataFrame(means).round(3).to_string())
# partial effect of |vk| on peak (out-of-plane decorrelation)
q = quality(d[d.tracer == "T"], pmin=0.0)
X = np.column_stack([np.ones(len(q)), np.hypot(q.vi, q.vj), np.abs(q.vk)])
beta = np.linalg.lstsq(X, 1 - q.peak.values, rcond=None)[0]
print("1-peak ~ a + b|u_inplane| + c|vk| (T tracer, contrast>=0.3, peak>=0 filter off):", beta.round(5), "n=", len(q))
C.dump_provenance(out / "f11_analysis_provenance.json", {"script": "f11_analyze.py", "n_patches_total": int(len(d))})
