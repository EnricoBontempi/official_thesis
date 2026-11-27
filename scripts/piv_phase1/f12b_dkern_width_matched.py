"""F1.2 supplement: separate kernel SHAPE from kernel WIDTH in the D-kern comparison.

The specified top-hat (L_sr = W) is, after cell-overlap discretisation, 1.3-1.5x wider than the MHG
calibrated to 0.93 W. Here the top-hat width W' is tuned per axis so that its discrete L_sr equals
the MHG discrete L_sr (shape-only difference). TRAIN, C149 train_fit, velocity only.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).parent))
import common as C, piv_kernels as K

out = C.REPO / "results/piv/phase1/f12"
reg = json.load(open(out / "kernel_registry.json"))
g = C.load_geometry(); dx, dy = g["dx"], g["dy"]
d = C.load_role("train_fit"); vi, vj = d["vi"].astype(np.float64), d["vj"].astype(np.float64)
sims = d["sim_ids"]; conds = list(dict.fromkeys(sims)); umag = np.hypot(vi, vj)
vref = {c: np.percentile(umag[sims == c], 99.9) for c in conds}
sigma_u = {c: 0.10 * vref[c] / 6.0 for c in conds}

du_dy, du_dx = np.gradient(vi, dy, dx, axis=(1, 2)); dv_dy, dv_dx = np.gradient(vj, dy, dx, axis=(1, 2))
G = np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
hg = np.zeros(vi.shape, bool)
for c in conds:
    m = sims == c; hg[m] = G[m] >= np.percentile(G[m], 90)

rows = []; matched = []
for W in (1.0, 1.3, 1.5, 1.9):
    mats_m, mats_t = {}, {}
    for ax, D in (("x", dx), ("y", dy)):
        r = [q for q in reg["rows"] if q["W_mm"] == W and q["axis"] == ax][0]
        s, w = K.mhg_weights(r["mhg_sigma_used_mm"], D); Lm = K.lsr_discrete(w, D)
        f = lambda Wp: K.lsr_discrete(K.tophat_weights(Wp, D)[1], D) - Lm
        Wp = brentq(f, 1e-3, 3 * W)
        st, wt = K.tophat_weights(Wp, D)
        matched.append({"W_mm": W, "axis": ax, "mhg_Lsr_disc": Lm, "tophat_W_matched_mm": Wp, "tophat_W_over_W": Wp / W,
                        "th_matched_weights": wt.tolist(), "mhg_weights": w.tolist()})
        mats_m[ax] = K.axis_matrix(128, w, s, D, 0); mats_t[ax] = K.axis_matrix(128, wt, st, D, 0)
    my, mx = K.valid_margin(W, dy), K.valid_margin(W, dx)
    vm = np.zeros((128, 128), bool); vm[my:128 - my, mx:128 - mx] = True
    for comp, arr in (("vi", vi), ("vj", vj)):
        fm = np.matmul(np.matmul(mats_m["y"], arr), mats_m["x"].T); ft = np.matmul(np.matmul(mats_t["y"], arr), mats_t["x"].T)
        diff = fm - ft
        for rname, rmask in (("all", np.ones_like(hg)), ("HG_top10pct_gradient", hg)):
            for c in conds + ["POOLED"]:
                sm = (sims == c) if c != "POOLED" else np.ones(len(sims), bool)
                e = diff[sm][(rmask[sm] & vm[None])]
                rms = float(np.sqrt(np.mean(e**2)))
                su = sigma_u[c] if c != "POOLED" else float(np.mean(list(sigma_u.values())))
                rows.append({"W_mm": W, "component": comp, "region": rname, "condition": c, "rms_mhg_minus_th_matched": rms,
                             "sigma_u": su, "ratio": rms / su})
with open(out / "dkern_width_matched.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(matched, open(out / "tophat_width_matched.json", "w"), indent=1)
C.dump_provenance(out / "f12b_provenance.json", {"script": "f12b_dkern_width_matched.py"})
import pandas as pd
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(df[df.condition == "POOLED"].pivot_table(index=["W_mm", "component"], columns="region", values="ratio").round(3))
q = df[(df.condition != "POOLED") & (df.region == "HG_top10pct_gradient")]
print(q.groupby(["W_mm", "component"]).ratio.agg(["min", "median", "max"]).round(3))
print(pd.DataFrame(matched)[["W_mm", "axis", "mhg_Lsr_disc", "tophat_W_matched_mm", "tophat_W_over_W"]].round(3))
