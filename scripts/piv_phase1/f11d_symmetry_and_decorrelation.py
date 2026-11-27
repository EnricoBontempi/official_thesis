"""F1.1 supplement (auxiliary): (a) mirror symmetry of the time-mean components about the grid centre lines (no free
centre search), including T as context; (b) which velocity component drives loss of scalar-pattern correlation
between frames (a structure that leaves the plane decorrelates), with a cluster bootstrap over (condition, frame)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import common as C
out = C.REPO / "results/piv/phase1/f11"
M = np.load(out / "mean_maps_train_conditions.npz")
sims = sorted({k.split("|")[0] for k in M.files})
rows = []
for s in sims:
    for comp in ("vi", "vj", "vk", "T"):
        f = M[f"{s}|{comp}"]
        f = f - f.mean()   # remove the spatial mean: otherwise a large offset makes any map look symmetric
        for name, ax in (("about_row_centre(63.5): rows mirrored", 0), ("about_col_centre(63.5): cols mirrored", 1)):
            fl = np.flip(f, axis=ax)
            sym, anti = 0.5 * (f + fl), 0.5 * (f - fl)
            tot = (f**2).sum()
            rows.append({"sim": s, "component": comp, "mirror": name, "sym_energy_frac": float((sym**2).sum() / tot),
                         "anti_energy_frac": float((anti**2).sum() / tot), "mean_m_s": float(M[f"{s}|{comp}"].mean())})
df = pd.DataFrame(rows); df.to_csv(out / "mean_field_symmetry_centre_lines.csv", index=False)
pd.set_option("display.width", 200)
print(df[df.component != "T"].pivot_table(index=["component", "mirror"], values=["sym_energy_frac", "anti_energy_frac"], aggfunc=["median", "min", "max"]).round(3))
print(df[df.component == "T"].groupby("mirror")[["sym_energy_frac"]].median().round(3))
# signs of the mean components (means over the whole map) per condition
print(df[df.mirror.str.startswith("about_row")].pivot(index="sim", columns="component", values="mean_m_s").round(2))

# ---- (b) decorrelation drivers
d = pd.read_csv(out / "displacement_lags_passive_tracers.csv")
d = d[(d.edge == 0)]
d["loss"] = 1 - d.peak
rng = np.random.default_rng(20261127)
res = []
for (tn, lag), s in d.groupby(["tracer", "lag"]):
    X = np.column_stack([np.abs(s.vi_ms / lag), np.abs(s.vj_ms / lag), np.abs(s.vk_ms / lag)])  # |v| in m/s*0.25 ... same scale
    X = (X - X.mean(0)) / X.std(0)
    X = np.column_stack([np.ones(len(s)), X]); y = s.loss.values
    keys = (s.sim + "|" + s.frame.astype(str)).values; uk = np.unique(keys); idx_by = {k: np.where(keys == k)[0] for k in uk}
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    for _ in range(300):
        ii = np.concatenate([idx_by[k] for k in rng.choice(uk, len(uk))])
        bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    res.append({"tracer": tn, "lag": lag, "n": len(s), **{f"coef_|{n}|_per_sd": b0[i + 1] for i, n in enumerate(("vi", "vj", "vk"))},
                **{f"lo_{n}": lo[i + 1] for i, n in enumerate(("vi", "vj", "vk"))}, **{f"hi_{n}": hi[i + 1] for i, n in enumerate(("vi", "vj", "vk"))},
                "mean_loss": float(y.mean())})
r = pd.DataFrame(res); r.to_csv(out / "decorrelation_drivers.csv", index=False)
print(r.round(4).to_string())
C.dump_provenance(out / "f11d_provenance.json", {"script": "f11d_symmetry_and_decorrelation.py"})
