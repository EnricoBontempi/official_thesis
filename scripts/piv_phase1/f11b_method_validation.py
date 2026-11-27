"""F1.1 supplement: (1) validate the displacement estimator on a synthetic known shift; (2) lag scaling
and passive-like tracers (CH4/NH3 mole fraction) on TRAIN conditions. Auxiliary only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, yaml
from scipy.ndimage import shift as ndshift
sys.path.insert(0, str(Path(__file__).parent))
import common as C
import importlib.util
spec = importlib.util.spec_from_file_location("f11", Path(__file__).parent / "f11_orientation_check.py")
src = open(Path(__file__).parent / "f11_orientation_check.py").read()
# reuse ncc_shift definition only
ns = {"np": np, "fftconvolve": __import__("scipy.signal", fromlist=["fftconvolve"]).fftconvolve, "P": 32, "R_": 14}
exec(src[src.index("def ncc_shift"):src.index("g = C.load_geometry()")], ns)
ncc_shift = ns["ncc_shift"]
out = C.REPO / "results/piv/phase1/f11"

# (1) synthetic: smooth random field shifted by known (sy, sx) cells
rng = np.random.default_rng(1)
from scipy.ndimage import gaussian_filter
f = gaussian_filter(rng.standard_normal((128, 128)), 4); f /= f.std()
res = []
for sy, sx in [(0, 0), (3.0, -2.0), (-6.5, 4.25), (9.0, 1.0), (-2.0, -8.0)]:
    g2 = ndshift(f, (sy, sx), order=3, mode="nearest")
    est = [ncc_shift(f, g2, cy, cx) for cy in range(48, 81, 8) for cx in range(48, 81, 8)]
    est = np.array([[e[0], e[1]] for e in est])
    res.append({"true": [sy, sx], "est_median": np.median(est, 0).round(3).tolist(), "est_std": est.std(0).round(3).tolist()})
print("synthetic validation:", json.dumps(res))
json.dump(res, open(out / "synthetic_shift_validation.json", "w"), indent=1)

# (2) real TRAIN: lags and tracers
RAW = C.DEFAULT_DATA_ROOT / "data/raw/combustion/hf_dataset/numerical"
oc = yaml.safe_load(open(C.REPO / "data/manifests/operating_conditions.yaml"))["conditions"]
shard = {c["sim_id"]: c["shard_filename"] for c in oc}
g = C.load_geometry(); dx, dy = g["dx"], g["dy"]
tracer_ch = {"0NH3_1.h5": {"CH4": 2}, "40NH3_1.h5": {"NH3": 7, "CH4": 2}, "80NH3_1.05.h5": {"NH3": 7}}
rows = []
for sim, trs in tracer_ch.items():
    C.refuse_test(sim_id=sim)
    with pa.memory_map(str(RAW / shard[sim]), "r") as mm:
        tb = pa.ipc.open_stream(mm).read_all()
    row = {c: tb.column(c)[0].as_py() for c in tb.column_names}
    arr = np.frombuffer(row["numerical"], dtype=np.float32).reshape(2001, 128, 128, 15)
    for f0 in [100 * k for k in range(1, 14)]:
        for lag in (1, 2, 4):
            A, B = arr[f0], arr[f0 + lag]
            for tn, ch in trs.items():
                a_, b_ = A[:, :, ch], B[:, :, ch]
                for cy in range(16, 113, 8):
                    for cx in range(16, 113, 8):
                        hh = 16
                        c = a_[cy-hh:cy+hh, cx-hh:cx+hh].std() / (a_.std() + 1e-30)
                        if c < 0.3: continue
                        r = ncc_shift(a_, b_, cy, cx)
                        if r is None: continue
                        sy, sx, pk, ed = r
                        vel = {k: float(0.5 * (A[cy-hh:cy+hh, cx-hh:cx+hh, 11 + n].mean() + B[cy-hh:cy+hh, cx-hh:cx+hh, 11 + n].mean())) for n, k in enumerate(("vi", "vj", "vk"))}
                        rows.append({"sim": sim, "frame": f0, "lag": lag, "tracer": tn, "cy": cy, "cx": cx, "dx_mm": sx*dx, "dy_mm": sy*dy,
                                     "peak": pk, "edge": int(ed), **{f"{k}_ms": v * lag * 0.25 for k, v in vel.items()},
                                     "Tmean": float(A[cy-hh:cy+hh, cx-hh:cx+hh, 10].mean())})
    del arr
df = pd.DataFrame(rows); df.to_csv(out / "displacement_lags_passive_tracers.csv", index=False)
q = df[(df.edge == 0) & (df.peak >= 0.9)]
print(len(df), len(q))
reg_rows = []
for (tn, lag), s in q.groupby(["tracer", "lag"]):
    X = np.column_stack([np.ones(len(s)), s.vi_ms, s.vj_ms, s.vk_ms])
    line = f"{tn} lag{lag} n={len(s)} "
    for t in ("dx_mm", "dy_mm"):
        b = np.linalg.lstsq(X, s[t].values, rcond=None)[0]
        r2 = 1 - ((s[t] - X @ b)**2).sum() / ((s[t] - s[t].mean())**2).sum()
        line += f"| {t}: coef(vi,vj,vk)={b[1:].round(2)} R2={r2:.2f} "
        reg_rows.append({"tracer": tn, "lag": lag, "n": len(s), "target": t, "intercept_mm": b[0], "coef_dt_vi": b[1], "coef_dt_vj": b[2],
                         "coef_dt_vk": b[3], "R2": r2, "mean_abs_target_mm": float(np.abs(s[t]).mean())})
    print(line)
pd.DataFrame(reg_rows).to_csv(out / "lag_joint_regressions.csv", index=False)
C.dump_provenance(out / "f11b_provenance.json", {"script": "f11b_method_validation.py"})
