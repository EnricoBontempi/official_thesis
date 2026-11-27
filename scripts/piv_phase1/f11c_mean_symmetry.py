"""F1.1 supplement (velocity-only except for a context map of T): symmetry of the time-mean fields.

For a swirl burner cut by a plane through its axis, the time-mean axial component is symmetric about the axis line,
the in-plane radial component and the out-of-plane azimuthal component are antisymmetric (signed components).
We reflect the mean maps of Velocity[i], [j], [k] about a vertical line (column c0) or horizontal line (row r0)
and report the symmetric / antisymmetric energy fractions. TRAIN conditions only, frames 0..1480 step 10.
This is an auxiliary diagnostic: it presupposes an axisymmetric mean flow and a slice through the axis.
"""
import json, sys, csv
from pathlib import Path
import numpy as np, pyarrow as pa, yaml
sys.path.insert(0, str(Path(__file__).parent))
import common as C
out = C.REPO / "results/piv/phase1/f11"
RAW = C.DEFAULT_DATA_ROOT / "data/raw/combustion/hf_dataset/numerical"
oc = yaml.safe_load(open(C.REPO / "data/manifests/operating_conditions.yaml"))["conditions"]
shard = {c["sim_id"]: c["shard_filename"] for c in oc}
conds = ["0NH3_1.h5", "40NH3_1.h5", "80NH3_1.05.h5", "20NH3_0.9.h5"]
idx = list(range(0, 1481, 10))
means = {}
for sim in conds:
    C.refuse_test(sim_id=sim)
    p = RAW / shard[sim]
    sha = C.sha256_file(p)
    with pa.memory_map(str(p), "r") as mm:
        tb = pa.ipc.open_stream(mm).read_all()
    row = {c: tb.column(c)[0].as_py() for c in tb.column_names}
    arr = np.frombuffer(row["numerical"], dtype=np.float32).reshape(2001, 128, 128, 15)
    sel = arr[idx][:, :, :, [10, 11, 12, 13]].astype(np.float64)
    means[sim] = {"T": sel[..., 0].mean(0), "vi": sel[..., 1].mean(0), "vj": sel[..., 2].mean(0), "vk": sel[..., 3].mean(0),
                  "vi_rms": sel[..., 1].std(0), "vj_rms": sel[..., 2].std(0), "vk_rms": sel[..., 3].std(0)}
    C.READ_LOG.append({"file": str(p), "sha256": sha, "frames": f"0..1480 step 10 ({len(idx)})"})
    del arr
np.savez_compressed(out / "mean_maps_train_conditions.npz", **{f"{s}|{k}": v for s, d in means.items() for k, v in d.items()})

def reflect_energy(f, c0, axis):
    """Reflect about index c0 (may be half-integer) along axis; return (sym_frac, anti_frac) over the overlap."""
    n = f.shape[axis]
    f = np.moveaxis(f, axis, -1)
    d = np.arange(1, n)
    lo = np.floor(c0 - d / 2 * 2 / 2) if False else None
    # positions c0 - t and c0 + t, t = 0.5, 1, 1.5 ...
    ts = np.arange(0.5 if (2 * c0) % 2 else 1.0, min(c0, n - 1 - c0) + 1e-9, 1.0)
    if len(ts) < 20: return None
    L = np.stack([f[..., int(round(c0 - t))] for t in ts], -1)
    R = np.stack([f[..., int(round(c0 + t))] for t in ts], -1)
    s, a = 0.5 * (L + R), 0.5 * (L - R)
    tot = (s**2).sum() + (a**2).sum()
    return float((s**2).sum() / tot), float((a**2).sum() / tot), len(ts)

rows = []
for sim, m in means.items():
    for comp in ("vi", "vj", "vk"):
        f = m[comp] - 0.0   # raw mean map (no spatial mean removal: signed components)
        for axis, name in ((1, "mirror_about_vertical_line(col c0)"), (0, "mirror_about_horizontal_line(row r0)")):
            best = {"sym": (-1, None), "anti": (-1, None)}
            for c0 in np.arange(40, 88.01, 0.5):
                r = reflect_energy(f, c0, axis)
                if r is None: continue
                if r[0] > best["sym"][0]: best["sym"] = (r[0], float(c0))
                if r[1] > best["anti"][0]: best["anti"] = (r[1], float(c0))
            rows.append({"sim": sim, "component": comp, "mirror": name, "best_sym_frac": best["sym"][0], "c0_sym": best["sym"][1],
                         "best_anti_frac": best["anti"][0], "c0_anti": best["anti"][1],
                         "mean_value_m_s": float(f.mean()), "mean_abs_m_s": float(np.abs(f).mean())})
with open(out / "mean_field_symmetry.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import pandas as pd
pd.set_option("display.width", 220)
df = pd.DataFrame(rows)
print(df.groupby(["component", "mirror"])[["best_sym_frac", "best_anti_frac", "mean_value_m_s", "mean_abs_m_s"]].median().round(3))
print(df[df.sim == "0NH3_1.h5"].round(3).to_string())
C.dump_provenance(out / "f11c_provenance.json", {"script": "f11c_mean_symmetry.py", "conds": conds})
