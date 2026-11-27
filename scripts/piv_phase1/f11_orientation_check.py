"""F1.1 AUXILIARY orientation check (never a proof).

Compares the apparent displacement of scalar structures between consecutive frames (dt = 0.25 ms)
with the velocity components Velocity[i], [j], [k], on TRAIN conditions only. T and OH are used only
for this qualification, never as inputs. Raw Arrow shards are read only for TRAIN sim_ids.

Method: template matching (normalised cross-correlation, sub-cell parabolic peak) of 32x32-cell patches
of frame f in frame f+1 within +-14 cells. Displacement in mm uses the actual cell sizes
(dx = 0.581 mm along columns, dy = 0.758 mm along rows; both axes ascend with the array index).
"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import yaml
from scipy.signal import fftconvolve
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--conds", nargs="+", default=["0NH3_1.h5", "40NH3_1.h5", "80NH3_1.05.h5"])
ap.add_argument("--frames", nargs="+", type=int, default=[100 * k for k in range(1, 15)])
ap.add_argument("--out", default=str(C.REPO / "results/piv/phase1/f11"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

RAW = C.DEFAULT_DATA_ROOT / "data/raw/combustion/hf_dataset/numerical"
oc = yaml.safe_load(open(C.REPO / "data/manifests/operating_conditions.yaml"))["conditions"]
shard = {c["sim_id"]: c["shard_filename"] for c in oc}
inv = json.load(open(C.REPO / "data/manifests/remote_file_inventory.json"))["entries"]
inv_sha = {}
for e in inv:
    nm = e.get("path") or e.get("name") or e.get("filename") or ""
    if "numerical" in nm and nm.endswith(".arrow"):
        inv_sha[Path(nm).name] = (e.get("lfs") or {}).get("sha256") or e.get("sha256")

CH = {"T": 10, "OH": 8, "CO2": 4, "vi": 11, "vj": 12, "vk": 13}
P, R_, STEP = 32, 14, 8
dtt = 0.25e-3

def read_frames(sim_id, idx):
    C.refuse_test(sim_id=sim_id)
    path = RAW / shard[sim_id]
    sha = C.sha256_file(path)
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_stream(mm).read_all()
    row = {c: table.column(c)[0].as_py() for c in table.column_names}
    sh = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    arr = np.frombuffer(row["numerical"], dtype=np.float32).reshape(*sh, int(row["numerical_channels"]))
    x = np.frombuffer(row["x"], dtype=np.float64); y = np.frombuffer(row["y"], dtype=np.float64)
    fr = {i: np.array(arr[i]) for i in idx}
    C.READ_LOG.append({"file": str(path), "sha256": sha, "inventory_lfs_sha256": inv_sha.get(path.name), "frames": sorted(idx)})
    return fr, x, y, sha

def ncc_shift(Af, Bf, cy, cx):
    """Template = A[cy-P/2:cy+P/2, cx-P/2:cx+P/2]; search in B within +-R_ cells, clipped to the domain.
    Returns (sy, sx, peak, edge) in cells; edge=True if the peak lies on the boundary of the searched range."""
    h = P // 2
    T0 = Af[cy - h:cy + h, cx - h:cx + h].astype(np.float64)
    syl, syh = max(-R_, h - cy), min(R_, 128 - h - cy)
    sxl, sxh = max(-R_, h - cx), min(R_, 128 - h - cx)
    Rg = Bf[cy + syl - h:cy + syh + h, cx + sxl - h:cx + sxh + h].astype(np.float64)
    Tz = T0 - T0.mean(); tn = np.sqrt((Tz**2).sum())
    if tn < 1e-12 * max(1.0, abs(T0).max()): return None
    num = fftconvolve(Rg, Tz[::-1, ::-1], mode="valid")
    ny, nx = num.shape
    ii = np.pad(Rg, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(Rg**2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    def box(I):
        return I[P:P + ny, P:P + nx] - I[0:ny, P:P + nx] - I[P:P + ny, 0:nx] + I[0:ny, 0:nx]
    s1, s2 = box(ii), box(ii2)
    var = np.maximum(s2 - s1**2 / (P * P), 1e-18)
    ncc = num / (tn * np.sqrt(var))
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak = ncc[iy, ix]
    def sub(v, i):
        if 0 < i < len(v) - 1:
            a_, b_, c_ = v[i - 1], v[i], v[i + 1]
            den = a_ - 2 * b_ + c_
            return 0.0 if den == 0 else 0.5 * (a_ - c_) / den
        return 0.0
    dyf = sub(ncc[:, ix], iy); dxf = sub(ncc[iy, :], ix)
    edge = iy in (0, ny - 1) or ix in (0, nx - 1)
    return (iy + syl) + dyf, (ix + sxl) + dxf, float(peak), edge

g = C.load_geometry(); dx, dy = g["dx"], g["dy"]
st = C.load_stats()
rows = []
ident = []
t0 = time.time()
centers = range(P // 2, 128 - P // 2 + 1, STEP)
for sim in a.conds:
    fr, x, y, sha = read_frames(sim, set(a.frames) | {f + 1 for f in a.frames})
    cache = C.load_role("train_fit")
    for f in a.frames:
        # identity cache vs raw (cache holds frames that are multiples of 10)
        m = np.where((cache["sim_ids"] == sim) & (cache["frame_idx"] == f))[0]
        if m.size:
            for k in ("vi", "vj"):
                ident.append({"sim": sim, "frame": f, "comp": k,
                              "max_abs_diff_m_s": float(np.max(np.abs(cache[k][m[0]] - fr[f][:, :, CH[k]])))})
        A, B = fr[f], fr[f + 1]
        for tracer in ("T", "OH"):
            a_, b_ = A[:, :, CH[tracer]], B[:, :, CH[tracer]]
            for cy in centers:
                for cx in centers:
                    hh = P // 2
                    patch_std = float(a_[cy - hh:cy + hh, cx - hh:cx + hh].std())
                    contrast = patch_std / float(a_.std() + 1e-30)
                    res = ncc_shift(a_, b_, cy, cx)
                    if res is None: continue
                    sy, sx, peak, edge = res
                    sl = (slice(cy - hh, cy + hh), slice(cx - hh, cx + hh))
                    vel = {k: float(0.5 * (A[sl][:, :, CH[k]].mean() + B[sl][:, :, CH[k]].mean())) for k in ("vi", "vj", "vk")}
                    rows.append({"sim": sim, "frame": f, "tracer": tracer, "cy": cy, "cx": cx, "dx_mm": sx * dx, "dy_mm": sy * dy,
                                 "peak": peak, "edge": int(edge), "patch_std": patch_std, "contrast": contrast,
                                 "Tmean": float(A[sl][:, :, CH["T"]].mean()), **{k: v for k, v in vel.items()},
                                 "vi_ms": vel["vi"] * dtt * 1e3, "vj_ms": vel["vj"] * dtt * 1e3, "vk_ms": vel["vk"] * dtt * 1e3})
    print(sim, "done", f"{time.time()-t0:.0f}s", flush=True)
    del fr

with open(out / "displacement_patches.csv", "w", newline="") as fcsv:
    w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(ident, open(out / "cache_vs_raw_identity.json", "w"), indent=1)
C.dump_provenance(out / "f11_provenance.json", {"script": "f11_orientation_check.py", "conds": a.conds, "frames": a.frames,
                  "patch": P, "search": R_, "step": STEP, "runtime_s": time.time() - t0})
print("identity max diff:", max(i["max_abs_diff_m_s"] for i in ident) if ident else None, "n_rows", len(rows))
