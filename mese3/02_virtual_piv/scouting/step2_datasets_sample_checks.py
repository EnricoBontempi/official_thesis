#!/usr/bin/env python3
"""Checks on small samples reported in `step2_datasets.md` §4 (mission 02, step 2).

Subcommands (all read local files previously obtained with `step2_datasets_remote_zip.py`
or a direct download; nothing is downloaded here):

  neal-dat FILE.dat [...]      Tecplot reference fields of the Neal et al. PIV-UQ database:
                               grid, spacing, flag counts, zero/NaN coincidence, valid bbox.
  tif FILE.tif [...]           Image shape, dtype and intensity percentiles (needs Pillow).
  mrb-zip Data_MRB_PIV_Avg.zip Darmstadt MRB mean fields: per-file NaN/zero counts and the
                               coincidence of vx==0 with vy==NaN (needs scipy).
  arrow-head HEAD.bin TAIL.bin TAIL_OFFSET
                               (shape_h/shape_w buffers are not in HEAD/TAIL; H and W are
                               inferred from the x grid, which is constant along rows)
                               RealPDEBench Arrow shard: HEAD.bin = first bytes of the shard
                               (>= 2 MiB), TAIL.bin = bytes starting at absolute offset
                               TAIL_OFFSET that cover the x, y, t buffers. Recovers the record
                               batch buffer table and reads u (first frames), x, y, t of row 0
                               (needs pyarrow for the schema).
  tecplot3d FILE.dat           Yang et al. single-zone Tecplot point file: grid and exact-zero
                               rows per z plane.

Dependencies: Python >= 3.9, numpy; optional pillow, scipy, pyarrow as noted.
Run as `python3 -I step2_datasets_sample_checks.py <subcommand> ...`.
"""
import json
import re
import struct
import sys
import zipfile

import numpy as np


def neal_dat(paths):
    out = []
    for fn in paths:
        lines = open(fn).read().splitlines()
        I = int(re.search(r"I=\s*(\d+)", lines[2]).group(1))
        J = int(re.search(r"J=\s*(\d+)", lines[2]).group(1))
        a = np.loadtxt(fn, skiprows=3)
        X, Y = a[:, 0].reshape(J, I), a[:, 1].reshape(J, I)
        flag, vel = a[:, 5], a[:, 2:5]
        zero = (vel == 0).all(1)
        F = flag.reshape(J, I)
        rows, cols = np.where(F.any(1))[0], np.where(F.any(0))[0]
        sub = F[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        m = flag == 1
        out.append({
            "file": fn.split("/")[-1], "header": lines[:3], "I": I, "J": J, "rows": len(a),
            "X_range": [X.min(), X.max()], "dX": np.unique(np.diff(X[0])).tolist(),
            "Y_range": [Y.min(), Y.max()], "dY": np.unique(np.diff(Y[:, 0])).tolist(),
            "flag_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(flag, return_counts=True))},
            "zero_vectors": int(zero.sum()), "flag0_and_zero": int((zero & (flag == 0)).sum()),
            "flag1_and_zero": int((zero & (flag == 1)).sum()), "nan": int(np.isnan(vel).sum()),
            "valid_mean_px": vel[m].mean(0).round(3).tolist(), "valid_std_px": vel[m].std(0).round(3).tolist(),
            "flag0_inside_valid_bbox": int((sub == 0).sum()), "valid_bbox_size": int(sub.size),
        })
    return out


def tif(paths):
    from PIL import Image
    out = []
    for fn in paths:
        im = Image.open(fn)
        a = np.array(im)
        out.append({"file": fn.split("/")[-1], "mode": im.mode, "shape": a.shape, "dtype": str(a.dtype),
                    "min": int(a.min()), "max": int(a.max()), "p50": float(np.percentile(a, 50)),
                    "p99": float(np.percentile(a, 99))})
    return out


def mrb_zip(path):
    import io
    import scipy.io as sio
    z = zipfile.ZipFile(path)
    out = []
    for n in sorted(z.namelist()):
        d = sio.loadmat(io.BytesIO(z.read(n)))
        vx, vy, x, y = d["vx"], d["vy"], d["x2D"].ravel(), d["y2D"].ravel()
        nx, ny, zx, zy = np.isnan(vx), np.isnan(vy), vx == 0, vy == 0
        out.append({"file": n, "shape": vx.shape, "dx": round(float(np.diff(x).mean()), 5),
                    "dy": round(float(np.diff(y).mean()), 5), "x_range": [float(x.min()), float(x.max())],
                    "y_range": [float(y.min()), float(y.max())], "nan_vx": int(nx.sum()), "nan_vy": int(ny.sum()),
                    "zero_vx": int(zx.sum()), "zero_vy": int(zy.sum()),
                    "zero_vx_equals_nan_vy": bool((zx == ny).all()) if zx.any() else None,
                    "nan_vx_equals_nan_vy": bool((nx == ny).all())})
    return out


def arrow_head(head, tail, tail_offset):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    b = open(head, "rb").read()
    r = pa.BufferReader(b)
    sch = ipc.read_schema(ipc.read_message(r))
    pos = r.tell()
    _, ln = struct.unpack("<Ii", b[pos:pos + 8])
    meta = b[pos + 8:pos + 8 + ln]
    body0 = pos + 8 + ln
    # longest chain of (offset,length) int64 pairs starting at offset 0 = buffer table
    best = []
    for s in range(0, len(meta) - 32, 8):
        k, run = s, []
        while k + 16 <= len(meta):
            o, l = struct.unpack("<qq", meta[k:k + 16])
            if l < 0 or o < 0 or (not run and o != 0):
                break
            if run and o not in ((run[-1][0] + run[-1][1] + 7) // 8 * 8, run[-1][0] + run[-1][1]):
                break
            run.append((o, l)); k += 16
        if len(run) > len(best):
            best = run
    names = sch.names
    # binary columns have 3 buffers (validity, offsets, data); int32 columns 2 (validity, data)
    nbuf = {"string": 3, "binary": 3, "int32": 2}
    bufmap, i = {}, 0
    for nme, t in zip(names, sch.types):
        k = nbuf[str(t)]
        bufmap[nme] = best[i:i + k]; i += k
    tb = open(tail, "rb").read()

    def tail_bytes(col):
        o, l = bufmap[col][-1]
        a = body0 + o - tail_offset
        return tb[a:a + l]

    nrows = len(bufmap["u"][1]) and bufmap["u"][1][1] // 4 - 1  # int32 offsets: nrows+1 entries
    xf = np.frombuffer(tail_bytes("x"), dtype="<f8")
    HW = len(xf) // nrows
    xf = xf[:HW]
    # x is constant along rows and increases along columns: W = first wrap-around of x
    wraps = np.where(np.diff(xf) < 0)[0]
    W = int(wraps[0] + 1) if wraps.size else HW
    H = HW // W
    u_off = bufmap["u"][2][0]
    nfr = (len(b) - body0 - u_off) // (H * W * 4)
    u = np.frombuffer(b[body0 + u_off:body0 + u_off + nfr * H * W * 4], dtype="<f4").reshape(nfr, H, W)
    x = xf.reshape(H, W)
    y = np.frombuffer(tail_bytes("y"), dtype="<f8")[:H * W].reshape(H, W)
    t = np.frombuffer(tail_bytes("t"), dtype="<f8")
    T = bufmap["u"][2][1] // nrows // (H * W * 4)  # frames per row from the u buffer length
    t = t[:T]
    sim_ids = re.findall(rb"[0-9]{3,}\.h5", b[body0:body0 + 200])
    return {"schema": names, "rows_in_batch": nrows, "row0_sim_id": sim_ids[0].decode() if sim_ids else None,
            "H": H, "W": W, "u_frames_read": int(nfr), "u_nan": int(np.isnan(u).sum()), "u_zero": int((u == 0).sum()),
            "u_min": float(u.min()), "u_max": float(u.max()),
            "x_range": [float(x.min()), float(x.max())], "dx_unique": np.unique(np.round(np.diff(x, axis=1), 6)).tolist(),
            "y_range": [float(y.min()), float(y.max())], "dy_unique": np.unique(np.round(np.diff(y, axis=0), 6)).tolist(),
            "t_first": t[:3].tolist(), "t_last": t[-2:].tolist(), "n_t": len(t),
            "dt_unique": np.unique(np.round(np.diff(t), 6)).tolist()}


def tecplot3d(fn):
    a = np.loadtxt(fn, skiprows=1)
    z0 = (a[:, 3:6] == 0).all(1)
    zs = np.unique(a[:, 2])
    return {"points": a.shape[0], "header": open(fn).readline().strip(),
            "x": [float(a[:, 0].min()), float(a[:, 0].max()), len(np.unique(a[:, 0]))],
            "y": [float(a[:, 1].min()), float(a[:, 1].max()), len(np.unique(a[:, 1]))],
            "z_planes": zs.tolist(), "all_zero_velocity_fraction": round(float(z0.mean()), 4),
            "zero_fraction_by_z": {str(z): round(float(z0[a[:, 2] == z].mean()), 3) for z in zs}}


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    cmd, args = sys.argv[1], sys.argv[2:]
    res = {"neal-dat": lambda: neal_dat(args), "tif": lambda: tif(args), "mrb-zip": lambda: mrb_zip(args[0]),
           "arrow-head": lambda: arrow_head(args[0], args[1], int(args[2])),
           "tecplot3d": lambda: tecplot3d(args[0])}[cmd]()
    json.dump(res, sys.stdout, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print()


if __name__ == "__main__":
    main()
