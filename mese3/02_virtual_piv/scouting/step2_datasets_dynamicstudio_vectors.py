#!/usr/bin/env python3
"""Decoder and counts for Dantec DynamicStudio 3.41 `.vectors` files (Cardiff dataset S2D-01).

Purpose (step 2 scouting, mission 02): reproduce the checks reported in
`step2_datasets.md` §3–4 for B-PIV-S2D-01. This is an UNOFFICIAL, reverse-engineered
reader of the .NET BinaryFormatter serialisation used by DynamicStudio 3.41 for
`Dynamix.Datasets.Vectors.VectorMap`. It was checked only for internal consistency on
the Cardiff files listed in `step2_datasets_samples.sha256`; it has NOT been compared
with an official DynamicStudio export. The numeric meaning of the `Status` field is
NOT documented in the dataset: this script reports raw codes only.

What the decoder relies on (observed in the files, not in any documentation):
  * a BinaryArray record (type 0x07) of rank 2 whose lengths (n0, n1) give the vector grid;
  * the array is followed by n0*n1 MemberReference records (0x09 + int32 object id);
  * the first referenced object is a ClassWithMembersAndTypes record (0x05) for class
    `Dynamix.Datasets.Vectors.Vector` with 14 members U,V,W (double), X,Y,Z (single),
    Status (class VectorStatus, int32 value__), HP1,WP1,HP2,WP2 (int32), U2,V2,Exp (double);
  * the other objects are ClassWithId records (0x01, object id, metadata id).
Array element k is mapped to indices (k // n1, k % n1) assuming row-major order; this
mapping is an assumption checked only through the X/Y pixel coordinates stored in
valid vectors (see `grid_check` in the output).

Dependencies: Python >= 3.9, numpy. No network access.

Usage:
  python3 -I step2_datasets_dynamicstudio_vectors.py summary FILE.vectors [...]
  python3 -I step2_datasets_dynamicstudio_vectors.py compare RAW.vectors RANGE.vectors MOVAVG.vectors
  python3 -I step2_datasets_dynamicstudio_vectors.py nanregion RAW_frameA.vectors RAW_frameB.vectors [...]
      (spatial overlap of the non-finite U vectors across frames of the same level)
Outputs JSON on stdout.
"""
import json
import struct
import sys
from collections import Counter

import numpy as np

CLASS_NAME = b"Dynamix.Datasets.Vectors.Vector"
STATUS_NAME = b"Dynamix.Datasets.Vectors.VectorStatus"


def _i32(d, p):
    return struct.unpack_from("<i", d, p)[0]


def _find_array(d):
    """Return (offset_after_header, n0, n1, refs) for the rank-2 Vector array."""
    for p in range(len(d) - 40):
        if d[p] != 7:
            continue
        rank = _i32(d, p + 6)
        if rank != 2:
            continue
        n0, n1 = struct.unpack_from("<ii", d, p + 10)
        if not (0 < n0 < 10000 and 0 < n1 < 10000):
            continue
        # header: type(1) id(4) arrtype(1) rank(4) lengths(8) typeEnum(1)=4 class name + libid
        q = p + 18
        if d[q] != 4:
            continue
        ln = d[q + 1]
        if d[q + 2:q + 2 + ln] != CLASS_NAME:
            continue
        q = q + 2 + ln + 4
        refs = []
        for _ in range(n0 * n1):
            if d[q] != 9:
                raise ValueError(f"expected MemberReference at {q}, got {d[q]:#x}")
            refs.append(_i32(d, q + 1))
            q += 5
        return q, n0, n1, refs
    raise ValueError("rank-2 Vector array not found")


def _read_status(d, p, status_by_id):
    """Parse the nested VectorStatus object starting at p; return (value, next_pos)."""
    t = d[p]
    if t == 0x05:  # ClassWithMembersAndTypes (first occurrence)
        oid = _i32(d, p + 1)
        ln = d[p + 5]
        name = d[p + 6:p + 6 + ln]
        if name != STATUS_NAME:
            raise ValueError("unexpected status class name")
        q = p + 6 + ln
        nmem = _i32(d, q)
        q += 4
        for _ in range(nmem):  # member names
            q += 1 + d[q]
        q += nmem  # binary type enums
        q += nmem  # primitive type codes (all primitives here)
        q += 4  # library id
        val = _i32(d, q)
        status_by_id[oid] = val
        return val, q + 4
    if t == 0x01:  # ClassWithId
        oid = _i32(d, p + 1)
        val = _i32(d, p + 9)
        status_by_id[oid] = val
        return val, p + 13
    if t == 0x09:  # MemberReference to an earlier status object
        oid = _i32(d, p + 1)
        return status_by_id[oid], p + 5
    raise ValueError(f"unexpected status record type {t:#x} at {p}")


def decode(path):
    d = open(path, "rb").read()
    _, n0, n1, refs = _find_array(d)
    c = d.find(b"\x1f" + CLASS_NAME + b"\x0e\x00\x00\x00")
    if c < 0:
        raise ValueError("Vector class definition not found")
    rec = c - 5  # record type byte + object id precede the name
    if d[rec] != 0x05:
        raise ValueError("class definition record not found")
    first_id = _i32(d, rec + 1)
    q = c + 1 + len(CLASS_NAME) + 4
    for _ in range(14):
        q += 1 + d[q]
    q += 14  # binary type enums
    q += 6  # U,V,W double; X,Y,Z single
    q += 1 + d[q] + 4  # VectorStatus class name + library id
    q += 7  # HP1..WP2 int32, U2,V2,Exp double
    q += 4  # library id of the Vector class record
    status_by_id = {}
    recs = {}
    oid = first_id
    p = q
    n = n0 * n1
    for k in range(n):
        if k > 0:
            if d[p] != 0x01:
                raise ValueError(f"record {k}: expected ClassWithId at {p}, got {d[p]:#x}")
            oid = _i32(d, p + 1)
            p += 9
        U, V, W = struct.unpack_from("<3d", d, p)
        X, Y, Z = struct.unpack_from("<3f", d, p + 24)
        st, p = _read_status(d, p + 36, status_by_id)
        hp = struct.unpack_from("<4i", d, p)
        u2, v2, ex = struct.unpack_from("<3d", d, p + 16)
        p += 40
        recs[oid] = (U, V, W, X, Y, Z, st, *hp, u2, v2, ex)
    missing = [r for r in refs if r not in recs]
    if missing:
        raise ValueError(f"{len(missing)} referenced objects not decoded")
    A = np.array([recs[r] for r in refs], dtype=float)
    out = {
        "n0": n0, "n1": n1, "n_vectors": n, "bytes_consumed_to": p, "file_size": len(d),
        "first_object_id": first_id,
    }
    cols = ["U", "V", "W", "X", "Y", "Z", "status", "HP1", "WP1", "HP2", "WP2", "U2", "V2", "Exp"]
    return out, {c: A[:, i].reshape(n0, n1) for i, c in enumerate(cols)}


def summary(path):
    meta, a = decode(path)
    st = a["status"].astype(int)
    U = a["U"]
    fin = np.isfinite(U)
    valid = (st == 0) & fin
    ii, jj = np.nonzero(valid)
    gx = {
        "X_minus_16i_unique": sorted(set(np.round(a["X"][valid] - 16 * ii, 3).tolist()))[:5],
        "Y_minus_16j_unique": sorted(set(np.round(a["Y"][valid] - 16 * jj, 3).tolist()))[:5],
    }
    by = {}
    for code in sorted(set(st.ravel().tolist())):
        m = st == code
        by[str(code)] = {
            "count": int(m.sum()),
            "finite_U": int(np.isfinite(U[m]).sum()),
            "X_and_Y_zero": int(((a["X"][m] == 0) & (a["Y"][m] == 0)).sum()),
        }
    meta.update({"file": path.split("/")[-1], "status_counts": by,
                 "finite_U_total": int(fin.sum()), "grid_check": gx})
    return meta, a


def neighbour_mean(U, valid, i, j, half=2):
    sl = (slice(max(0, i - half), i + half + 1), slice(max(0, j - half), j + half + 1))
    w = valid[sl].copy()
    w[min(i, half), min(j, half)] = False
    vals = U[sl][w]
    return float(vals.mean()) if vals.size else float("nan")


def compare(raw, rng, mov):
    (m0, a0), (m1, a1), (m2, a2) = summary(raw), summary(rng), summary(mov)
    s0, s1, s2 = (x["status"].astype(int) for x in (a0, a1, a2))
    trans = Counter(zip(s0.ravel().tolist(), s1.ravel().tolist(), s2.ravel().tolist()))
    f0, f2 = np.isfinite(a0["U"]), np.isfinite(a2["U"])
    changed = f0 & f2 & ((np.abs(a0["U"] - a2["U"]) > 1e-9) | (np.abs(a0["V"] - a2["V"]) > 1e-9))
    newly_finite = (~f0) & f2
    # consistency check: how far is a code-16 vector of the final field from the 5x5 mean of
    # its neighbours in the final field (code-0 only, or all finite)? Indicative only: the
    # DynamicStudio substitution rule (iterations=3, area 5x5 per XML) is not reproduced.
    valid2 = (s2 == 0) & f2
    tgt = list(zip(*np.nonzero((s2 == 16) & (s0 == 0))))
    nb = {}
    for label, pool in (("code0_neighbours", valid2), ("all_finite_neighbours", f2)):
        r = []
        for i, j in tgt:
            mu = neighbour_mean(a2["U"], pool, i, j)
            mv = neighbour_mean(a2["V"], pool, i, j)
            r.append(np.hypot(a2["U"][i, j] - mu, a2["V"][i, j] - mv))
        r = np.array(r)
        nb[label] = {"median": float(np.nanmedian(r)) if r.size else None,
                     "p90": float(np.nanpercentile(r, 90)) if r.size else None}
    chg = np.array([np.hypot(a0["U"][i, j] - a2["U"][i, j], a0["V"][i, j] - a2["V"][i, j]) for i, j in tgt])
    res = np.array([0.0] * len(tgt))
    # magnitude scale of valid vectors for reference
    mag = np.hypot(a2["U"], a2["V"])[valid2]
    return {
        "levels": [m0, m1, m2],
        "same_shape": a0["U"].shape == a1["U"].shape == a2["U"].shape,
        "transitions_raw_range_movavg": {f"{k[0]}->{k[1]}->{k[2]}": v for k, v in sorted(trans.items())},
        "vectors_changed_raw_to_movavg": int(changed.sum()),
        "changed_final_status_counts": dict(Counter(s2[changed].tolist())),
        "nonfinite_raw_to_finite_movavg": int(newly_finite.sum()),
        "newly_finite_final_status_counts": dict(Counter(s2[newly_finite].tolist())),
        "neighbour_check_code16_from_code0": {
            "n": int(res.size),
            "abs_diff_final_value_to_5x5_mean_in_final_field": nb,
            "median_abs_change_raw_to_final": float(np.median(chg)) if chg.size else None,
            "median_magnitude_code0_vectors": float(np.median(mag)),
            "units": "same as U,V in file (unit not decoded; pixels per frame pair assumed, not verified)",
        },
    }


def nanregion(paths):
    masks = {}
    for pth in paths:
        _, a = decode(pth)
        masks[pth.split("/")[-1]] = ~np.isfinite(a["U"])
    m = list(masks.values())
    inter, union = np.logical_and.reduce(m), np.logical_or.reduce(m)
    return {"nonfinite_per_file": {k: int(v.sum()) for k, v in masks.items()},
            "intersection": int(inter.sum()), "union": int(union.sum())}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("summary", "compare", "nanregion"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "summary":
        out = [summary(p)[0] for p in sys.argv[2:]]
    elif sys.argv[1] == "nanregion":
        out = nanregion(sys.argv[2:])
    else:
        out = compare(*sys.argv[2:5])
    json.dump(out, sys.stdout, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print()


if __name__ == "__main__":
    main()
