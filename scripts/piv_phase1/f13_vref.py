"""F1.3: per-condition reference velocity v_ref,c = p99.9 of |u_planar| and the resulting sigma_u.

Velocity-only statistics. Roles read: train_fit (C149 pool, the training role of the campaign),
train_fit of the historical A38 cache (cross-check), seen_train_eval (TRAIN conditions), and
val_eval (VAL conditions; authorised by roadmap F1.3 for velocity-only statistics).
TEST is never read. T is never read here.
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

out = C.REPO / "results/piv/phase1/f13"; out.mkdir(parents=True, exist_ok=True)
SIGMA_PX = {"main_0.10": 0.10, "lo_0.06": 0.06, "hi_0.15": 0.15}
DMAX = 6.0
rng = np.random.default_rng(20261127)

roles = [("train_fit_C149", "train_fit", C.CACHE_C149), ("train_fit_A38", "train_fit", C.CACHE_A38),
         ("seen_train_eval", "seen_train_eval", C.CACHE_C149), ("val_eval", "val_eval", C.CACHE_C149)]
rows = []
vals = {}
for label, role, cache in roles:
    d = C.load_role(role, cache=cache)
    u = np.hypot(d["vi"].astype(np.float64), d["vj"].astype(np.float64))
    meta = {r["sim_id"]: (r["nh3_fraction"], r["phi"], r["split"]) for r in d["frames"]}
    for c in dict.fromkeys(d["sim_ids"]):
        m = d["sim_ids"] == c
        uu = u[m]; n = uu.shape[0]
        v999 = float(np.percentile(uu, 99.9))
        # frame-bootstrap (resample frames) of the p99.9
        bs = []
        for _ in range(200):
            idx = rng.integers(0, n, n)
            bs.append(np.percentile(uu[idx], 99.9))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        vz = {"p50": float(np.percentile(uu, 50)), "p99": float(np.percentile(uu, 99)), "p99.9": v999,
              "max": float(uu.max())}
        # per-frame p99.9 spread
        pf = np.percentile(uu.reshape(n, -1), 99.9, axis=1)
        row = {"role": label, "split": meta[c][2], "condition": c, "nh3": meta[c][0], "phi": meta[c][1], "n_frames": n,
               "p50": vz["p50"], "p99": vz["p99"], "v_ref_p99.9": v999, "boot95_lo": float(lo), "boot95_hi": float(hi),
               "max": vz["max"], "perframe_p99.9_median": float(np.median(pf)), "perframe_p99.9_min": float(pf.min()),
               "perframe_p99.9_max": float(pf.max())}
        for k, s in SIGMA_PX.items():
            row[f"sigma_u_{k}_m_s"] = s * v999 / DMAX
        rows.append(row)
        vals[(label, c)] = v999
    print(label, "done", flush=True)
    # pooled
    uu = u.reshape(-1)
    print("  pooled p99.9 = %.2f" % np.percentile(uu, 99.9))

with open(out / "vref_sigma_u_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# consistency summary: C149 vs A38 on TRAIN conditions; train vs seen
def col(label): return {r["condition"]: r["v_ref_p99.9"] for r in rows if r["role"] == label}
a, b, s = col("train_fit_C149"), col("train_fit_A38"), col("seen_train_eval")
rel_ab = {c: a[c] / b[c] - 1 for c in a}
rel_as = {c: s[c] / a[c] - 1 for c in a}
summ = {"C149_vs_A38_relative_diff": {"mean": float(np.mean(list(rel_ab.values()))), "max_abs": float(np.max(np.abs(list(rel_ab.values())))),
                                      "per_condition": rel_ab},
        "seen_vs_C149_relative_diff": {"mean": float(np.mean(list(rel_as.values()))), "max_abs": float(np.max(np.abs(list(rel_as.values())))),
                                       "per_condition": rel_as},
        "train_C149_vref_min_max_mean": [min(a.values()), max(a.values()), float(np.mean(list(a.values())))],
        "val_vref": col("val_eval"), "d_max_px": DMAX, "sigma_px": SIGMA_PX}
json.dump(summ, open(out / "vref_consistency.json", "w"), indent=1)
C.dump_provenance(out / "f13_provenance.json", {"script": "f13_vref.py", "bootstrap_seed": 20261127, "roles": [r[0] for r in roles]})
print(json.dumps({k: summ[k] for k in ("train_C149_vref_min_max_mean",)}))
print("val", summ["val_vref"])
print("C149 vs A38 max abs %.3f ; seen vs C149 max abs %.3f" % (summ["C149_vs_A38_relative_diff"]["max_abs"], summ["seen_vs_C149_relative_diff"]["max_abs"]))
