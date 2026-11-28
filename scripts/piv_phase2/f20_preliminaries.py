"""F2.0 - preliminaries updated after R1 (CPU; TRAIN + velocity-only VAL; never TEST).

(a) v_ref,c = p99.9 of |u_planar| over ALL nodes of frames 0:2001:10 (201 frames) of the raw record of every
    TRAIN and VAL condition (velocity only), with a block-bootstrap 95 % CI (blocks of 10 consecutive sampled frames).
(b) Kernel registry (MHG response, MHG noise, top-hat) with the R1 steps (rows 0.581 / cols 0.758 mm) and the
    SENS-SWAP steps, always calibrated to the discrete target within 1 %.
(c) Content of windows B-in / B-down / B-off (velocity metrics; T only as a declared diagnostic).
(d) Index mapping the F1.2 swapcells tables onto the R1 convention.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
import common2 as C2
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(C.REPO / "results/piv/phase2/f20"))
ap.add_argument("--n-boot", type=int, default=400)
ap.add_argument("--skip-vref", action="store_true")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
g = C.load_geometry()
steps = {"rows": g["dx"], "cols": g["dy"]}   # NB: Phase-1 loader names: dx = 0.581 mm, dy = 0.758 mm (x/y vectors)
assert abs(steps["rows"] - 0.5810574316105278) < 1e-12 and abs(steps["cols"] - 0.7581226848528722) < 1e-12

# ------------------------------------------------------------------------------ (a) v_ref on the full record
splits = C.load_splits()
conds = list(splits["train"]["sim_ids"]) + list(splits["val"]["sim_ids"])
role_of = {c: "TRAIN" for c in splits["train"]["sim_ids"]} | {c: "VAL" for c in splits["val"]["sim_ids"]}
RAW = C.DEFAULT_DATA_ROOT / "data/raw/combustion/hf_dataset/numerical"
oc = {c["sim_id"]: c for c in yaml.safe_load(open(C.REPO / "data/manifests/operating_conditions.yaml"))["conditions"]}
inv = json.load(open(C.REPO / "data/manifests/remote_file_inventory.json"))["entries"]
inv_sha = {}
for e in inv:
    nm = e.get("path") or e.get("name") or e.get("filename") or ""
    if nm.endswith(".arrow") and "numerical" in nm:
        inv_sha[Path(nm).name] = (e.get("lfs") or {}).get("sha256") or e.get("sha256")
FRAMES = list(range(0, 2001, 10))
assert len(FRAMES) == 201
BLOCK = 10
TOPK = 4500
rng = np.random.default_rng(20261128)
rows = []
if not a.skip_vref:
    for c in conds:
        C.refuse_test(sim_id=c)
        path = RAW / oc[c]["shard_filename"]
        sha = C.sha256_file(path)
        with pa.memory_map(str(path), "r") as mm:
            tb = pa.ipc.open_stream(mm).read_all()
        row = {k: tb.column(k)[0].as_py() for k in tb.column_names}
        arr = np.frombuffer(row["numerical"], dtype=np.float32).reshape(2001, 128, 128, 15)
        vel = arr[FRAMES][:, :, :, [11, 12]].astype(np.float64)
        del arr, row, tb
        u = np.hypot(vel[..., 0], vel[..., 1]).reshape(len(FRAMES), -1)
        C.READ_LOG.append({"file": str(path), "sha256": sha, "inventory_lfs_sha256": inv_sha.get(path.name),
                           "channels": "Velocity[i], Velocity[j] only", "frames": "0:2001:10"})
        v999 = float(np.percentile(u, 99.9))
        # blocks of 10 consecutive sampled frames (last block has 1 frame)
        blocks = [u[i:i + BLOCK] for i in range(0, len(FRAMES), BLOCK)]
        tops = [np.sort(b.ravel())[-TOPK:] for b in blocks]
        sizes = np.array([b.size for b in blocks])
        bs = []
        for _ in range(a.n_boot):
            pick = rng.integers(0, len(blocks), len(blocks))
            n_tot = sizes[pick].sum()
            r = int(np.ceil(n_tot * 0.001))  # values strictly above the p99.9 order statistic
            cat = np.concatenate([tops[i] for i in pick])
            k = int((1 - 0.001) * (n_tot - 1))  # linear-interpolation index on the full sorted sample
            # rank from top of the interpolated percentile
            frac = (1 - 0.001) * (n_tot - 1); lo_i = int(np.floor(frac)); hi_i = int(np.ceil(frac)); w = frac - lo_i
            top_lo, top_hi = n_tot - 1 - lo_i, n_tot - 1 - hi_i  # ranks from the top (0 = max)
            srt = np.sort(cat)[::-1]
            assert max(top_lo, top_hi) < len(srt), "TOPK too small"
            bs.append((1 - w) * srt[top_lo] + w * srt[top_hi])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"condition": c, "role": role_of[c], "nh3": oc[c]["nh3_pct"] / 100, "phi": oc[c]["equivalence_ratio"],
                     "n_frames": len(FRAMES), "n_nodes": int(u.size), "p50": float(np.percentile(u, 50)), "p99": float(np.percentile(u, 99)),
                     "v_ref_p99.9": v999, "boot95_lo": float(lo), "boot95_hi": float(hi), "boot_se": float(np.std(bs)),
                     "max": float(u.max()), "shard": path.name, "shard_sha256": sha, "sha_matches_inventory": sha == inv_sha.get(path.name)})
        print(c, f"{v999:.2f} [{lo:.2f},{hi:.2f}] {time.time()-t0:.0f}s", flush=True)
    vref = pd.DataFrame(rows)
    for tag, spx in (("lo_0.06", 0.06), ("main_0.10", 0.10), ("hi_0.15", 0.15)):
        vref[f"sigma_u_{tag}_m_s"] = spx * vref["v_ref_p99.9"] / 6.0
    # compare with the Phase 1 pool C149 values
    p1 = pd.read_csv(C.REPO / "results/piv/phase1/f13/vref_sigma_u_table.csv")
    p1 = p1[p1.role.isin(["train_fit_C149", "val_eval"])].set_index("condition")["v_ref_p99.9"]
    vref["phase1_pool_v_ref"] = vref.condition.map(p1)
    vref["rel_diff_vs_phase1_pool"] = vref["v_ref_p99.9"] / vref["phase1_pool_v_ref"] - 1
    vref.to_csv(out / "vref_full_record.csv", index=False)
    print("v_ref range", vref["v_ref_p99.9"].min(), vref["v_ref_p99.9"].max(), "mean", vref["v_ref_p99.9"].mean())

# ------------------------------------------------------------------------------ (b) kernel registry
Ws = [1.0, 1.3, 1.5, 1.9]
LAMS = [2, 3, 4, 6, 10]
reg = {"primary": {}, "swap": {}}
flat = []
problems = []
for conv in ("primary", "swap"):
    st = P._steps(steps, conv)
    for W in Ws:
        entry = {"response_mhg": {}, "noise": {}, "response_tophat": {}, "window": {}}
        for axis, D in zip(("rows", "cols"), st):
            entry["window"][axis] = {"margin_nodes": P.window_margin(W, D), "step_mm": D, "W_over_step": W / D,
                                      "exact_ratio_W_over_2step": W / (2 * D)}
            for which, kind, tgt in (("response_mhg", "mhg", P.LSR_RESPONSE_OVER_W * W), ("noise", "mhg", P.LSR_NOISE_OVER_W * W),
                                     ("response_tophat", "tophat", W)):
                try:
                    c = P.calibrate(kind, tgt, D)
                    k, w = P.kernel_weights(kind, c["param"], D)
                    H = P.transfer_function(k, w, D, LAMS)
                    ac = P.autocorr_discrete(w, 4)
                    e = {"param_mm": c["param"], "param_nominal_mm": c["param_nominal"], "lsr_target_mm": c["lsr_target"],
                         "lsr_disc_mm": c["lsr_disc"], "lsr_disc_nominal_mm": c["lsr_disc_nominal"], "err_rel": c["err_rel"],
                         "half_nodes": c["half_nodes"], "neg_mass": c["neg_mass"],
                         "support_mm": (P.TRUNC_SIGMA * c["param"]) if kind == "mhg" else c["param"] / 2,
                         "H": {str(l): float(h) for l, h in zip(LAMS, H)},
                         "autocorr_lag_nodes": [float(x) for x in ac], "autocorr_lag_mm": [m * D for m in range(5)]}
                except P.PivError as ex:
                    e = {"error": str(ex)}; problems.append({"conv": conv, "W": W, "axis": axis, "which": which, "error": str(ex)})
                entry[which][axis] = e
                flat.append({"convention": conv, "W_mm": W, "axis": axis, "kernel": which, "step_mm": D, **({k_: v for k_, v in e.items() if not isinstance(v, (dict, list))})})
        reg[conv][f"{W:.1f}"] = entry
json.dump({"steps_mm": steps, "tolerance": P.CAL_TOL, "kernels": reg, "problems": problems}, open(out / "kernel_registry_v2.json", "w"), indent=1)
pd.DataFrame(flat).to_csv(out / "kernel_registry_v2.csv", index=False)
print("calibration problems:", problems)
# monotonicity / coherence of the discrete L_sr(param) maps used by the calibration (top-hat could in principle be non-monotone)
mono = []
for D in (steps["rows"], steps["cols"]):
    ws = np.linspace(0.05, 3.0, 600)
    Lt = np.array([P.lsr_discrete(P.tophat_weights(x, D)[1], D) for x in ws])
    sg = np.linspace(0.02, 1.6, 3000)
    Lm = np.array([P.lsr_discrete(P.mhg_weights(x, D)[1], D) for x in sg])
    dec = np.where(np.diff(Lm) < -1e-9)[0]
    roots = {}
    for W in Ws:
        for nm, t in (("response", P.LSR_RESPONSE_OVER_W * W), ("noise", P.LSR_NOISE_OVER_W * W)):
            roots[f"W{W}_{nm}"] = int((np.diff(np.sign(Lm - t)) != 0).sum())
        roots[f"W{W}_tophat"] = int((np.diff(np.sign(Lt - W)) != 0).sum())
    mono.append({"step_mm": D, "tophat_L_vs_width_monotone_nondecreasing": bool(np.all(np.diff(Lt) >= -1e-9)),
                 "mhg_L_vs_sigma_non_monotone_only_below_sigma_mm": float(sg[dec].max()) if len(dec) else None,
                 "mhg_max_decrease_mm": float(min(0.0, np.diff(Lm).min())), "roots_per_target": roots,
                 "all_targets_have_unique_root": bool(all(v == 1 for v in roots.values()))})
json.dump(mono, open(out / "kernel_calibration_monotonicity.json", "w"), indent=1)
print(mono)

# ------------------------------------------------------------------------------ (c) windows
d = C.load_role("train_fit", want_T=True)
u = np.hypot(d["vi"], d["vj"]); T = d["T"]
res = {"slice_full": {"frac_u_gt_20": float((u > 20).mean()), "mean_speed": float(u.mean()), "rms_vj": float(d["vj"].std()),
                      "frac_T_lt_1800_DIAGNOSTIC": float((T < 1800).mean())}}
for conv in ("primary", "swap"):
    st = P._steps(steps, conv)
    for fid in ("B-in", "B-down", "B-off"):
        r0, r1, c0, c1 = P.field_rect(fid, st)
        sl = (slice(None), slice(r0, r1 + 1), slice(c0, c1 + 1))
        mr, mc = P.window_margin(1.5, st[0]), P.window_margin(1.5, st[1])
        nr, nc = r1 - r0 + 1, c1 - c0 + 1
        res[f"{fid}|{conv}"] = {"rows": [r0, r1], "cols": [c0, c1], "n_rows": nr, "n_cols": nc,
                                 "extent_mm": [(nr - 1) * st[0], (nc - 1) * st[1]], "n_valid_vectors_W1.5": (nr - 2 * mr) * (nc - 2 * mc),
                                 "margins_rows_cols": [mr, mc], "frac_u_gt_20": float((u[sl] > 20).mean()), "mean_speed": float(u[sl].mean()),
                                 "rms_vj": float(d["vj"][sl].std()), "rms_vi": float(d["vi"][sl].std()),
                                 "frac_T_lt_1800_DIAGNOSTIC": float((T[sl] < 1800).mean())}
res["A"] = {"n_valid_vectors_W1.3": (128 - 2 * P.window_margin(1.3, steps["rows"])) * (128 - 2 * P.window_margin(1.3, steps["cols"])),
            "margins_rows_cols_W1.3": [P.window_margin(1.3, steps["rows"]), P.window_margin(1.3, steps["cols"])]}
json.dump(res, open(out / "b_windows_content.json", "w"), indent=1)
for k, v in res.items(): print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})

# ------------------------------------------------------------------------------ (d) swapcells index
sw = json.load(open(C.REPO / "results/piv/phase1/f12_swapcells/kernel_registry.json"))["rows"]
idx = []
for r in sw:
    W = r["W_mm"]
    # swapcells: axis 'x' = array columns with cell 0.758 mm; axis 'y' = array rows with cell 0.581 mm  ==  R1 convention
    axis = "cols" if r["axis"] == "x" else "rows"
    new = reg["primary"][f"{W:.1f}"]
    n_resp = new["response_mhg"][axis]["param_mm"]; n_noise = new["noise"][axis]["param_mm"]
    idx.append({"W_mm": W, "axis_R1": axis, "step_mm": r["cell_mm"], "step_matches_R1": abs(r["cell_mm"] - (steps["cols"] if axis == "cols" else steps["rows"])) < 1e-9,
                "sigma_resp_swapcells": r["mhg_sigma_used_mm"], "sigma_resp_R1": n_resp, "resp_rel_diff": n_resp / r["mhg_sigma_used_mm"] - 1,
                "sigma_noise_swapcells": r["noise_sigma_used_mm"], "sigma_noise_R1": n_noise, "noise_rel_diff": n_noise / r["noise_sigma_used_mm"] - 1,
                "swapcells_resp_Lsr_err_pct": 100 * (r["mhg_Lsr_disc_used_mm"] / (0.93 * W) - 1),
                "filter_table_reusable": bool(abs(n_resp / r["mhg_sigma_used_mm"] - 1) < 5e-3)})
pd.DataFrame(idx).to_csv(out / "swapcells_index.csv", index=False)
print(pd.DataFrame(idx)[["W_mm", "axis_R1", "step_matches_R1", "resp_rel_diff", "noise_rel_diff", "filter_table_reusable"]].round(4).to_string())
C2.dump_provenance(out / "f20_provenance.json", {"script": "f20_preliminaries.py", "runtime_s": time.time() - t0, "bootstrap_seed": 20261128,
                  "n_boot": a.n_boot, "block_frames": BLOCK})
print("done", time.time() - t0)
