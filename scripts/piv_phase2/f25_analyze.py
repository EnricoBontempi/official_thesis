"""F2.5 - Cardiff (isothermal premixed swirl burner, DynamicStudio 3.41): processing chain and substitution rule. OPTIONAL.

Validates the (unofficial) decoder, describes how the real three-stage chain (AdaptivePIV -> ValRange -> ValMovAvg) transforms the
vectors over the 514 frames, and compares the substitution rule of the generator (iterative 3x3 mean of known neighbours) with what
the real chain produced.  Fractions and morphology of the codes describe THIS isothermal dataset only: NOTHING is transferred to
combustion or to SENS-REJ.  Data are read from ~/piv_qualification_data (outside the repository).
"""
import importlib.util, json, os, sys, hashlib, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import convolve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
import common2 as C2
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P

ROOT = Path(os.path.expanduser("~/piv_qualification_data/cardiff_d2017/data/d.2017.0011500240/PIVImages/20170303_143734/HiSense Mk II #1.4wewffbs"))
TOOLS = Path(os.path.expanduser("~/piv_qualification_data/tools"))
out = C.REPO / "results/piv/phase2/f25"; (out / "figures").mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("dsv", TOOLS / "step2_datasets_dynamicstudio_vectors.py")
dsv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dsv)
S1 = ROOT / "I.4xd8gr7l/0h6c/data"; S2 = ROOT / "I.4xd8gr7l/0h6c/bled/data"; S3 = ROOT / "I.4xd8gr7l/0h6c/bled/h4h3/data"
N = 514
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
K3 = np.ones((3, 3)); K3[1, 1] = 0
K5 = np.ones((5, 5)); K5[2, 2] = 0


def nbr_mean(F, known, K):
    cnt = convolve(known.astype(float), K, mode="constant"); tot = convolve(np.where(known, F, 0.0), K, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def one(k):
    r = {}
    try:
        _, a0 = dsv.decode(str(S1 / f"AdaptivePIV#{k}.vectors")); _, a1 = dsv.decode(str(S2 / f"ValRange#{k}.vectors")); _, a2 = dsv.decode(str(S3 / f"ValMovAvg#{k}.vectors"))
    except Exception as e:
        return {"k": k, "error": repr(e)}
    s = [x["status"].astype(int) for x in (a0, a1, a2)]
    fin = [np.isfinite(x["U"]) for x in (a0, a1, a2)]
    r["k"] = k
    r["n"] = int(s[0].size)
    for st in range(3):
        for code, c in Counter(s[st].ravel().tolist()).items():
            r[f"stage{st+1}_code{code}"] = c
        r[f"stage{st+1}_nonfinite"] = int((~fin[st]).sum())
    r["trans"] = {f"{x}>{y}>{z}": c for (x, y, z), c in Counter(zip(s[0].ravel().tolist(), s[1].ravel().tolist(), s[2].ravel().tolist())).items()}
    code_final = s[2]
    sub = (code_final == 16) & fin[2]
    known = (code_final == 0) & fin[2]
    r["n_final_code16"] = int(sub.sum()); r["n_final_code0"] = int(known.sum())
    unchanged_ok = known & fin[0] & (s[0] == 0)
    r["code0_changed_raw_to_final"] = int((unchanged_ok & ((np.abs(a0["U"] - a2["U"]) > 1e-9) | (np.abs(a0["V"] - a2["V"]) > 1e-9))).sum())
    # substitution comparison on the final field (U, V), neighbours = known (code 0) vectors of the final field
    d = {}
    for comp in ("U", "V"):
        F = a2[comp]
        m3 = nbr_mean(F, known, K3); m5 = nbr_mean(F, known, K5)
        # generator rule: iterative 3x3 mean with the code-16 vectors as unknown (domain = finite vectors of the final field)
        dom = fin[2]
        try:
            it, passes = P.substitute_iterative_3x3(np.where(known, F, 0.0), known, dom)
        except Exception:
            it, passes = np.full(F.shape, np.nan), -1
        d[comp] = {"v": F[sub], "m3": m3[sub], "m5": m5[sub], "it": it[sub], "raw": a0[comp][sub], "passes": passes}
        # reference variability: difference between a known vector and the mean of its known neighbours (leave-one-out)
        mk3 = nbr_mean(F, known, K3)
        d[comp]["loo3"] = (F - mk3)[known]
    r["sub"] = {c: {kk: (vv.astype(np.float32) if hasattr(vv, "astype") else vv) for kk, vv in d[c].items()} for c in d}
    r["stage1_nonfinite_mask"] = (~fin[0]) if k in (0, 1, 257, 513) else None
    r["frame_arrays"] = {c: (a2[c].astype(np.float32), a0[c].astype(np.float32), s[2].astype(np.int16), s[0].astype(np.int16), a2["X"].astype(np.float32), a2["Y"].astype(np.float32)) for c in ("U",)} if k in (257,) else None
    return r


if __name__ == "__main__":
    t0 = time.time()
    # ---- reproduction of the scout's sha256
    man = pd.read_csv(TOOLS / "step2_datasets_samples_manifest.tsv", sep="\t")
    man = man[man.dataset_id == "S2D-01"]
    known_sha = set(man.sha256)
    repro = []
    for stage, d_, name in ((1, S1, "AdaptivePIV"), (2, S2, "ValRange"), (3, S3, "ValMovAvg")):
        for k in (0, 1, 257, 513):
            p = d_ / f"{name}#{k}.vectors"
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            repro.append({"stage": stage, "frame": k, "file": name, "sha256": sha, "in_scout_manifest": sha in known_sha})
    pd.DataFrame(repro).to_csv(out / "scout_sha256_reproduction.csv", index=False)
    print("scout sha reproduction:", sum(r["in_scout_manifest"] for r in repro), "/", len(repro), "(manifest has", len(known_sha), "S2D-01 entries)", flush=True)

    with Pool(10) as pool:
        res = pool.map(one, range(N), chunksize=4)
    errs = [r for r in res if "error" in r]
    res = [r for r in res if "error" not in r]
    print("decoded frames", len(res), "decode errors", len(errs), f"{time.time()-t0:.0f}s", flush=True)

    # ---- (1) decoder validation: mean of decoded final fields vs stored statistics
    stat_path = ROOT / "I.4xd8gr7l/0h6c/bled/h4h3/99f/data/StatisticsVec#0.vectors"
    val = {"statistics_file": str(stat_path.relative_to(ROOT)), "bytes": stat_path.stat().st_size}
    try:
        meta, A = dsv.decode(str(stat_path))
        val["decoded"] = True; val["grid"] = [meta["n0"], meta["n1"]]
        Us = np.stack([dsv.decode(str(S3 / f"ValMovAvg#{k}.vectors"))[1]["U"] for k in range(N)]); Vs = np.stack([dsv.decode(str(S3 / f"ValMovAvg#{k}.vectors"))[1]["V"] for k in range(N)])
        mU, mV = np.nanmean(Us, 0), np.nanmean(Vs, 0)
        for c in ("U", "V", "W", "U2", "V2", "Exp"):
            val[f"stat_{c}_finite"] = int(np.isfinite(A[c]).sum())
        ok = np.isfinite(A["U"]) & np.isfinite(mU)
        val["mean_U_vs_stat_U"] = {"n": int(ok.sum()), "max_abs_diff": float(np.abs(A["U"][ok] - mU[ok]).max()) if ok.any() else None,
                                  "corr": float(np.corrcoef(A["U"][ok], mU[ok])[0, 1]) if ok.sum() > 2 else None, "rms_diff": float(np.sqrt(np.mean((A["U"][ok] - mU[ok]) ** 2))) if ok.any() else None}
        ok = np.isfinite(A["V"]) & np.isfinite(mV)
        val["mean_V_vs_stat_V"] = {"n": int(ok.sum()), "max_abs_diff": float(np.abs(A["V"][ok] - mV[ok]).max()) if ok.any() else None, "corr": float(np.corrcoef(A["V"][ok], mV[ok])[0, 1]) if ok.sum() > 2 else None}
        val["stat_status_counts"] = dict(Counter(A["status"].astype(int).ravel().tolist()))
        val["U_range_stat"] = [float(np.nanmin(A["U"])), float(np.nanmax(A["U"]))]
        val["U_range_mean_of_frames"] = [float(np.nanmin(mU)), float(np.nanmax(mU))]
    except Exception as e:
        val["decoded"] = False; val["error"] = repr(e)
    # ---- (2) NaN region vs MaskDefine
    mk = {}
    try:
        raw = (ROOT / "D.4xd8dily/data/MaskDefine#0.image").read_bytes()
        mk["bytes"] = len(raw)
        # find image size in AcquiredImageEnsemble.xml
        xml = (ROOT / "AcquiredImageEnsemble.xml").read_text(errors="ignore")
        import re
        wh = re.findall(r'(?i)(?:width|height)\D{1,20}(\d{3,4})', xml)[:6]; mk["xml_width_height_candidates"] = wh
        # candidate geometry 1344 x 1024, one byte per pixel, payload at the end of the serialised object
        nonf = [r["stage1_nonfinite_mask"] for r in res if r["stage1_nonfinite_mask"] is not None]
        mk["stage1_nonfinite_counts_frames_0_1_257_513"] = [int(m.sum()) for m in nonf]
        inter = np.logical_and.reduce(nonf); union = np.logical_or.reduce(nonf)
        mk["nonfinite_intersection"] = int(inter.sum()); mk["nonfinite_union"] = int(union.sum())
        W_, H_ = 1344, 1024
        payload = np.frombuffer(raw[-W_ * H_:], dtype=np.uint8) if len(raw) >= W_ * H_ else None
        if payload is not None:
            img = payload.reshape(H_, W_)
            mk["mask_payload_unique_values"] = np.unique(img).tolist()[:6]
            # vector grid: 61 x 81 at 16 px; node (i,j) -> pixel (16*j .. , 16*i ..); sample the mask at the node centres
            ny, nx = inter.shape
            ii, jj = np.mgrid[0:ny, 0:nx]
            for sh_name, (y0, x0) in {"origin_0": (0, 0), "origin_8": (8, 8)}.items():
                yy = np.clip(16 * ii + y0, 0, H_ - 1); xx = np.clip(16 * jj + x0, 0, W_ - 1)
                for nm, vv in (("value0", img[yy, xx] == 0), ("value_nonzero", img[yy, xx] != 0)):
                    both = (vv & inter).sum(); mk[f"{sh_name}_{nm}_overlap_with_nonfinite_intersection"] = {"mask_nodes": int(vv.sum()), "intersection_nodes": int(inter.sum()), "overlap": int(both),
                                                                                                         "jaccard": float(both / max(1, (vv | inter).sum()))}
        else:
            mk["mask_payload"] = "too short for 1344x1024 bytes"
    except Exception as e:
        mk["error"] = repr(e)
    json.dump({"decoder_validation": val, "mask_check": mk, "decode_errors": [e["k"] for e in errs]}, open(out / "decoder_and_mask_validation.json", "w"), indent=1, default=str)
    print("decoder validation:", json.dumps(val, default=str)[:600], flush=True)
    print("mask:", json.dumps(mk, default=str)[:600], flush=True)

    # ---- (3) chain behaviour over the 514 frames
    codes = sorted({int(c.split("code")[1]) for r in res for c in r if c.startswith("stage") and "code" in c})
    rows = []
    for r in res:
        row = {"frame": r["k"], "n": r["n"]}
        for st in (1, 2, 3):
            row[f"stage{st}_nonfinite"] = r[f"stage{st}_nonfinite"]
            for c in codes: row[f"stage{st}_code{c}"] = r.get(f"stage{st}_code{c}", 0)
        row["final_code16"] = r["n_final_code16"]; row["final_code0"] = r["n_final_code0"]; row["code0_changed_raw_to_final"] = r["code0_changed_raw_to_final"]
        rows.append(row)
    fr = pd.DataFrame(rows); fr.to_csv(out / "chain_per_frame_counts.csv", index=False)
    tr = Counter()
    for r in res: tr.update(r["trans"])
    tot = sum(tr.values())
    pd.DataFrame([{"stage1>stage2>stage3_codes": k, "count": v, "share_of_all_vector_slots": v / tot} for k, v in sorted(tr.items(), key=lambda x: -x[1])]).to_csv(out / "chain_code_transitions.csv", index=False)
    summ = {"frames": len(res), "vectors_per_frame": res[0]["n"], "codes_seen": codes,
            "mean_share": {f"stage{st}_code{c}": float(fr[f"stage{st}_code{c}"].mean() / fr.n.mean()) for st in (1, 2, 3) for c in codes},
            "mean_share_nonfinite": {f"stage{st}": float(fr[f"stage{st}_nonfinite"].mean() / fr.n.mean()) for st in (1, 2, 3)},
            "final_code16_share_mean_min_max": [float(fr.final_code16.mean() / fr.n.mean()), float(fr.final_code16.min() / fr.n.mean()), float(fr.final_code16.max() / fr.n.mean())],
            "code0_vectors_changed_raw_to_final_total": int(fr.code0_changed_raw_to_final.sum()),
            "note": "describes THIS isothermal dataset; not transferable to combustion or to SENS-REJ"}
    # ---- (4) substitution rule
    cmp_rows = {}
    for comp in ("U", "V"):
        cat = lambda key: np.concatenate([r["sub"][comp][key] for r in res]).astype(np.float64)
        v, m3, m5, it, raw = cat("v"), cat("m3"), cat("m5"), cat("it"), cat("raw")
        loo = cat("loo3")
        ok = np.isfinite(v) & np.isfinite(m3) & np.isfinite(m5) & np.isfinite(it)
        cmp_rows[comp] = {"n_substituted_vectors": int(v.size), "n_compared": int(ok.sum()),
                          "rms_actual_minus_3x3_mean_known": float(np.sqrt(np.mean((v[ok] - m3[ok]) ** 2))),
                          "rms_actual_minus_5x5_mean_known": float(np.sqrt(np.mean((v[ok] - m5[ok]) ** 2))),
                          "rms_actual_minus_generator_iterative_3x3": float(np.sqrt(np.mean((v[ok] - it[ok]) ** 2))),
                          "median_abs_actual_minus_3x3": float(np.median(np.abs(v[ok] - m3[ok]))), "median_abs_actual_minus_5x5": float(np.median(np.abs(v[ok] - m5[ok]))),
                          "median_abs_actual_minus_iterative_3x3": float(np.median(np.abs(v[ok] - it[ok]))),
                          "rms_valid_vector_minus_3x3_mean_of_neighbours (leave-one-out variability scale)": float(np.sqrt(np.nanmean(loo ** 2))),
                          "rms_actual_minus_raw_stage1_value (how much the chain changed the substituted vectors)": float(np.sqrt(np.nanmean((v[np.isfinite(raw) & ok] - raw[np.isfinite(raw) & ok]) ** 2))) if np.isfinite(raw).any() else None,
                          "frac_with_nonfinite_stage1_value": float(1 - np.isfinite(raw).mean())}
    summ["substitution_comparison"] = cmp_rows
    json.dump(summ, open(out / "chain_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ["mean_share"], indent=0)[:800]); print(json.dumps(cmp_rows, indent=1)[:1800], flush=True)

    # ---- figure
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.8), constrained_layout=True)
    top = pd.read_csv(out / "chain_code_transitions.csv").head(8)
    axs[0].barh(range(len(top)), 100 * top.share_of_all_vector_slots, color=BLUE); axs[0].set_yticks(range(len(top))); axs[0].set_yticklabels(top["stage1>stage2>stage3_codes"], fontsize=7.5); axs[0].invert_yaxis()
    axs[0].set_xlabel("% of all vector slots (514 frames x 4941)"); axs[0].set_title("most frequent status paths stage1>stage2>stage3 (codes as decoded)", fontsize=8.5, loc="left")
    axs[1].plot(fr.frame, 100 * fr.final_code16 / fr.n, color=ORANGE, lw=0.8, label="final code 16"); axs[1].plot(fr.frame, 100 * fr.stage1_nonfinite / fr.n, color=GREY, lw=0.8, label="stage 1 non-finite (fixed region)")
    axs[1].set_xlabel("frame"); axs[1].set_ylabel("% of vectors"); axs[1].legend(frameon=False, fontsize=7.5); axs[1].set_title("variation between frames (this isothermal dataset only)", fontsize=8.5, loc="left")
    sub = {c: [np.concatenate([r["sub"][c][k] for r in res]).astype(np.float64) for k in ("v", "m3", "m5", "it")] for c in ("U",)}
    v, m3, m5, it = sub["U"]
    bins = np.linspace(-1.5, 1.5, 61)
    for arr, col, lab in ((v - m3, BLUE, "actual - 3x3 mean (known)"), (v - m5, GREEN, "actual - 5x5 mean (known)"), (v - it, VERM, "actual - generator rule (iterative 3x3)")):
        axs[2].hist(arr[np.isfinite(arr)], bins=bins, histtype="step", color=col, label=lab, density=True)
    axs[2].set_xlabel("U difference (file units, pixels per pair assumed)"); axs[2].legend(frameon=False, fontsize=7); axs[2].set_title("substituted vectors vs candidate rules", fontsize=8.5, loc="left")
    fig.suptitle("Fig. N3 — Cardiff (isothermal): what the real validation/substitution chain does. Descriptive; no frequency is transferred to combustion", fontsize=9.5, x=0.005, ha="left")
    fig.savefig(out / "figures/figN3_cardiff_chain.png"); plt.close(fig)
    C2.dump_provenance(out / "f25_provenance.json", {"script": "f25_analyze.py", "runtime_s": time.time() - t0, "decoder": "step2_datasets_dynamicstudio_vectors.py @ ee9948e (unofficial)"})
    print("done", time.time() - t0)
