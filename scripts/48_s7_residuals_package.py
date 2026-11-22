#!/usr/bin/env python3
"""Package of the S7 residuals: neural swaps, A38 Gappy swap, C149 baselines, batch diagnostic. CPU only, no inference.

Refuses unless every input is COMPLETE/DIAG_OK and consistent with the frozen matrix. Descriptive only: no winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("seen_train_eval", "val_eval")
ROLE_LABEL = {"seen_train_eval": "seen220", "val_eval": "VAL123"}
COLOR = {"sparse_unet": "#2a6fb0", "adapted_glu": "#d9731a"}


def _cfg(f, p):
    return f"{'U-Net' if f == 'sparse_unet' else 'GLU'} p_{'on' if p else 'off'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--swap-root", type=Path, required=True)
    ap.add_argument("--gappy-swap-root", type=Path, required=True)
    ap.add_argument("--baselines-root", type=Path, required=True)
    ap.add_argument("--diag-root", type=Path, required=True)
    ap.add_argument("--package-tables", type=Path, required=True)
    ap.add_argument("--historical-summary", type=Path, default=ROOT / "results/tables/month2_baseline_summary.csv")
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    issues: list[str] = []
    # ---------------- neural swap
    rows, cond_rows, frame_rows = [], [], []
    for e in matrix["runs"]:
        d = args.swap_root / e["run_id"]
        if not (d / "COMPLETE.json").exists():
            issues.append(f"{e['run_id']}: swap COMPLETE.json missing")
            continue
        s = json.loads((d / "swap_summary.json").read_text())
        if s["checks_failed"] or s["tags"]["weights_sha256"] != e["checkpoints"]["best"]["weights_sha256"] or s["tags"]["model_seed"] != e["model_seed"]:
            issues.append(f"{e['run_id']}: swap identity/checks")
        for role in ROLES:
            r = s["roles"][role]
            rows.append({"run_id": e["run_id"], "family": e["family"], "p_conditioning": e["p_conditioning"], "model_seed": e["model_seed"],
                         "best_epoch": e["checkpoints"]["best"]["epoch"], "role": role, "n_frames": r["n_frames"],
                         "aligned_macro_K": r["aligned_macro_K"], "aligned_replay_macro_K": r["aligned_replay_macro_K"],
                         "aligned_replay_max_abs_frame_delta_K": r["aligned_replay_max_abs_frame_delta_K"],
                         "swapped_macro_K": r["swapped_macro_K"], "delta_swap_minus_aligned_K": r["swapped_macro_K"] - r["aligned_macro_K"],
                         "donor_reference_macro_K": r["donor_reference_macro_K"]})
            sw = pd.read_csv(d / f"per_frame_swap_{role}.csv", usecols=["sim_id", "frame_index", "rmse_K", "mask_hash"])
            al = pd.read_csv(args.eval_root / e["run_id"] / "per_frame" / "best" / "cov0.05" / role / "rep0.csv", usecols=["sim_id", "frame_index", "rmse_K", "mask_hash"])
            dn = pd.read_csv(d / f"per_frame_donor_reference_{role}.csv", usecols=["sim_id", "frame_index", "rmse_K"])
            m = sw.merge(al, on=["sim_id", "frame_index"], suffixes=("_swap", "_aligned")).merge(dn.rename(columns={"rmse_K": "rmse_K_donor"}), on=["sim_id", "frame_index"])
            if len(m) != len(sw) or not (m["mask_hash_swap"] == m["mask_hash_aligned"]).all():
                issues.append(f"{e['run_id']}/{role}: swap frames or mask_hash differ from the aligned reference")
            for sim, g in m.groupby("sim_id"):
                cond_rows.append({"run_id": e["run_id"], "family": e["family"], "p_conditioning": e["p_conditioning"], "model_seed": e["model_seed"], "role": role,
                                  "sim_id": sim, "n_frames": len(g), "aligned_K": g["rmse_K_aligned"].mean(), "swapped_K": g["rmse_K_swap"].mean(),
                                  "donor_reference_K": g["rmse_K_donor"].mean(), "delta_swap_minus_aligned_K": (g["rmse_K_swap"] - g["rmse_K_aligned"]).mean(),
                                  "frac_frames_swapped_below_aligned": float((g["rmse_K_swap"] < g["rmse_K_aligned"]).mean()),
                                  "frac_frames_swapped_below_donor_ref": float((g["rmse_K_swap"] < g["rmse_K_donor"]).mean())})
    swap = pd.DataFrame(rows)
    swap_c = pd.DataFrame(cond_rows)
    don = swap.groupby("role")["donor_reference_macro_K"].agg(["min", "max"])
    if len(swap) and (don["max"] - don["min"]).max() > 1e-9:
        issues.append("donor reference differs between runs (it must be model independent)")
    # ---------------- A38 Gappy swap
    g = args.gappy_swap_root
    gap = pd.DataFrame()
    if not (g / "COMPLETE.json").exists():
        issues.append("gappy swap COMPLETE.json missing")
    else:
        gs = json.loads((g / "swap_summary.json").read_text())
        if gs["checks_failed"]:
            issues.append("gappy swap checks failed")
        gap = pd.DataFrame([{"context": "A38 historical frozen artifact (760-frame fit)", "role": k.split("/")[0], "p_conditioning": k.split("/")[1] == "p_on", **v}
                            for k, v in gs["roles"].items()])
        gap["delta_swap_minus_aligned_K"] = gap["swapped_macro_K"] - gap["aligned_macro_K"]
        gap["artifact_sha256"] = gs["artifact_sha256"]
    # ---------------- baselines
    b = args.baselines_root
    if not (b / "COMPLETE.json").exists():
        issues.append("baselines COMPLETE.json missing")
    base = pd.read_csv(b / "summary_baselines_c149.csv") if (b / "summary_baselines_c149.csv").exists() else pd.DataFrame()
    fitprov = json.loads((b / "fit_provenance.json").read_text()) if (b / "fit_provenance.json").exists() else {}
    ctl = json.loads((b / "control_a38_subset.json").read_text()) if (b / "control_a38_subset.json").exists() else {}
    if not ctl.get("ok"):
        issues.append("A38-subset control not ok")
    # ---------------- batch diag
    dg = args.diag_root
    diag = pd.read_csv(dg / "batch_diag_summary.csv") if (dg / "DIAG_OK.json").exists() else pd.DataFrame()
    if diag.empty:
        issues.append("batch diagnostic DIAG_OK missing")
    if issues:
        print("PACKAGE REFUSED:")
        [print(" -", i) for i in issues]
        return 1
    args.dest.mkdir(parents=True)
    t, f = args.dest / "tables", args.dest / "figures"
    t.mkdir()
    f.mkdir()
    swap.to_csv(t / "swap_neural_by_run.csv", index=False)
    swap_c.to_csv(t / "swap_neural_by_condition.csv", index=False)
    gap.to_csv(t / "swap_gappypod_a38.csv", index=False)
    fam = swap.groupby(["family", "p_conditioning", "role"])[["aligned_macro_K", "swapped_macro_K", "delta_swap_minus_aligned_K", "donor_reference_macro_K"]].agg(["mean", "min", "max"])
    fam.columns = ["_".join(c) for c in fam.columns]
    fam.reset_index().to_csv(t / "swap_neural_family_p_seed_dispersion.csv", index=False)
    base.to_csv(t / "baselines_c149_macro.csv", index=False)
    hist = pd.read_csv(args.historical_summary)
    hist = hist[(hist.K == 0) & np.isclose(hist.site_fraction, 0.05) & hist.method.isin(["GlobalTrainMean", "ParamConditionedMean", "GappyPOD_free"])]
    hist = hist[["method", "p_conditioning", "role", "site_fraction", "rmse_K", "abs_bias_K", "centered_rmse_K"]].assign(fit_pool="A38_historical_760")
    hist.to_csv(t / "baselines_a38_historical_5pct.csv", index=False)
    cmp = base[np.isclose(base.site_fraction, 0.05)].merge(hist, on=["method", "p_conditioning", "role", "site_fraction"], suffixes=("_c149", "_a38"))
    cmp["delta_c149_minus_a38_rmse_K"] = cmp["rmse_K_c149"] - cmp["rmse_K_a38"]
    cmp.to_csv(t / "baselines_pool_effect_c149_vs_a38_5pct.csv", index=False)
    glob = pd.read_csv(args.package_tables / "global_by_setting.csv")
    glob = glob[glob.checkpoint_kind == "best"]
    neural = glob.groupby(["family", "p_conditioning", "role", "coverage"])[["rmse_K", "centered_rmse_K", "abs_bias_K"]].agg(["mean", "min", "max"])
    neural.columns = ["_".join(c) for c in neural.columns]
    neural.reset_index().to_csv(t / "neural_best_family_p_summary.csv", index=False)
    diag.to_csv(t / "batch_diagnostic_summary.csv", index=False)
    (args.diag_root / "batch_diag_rows.csv").exists() and pd.read_csv(args.diag_root / "batch_diag_rows.csv").to_csv(t / "batch_diagnostic_rows.csv.gz", index=False, compression="gzip")
    (t / "c149_baseline_fit_provenance.json").write_text(json.dumps({**fitprov, "control_a38_subset": ctl}, indent=2) + "\n")
    # ---------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, role in zip(axes, ROLES):
        s = swap[swap.role == role].reset_index(drop=True)
        for i, r in s.iterrows():
            c = COLOR[r["family"]]
            ax.plot([i, i], [r["aligned_macro_K"], r["swapped_macro_K"]], color=c, lw=1, alpha=0.6)
            ax.scatter(i, r["aligned_macro_K"], color=c, marker="o", s=26)
            ax.scatter(i, r["swapped_macro_K"], color=c, marker="s", s=26)
        ax.axhline(s["donor_reference_macro_K"].iloc[0], color="#555555", ls="--", lw=1, label="RMSE(T_donor, T_recipient)")
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([f"{_cfg(r['family'], r['p_conditioning'])} s{r['model_seed']}" for _, r in s.iterrows()], rotation=70, ha="right", fontsize=7)
        ax.set_title(ROLE_LABEL[role])
        ax.set_ylabel("macro RMSE (K), best, 5%, replica 0")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Snapshot swap: aligned (circle) vs donor-velocity swapped (square); dashed = donor temperature as prediction")
    fig.savefig(f / "swap_aligned_vs_swapped.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    v = swap_c[swap_c.role == "val_eval"]
    conds = sorted(v.sim_id.unique())
    cfgs = [("sparse_unet", True), ("sparse_unet", False), ("adapted_glu", True), ("adapted_glu", False)]
    for ci, c in enumerate(conds):
        for i, (fa, p) in enumerate(cfgs):
            h = v[(v.sim_id == c) & (v.family == fa) & (v.p_conditioning == p)]
            x = ci * 5 + i
            ax.scatter([x] * len(h), h["delta_swap_minus_aligned_K"], color=COLOR[fa], facecolors="none" if not p else COLOR[fa], s=30)
    ax.set_xticks([ci * 5 + 1.5 for ci in range(len(conds))])
    ax.set_xticklabels(conds)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_ylabel("swapped - aligned RMSE (K), one point per seed")
    ax.set_title("VAL123 by condition: every condition and seed shown")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLOR[fa], mfc="none" if not p else COLOR[fa], label=_cfg(fa, p)) for fa, p in cfgs], frameon=False, ncol=4)
    fig.savefig(f / "swap_by_condition_val.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, role in zip(axes, ("train_fit", "seen_train_eval", "val_eval")):
        for (run, fa, p), gg in glob[glob.role == role].groupby(["run_id", "family", "p_conditioning"]):
            gg = gg.sort_values("coverage")
            ax.plot(gg["coverage"] * 100, gg["rmse_K"], "-" if p else "--", color=COLOR[fa], lw=0.8, alpha=0.45)
        for (m, pp), gg in base[base.role == role].groupby(["method", "p_conditioning"]):
            gg = gg.sort_values("site_fraction")
            ax.plot(gg["site_fraction"] * 100, gg["rmse_K"], ":" if not pp else "-.", color="#222222" if m == "GappyPOD_free" else "#8a8a8a", lw=2,
                    marker="x", label=f"{m} {'p_on' if pp else 'p_off'} (C149 fit)")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 20])
        ax.set_xticklabels(["1%", "5%", "20%"])
        ax.set_title({"train_fit": "reference760", "seen_train_eval": "seen220", "val_eval": "VAL123"}[role])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("macro RMSE (K)")
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Neural best (blue U-Net, orange GLU, thin lines, 12 runs) vs C149-fitted baselines (thick)")
    fig.savefig(f / "neural_vs_baselines_c149_coverage.png", dpi=140)
    plt.close(fig)
    manifest = {"n_swap_runs": int(swap.run_id.nunique()), "baselines_control_ok": bool(ctl.get("ok")),
                "tables": sorted(p.name for p in t.iterdir()), "figures": sorted(p.name for p in f.iterdir())}
    (args.dest / "residuals_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
