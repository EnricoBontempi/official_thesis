#!/usr/bin/env python3
"""Field panels for the three VAL conditions (frame 1000) from the stored NPZ of the final evaluation, plus a
consistency table (field vs checkpoint vs per-frame CSV), and the corrected centered-vs-bias figure.

No inference. All 12 runs are shown (no seed selection). Common scales are declared in the figure:
temperature = [min, max] of the truth over the three VAL frames (predictions are clipped for display only);
signed error = symmetric, +-(max |error| over ALL panels of the figure set), the same for every figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPTS = ("best", "last", "best_seen_diagnostic")
ORDER = [("sparse_unet", True), ("sparse_unet", False), ("adapted_glu", True), ("adapted_glu", False)]


def _label(e: dict) -> str:
    fam = "U-Net" if e["family"] == "sparse_unet" else "GLU"
    return f"{fam} {'p_on' if e['p_conditioning'] else 'p_off'} s{e['model_seed']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--package-tables", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    runs = sorted(matrix["runs"], key=lambda e: (ORDER.index((e["family"], bool(e["p_conditioning"]))), e["model_seed"]))
    args.dest.mkdir(parents=True)
    # ---- load + consistency
    data, cons = {}, []
    for e in runs:
        for ck in CKPTS:
            f = args.eval_root / e["run_id"] / "fields" / f"val_frame1000_{ck}_cov05_rep0.npz"
            z = np.load(f, allow_pickle=True)
            spec = e["checkpoints"][ck]
            if str(z["checkpoint_kind"]) != ck or str(z["run_id"]) != e["run_id"]:
                raise SystemExit(f"{f}: tags do not match the run/checkpoint")
            sims = [str(s) for s in z["sim_id"]]
            csv = pd.read_csv(args.eval_root / e["run_id"] / "per_frame" / ck / "cov0.05" / "val_eval" / "rep0.csv",
                              usecols=["sim_id", "frame_index", "rmse_K", "weights_sha256", "epoch"])
            if set(csv["weights_sha256"]) != {spec["weights_sha256"]} or set(csv["epoch"]) != {spec["epoch"]}:
                raise SystemExit(f"{f}: CSV identity differs from the matrix")
            for i, s in enumerate(sims):
                err = z["pred_T_K"][i] - z["true_T_K"][i]
                r = float(np.sqrt(np.mean(err ** 2)))
                hit = csv[(csv.sim_id == s) & (csv.frame_index == 1000)]
                cons.append({"run_id": e["run_id"], "checkpoint_kind": ck, "sim_id": s, "epoch": spec["epoch"],
                             "weights_sha256": spec["weights_sha256"][:12], "rmse_from_field_K": r,
                             "rmse_from_csv_K": float(hit["rmse_K"].iloc[0]) if len(hit) == 1 else float("nan"),
                             "delta_K": r - float(hit["rmse_K"].iloc[0]) if len(hit) == 1 else float("nan"),
                             "shared_forward_from": spec.get("reuse_forward_from")})
            data[(e["run_id"], ck)] = (sims, z["pred_T_K"], z["true_T_K"])
    cdf = pd.DataFrame(cons)
    cdf.to_csv(args.dest / "field_consistency.csv", index=False)
    conds = sorted(set(data[(runs[0]["run_id"], "best")][0]))
    truth = {}
    for e in runs:
        sims, _, tr = data[(e["run_id"], "best")]
        for i, s in enumerate(sims):
            truth.setdefault(s, tr[i])
            if not np.array_equal(truth[s], tr[i]):
                raise SystemExit("truth differs between runs for " + s)
    vmin = min(float(t.min()) for t in truth.values())
    vmax = max(float(t.max()) for t in truth.values())
    emax = max(float(np.abs(data[k][1][i] - data[k][2][i]).max()) for k in data for i in range(3))
    emax_true = emax
    emax = 300.0   # declared display scale (clipped); the true maximum is reported in field_summary.json
    for ck in CKPTS:
        fig, axes = plt.subplots(len(runs) + 1, 2 * len(conds), figsize=(2.0 * 2 * len(conds) + 1.2, 1.55 * (len(runs) + 1)),
                                 layout="constrained", squeeze=False)
        for ci, c in enumerate(conds):
            ax = axes[0][2 * ci]
            imT = ax.imshow(truth[c], origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{c.replace('.h5', '')}\ntruth", fontsize=8)
            axes[0][2 * ci + 1].axis("off")
        for ri, e in enumerate(runs, start=1):
            sims, pred, tr = data[(e["run_id"], ck)]
            for ci, c in enumerate(conds):
                i = sims.index(c)
                a1, a2 = axes[ri][2 * ci], axes[ri][2 * ci + 1]
                a1.imshow(np.clip(pred[i], vmin, vmax), origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
                imE = a2.imshow(pred[i] - tr[i], origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax)
                r = float(np.sqrt(np.mean((pred[i] - tr[i]) ** 2)))
                a2.set_title(f"RMSE {r:.0f} K", fontsize=7, pad=1)
                if ci == 0:
                    a1.set_ylabel(_label(e), fontsize=7)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(imT, ax=axes.ravel().tolist(), shrink=0.25, label="T (K)", location="right", pad=0.01)
        fig.colorbar(imE, ax=axes.ravel().tolist(), shrink=0.25, label="signed error pred-truth (K, clipped)", location="right", pad=0.01)
        fig.suptitle(f"VAL frame 1000, 5% coverage, mask replica 0, checkpoint {ck}: all 12 runs\n"
                     f"T scale [{vmin:.0f}, {vmax:.0f}] K (pred clipped), error scale +-{emax:.0f} K (display clip; true max |error| {emax_true:.0f} K), common to every panel", fontsize=9)
        fig.savefig(args.dest / f"fields_val_frame1000_{ck}.png", dpi=110)
        plt.close(fig)
    # ---- corrected centered-vs-bias figure from the package tables
    spec = importlib.util.spec_from_file_location("s7agg40", ROOT / "scripts/40_s7_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    glob = pd.read_csv(args.package_tables / "global_by_setting.csv")
    glob["p_conditioning"] = glob["p_conditioning"].astype(bool)
    mod.fig_error_parts(glob, args.dest / "centered_vs_bias_5pct_best.png")
    summary = {"n_runs": len(runs), "t_scale_K": [vmin, vmax], "error_scale_display_K": emax, "max_abs_error_K": emax_true,
               "max_abs_delta_field_vs_csv_K": float(cdf["delta_K"].abs().max()),
               "by_family_max_abs_delta_K": {f: float(g["delta_K"].abs().max()) for f, g in
                                             cdf.assign(f=cdf["run_id"].str.contains("glu").map({True: "adapted_glu", False: "sparse_unet"})).groupby("f")}}
    (args.dest / "field_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
