#!/usr/bin/env python3
"""Astra A3 figures: three VAL conditions, frame 1000, mask replica 0. NO inference.

Fields come from the persisted ``val_frame1000_rep0_fields.npz`` (free base prediction + truth,
kelvin). Oracle / AO9 panels are free + the scalar offset stored in the per-frame CSV
(delta_oracle_K / delta_ao9_K; no clipping, same as the evaluator). Every panel is cross-checked
against the CSV RMSE (assert). Only the four neural models have persisted fields; baseline fields
were not saved (would need re-running them) and are not drawn.

Common scales: temperature panels share one [vmin, vmax] across ALL panels of ALL figures;
signed-error panels share one symmetric +-limit across all panels of all figures (both computed
from the data, rounded outward to 50 K; no per-panel autoscale).
Orientation: array indexing, row 0 at top (site index = row*128 + col).
Declared thresholds (frozen config): hot >= 2100 K, peak >= 2200 K, jet < 1800 K.
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

RUNS = [("s5_sparse_unet_pon_seed0", "U-Net p_on"), ("s5_sparse_unet_poff_seed0", "U-Net p_off"),
        ("s5_adapted_glu_pon_seed0", "GLU p_on"), ("s5_adapted_glu_poff_seed0", "GLU p_off")]
TREAT = [("free", "free"), ("oracle", "oracle offset"), ("ao9", "AO9 offset (9 ideal T)")]
HOT, PEAK, JET = 2100.0, 2200.0, 1800.0
FRAME = 1000


def up50(x: float) -> float:
    return float(np.ceil(x / 50.0) * 50.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra-root", required=True)
    ap.add_argument("--out", required=True, help="package dir (figures/ written inside; tables/ must exist)")
    a = ap.parse_args()
    astra, out = Path(a.astra_root), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    data, csv = {}, {}
    for run, _ in RUNS:
        z = np.load(astra / run / "val_frame1000_rep0_fields.npz")
        data[run] = {k: np.asarray(z[k], np.float64) for k in z.files}
        c = pd.read_csv(astra / run / "per_frame_val_eval.csv.gz")
        csv[run] = c[(c.replica_id == 0) & (c.frame_index == FRAME)]
    sims = sorted({k.split("__")[0] for k in data[RUNS[0][0]] if "__true_K" in k})
    t9 = data[RUNS[0][0]]["t9_sites"]
    for run, _ in RUNS:
        assert np.array_equal(data[run]["t9_sites"], t9)
    rc = np.stack([t9 // 128, t9 % 128], 1)

    # fields per (run, sim, treatment); consistency vs CSV
    fields, rows = {}, []
    for run, lab in RUNS:
        for sim in sims:
            pred, tru = data[run][f"{sim}__pred_K"], data[run][f"{sim}__true_K"]
            assert pred.shape == (128, 128) and np.isfinite(pred).all() and np.isfinite(tru).all()
            cs = csv[run][csv[run].sim_id == sim].set_index("treatment")
            assert set(cs.index) == {"free", "oracle", "ao9"}, (run, sim, cs.index)
            delta = {"free": 0.0, "oracle": float(cs.loc["free", "delta_oracle_K"]), "ao9": float(cs.loc["free", "delta_ao9_K"])}
            for t, _ in TREAT:
                f = pred + delta[t]
                rmse = float(np.sqrt(np.mean((f - tru) ** 2)))
                ref = float(cs.loc[t, "all_rmse_K"])
                assert abs(rmse - ref) < 5e-3, (run, sim, t, rmse, ref)  # float32 npz vs float64 CSV
                fields[(run, sim, t)] = f
                rows.append({"model": run, "sim_id": sim, "treatment": t, "delta_applied_K": delta[t],
                             "all_rmse_K_from_npz": rmse, "all_rmse_K_csv": ref, "abs_diff_K": abs(rmse - ref),
                             "frame_index": FRAME, "replica_id": 0})
    pd.DataFrame(rows).to_csv(out / "tables" / "figure_panel_check.csv", index=False, float_format="%.8g")

    allT = np.concatenate([v.ravel() for v in fields.values()] + [data[r][f"{s}__true_K"].ravel() for r, _ in RUNS for s in sims])
    vmin, vmax = np.floor(allT.min() / 50) * 50, up50(allT.max())
    elim = up50(max(np.abs(fields[(r, s, t)] - data[r][f"{s}__true_K"]).max() for r, _ in RUNS for s in sims for t, _ in TREAT))
    scale = {"T_vmin_K": float(vmin), "T_vmax_K": float(vmax), "error_abs_limit_K": float(elim),
             "thresholds_K": {"hot_ge": HOT, "peak_ge": PEAK, "jet_lt": JET}, "frame_index": FRAME, "replica": 0,
             "t9_sites_flat": t9.tolist(), "orientation": "array index, row 0 at top"}
    (out / "tables" / "figure_scales.json").write_text(json.dumps(scale, indent=2) + "\n")

    for sim in sims:
        tru = data[RUNS[0][0]][f"{sim}__true_K"]
        # ---------- temperature fields
        fig, ax = plt.subplots(len(RUNS), 4, figsize=(11.5, 11.2), constrained_layout=True)
        for i, (run, lab) in enumerate(RUNS):
            panels = [("truth", tru)] + [(tl, fields[(run, sim, t)]) for t, tl in TREAT]
            for j, (tl, f) in enumerate(panels):
                x = ax[i, j]
                im = x.imshow(f, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
                x.contour(f, levels=[HOT], colors="white", linewidths=0.7)
                x.scatter(rc[:, 1], rc[:, 0], s=22, facecolors="none", edgecolors="cyan", linewidths=0.9)
                x.set_xticks([]); x.set_yticks([])
                if j == 0:
                    x.set_ylabel(lab, fontsize=10)
                r_ = float(np.sqrt(np.mean((f - tru) ** 2)))
                x.set_title(tl if j == 0 else f"{tl}\nRMSE {r_:.1f} K", fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.5, location="right"); cb.set_label("T [K] (common scale, all panels)")
        fig.suptitle(f"{sim} - VAL frame {FRAME}, mask replica 0 - white contour: T=2100 K (hot); cyan circles: the 9 ideal T sites\n"
                     "exploratory (VAL influenced S5 selection); oracle uses the full true T (diagnostic)", fontsize=9)
        fig.savefig(out / "figures" / f"val_{sim.replace('.h5', '')}_frame{FRAME}_rep0_fields.png", dpi=130)
        plt.close(fig)
        # ---------- signed error (pred - true)
        fig, ax = plt.subplots(len(RUNS), 3, figsize=(8.8, 11.2), constrained_layout=True)
        for i, (run, lab) in enumerate(RUNS):
            for j, (t, tl) in enumerate(TREAT):
                e = fields[(run, sim, t)] - tru
                x = ax[i, j]
                im = x.imshow(e, cmap="RdBu_r", vmin=-elim, vmax=elim, interpolation="nearest")
                x.contour(tru, levels=[HOT], colors="black", linewidths=0.7)
                x.scatter(rc[:, 1], rc[:, 0], s=22, facecolors="none", edgecolors="limegreen", linewidths=0.9)
                x.set_xticks([]); x.set_yticks([])
                if j == 0:
                    x.set_ylabel(lab, fontsize=10)
                x.set_title(f"{tl}\nbias {e.mean():+.1f} K, RMSE {np.sqrt(np.mean(e**2)):.1f} K", fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.5, location="right"); cb.set_label("T_pred - T_true [K] (common symmetric scale)")
        fig.suptitle(f"{sim} - VAL frame {FRAME}, replica 0 - black contour: TRUE T=2100 K; green circles: 9 T sites\n"
                     "bias/RMSE over the full 128x128 field; exploratory", fontsize=9)
        fig.savefig(out / "figures" / f"val_{sim.replace('.h5', '')}_frame{FRAME}_rep0_signed_error.png", dpi=130)
        plt.close(fig)
    print(json.dumps({"scales": scale, "figures": sorted(p.name for p in (out / "figures").glob("*.png"))}, indent=1)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
