#!/usr/bin/env python3
"""Aggregate the verified Block II outputs (CPU): tables, figures, INDEX.md, FINAL_SUMMARY.md.

Refuses incomplete matrices, duplicate keys, non-comparable inputs and missing gates. No significance tests,
no thresholds. Unfavourable results stay in the package.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.nonideal_observations import file_sha256, load_degradation_spec  # noqa: E402
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402

CELLS = list(ne.CELL_ORDER)
COLORS = {("sparse_unet", "p_on"): "#1b6ca8", ("sparse_unet", "p_off"): "#7fb3d5",
          ("adapted_glu", "p_on"): "#c0392b", ("adapted_glu", "p_off"): "#e6a09a"}
MARK = {0: "o", 1: "s", 2: "^"}
ROLE_LABEL = {"seen_train_eval": "seen220", "val_eval": "VAL123"}


def load_frames(eval_root: Path, spec) -> pd.DataFrame:
    parts = []
    for r in spec.raw["models"]["runs"]:
        d = eval_root / r["run_id"]
        if not (d / "COMPLETE.json").exists():
            raise SystemExit(f"{r['run_id']}: not COMPLETE")
        df = pd.concat([pd.read_csv(p) for p in sorted((d / "per_frame").rglob("rep*.csv"))], ignore_index=True)
        ne.check_run_frame_table(df, family=r["family"], expect_diag=(r["family"] == "adapted_glu"))
        parts.append(df)
    allf = pd.concat(parts, ignore_index=True)
    if allf.duplicated(ne.KEY_COLS).any():
        raise SystemExit("duplicate keys across runs")
    nominal = allf[allf["cell"] != ne.DIAG_CELL]
    if len(nominal) != ne.NOMINAL_TOTAL:
        raise SystemExit(f"{len(nominal)} nominal rows != {ne.NOMINAL_TOTAL}")
    return allf


def build_tables(frames: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nominal = frames[frames["cell"] != ne.DIAG_CELL]
    pc = ne.per_condition_table(nominal)
    macro = ne.macro_table(pc)
    mcols = [c for c in macro.columns if c.endswith("_K") or c.endswith("iou")]
    keys_m = ["run_id", "family", "variant", "model_seed", "role", "replica"]
    d_macro = ne.delta_vs_r0(macro, keys_m, [c for c in mcols if c in ("rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "jet_rmse_K", "high_rmse_K", "peak_rmse_K", "high_iou")])
    keys_c = ["run_id", "family", "variant", "model_seed", "role", "replica", "condition"]
    d_cond = ne.delta_vs_r0(pc, keys_c, ["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K"])
    # mean over mask replicas (descriptive; replicas are not independent training runs)
    per_model = d_macro.groupby(["run_id", "family", "variant", "model_seed", "role", "cell"])[
        [c for c in d_macro.columns if c.startswith(("d_", "rel_"))]].mean().reset_index()
    macro_rep = macro.groupby(["run_id", "family", "variant", "model_seed", "role", "cell"])[["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K"]].mean().reset_index()
    summ = ne.seed_summary(per_model, ["d_rmse_K", "rel_rmse_K", "d_mae_K", "d_bias_K"], ["family", "variant", "role", "cell"])
    abs_summ = ne.seed_summary(macro_rep, ["rmse_K", "mae_K", "bias_K"], ["family", "variant", "role", "cell"])
    contrasts = ne.structure_contrasts(macro)
    pon_poff = ne.pon_poff_contrast(per_model, "d_rmse_K")
    fam = per_model.groupby(["variant", "model_seed", "role", "cell"]).apply(
        lambda g: pd.Series({"d_rmse_unet": g.loc[g.family == "sparse_unet", "d_rmse_K"].mean(),
                             "d_rmse_glu": g.loc[g.family == "adapted_glu", "d_rmse_K"].mean(),
                             "rel_unet": g.loc[g.family == "sparse_unet", "rel_rmse_K"].mean(),
                             "rel_glu": g.loc[g.family == "adapted_glu", "rel_rmse_K"].mean()}), include_groups=False).reset_index()
    regional = macro[macro["cell"].isin(["G1", "N1G1"])][keys_m[:4] + ["cell", "role", "replica", "g1in_rmse_K", "g1out_rmse_K", "g1in_bias_K", "g1out_bias_K", "g1in_n", "g1out_n"]]
    removed = nominal.groupby(["cell", "role", "replica"])[["n_original", "n_surviving", "n_removed", "removed_site_fraction", "area_fraction_effective", "area_mm2"]].agg(["mean", "min", "max"])
    removed.columns = ["__".join(c) for c in removed.columns]
    diag = frames[(frames["cell"].isin(["R0", ne.DIAG_CELL])) & (frames["family"] == "adapted_glu")]
    glu_batch = diag.pivot_table(index=["run_id", "role", "replica", "condition", "frame_index"], columns="cell", values="rmse_K").reset_index()
    glu_batch["delta_K"] = glu_batch["R0"] - glu_batch[ne.DIAG_CELL]
    glu_batch_s = glu_batch.groupby(["run_id", "role", "replica"])["delta_K"].agg(["mean", lambda s: s.abs().max()]).reset_index()
    glu_batch_s.columns = ["run_id", "role", "replica", "mean_delta_K", "max_abs_frame_delta_K"]
    return {"per_condition": pc, "macro_by_replica": macro, "delta_vs_R0_by_replica": d_macro, "delta_vs_R0_by_condition": d_cond,
            "delta_vs_R0_per_model": per_model, "absolute_per_model": macro_rep, "delta_seed_summary": summ,
            "absolute_seed_summary": abs_summ, "structure_contrasts": contrasts, "pon_minus_poff": pon_poff,
            "family_contrast_by_seed_index": fam, "regional_g1": regional, "sites_and_area": removed.reset_index(),
            "glu_batching_diagnostic": glu_batch_s}


def _panel(ax, df, col, role, cells, title):
    for (fam, var), c in COLORS.items():
        sub = df[(df.family == fam) & (df.variant == var) & (df.role == role) & (df.cell.isin(cells))]
        for seed, m in MARK.items():
            s = sub[sub.model_seed == seed]
            ax.scatter([cells.index(x) + (0.18 if var == "p_off" else -0.18) + (0.04 if fam == "adapted_glu" else 0) for x in s.cell], s[col], color=c, marker=m, s=22, alpha=.9)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, fontsize=8); ax.set_title(title, fontsize=9)


def figures(tables, frames, eval_root: Path, dest: Path, spec) -> list[str]:
    fdir = dest / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    out = []
    ab, pm = tables["absolute_per_model"], tables["delta_vs_R0_per_model"]
    for name, df, col, cells, ylabel in (("fig1_macro_rmse_by_cell", ab, "rmse_K", CELLS, "macro RMSE (K)"),
                                         ("fig2_deterioration_vs_R0", pm, "d_rmse_K", CELLS[1:], "RMSE - R0 (K), same model")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, role in zip(axes, ne.EVAL_ROLES):
            _panel(ax, df, col, role, cells, ROLE_LABEL[role]); ax.set_ylabel(ylabel)
        h = [plt.Line2D([], [], color=c, marker="o", ls="", label=f"{'U-Net' if f == 'sparse_unet' else 'GLU'} {v}") for (f, v), c in COLORS.items()]
        h += [plt.Line2D([], [], color="k", marker=m, ls="", label=f"seed {s}") for s, m in MARK.items()]
        axes[0].legend(handles=h, fontsize=7, ncol=2)
        fig.savefig(fdir / f"{name}.png", dpi=140); plt.close(fig); out.append(f"figures/{name}.png")
    dc = tables["delta_vs_R0_by_condition"]; dc = dc[dc.role == "val_eval"].groupby(["family", "variant", "model_seed", "cell", "condition"])["d_rmse_K"].mean().reset_index()
    conds = sorted(dc.condition.unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.8), constrained_layout=True, sharey=True)
    for ax, c in zip(np.atleast_1d(axes), conds):
        _panel(ax, dc[dc.condition == c].assign(role="val_eval"), "d_rmse_K", "val_eval", CELLS[1:], c.replace(".h5", ""))
    np.atleast_1d(axes)[0].set_ylabel("RMSE - R0 (K), VAL condition")
    fig.savefig(fdir / "fig3_val_by_condition.png", dpi=140); plt.close(fig); out.append("figures/fig3_val_by_condition.png")
    sc = tables["structure_contrasts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    cols = ["N2_minus_N1", "G1_minus_G1eq", "N1G1_minus_G1", "N1G1_minus_N1", "interaction_N1G1_N1_G1_R0"]
    for ax, role in zip(axes, ne.EVAL_ROLES):
        for i, c in enumerate(cols):
            for (f, v), col in COLORS.items():
                s = sc[(sc.family == f) & (sc.variant == v) & (sc.role == role)]
                ax.scatter([i + (0.18 if v == "p_off" else -0.18)] * len(s), s[c], color=col, s=20, marker="o")
        ax.axhline(0, color="gray", lw=.6); ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=6)
        ax.set_title(ROLE_LABEL[role] + ": level vs structure (K)", fontsize=9)
    fig.savefig(fdir / "fig4_level_vs_structure.png", dpi=140); plt.close(fig); out.append("figures/fig4_level_vs_structure.png")
    # fields: seed-0 model of each family x p, fixed frames (spec rule), VAL conditions
    for r in spec.raw["models"]["runs"]:
        if int(r["model_seed"]) != 0:
            continue
        p = eval_root / r["run_id"] / "fields_figure_frames.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        keys = sorted({tuple(k.split("|")[2:4]) for k in z.files if k.startswith("R0|val_eval")})
        if not keys:
            continue
        cells = CELLS
        fig, axes = plt.subplots(len(keys), len(cells) + 1, figsize=(2.3 * (len(cells) + 1), 2.3 * len(keys)), constrained_layout=True, squeeze=False)
        for i, (sim, fr) in enumerate(keys):
            true = z[f"R0|val_eval|{sim}|{fr}|true_T_K"]
            axes[i, 0].imshow(true, origin="upper", cmap="inferno"); axes[i, 0].set_title(f"true T\n{sim.replace('.h5','')} f{fr}", fontsize=6)
            for j, c in enumerate(cells):
                err = np.abs(z[f"{c}|val_eval|{sim}|{fr}|pred_T_K"] - true)
                ax = axes[i, j + 1]; ax.imshow(err, origin="upper", cmap="magma", vmin=0, vmax=400)
                rem = z[f"{c}|val_eval|{sim}|{fr}|removed_sites"]
                if rem.size: ax.scatter(rem % 128, rem // 128, s=1.5, c="cyan", marker=".")
                reg = z[f"{c}|val_eval|{sim}|{fr}|region_mask"]
                if reg.size: ax.contour(reg.astype(float), levels=[0.5], colors="w", linewidths=.5)
                ax.set_title(f"|err| {c}", fontsize=6)
        for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{r['run_id']} (fixed frames; |err| 0-400 K; cyan = removed sites, white = G1 region)", fontsize=8)
        fig.savefig(fdir / f"fig5_fields_{r['run_id']}.png", dpi=130); plt.close(fig); out.append(f"figures/fig5_fields_{r['run_id']}.png")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--verify-dir", type=Path, required=True)
    a = ap.parse_args()
    if not (a.verify_dir / "VERIFY_OK.json").exists():
        raise SystemExit("VERIFY_OK.json missing: refusing to aggregate")
    spec = load_degradation_spec(a.spec)
    a.dest.mkdir(parents=True, exist_ok=False)
    frames = load_frames(a.eval_root, spec)
    t = build_tables(frames)
    (a.dest / "tables").mkdir()
    for k, v in t.items():
        v.to_csv(a.dest / "tables" / f"{k}.csv", index=False)
    frames.to_csv(a.dest / "tables" / "per_frame_all.csv.gz", index=False)
    figs = figures(t, frames, a.eval_root, a.dest, spec)
    s = t["delta_seed_summary"]
    lines = ["# Block II — frozen models on degraded velocity observations", "",
             "Descriptive scenario study (synthetic degradations, not PIV validation). VAL123 was used for selection/development: not an independent confirmation. No tests, no thresholds.", "",
             f"Rows: {int((frames['cell'] != ne.DIAG_CELL).sum())} nominal frame evaluations (expected {ne.NOMINAL_TOTAL}); {int((frames['cell'] == ne.DIAG_CELL).sum())} technical GLU replay rows counted separately.", "",
             "## Mean deterioration (RMSE_scenario - RMSE_R0, K) — mean over seeds [min, max]", ""]
    for role in ne.EVAL_ROLES:
        lines += [f"### {ROLE_LABEL[role]}", "", "| family | p | " + " | ".join(CELLS[1:]) + " |", "|---|---|" + "---|" * 5]
        for (f, v), _ in COLORS.items():
            row = []
            for c in CELLS[1:]:
                q = s[(s.family == f) & (s.variant == v) & (s.role == role) & (s.cell == c)].iloc[0]
                row.append(f"{q['d_rmse_K__mean']:.2f} [{q['d_rmse_K__min']:.2f}, {q['d_rmse_K__max']:.2f}]")
            lines.append(f"| {f} | {v} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Figures", ""] + [f"- `{f}`" for f in figs] + ["", "## Tables", ""] + [f"- `tables/{k}.csv`" for k in t] + ["- `tables/per_frame_all.csv.gz`", ""]
    (a.dest / "INDEX.md").write_text("\n".join(lines) + "\n")
    manifest = {p.relative_to(a.dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.dest.rglob("*")) if p.is_file()}
    (a.dest / "package_manifest.json").write_text(json.dumps({"spec_sha256": file_sha256(a.spec), "files": manifest}, indent=2) + "\n")
    (a.dest / "COMPLETE.json").write_text(json.dumps({"ok": True, "n_files": len(manifest)}, indent=2) + "\n")
    (a.dest / "FINAL_SUMMARY.md").write_text("\n".join(lines[:4] + ["", f"Package: {a.dest}", "Status: COMPLETE (verify, gates and completeness checks passed)."]) + "\n")
    print("AGGREGATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
