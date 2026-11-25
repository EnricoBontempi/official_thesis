#!/usr/bin/env python3
"""Aggregate the 30 specialist evaluations against the original clean baselines (CPU).

Matrix: TRAIN regime x EVAL scenario, per family, role, checkpoint kind (best, last), seeds visible.
Refuses incomplete matrices, duplicate keys, non-comparable inputs and missing gates; the reason goes to BLOCK_REPORT.md.
No significance tests and no thresholds.
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
from combustion_recon.data.nonideal_observations import file_sha256  # noqa: E402
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402

REGIMES = ["N1", "N2", "G1", "G1-eq", "N1G1"]
CELLS = list(ne.CELL_ORDER)
FAM = {"sparse_unet": "U-Net", "adapted_glu": "GLU"}
ROLE = {"seen_train_eval": "seen220", "val_eval": "VAL123"}
BASELINES = ROOT / "configs/nonideal/specialist_baselines_v1.json"


class Refuse(Exception):
    pass


def load_specialists(freeze: dict, eval_root: Path) -> pd.DataFrame:
    parts, problems = [], []
    for rid, fr in freeze["runs"].items():
        if not fr.get("valid"):
            problems.append(f"{rid}: invalid training ({'; '.join(fr.get('problems', []))})")
            continue
        for kind in ("best", "last"):
            d = eval_root / rid / kind
            src = d
            if kind == "last" and fr.get("last_same_as_best"):
                if not (d / "SAME_AS_BEST.json").is_file():
                    problems.append(f"{rid}/last: SAME_AS_BEST marker missing")
                    continue
                src = eval_root / rid / "best"
            if not (src / "COMPLETE.json").is_file():
                problems.append(f"{rid}/{kind}: no COMPLETE.json")
                continue
            gate = json.loads((src / "gate.json").read_text())
            if not gate.get("all_pass"):
                problems.append(f"{rid}/{kind}: gate not passed")
                continue
            df = pd.concat([pd.read_csv(p) for p in sorted((src / "per_frame").rglob("rep*.csv"))], ignore_index=True)
            try:
                ne.check_run_frame_table(df, family=fr["family"], expect_diag=False)
            except ne.NonidealEvalError as exc:
                problems.append(f"{rid}/{kind}: {exc}")
                continue
            if df["spec_sha256"].nunique() != 1 or df["spec_sha256"].iloc[0] != freeze["degradation_spec_sha256"]:
                problems.append(f"{rid}/{kind}: spec hash differs")
                continue
            df = df.copy()
            df["checkpoint_kind"] = kind
            df["train_regime"] = fr["regime"]
            parts.append(df)
    if problems:
        raise Refuse("\n".join(problems))
    allf = pd.concat(parts, ignore_index=True)
    if len(allf) != 30 * 2 * ne.NOMINAL_PER_MODEL:
        raise Refuse(f"{len(allf)} rows != {30 * 2 * ne.NOMINAL_PER_MODEL}")
    if allf.duplicated(ne.KEY_COLS + ["checkpoint_kind"]).any():
        raise Refuse("duplicate keys")
    return allf


def check_fields(freeze: dict, eval_root: Path) -> list[dict]:
    """Figure fields of the 30 best checkpoints: present, hash-correct, 6 cells x 3 VAL conditions, RMSE consistent with the CSVs."""
    idx, problems = [], []
    for rid, fr in freeze["runs"].items():
        d = eval_root / rid / "best"
        mp = d / "fields_manifest.json"
        if not mp.is_file():
            problems.append(f"{rid}: fields_manifest.json missing")
            continue
        m = json.loads(mp.read_text())
        npz = d / m["file"]
        if not npz.is_file() or hashlib.sha256(npz.read_bytes()).hexdigest() != m["file_sha256"]:
            problems.append(f"{rid}: fields file missing or hash differs")
            continue
        ents = m["entries"]
        if len(ents) != 18 or len({e["cell"] for e in ents}) != 6 or len({e["condition"] for e in ents}) != 3:
            problems.append(f"{rid}: fields entries {len(ents)} (expected 6 cells x 3 VAL conditions)")
            continue
        if any(abs(e["rmse_K_from_fields"] - e["rmse_K_per_frame_csv"]) > 5e-3 for e in ents):
            problems.append(f"{rid}: field RMSE disagrees with the per-frame CSV")
            continue
        if m["identity"]["checkpoint_weights_sha256"] != fr["best"]["weights_sha256"] or m["identity"]["run_id"] != rid:
            problems.append(f"{rid}: fields identity differs from the freeze")
            continue
        idx.append({"run_id": rid, "family": fr["family"], "regime": fr["regime"], "seed": fr["seed"], "fields_npz": str(npz),
                    "fields_manifest": str(mp), "fields_sha256": m["file_sha256"], "n_entries": len(ents)})
    if problems:
        raise Refuse("\n".join(problems))
    return idx


def load_baselines() -> tuple[pd.DataFrame, dict]:
    man = json.loads(BASELINES.read_text())
    p = Path(man["block2_per_frame_csv"])
    if not p.is_file():
        raise Refuse(f"baseline per-frame CSV not found: {p}")
    if file_sha256(p) != man["block2_per_frame_csv_sha256"]:
        raise Refuse("baseline per-frame CSV hash differs from the manifest")
    f = pd.read_csv(p)
    f = f[(f["cell"] != ne.DIAG_CELL) & f["run_id"].isin(man["runs"])]
    if len(f) != 6 * ne.NOMINAL_PER_MODEL:
        raise Refuse(f"baseline rows {len(f)} != {6 * ne.NOMINAL_PER_MODEL}")
    return f, man


def build(spec_frames: pd.DataFrame, base_frames: pd.DataFrame, freeze: dict) -> dict[str, pd.DataFrame]:
    sp = spec_frames
    keys = ["run_id", "family", "variant", "model_seed", "checkpoint_kind", "train_regime", "cell", "role", "replica"]
    pc = sp.groupby(keys + ["condition"], dropna=False)[["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K"]].mean().reset_index()
    macro = pc.groupby(keys, dropna=False)[["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K"]].mean().reset_index()
    bpc = base_frames.groupby(["run_id", "family", "model_seed", "cell", "role", "replica", "condition"])[["rmse_K", "mae_K", "bias_K"]].mean().reset_index()
    bmac = bpc.groupby(["run_id", "family", "model_seed", "cell", "role", "replica"])[["rmse_K", "mae_K", "bias_K"]].mean().reset_index()
    bmac = bmac.rename(columns={"rmse_K": "base_rmse_K", "mae_K": "base_mae_K", "bias_K": "base_bias_K", "run_id": "baseline_run_id"})
    m = macro.merge(bmac, on=["family", "model_seed", "cell", "role", "replica"], how="left", validate="many_to_one")
    if m["base_rmse_K"].isna().any():
        raise Refuse("a specialist row has no baseline")
    r0b = bmac[bmac["cell"] == "R0"].rename(columns={"base_rmse_K": "base_R0_rmse_K"})[["family", "model_seed", "role", "replica", "base_R0_rmse_K"]]
    r0s = macro[macro["cell"] == "R0"].rename(columns={"rmse_K": "spec_R0_rmse_K"})[["run_id", "checkpoint_kind", "role", "replica", "spec_R0_rmse_K"]]
    m = m.merge(r0b, on=["family", "model_seed", "role", "replica"]).merge(r0s, on=["run_id", "checkpoint_kind", "role", "replica"])
    # mean over mask replicas (descriptive; replicas are not independent trainings)
    rk = ["run_id", "family", "model_seed", "checkpoint_kind", "train_regime", "cell", "role"]
    num = ["rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K", "base_rmse_K", "base_R0_rmse_K", "spec_R0_rmse_K"]
    pm = m.groupby(rk)[num].mean().reset_index()
    pm["gain_vs_baseline_K"] = pm["base_rmse_K"] - pm["rmse_K"]            # > 0: specialist better than the clean-trained original
    pm["delta_vs_baseline_K"] = pm["rmse_K"] - pm["base_rmse_K"]
    pm["baseline_deterioration_K"] = pm["base_rmse_K"] - pm["base_R0_rmse_K"]
    pm["specialist_deterioration_K"] = pm["rmse_K"] - pm["spec_R0_rmse_K"]
    pm["clean_cost_K"] = pm["spec_R0_rmse_K"] - pm["base_R0_rmse_K"]       # specialist minus original on clean input
    nz = pm["baseline_deterioration_K"].abs() > 1e-9
    pm["fraction_of_baseline_deterioration_recovered"] = np.where(nz & (pm["cell"] != "R0"), pm["gain_vs_baseline_K"] / pm["baseline_deterioration_K"].where(nz), np.nan)
    vals = ["rmse_K", "gain_vs_baseline_K", "delta_vs_baseline_K", "specialist_deterioration_K", "baseline_deterioration_K", "clean_cost_K"]
    g = pm.groupby(["family", "checkpoint_kind", "train_regime", "role", "cell"])[vals]
    summ = pd.concat([getattr(g, f)().add_suffix(f"__{f}") for f in ("mean", "min", "max")], axis=1).reset_index()
    summ["n_seeds"] = pm.groupby(["family", "checkpoint_kind", "train_regime", "role", "cell"])["model_seed"].nunique().values
    own = pm[pm["cell"] == pm["train_regime"]]
    train = pd.DataFrame([{k: v for k, v in fr.items() if k in ("run_id", "family", "regime", "seed", "state", "epochs_completed", "best_val_epoch",
                                                                "best_val_metric", "specialist_val_at_best", "n_evals", "clip_events", "last_same_as_best")}
                          | {"attempts": fr["ledger"]["attempts"], "extension": fr["ledger"]["extension"], "signals": fr["ledger"]["signals"],
                             "best_epoch_file": fr["best"]["epoch"], "last_epoch_file": fr["last"]["epoch"]} for fr in freeze["runs"].values()])
    return {"per_condition": pc, "macro_by_replica": m, "per_run_scenario": pm, "seed_summary": summ, "diagonal_own_scenario": own, "training_diagnostics": train}


def heatmaps(summ: pd.DataFrame, dest: Path, kind: str = "best") -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    allv = summ[summ["checkpoint_kind"] == kind]["gain_vs_baseline_K__mean"]
    lim = float(np.nanmax(np.abs(allv))) or 1.0
    for i, fam in enumerate(("sparse_unet", "adapted_glu")):
        for j, role in enumerate(ne.EVAL_ROLES):
            s = summ[(summ.family == fam) & (summ.role == role) & (summ.checkpoint_kind == kind)]
            M = np.full((len(REGIMES), len(CELLS)), np.nan)
            for a, r in enumerate(REGIMES):
                for b, c in enumerate(CELLS):
                    q = s[(s.train_regime == r) & (s.cell == c)]
                    if len(q):
                        M[a, b] = q["gain_vs_baseline_K__mean"].iloc[0]
            ax = axes[i, j]
            im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(CELLS, fontsize=8)
            ax.set_yticks(range(len(REGIMES))); ax.set_yticklabels(REGIMES, fontsize=8)
            ax.set_title(f"{FAM[fam]} {ROLE[role]} ({kind})", fontsize=9)
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    if np.isfinite(M[a, b]):
                        ax.text(b, a, f"{M[a, b]:+.1f}", ha="center", va="center", fontsize=7)
            ax.set_xlabel("EVAL scenario"); ax.set_ylabel("TRAIN regime")
    fig.colorbar(im, ax=axes, shrink=0.8, label="gain = RMSE_original - RMSE_specialist (K, mean of 3 seeds; > 0 specialist better)")
    p = dest / "figures" / f"fig1_gain_matrix_{kind}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140); plt.close(fig)
    return f"figures/{p.name}"


def own_and_clean(pm: pd.DataFrame, dest: Path) -> list[str]:
    out = []
    pmb = pm[pm.checkpoint_kind == "best"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    for i, fam in enumerate(("sparse_unet", "adapted_glu")):
        for j, role in enumerate(ne.EVAL_ROLES):
            ax = axes[i, j]
            for k, r in enumerate(REGIMES):
                s = pmb[(pmb.family == fam) & (pmb.role == role) & (pmb.train_regime == r) & (pmb.cell == r)]
                ax.scatter([k - 0.12] * len(s), s["base_rmse_K"], color="#888", marker="o", s=22, label="original (clean)" if k == 0 else None)
                ax.scatter([k + 0.12] * len(s), s["rmse_K"], color="#c0392b", marker="s", s=22, label="specialist" if k == 0 else None)
                for _, q in s.iterrows():
                    ax.plot([k - 0.12, k + 0.12], [q["base_rmse_K"], q["rmse_K"]], color="#bbb", lw=.6)
            ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(REGIMES)
            ax.set_title(f"{FAM[fam]} {ROLE[role]}: own scenario", fontsize=9); ax.set_ylabel("macro RMSE (K)")
    axes[0, 0].legend(fontsize=8)
    p = dest / "figures" / "fig2_own_scenario.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(f"figures/{p.name}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True, sharey=True)
    for ax, role in zip(axes, ne.EVAL_ROLES):
        for fam, col in (("sparse_unet", "#1b6ca8"), ("adapted_glu", "#c0392b")):
            for k, r in enumerate(REGIMES):
                s = pmb[(pmb.family == fam) & (pmb.role == role) & (pmb.train_regime == r) & (pmb.cell == "R0")]
                ax.scatter([k + (0.1 if fam == "adapted_glu" else -0.1)] * len(s), s["clean_cost_K"], color=col, s=22, label=FAM[fam] if k == 0 else None)
        ax.axhline(0, color="gray", lw=.6); ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(REGIMES)
        ax.set_title(f"{ROLE[role]}: clean-input cost (specialist - original, K)", fontsize=9)
    axes[0].legend(fontsize=8)
    p = dest / "figures" / "fig3_clean_cost.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(f"figures/{p.name}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args()
    freeze = json.loads(a.freeze.read_text())
    block = a.dest.parent / "BLOCK_REPORT.md"
    try:
        if freeze["n_valid"] != freeze["n_runs"] or freeze["problems"]:
            raise Refuse("training validation problems:\n" + "\n".join(freeze["problems"]))
        sp = load_specialists(freeze, a.eval_root)
        fields_idx = check_fields(freeze, a.eval_root)
        base, bman = load_baselines()
        t = build(sp, base, freeze)
    except Refuse as exc:
        block.write_text("# BLOCK_REPORT — specialist aggregation refused\n\nOutputs are kept as diagnostic. No retry, no new training.\n\n" + str(exc) + "\n")
        print("REFUSED:\n" + str(exc))
        return 3
    a.dest.mkdir(parents=True, exist_ok=False)
    (a.dest / "tables").mkdir()
    for k, v in t.items():
        v.to_csv(a.dest / "tables" / f"{k}.csv", index=False)
    (a.dest / "fields_index.json").write_text(json.dumps({
        "format": "nonideal_specialist_fields_index_v1", "n_runs": len(fields_idx), "eval_root": str(a.eval_root),
        "per_frame_csvs": "<eval_root>/<run_id>/best/per_frame/<cell>/<role>/rep<k>.csv (best); last: metrics only (SAME_AS_BEST.json when identical)",
        "freeze": str(a.freeze), "tables": str(a.dest / "tables"), "runs": fields_idx}, indent=1) + "\n")
    figs = [heatmaps(t["seed_summary"], a.dest, "best"), heatmaps(t["seed_summary"], a.dest, "last")] + own_and_clean(t["per_run_scenario"], a.dest)
    s = t["seed_summary"]
    lines = ["# Specialist retraining — recovery, clean cost and transfer", "",
             "Synthetic regimes (not a PIV validation). Selection = ORIGINAL clean VAL macro (first absolute minimum); the specialist VAL metric is only recorded. "
             "VAL123 was used for selection. No tests, no thresholds. Baselines = original clean models of the same family/seed (not retrained).", "",
             f"Rows: {len(sp)} specialist frame evaluations (30 runs x 2 checkpoints x {ne.NOMINAL_PER_MODEL}); baselines {len(base)}.", "",
             "## Own scenario, best checkpoint: gain = RMSE_original - RMSE_specialist (K), mean of 3 seeds [min, max]", ""]
    for role in ne.EVAL_ROLES:
        lines += [f"### {ROLE[role]}", "", "| family | " + " | ".join(REGIMES) + " |", "|---|" + "---|" * len(REGIMES)]
        for fam in ("sparse_unet", "adapted_glu"):
            row = []
            for r in REGIMES:
                q = s[(s.family == fam) & (s.checkpoint_kind == "best") & (s.role == role) & (s.train_regime == r) & (s.cell == r)].iloc[0]
                row.append(f"{q['gain_vs_baseline_K__mean']:+.2f} [{q['gain_vs_baseline_K__min']:+.2f}, {q['gain_vs_baseline_K__max']:+.2f}]")
            lines.append(f"| {FAM[fam]} | " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Clean-input cost (specialist - original RMSE on R0, K), best", "", "| family | role | " + " | ".join(REGIMES) + " |", "|---|---|" + "---|" * len(REGIMES)]
    for fam in ("sparse_unet", "adapted_glu"):
        for role in ne.EVAL_ROLES:
            row = []
            for r in REGIMES:
                q = s[(s.family == fam) & (s.checkpoint_kind == "best") & (s.role == role) & (s.train_regime == r) & (s.cell == "R0")].iloc[0]
                row.append(f"{q['clean_cost_K__mean']:+.2f} [{q['clean_cost_K__min']:+.2f}, {q['clean_cost_K__max']:+.2f}]")
            lines.append(f"| {FAM[fam]} | {ROLE[role]} | " + " | ".join(row) + " |")
    lines += ["", "## Figures", ""] + [f"- `{f}`" for f in figs] + ["", "## Tables", ""] + [f"- `tables/{k}.csv`" for k in t] + [""]
    (a.dest / "INDEX.md").write_text("\n".join(lines) + "\n")
    man = {p.relative_to(a.dest).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.dest.rglob("*")) if p.is_file()}
    (a.dest / "package_manifest.json").write_text(json.dumps({"freeze_sha256": file_sha256(a.freeze), "baselines": bman["block2_per_frame_csv_sha256"], "files": man}, indent=2) + "\n")
    (a.dest / "COMPLETE.json").write_text(json.dumps({"ok": True, "n_files": len(man)}) + "\n")
    (a.dest / "FINAL_SUMMARY.md").write_text("\n".join(lines[:4] + ["", f"Package: {a.dest}", "Status: COMPLETE (30/30 runs valid, gates passed, matrices complete)."]) + "\n")
    print("AGGREGATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
