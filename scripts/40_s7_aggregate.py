#!/usr/bin/env python3
"""Aggregate the finished S7 evaluation (12 runs) into tables, figures and a reviewer index.

Refuses to write anything unless every run is COMPLETE with a passed 18/18 gate and every
consistency check passes (keys, mask hashes against the verified payload, identities from the
frozen matrix, shared forwards, provenance). Ranks nothing and picks no winner: all contrasts
are descriptive. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.sparse_eval_masks import check_eval_mask_contract  # noqa: E402
from combustion_recon.evaluation import s7_package as pkg  # noqa: E402

CKPTS = ("best", "last", "best_seen_diagnostic")
ROLE_LABEL = {"train_fit": "reference760", "seen_train_eval": "seen220", "val_eval": "VAL123"}
COLOR = {"sparse_unet": "#2a6fb0", "adapted_glu": "#d9731a"}
LABEL = {"sparse_unet": "U-Net", "adapted_glu": "GLU"}


def _cfg(family: str, p: bool) -> str:
    return f"{LABEL[family]} p_{'on' if p else 'off'}"


def _style(p: bool) -> str:
    return "-" if p else "--"


def fig_coverage(glob: pd.DataFrame, ckpt: str, dest: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharey=False)
    for ax, role in zip(axes, pkg.ROLES):
        sub = glob[(glob.checkpoint_kind == ckpt) & (glob.role == role)]
        for (run, fam, p), g in sub.groupby(["run_id", "family", "p_conditioning"]):
            g = g.sort_values("coverage")
            ax.plot(g["coverage"] * 100, g["rmse_K"], _style(p), color=COLOR[fam], marker="o", ms=3, lw=1.2, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 20])
        ax.set_xticklabels(["1%", "5%", "20%"])
        ax.set_title(ROLE_LABEL[role])
        ax.set_xlabel("sensor coverage (eval)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("macro RMSE (K)")
    handles = [plt.Line2D([], [], color=COLOR[f], ls=_style(p), label=_cfg(f, p)) for f in COLOR for p in (True, False)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"S7 final evaluation, checkpoint {ckpt}: one line per run (3 seeds per configuration)")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_points_5pct(glob: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    configs = [(f, p) for f in COLOR for p in (True, False)]
    marks = {"best": "o", "last": "s", "best_seen_diagnostic": "^"}
    for ax, role in zip(axes, pkg.ROLES):
        sub = glob[(glob.coverage == 0.05) & (glob.role == role)]
        for i, (f, p) in enumerate(configs):
            for j, ck in enumerate(CKPTS):
                g = sub[(sub.family == f) & (sub.p_conditioning == p) & (sub.checkpoint_kind == ck)]
                ax.scatter([i + (j - 1) * 0.22] * len(g), g["rmse_K"], marker=marks[ck], color=COLOR[f],
                           facecolors="none" if not p else COLOR[f], s=30)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_cfg(f, p) for f, p in configs], rotation=20)
        ax.set_title(ROLE_LABEL[role])
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("macro RMSE at 5% (K), one point per seed")
    handles = [plt.Line2D([], [], marker=m, ls="", color="k", label=k) for k, m in marks.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_paired(summary: pd.DataFrame, dest: Path, role: str = "val_eval") -> None:
    sub = summary[(summary.metric == "rmse_K") & (summary.role == role) & (summary.coverage == 0.05)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax, kind, title in ((axes[0], "family", "GLU minus U-Net (same p, same seed index)"),
                            (axes[1], "p", "p_on minus p_off (same family, same seed)")):
        s = sub[sub.contrast == kind]
        labels = sorted(s["label"].unique())
        for i, lab in enumerate(labels):
            for j, ck in enumerate(CKPTS):
                row = s[(s.label == lab) & (s.checkpoint_kind == ck)]
                if len(row):
                    ax.scatter(row["mean_delta"], [i + (j - 1) * 0.2], marker={"best": "o", "last": "s", "best_seen_diagnostic": "^"}[ck],
                               color="#333333", s=26)
        ax.axvline(0, color="#999999", lw=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("mean over VAL conditions of the RMSE difference at 5% (K)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Descriptive paired differences (no test, no winner); circle=best, square=last, triangle=best_seen")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_condition(per_condition: pd.DataFrame, dest: Path) -> None:
    sub = per_condition[(per_condition.role == "val_eval") & (per_condition.coverage == 0.05) & (per_condition.checkpoint_kind == "best")]
    conds = sorted(sub["sim_id"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    configs = [(f, p) for f in COLOR for p in (True, False)]
    for ci, cond in enumerate(conds):
        for i, (f, p) in enumerate(configs):
            g = sub[(sub.sim_id == cond) & (sub.family == f) & (sub.p_conditioning == p)]
            x = ci * (len(configs) + 1) + i
            ax.scatter([x] * len(g), g["rmse_K"], color=COLOR[f], facecolors="none" if not p else COLOR[f], s=34)
            ax.hlines(g["rmse_K"].mean(), x - 0.3, x + 0.3, color=COLOR[f], lw=1)
    ax.set_xticks([ci * (len(configs) + 1) + 1.5 for ci in range(len(conds))])
    ax.set_xticklabels(conds)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[f], mfc="none" if not p else COLOR[f], label=_cfg(f, p)) for f, p in configs]
    ax.legend(handles=handles, frameon=False, ncol=4)
    ax.set_ylabel("RMSE at 5% (K), one point per seed; bar = seed mean")
    ax.set_title("VAL123 by condition, best checkpoint, 5% coverage")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_error_parts(glob: pd.DataFrame, dest: Path) -> None:
    sub = glob[(glob.coverage == 0.05) & (glob.checkpoint_kind == "best")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, role in zip(axes, pkg.ROLES):
        g = sub[sub.role == role]
        for (f, p), h in g.groupby(["family", "p_conditioning"]):
            ax.scatter(h["centered_rmse_K"], h["abs_bias_K"], color=COLOR[f], facecolors="none" if not p else COLOR[f], s=40)
        ax.set_xlabel("centered RMSE (K)")
        ax.set_title(ROLE_LABEL[role])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean over frames of per-frame |bias| (K)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[f], mfc="none" if not p else COLOR[f], label=_cfg(f, p)) for f in COLOR for p in (True, False)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig_costs(by_run: pd.DataFrame, eval_cost: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    b = by_run.sort_values("run_id")
    x = range(len(b))
    axes[0].bar(x, b["compute_h"], color="#4c6a92", label="compute")
    axes[0].bar(x, b["queue_wait_h"], bottom=b["compute_h"], color="#c9c9c9", label="queue wait")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(b["run_id"].str.replace("S7-C149-", "").str.replace("O-C149-", "O-"), rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("hours")
    axes[0].set_title("Training (all attempts); wait includes the array throttle")
    axes[0].legend(frameon=False)
    e = eval_cost.sort_values("run_id")
    y = range(len(e))
    axes[1].bar(y, e["compute_s"] / 60, color="#4c6a92", label="compute")
    axes[1].bar(y, e["queue_wait_s"] / 60, bottom=e["compute_s"] / 60, color="#c9c9c9", label="queue wait")
    axes[1].set_xticks(list(y))
    axes[1].set_xticklabels(e["run_id"].str.replace("S7-C149-", "").str.replace("O-C149-", "O-"), rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("minutes")
    axes[1].set_title("Final evaluation")
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json")
    ap.add_argument("--eval-sha", action="append", required=True)
    ap.add_argument("--costs-dir", type=Path, default=None, help="output of 41_s7_costs.py (optional)")
    args = ap.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    msha = pkg.sha256_file(args.matrix)
    sh = matrix["shared_inputs"]
    mask_root = Path(sh["mask_root"])
    check_eval_mask_contract(
        manifest_path=mask_root / "manifest.json", payload_root=mask_root,
        expected_manifest_file_sha256=sh["eval_mask_manifest_file_sha256"],
        expected_aggregate_payload_sha256=sh["eval_mask_aggregate_payload_sha256"],
    )
    payload = pkg.payload_mask_hashes(mask_root)
    table, runs, issues = pkg.load_and_check(args.eval_root, matrix, matrix_sha256=msha,
                                             allowed_eval_shas=args.eval_sha, payload_hashes=payload)
    if issues:
        print("AGGREGATION REFUSED. Issues:")
        for item in issues[:300]:
            print(" -", item)
        print(f"{len(issues)} issue(s); nothing written.")
        return 1
    tables = pkg.build_tables(table)
    glob = tables["global_by_setting"]
    summ = pkg.family_p_summary(glob)
    paired = pkg.paired_contrasts(tables["per_condition"])
    degr = pkg.degradations(glob, tables["per_condition"])
    nans = pkg.nan_report(table)
    tdir, fdir = args.dest / "tables", args.dest / "figures"
    tdir.mkdir(parents=True)
    fdir.mkdir()
    for name, frame in {**tables, **degr}.items():
        frame.to_csv(tdir / f"{name}.csv", index=False)
    summ.to_csv(tdir / "family_p_seed_dispersion.csv", index=False)
    paired["paired_summary"].to_csv(tdir / "paired_summary.csv", index=False)
    paired["paired_by_condition"].to_csv(tdir / "paired_by_condition.csv.gz", index=False)
    nans.to_csv(tdir / "undefined_regional_metric_frames.csv", index=False)
    pd.DataFrame(runs).to_csv(tdir / "completeness.csv", index=False)
    pkg.run_table(matrix).to_csv(tdir / "run_matrix.csv", index=False)
    costs_note = "not provided"
    figs = []
    for ck in CKPTS:
        fig_coverage(glob, ck, fdir / f"rmse_vs_coverage_{ck}.png")
        figs.append(f"rmse_vs_coverage_{ck}.png")
    fig_points_5pct(glob, fdir / "rmse_points_5pct.png")
    fig_paired(paired["paired_summary"], fdir / "paired_differences_val5.png")
    fig_condition(tables["per_condition"], fdir / "val_by_condition_5pct_best.png")
    fig_error_parts(glob, fdir / "centered_vs_bias_5pct_best.png")
    figs += ["rmse_points_5pct.png", "paired_differences_val5.png", "val_by_condition_5pct_best.png", "centered_vs_bias_5pct_best.png"]
    if args.costs_dir is not None and (args.costs_dir / "training_costs_by_run.csv").exists():
        for f in ("training_costs_by_attempt.csv", "training_costs_by_run.csv", "eval_costs_by_task.csv"):
            (tdir / f).write_bytes((args.costs_dir / f).read_bytes())
        fig_costs(pd.read_csv(tdir / "training_costs_by_run.csv"), pd.read_csv(tdir / "eval_costs_by_task.csv"), fdir / "costs_compute_vs_queue.png")
        figs.append("costs_compute_vs_queue.png")
        costs_note = "training_costs_by_*.csv, eval_costs_by_task.csv"
    five = glob[(glob.coverage == 0.05) & (glob.checkpoint_kind == "best")]
    wide = five.pivot_table(index=["run_id", "family", "p_conditioning", "model_seed", "epoch"], columns="role", values="rmse_K").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"train_fit": "ref760", "seen_train_eval": "seen220", "val_eval": "VAL123"})
    wide.to_csv(tdir / "headline_best_5pct_by_run.csv", index=False)
    gate_max = max(float(r["gate_max_abs_delta_K"] or 0.0) for r in runs)
    manifest = {
        "eval_git_shas": args.eval_sha, "matrix_sha256": msha, "n_runs": len(runs), "n_rows": int(len(table)),
        "n_csv": int(sum(r["n_csv"] for r in runs)), "gate_max_abs_delta_K": gate_max,
        "tables": sorted(p.name for p in tdir.iterdir()), "figures": sorted(p.name for p in fdir.iterdir()),
        "ranking": "none", "winner": "none",
    }
    (args.dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# S7 C149 — pacchetto di valutazione finale (12 run)", "",
        "Pacchetto tecnico per il revisore. Descrittivo: nessuna classifica, nessun vincitore, nessuna regola di selezione applicata (la proposta Opus non è approvata).",
        "Metriche = valutazione finale S7, non metriche del loop. VAL123 già usata nella selezione del best: non è conferma indipendente.", "",
        "## Completezza", "",
        f"- Run validi: {len(runs)}/12; CSV per-frame: {manifest['n_csv']} (81 per run); righe totali: {manifest['n_rows']}.",
        f"- Gate loop-parity al 5% (macro 0.05 K): tutti i run 18/18; massimo scarto macro registrato: {gate_max:.3g} K.",
        f"- Matrice congelata SHA256 `{msha}`; evaluator SHA {', '.join(args.eval_sha)}.",
        "- Controlli superati: chiavi uniche e complete; `mask_hash` per riga = payload verificato (1/5/20%); famiglia/p/seed/epoca/criterio/hash file e pesi = matrice; forward condivisi bit-identici alla sorgente; provenienza e hash input.", "",
        "## Tabelle (`tables/`)", "",
        "- `global_by_setting.csv` / `global_by_replica.csv`: macro per run × checkpoint (best, last, best_seen separati) × ruolo (reference760/seen220/VAL123) × copertura (1/5/20%); RMSE, MAE, bias, |bias|, centered, regioni.",
        "- `per_condition.csv`: stessa griglia per condizione (VAL: 3 condizioni), con min/max sulle repliche di maschera.",
        "- `regional_by_setting.csv`, `undefined_regional_metric_frames.csv`: metriche regionali e conteggio dei frame con regione vuota (NaN esplicito, esclusi dalle medie regionali).",
        "- `family_p_seed_dispersion.csv`: media/min/max/deviazione (ddof=1, n=3) sui seed per famiglia × p — dispersione descrittiva dei training, separata dalle maschere.",
        "- `paired_summary.csv`, `paired_by_condition.csv.gz`: differenze appaiate GLU−U-Net (stesso p, stesso indice di seed) e p_on−p_off (stessa famiglia, stesso seed), per condizione e riassunto (media/min/max, n condizioni migliori/peggiori). Nessun test, nessun vincitore; i seed di famiglie diverse hanno init diverse.",
        "- `coverage_degradation.csv` (1%/20% vs 5%), `role_gap.csv` (VAL−seen, VAL−reference760), `selection_shift.csv` (last−best, best_seen−best), `worst_val_condition.csv`: peggioramenti.",
        "- `headline_best_5pct_by_run.csv`, `run_matrix.csv`, `completeness.csv`.",
        f"- Costi: {costs_note} (calcolo e attesa separati; per i requeue l'attesa è dal requeue; l'attesa dell'array include il limite %4).", "",
        "## Figure (`figures/`)", "",
    ] + [f"- `{f}`" for f in sorted(p.name for p in fdir.iterdir())] + [
        "", "## Impegni S7 ancora MANCANTI (STOP F non dichiarato)", "",
        "1. Snapshot-swap neurali best@5% replica 0 su seen/VAL, stesso pairing/osservazioni/riferimento donatore, 12 run: non eseguiti (funzioni S6 family-aware; manca il wrapper S7).",
        "2. Residuo swap GappyPOD p_on/off S6: non eseguito.",
        "3. Baseline C149 con pool e ricetta di fit dichiarati: non eseguite (le baseline A38 restano storiche).",
        "4. Specifica `month2_s7_c149_protocol.yaml` con la regola di scelta del seguito (da approvare prima di leggere gli esiti), report `10_s7_c149_model_comparison.md`, decisione sul seguito.",
        "5. Verifica del COMPLETE scientifico non sostituisce la revisione: il pacchetto non interpreta i risultati.", "",
        "Nessuno sweep K, nessun training con T, nessun TEST.", "",
    ]
    (args.dest / "INDEX.md").write_text("\n".join(lines))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
