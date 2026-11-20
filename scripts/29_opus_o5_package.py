#!/usr/bin/env python3
"""Build the technical O5 package from a finished attempt2 output root.

Refuses to write anything unless all eight runs are complete and every check
passes. Trains nothing, ranks nothing, and does not conclude the study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation import opus_o5_package as pkg  # noqa: E402
from combustion_recon.evaluation import opus_o5_plots as plots  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md_table(frame: pd.DataFrame, floats: int = 2) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.itertuples(index=False):
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/opus_o5_matrix.json")
    parser.add_argument("--loop-curves", type=Path, default=ROOT / "results/month2/opus_data_study/o5_prep/loop_curves.csv")
    parser.add_argument("--eval-sha", action="append", required=True, help="declared evaluator/launcher SHA, repeatable")
    args = parser.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    runs = pkg.run_table(matrix)
    table, run_rows, issues = pkg.load_and_check(args.out_root, matrix, allowed_eval_shas=args.eval_sha)
    if issues:
        print("PACKAGE REFUSED. Issues:")
        for item in issues[:200]:
            print(" -", item)
        print(f"{len(issues)} issue(s); nothing written.")
        return 1
    loop = pd.read_csv(args.loop_curves)
    tables = pkg.build_tables(table, runs, matrix)
    loop_cmp = pkg.loop_comparison(tables["global_by_replica"], loop)
    cmp = pkg.comparisons(tables["per_condition"], runs)
    costs = pkg.setting_costs(args.out_root)
    nans = pkg.nan_report(table)

    tdir = args.dest / "tables"
    fdir = args.dest / "figures"
    tdir.mkdir(parents=True, exist_ok=False)
    for name, frame in {**tables, **cmp}.items():
        frame.to_csv(tdir / f"{name}.csv", index=False)
    loop_cmp.to_csv(tdir / "loop_vs_o5_5pct.csv", index=False)
    costs.to_csv(tdir / "setting_costs.csv", index=False)
    nans.to_csv(tdir / "undefined_regional_metric_frames.csv", index=False)
    pd.DataFrame(run_rows).to_csv(tdir / "completeness.csv", index=False)
    figures = plots.make_all(fdir, tables, loop, runs, cmp)

    g = tables["global_by_setting"]
    five = g[(g.coverage == 0.05)]
    wide = five.pivot_table(index=["run_id", "checkpoint_kind", "epoch"], columns="role", values="rmse_K").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"train_fit": "ref760", "seen_train_eval": "seen220", "val_eval": "VAL123"})
    order = {name: i for i, name in enumerate(pkg.RUN_ORDER)}
    wide = wide.sort_values(["run_id", "checkpoint_kind"], key=lambda s: s.map(order) if s.name == "run_id" else s)
    s = cmp["compare_summary"]
    s5 = s[(s.metric == "rmse_K") & (s.coverage == 0.05) & (s.role.isin(["seen_train_eval", "val_eval"])) & (s.family != "selection")]
    s5 = s5[["label", "checkpoint_a", "role", "mean_delta", "min_delta", "max_delta", "n_conditions", "n_a_lower", "n_a_higher"]]
    sel5 = s[(s.metric == "rmse_K") & (s.coverage == 0.05) & (s.role.isin(["seen_train_eval", "val_eval"])) & (s.family == "selection")]
    sel5 = sel5[["label", "role", "mean_delta", "min_delta", "max_delta", "n_conditions", "n_a_lower", "n_a_higher"]]
    lc = loop_cmp.groupby("role").agg(n=("delta_K", "size"), max_abs_delta_K=("delta_K", lambda v: float(v.abs().max())), n_within=("within_tolerance", "sum")).reset_index()
    text = [
        "# Tabelle O5, lettura descrittiva",
        "",
        "Generato da `scripts/29_opus_o5_package.py`. Nessun rango, nessun test, nessuna conclusione. RMSE in K, media sulle condizioni e sulle repliche.",
        "",
        "## RMSE globale al 5% per run e checkpoint",
        "",
        _md_table(wide),
        "",
        "## Delta appaiati al 5% (a meno b), per condizione poi media",
        "",
        "Dati: stessa p e seed 0. Seed: A38 seed 1 meno seed 0. p: on meno off. `n_a_lower` conta le condizioni in cui a ha RMSE minore.",
        "",
        _md_table(s5),
        "",
        "## Selezione: last, best_seen meno best (stesso run)",
        "",
        _md_table(sel5),
        "",
        "## Metriche regionali non definite su alcuni frame",
        "",
        f"{len(nans)} gruppi (run, checkpoint, ruolo, copertura, metrica) hanno frame con metrica regionale non definita (NaN, per esempio picco vuoto). Le medie li saltano. Elenco: `tables/undefined_regional_metric_frames.csv`. Le metriche di base (RMSE, MAE, bias, centered) sono finite ovunque.",
        "",
        f"## O5 al 5% contro il loop alla stessa epoca (tolleranza {pkg.LOOP_TOLERANCE_K} K)",
        "",
        "Train e seen: replica 0. VAL: media delle tre maschere.",
        "",
        _md_table(lc, 4),
        "",
    ]
    (args.dest / "TABLES.md").write_text("\n".join(text))
    files = sorted(p for p in args.dest.rglob("*") if p.is_file())
    manifest = {
        "out_root": str(args.out_root),
        "declared_eval_shas": args.eval_sha,
        "matrix_sha256": _sha(args.matrix),
        "loop_curves_sha256": _sha(args.loop_curves),
        "runs": run_rows,
        "loop_within_tolerance_all": bool(loop_cmp["within_tolerance"].all()),
        "files": {str(p.relative_to(args.dest)): _sha(p) for p in files},
        "ranking": "none",
    }
    (args.dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"package written to {args.dest}: {len(files)} files, {len(figures)} figures")
    print("loop within tolerance everywhere:", manifest["loop_within_tolerance_all"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
