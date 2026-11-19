#!/usr/bin/env python3
"""Write the S5 R0 tables from the saved curve extract.

The extract is read-only provenance. This script does not load checkpoints
and does not recompute predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from combustion_recon.evaluation.opus_r0_selection import (
    eval_points_from_s5_extract,
    r0_rows_for_points,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/month2/opus_data_study/r0_selection/s5_curves_source.json"
DEFAULT_OUT = ROOT / "results/month2/opus_data_study/r0_selection"


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tables(source_path: Path, out_dir: Path) -> list[dict]:
    payload = json.loads(source_path.read_text())
    rows: list[dict] = []
    for run_id, run in payload["runs"].items():
        points = eval_points_from_s5_extract(run)
        # Guard: CSV condition means reconstruct the recorded VAL macro.
        for point in points:
            conds = list(point["conditions"].values())
            recon = sum(conds) / float(len(conds))
            if abs(recon - float(point["val_mean"])) > 1e-6:
                raise SystemExit(
                    f"{run_id} epoch {point['epoch']}: condition mean {recon} "
                    f"!= recorded {point['val_mean']}"
                )
        rows.extend(r0_rows_for_points(run_id, points, summary=run["summary"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "rule",
        "epoch",
        "update",
        "score_name",
        "score",
        "val_mean_condition_rmse_K",
        "seen_mean_condition_rmse_K",
        "train_mean_condition_rmse_K",
        "lr",
        "conditions",
        "condition_rmse_K",
        "source",
        "note",
    ]
    csv_path = out_dir / "r0_selections.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["conditions"] = ",".join(row["conditions"])
            flat["condition_rmse_K"] = json.dumps(row["condition_rmse_K"], sort_keys=True)
            writer.writerow(flat)

    lines = [
        "# R0 — rianalisi delle quattro storie S5",
        "",
        "Selezioni calcolate sulle curve già registrate. Scheduler e early-stop hanno usato VAL: questa tabella non è una validazione indipendente e non crea checkpoint intermedi.",
        "",
        "| Run | Regola | Epoca | Update | Score | VAL | seen | Condizioni (RMSE K) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        cond = ", ".join(
            f"{name.removesuffix('.h5')} {value:.3f}"
            for name, value in sorted(row["condition_rmse_K"].items())
        )
        lines.append(
            "| {run} | {rule} | {epoch} | {update} | {score:.4f} | {val:.4f} | {seen} | {cond} |".format(
                run=row["run_id"],
                rule=row["rule"],
                epoch=row["epoch"],
                update=row["update"],
                score=row["score"],
                val=row["val_mean_condition_rmse_K"],
                seen="" if row["seen_mean_condition_rmse_K"] is None else f"{row['seen_mean_condition_rmse_K']:.4f}",
                cond=cond,
            )
        )
    lines.extend(
        [
            "",
            "Il criterio di ogni riga è `score_name` nel CSV. `original_best_val` e `original_last` restano le selezioni del training.",
            "L'ultima eval di ogni run manca da `eval_history.json` ed è stata unita dal CSV `evals/epoch_XXXX/val_eval_per_frame.csv.gz` e da `progress.json`. Il CSV `train_fit` di quell'epoca non c'è: il train all'arresto non è stato inventato.",
            "Leave-one-condition-out sceglie l'epoca che minimizza la media delle altre due condizioni VAL e riporta la condizione esclusa a quell'epoca. La curva di tutte e tre le condizioni era già visibile al training.",
            "",
        ]
    )
    (out_dir / "r0_selections.md").write_text("\n".join(lines))
    _write_provenance(payload, out_dir, source_path)
    return rows


def _write_provenance(payload: dict, out_dir: Path, source_path: Path) -> None:
    lines = [
        "# Provenienza R0",
        "",
        payload["extraction"]["note"],
        "",
        f"Sorgente committata: `{source_path.name}`.",
        "Script: `scripts/19_write_opus_r0_tables.py`.",
        "Nessun checkpoint è stato ricaricato per questa tabella. I pesi best/last restano solo sugli originali.",
        "",
        "| Run | git | best.pt | last.pt | best epoca | last epoca | stato |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for run_id, run in payload["runs"].items():
        sha = run["sha256"]
        summary = run["summary"]
        lines.append(
            "| {run} | `{git}` | `{best}` | `{last}` | {be} | {le} | {status} |".format(
                run=run_id,
                git=run["git_sha"],
                best=sha["checkpoints/best.pt"],
                last=sha["checkpoints/last.pt"],
                be=summary["best_val_epoch"],
                le=summary["epochs_completed"],
                status=summary["status"],
            )
        )
    lines.append("")
    (out_dir / "PROVENANCE.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = write_tables(args.source, args.out_dir)
    print(f"wrote {len(rows)} rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
