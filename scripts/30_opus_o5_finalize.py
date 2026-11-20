#!/usr/bin/env python3
"""Final O5 gate: check the eight runs, then build the package or write a block report.

Runs unattended on Engaging after the evaluation jobs end. It trains and
submits nothing, never ranks, and never calls the study scientifically closed.
Exit 0: package built. Exit 3: a valid result is missing, see BLOCK_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.evaluation import opus_o5_package as pkg  # noqa: E402
from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS  # noqa: E402

PY = sys.executable


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def sacct_table(job_ids: list[str]) -> str:
    out = _run(["sacct", "-j", ",".join(job_ids), "-X", "-P", "-o", "JobID,State,ExitCode,Start,End,Elapsed,NodeList"])
    return out.stdout if out.returncode == 0 else f"sacct unavailable: {out.stderr.strip()}"


def last_lines(path: Path, n: int = 3) -> str:
    if not path.exists():
        return "(no file)"
    return " / ".join(path.read_text(errors="replace").strip().splitlines()[-n:]) or "(empty)"


def block_report(dest: Path, out_root: Path, runs: list[dict], issues: list[str], jobs: list[str]) -> None:
    lines = [
        "# O5: blocco, nessun pacchetto prodotto",
        "",
        f"Generato {time.strftime('%Y-%m-%dT%H:%M:%S%z')}. Questo NON è un successo. Almeno un run non ha un risultato valido.",
        "",
        "## Stato Slurm",
        "",
        "```",
        sacct_table(jobs),
        "```",
        "",
        "## Run",
        "",
        "| run | COMPLETE.json | CSV | eval SHA |",
        "| --- | --- | ---: | --- |",
    ]
    for r in runs:
        lines.append(f"| {r['run_id']} | {r.get('complete_json')} | {r.get('n_csv')} | {r.get('eval_git_sha')} |")
    lines += ["", "## Problemi esatti", ""]
    lines += [f"- {item}" for item in issues[:300]]
    lines += ["", "## Ultime righe dei log (stdout, stderr)", ""]
    for path in sorted(out_root.glob("slurm-*.err")):
        out = path.with_suffix(".out")
        lines.append(f"- `{path.name}`: err: {last_lines(path)} | out: {last_lines(out, 2)}")
    lines += [
        "",
        "Nessun recupero automatico è implementato. Serve una diagnosi del run mancante e una submission mirata.",
    ]
    (dest / "BLOCK_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/opus_o5_matrix.json")
    parser.add_argument("--eval-sha", action="append", required=True)
    parser.add_argument("--jobs", default="29728465,29732630")
    parser.add_argument("--finalize-sha", default="unknown")
    args = parser.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    args.dest.mkdir(parents=True)
    matrix = json.loads(args.matrix.read_text())
    jobs = args.jobs.split(",")
    _, run_rows, issues = pkg.load_and_check(args.out_root, matrix, allowed_eval_shas=args.eval_sha)
    if issues:
        block_report(args.dest, args.out_root, run_rows, issues, jobs)
        print(f"BLOCKED: {len(issues)} issue(s); see {args.dest / 'BLOCK_REPORT.md'}")
        return 3

    package = args.dest / "package"
    cmd = [PY, str(ROOT / "scripts/29_opus_o5_package.py"), "--out-root", str(args.out_root), "--dest", str(package), "--matrix", str(args.matrix)]
    for sha in args.eval_sha:
        cmd += ["--eval-sha", sha]
    done = _run(cmd)
    (args.dest / "package_stdout.txt").write_text(done.stdout + "\n--- stderr\n" + done.stderr)
    if done.returncode != 0:
        block_report(args.dest, args.out_root, run_rows, ["package builder failed: see package_stdout.txt"], jobs)
        return 3
    field_notes = []
    for ckpt in CHECKPOINTS:
        res = _run([PY, str(ROOT / "scripts/28_opus_o5_figures.py"), "--out-root", str(args.out_root), "--dest", str(package / "figures" / f"val_frame1000_{ckpt}.png"), "--checkpoint", ckpt])
        field_notes.append(f"{ckpt}: {'ok' if res.returncode == 0 else 'FAILED ' + res.stderr[-300:]}")
    manifest = json.loads((package / "package_manifest.json").read_text())
    loop_ok = bool(manifest["loop_within_tolerance_all"])
    figures_ok = all(note.endswith("ok") for note in field_notes)

    ledger = (ROOT / "docs/month2/opus_ledger_o0_o6.md").read_text()
    o5_block = [
        "",
        "## Stato O5 verificato dal job finale",
        "",
        f"- Otto run con `COMPLETE.json` e controlli di completezza/provenienza superati (job finale, SHA `{args.finalize_sha}`).",
        f"- SHA evaluator/launcher dichiarati: {', '.join(args.eval_sha)}.",
        f"- Confronto con il loop al 5% entro 0.05 K su tutte le righe: **{loop_ok}**.",
        f"- Figure dei campi VAL frame 1000: {'; '.join(field_notes)}.",
        "- Report scientifico e decisione O6: **non fatti**, spettano a Neon e al coordinatore.",
    ]
    (args.dest / "LEDGER_O0_O6.md").write_text(ledger + "\n".join(o5_block) + "\n")

    run_lines = ["| run | pool | p | seed | training job | epoche best / best_seen / last | eval SHA |", "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for entry in matrix["runs"]:
        eps = " / ".join(str(entry["checkpoints"][c]["epoch"]) for c in ("best", "best_seen_diagnostic", "last"))
        sha = next(r["eval_git_sha"] for r in run_rows if r["run_id"] == entry["run_id"])
        run_lines.append(f"| {entry['run_id']} | {entry['pool_id']} | {'on' if entry['p_conditioning'] else 'off'} | {entry['model_seed']} | {entry['job_id']} | {eps} | `{sha[:10]}` |")
    index = [
        "# Indice tecnico O5 per Neon",
        "",
        "Pacchetto tecnico completo per la revisione. Non contiene il report scientifico né la decisione O6.",
        "",
        "## Dove",
        "",
        f"- Pacchetto: `{package}` (`tables/`, `figures/`, `TABLES.md`, `package_manifest.json` con SHA-256 di ogni file).",
        f"- Output grezzi per frame, directory per run: `{args.out_root}/<run_id>/`.",
        f"- Ledger O0-O6: `{args.dest / 'LEDGER_O0_O6.md'}`. Riepilogo: `{args.dest / 'FINAL_SUMMARY.md'}`.",
        "- R0 (già prodotto, non rifatto): `results/month2/opus_data_study/r0_selection/`. Limite: rianalisi sulle curve dei quattro S5, non una validazione indipendente; scheduler e early stop hanno usato VAL.",
        "- Bridge S5 (già prodotto): `results/month2/opus_data_study/bridge_s5/LEDGER.md`. Limite: l'ordine dei frame di S5 (`DataLoader(shuffle=True)`) differisce dallo stream Opus (`randperm`); il replay di inferenza ha delta 0.0 K entro 0.05 K ma `training_equivalence` resta `not_tested`. Gli S5 non sono righe di O5; i due riferimenti A38 seed 0 usano il runner nuovo.",
        "- Curve del loop (`training_loop_not_o5`): `results/month2/opus_data_study/o5_prep/loop_curves.csv`.",
        "",
        "## Run",
        "",
        *run_lines,
        "",
        "Il run O-B19-off-s0 è stato valutato dal job sostitutivo 29732630 (launcher `29b94345…`); gli altri sette dall'array 29728465 (launcher `91828680…`). Evaluator identico nei due; il pacchetto non descrive la campagna come una singola esecuzione. Il task 29728465_1 è fallito prima dell'evaluator per un controllo di accesso e non ha scritto output.",
        "",
        "## Tabelle (`package/tables/`)",
        "",
        "- `global_by_setting.csv`: run × checkpoint × ruolo × copertura, media sulle condizioni e sulle repliche, con min/max delle repliche per la RMSE.",
        "- `per_condition.csv`: stessa griglia per condizione.",
        "- `regional_by_setting.csv`, `regional_by_condition.csv`: metriche delle regioni calde, picco, getto, esterno; IoU e area relativa.",
        "- `compare_summary.csv`, `compare_by_condition.csv`: delta appaiati dati (B19/A38/C149, seed 0, stessa p), seed (A38 1 meno 0), p (on meno off) e selezione (last/best_seen meno best). Descrittivi, nessun test, nessun vincitore.",
        "- `loop_vs_o5_5pct.csv`: O5 al 5% contro il loop alla stessa epoca; train e seen replica 0, VAL media delle tre maschere, tolleranza 0.05 K.",
        "- `setting_costs.csv`, `completeness.csv`.",
        "",
        "## Figure (`package/figures/`)",
        "",
        "Curve del loop contro update e passaggi del pool con i punti O5; RMSE al 5% per checkpoint; curve di copertura 1/5/20%; VAL per condizione; metriche regionali per checkpoint; delta appaiati; campi VAL frame 1000 con scala comune (un frame, non l'insieme VAL). Scale comuni dichiarate nei titoli. Le figure vanno guardate dal revisore prima dell'uso.",
        "",
        "## Limiti dichiarati",
        "",
        "- 1% e 20% sono generalizzazioni della densità TRAIN 5%, non una curva ottimale del numero di sensori.",
        "- B19 usa il normalizzatore fit-only dei 760 frame; reference760 è la stessa diagnostica per tutti i pool.",
        "- Tre run hanno pesi identici fra `best` e `best_seen_diagnostic`: forward condiviso, identità e righe distinte e verificate uguali.",
        "- Due seed e tre maschere sono un metro descrittivo, non una distribuzione stimata. VAL ha influenzato selezione, scheduler e arresto.",
        "- Nessun TEST, refit, A4 o nuovo training.",
    ]
    (args.dest / "INDEX_FOR_NEON.md").write_text("\n".join(index) + "\n")

    tables_md = (package / "TABLES.md").read_text()
    summary = [
        "# Riepilogo finale tecnico O5",
        "",
        f"Generato {time.strftime('%Y-%m-%dT%H:%M:%S%z')} dal job finale (checkout SHA `{args.finalize_sha}`).",
        "",
        f"- Otto run validi: sì. Completezza (81 CSV e 3 campi per run), provenienza, identità dei checkpoint, maschere identiche, forward condivisi: tutti superati.",
        f"- Confronto col loop al 5% entro 0.05 K ovunque: **{loop_ok}**" + ("" if loop_ok else " (discrepanza: leggere `package/tables/loop_vs_o5_5pct.csv` prima di usare i risultati)"),
        f"- Figure dei campi: {'ok' if figures_ok else 'ERRORE: ' + '; '.join(field_notes)}.",
        "- Non fatto: report scientifico, decisione O6, controllo visivo delle figure.",
        "",
        "Il pacchetto non chiude Opus scientificamente.",
        "",
        tables_md,
    ]
    (args.dest / "FINAL_SUMMARY.md").write_text("\n".join(summary) + "\n")
    print(f"PACKAGE BUILT at {args.dest}; loop_ok={loop_ok}; figures_ok={figures_ok}")
    return 0 if figures_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
