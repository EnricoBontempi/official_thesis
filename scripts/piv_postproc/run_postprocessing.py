#!/usr/bin/env python3
"""Mission 02 (Virtual PIV) resumable CPU post-processing orchestrator.

Reads a ``piv_postproc_manifest_v1`` run manifest (no invented paths, job IDs, or
checkpoints), audits training & evaluation completeness (separating missing, failed,
incomplete, budget_exhausted, and plateau), aggregates confirmatory & exploratory
tables, parses Slurm accounting (``sacct -D``), renders curated main figures and the
structured diagnostic atlas, and writes provenance + SHA-256 package manifest.

Resumable via ``<dest>/.state/<stage>.json``: re-running with ``--resume`` skips
up-to-date stages and never duplicates rows or files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.piv_observations import load_piv_spec  # noqa: E402
from combustion_recon.evaluation.piv_accounting import collect_compute_accounting  # noqa: E402
from combustion_recon.evaluation.piv_aggregate import run_full_aggregation  # noqa: E402
from combustion_recon.evaluation.piv_eval import atomic_write_csv  # noqa: E402
from combustion_recon.evaluation.piv_manifest import (  # noqa: E402
    StageStateManager,
    atomic_write_json,
    atomic_write_text,
    audit_manifest,
    load_manifest,
    render_status_report_md,
    sha256_file,
    sha256_json,
)
from combustion_recon.evaluation.piv_plots_atlas import generate_diagnostic_atlas  # noqa: E402
from combustion_recon.evaluation.piv_plots_main import generate_main_figures  # noqa: E402


def _git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    head = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["status", "--porcelain"]) or ""
    return {
        "git_sha": head,
        "git_branch": branch,
        "working_tree_clean": len([ln for ln in porcelain.splitlines() if ln.strip()]) == 0,
    }


def _write_csv_gz(df: pd.DataFrame, path: Path) -> Path:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.csv.gz")
    try:
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] piv-postproc {msg}", flush=True)


def _render_index_md(
    *,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    conf_files: list[Path],
    expl_files: list[Path],
    main_figs: list[Path],
    atlas_sections: dict[str, list[Path]],
    dest: Path,
    strict_complete: bool,
    smoke: bool,
) -> str:
    s = audit["summary"]
    status_badge = (
        "NON-SCIENTIFIC SMOKE / TEST"
        if smoke
        else ("COMPLETO (pacchetto confermativo)" if strict_complete else "PARZIALE / DIAGNOSTICO (NON confermativo)")
    )
    lines = [
        "# Missione 02 (PIV virtuale) — Indice del pacchetto di post-processing",
        "",
        f"- **Stato del pacchetto:** `{status_badge}`",
        f"- **Manifest:** `{manifest.get('_manifest_path')}` (sha256 `{str(manifest.get('_manifest_sha256', ''))[:16]}...`)",
        f"- **Specifica osservazioni:** `{manifest.get('observation_spec_path')}` (sha256 `{manifest.get('observation_spec_sha256')}`)",
        f"- **Ruoli autorizzati:** `{', '.join(manifest.get('authorized_roles', []))}` (TEST rifiutato di default)",
        f"- **Run totali / richiesti:** `{s['n_runs_total']}` / `{s['n_runs_required']}`",
        f"- **Stati training:** plateau={s['training_status_counts'].get('plateau', 0)}, "
        f"budget_exhausted={s['training_status_counts'].get('budget_exhausted', 0)}, "
        f"incomplete={s['training_status_counts'].get('incomplete', 0)}, "
        f"failed={s['training_status_counts'].get('failed', 0)}, "
        f"missing={s['training_status_counts'].get('missing', 0)}",
        "",
        "> **Nota metodologica (roadmap §7.3):** le repliche di rumore (0, 1, 2) non sono campioni indipendenti; "
        "l'unità di replica statistica è il seed di training. I run `budget_exhausted` (1200 epoche senza plateau) "
        "sono artefatti validi inclusi con marcatore vuoto, **mai** dichiarati convergenti. "
        "I risultati su VAL123 sono di sviluppo (VAL è usata per la selezione).",
        "",
        "## 1. Stato dei run e provenienza",
        "",
        "- [STATUS_REPORT.md](STATUS_REPORT.md) — stato separato di mancanti, falliti, incompleti e `budget_exhausted`",
        "- [run_status_audit.csv](run_status_audit.csv)",
        "- [run_status_summary.json](run_status_summary.json)",
        "- [provenance.json](provenance.json)",
        "- [package_manifest.json](package_manifest.json)",
        "",
        "## 2. Analisi CONFERMATIVE (previste da `roadmap_tecnica_piv.md` e `decisione_R1.md`)",
        "",
        "### 2.1 Tabelle confermative",
        "",
    ]
    for p in sorted(conf_files):
        rel = p.relative_to(dest).as_posix()
        lines.append(f"- [`{rel}`]({rel})")
    lines += ["", "### 2.2 Figure principali (curate)", ""]
    if main_figs:
        for p in sorted(main_figs):
            rel = p.relative_to(dest).as_posix()
            lines.append(f"- [`{rel}`]({rel})")
    else:
        lines.append("- *(nessuna figura principale generata sui dati disponibili)*")

    lines += [
        "",
        "## 3. Analisi ESPLORATIVE — Atlante diagnostico",
        "",
        "### 3.1 Tabelle diagnostiche e accounting",
        "",
    ]
    for p in sorted(expl_files):
        rel = p.relative_to(dest).as_posix()
        lines.append(f"- [`{rel}`]({rel})")

    lines += ["", "### 3.2 Sezioni dell'atlante diagnostico", ""]
    for sec, paths in sorted(atlas_sections.items()):
        lines.append(f"#### `{sec}`")
        if paths:
            for p in sorted(paths):
                rel = p.relative_to(dest).as_posix()
                lines.append(f"- [`{rel}`]({rel})")
        else:
            lines.append("- *(nessun dato disponibile per questa sezione)*")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_pipeline(
    *,
    manifest_path: Path,
    dest: Path,
    resume: bool = True,
    allow_partial: bool = False,
    smoke: bool = False,
    n_boot: int = 300,
    skip_atlas: bool = False,
    force_stages: set[str] | None = None,
) -> int:
    force_stages = set(force_stages or ())
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    state = StageStateManager(dest)

    manifest = load_manifest(manifest_path, allow_test_with_r5_auth=False)
    spec_vref = None
    spec_path_raw = manifest.get("observation_spec_path")
    if spec_path_raw:
        sp = Path(spec_path_raw)
        if not sp.is_absolute():
            sp = ROOT / sp
        if sp.is_file():
            piv_spec = load_piv_spec(sp, expected_sha256=manifest.get("observation_spec_sha256"))
            spec_vref = piv_spec.raw.get("reference_velocity", {}).get("per_condition_m_s")

    # ---------------- Stage 1: audit
    audit = audit_manifest(manifest, smoke=smoke)
    audit_sig = sha256_json(
        {
            "manifest_sha256": manifest["_manifest_sha256"],
            "summary": audit["summary"],
            "smoke": smoke,
        }
    )
    audit_csv = dest / "run_status_audit.csv"
    audit_json = dest / "run_status_summary.json"
    status_md = dest / "STATUS_REPORT.md"
    if not (resume and "audit" not in force_stages and state.is_up_to_date("audit", audit_sig)):
        atomic_write_csv(audit["table"], audit_csv)
        atomic_write_json(audit_json, audit["summary"])
        atomic_write_text(status_md, render_status_report_md(audit))
        state.mark_done("audit", audit_sig, [audit_csv, audit_json, status_md])
        _log("stage audit: written")
    else:
        _log("stage audit: up-to-date (skipped)")

    strict_complete = bool(audit["summary"]["strict_complete"])
    if not strict_complete and not allow_partial:
        block_md = dest / "BLOCK_REPORT.md"
        atomic_write_text(
            block_md,
            "# BLOCK_REPORT — Missione 02 post-processing confermativo bloccato\n\n"
            "Almeno un run richiesto è mancante, fallito o incompleto (o presenta incoerenze di inizializzazione). "
            "Nessun `COMPLETE.json` è stato scritto. Rieseguire con `--allow-partial` per un pacchetto diagnostico parziale.\n\n"
            + render_status_report_md(audit),
        )
        _log(f"BLOCKED (strict mode): see {block_md}")
        return 3

    # ---------------- Stage 2: aggregate (confirmatory + exploratory tables)
    agg_sig = sha256_json(
        {
            "audit_sig": audit_sig,
            "strict": strict_complete and not allow_partial,
            "n_boot": n_boot,
        }
    )
    conf_dir = dest / "tables" / "confirmatory"
    expl_dir = dest / "tables" / "exploratory"
    conf_dir.mkdir(parents=True, exist_ok=True)
    expl_dir.mkdir(parents=True, exist_ok=True)

    # Always compute in-memory agg if downstream stages need it, but only rewrite files if signature changed
    agg = run_full_aggregation(
        manifest,
        audit,
        strict=(strict_complete and not allow_partial),
        spec_vref=spec_vref,
        n_boot=n_boot,
    )

    conf_files: list[Path] = []
    expl_files: list[Path] = []

    if not (resume and "aggregate" not in force_stages and state.is_up_to_date("aggregate", agg_sig)):
        conf = agg["confirmatory"]
        for name, obj in conf.items():
            if isinstance(obj, pd.DataFrame):
                if name in ("per_condition", "macro_by_replica"):
                    p = _write_csv_gz(obj, conf_dir / f"{name}.csv.gz")
                else:
                    p = conf_dir / f"{name}.csv"
                    atomic_write_csv(obj, p)
                conf_files.append(p)
            else:
                p = conf_dir / f"{name}.json"
                atomic_write_json(p, obj)
                conf_files.append(p)

        expl = agg["exploratory"]
        for name, obj in expl.items():
            if isinstance(obj, pd.DataFrame):
                if name in ("train_epoch_curves", "val_eval_curves", "paired_frame_deltas"):
                    p = _write_csv_gz(obj, expl_dir / f"{name}.csv.gz")
                else:
                    p = expl_dir / f"{name}.csv"
                    atomic_write_csv(obj, p)
                expl_files.append(p)
            else:
                p = expl_dir / f"{name}.json"
                atomic_write_json(p, obj)
                expl_files.append(p)

        state.mark_done("aggregate", agg_sig, conf_files + expl_files)
        _log(f"stage aggregate: wrote {len(conf_files)} confirmatory + {len(expl_files)} exploratory files")
    else:
        conf_files = sorted(conf_dir.glob("*"))
        expl_files = sorted(p for p in expl_dir.glob("*") if not p.name.startswith("compute_accounting_") and p.name != "eval_costs_by_config.csv")
        _log("stage aggregate: up-to-date (skipped)")

    # ---------------- Stage 3: accounting
    acc = collect_compute_accounting(manifest, audit["table"])
    acc_sig = sha256_json(
        {
            "audit_sig": audit_sig,
            "has_sacct": acc["has_sacct"],
            "n_jobs": len(acc["jobs"]),
            "n_runs": len(acc["by_run"]),
        }
    )
    acc_files = [
        expl_dir / "compute_accounting_jobs.csv",
        expl_dir / "compute_accounting_phases.csv",
        expl_dir / "compute_accounting_by_run.csv",
        expl_dir / "eval_costs_by_config.csv",
    ]
    if not (resume and "accounting" not in force_stages and state.is_up_to_date("accounting", acc_sig)):
        atomic_write_csv(acc["jobs"], acc_files[0])
        atomic_write_csv(acc["phases"], acc_files[1])
        atomic_write_csv(acc["by_run"], acc_files[2])
        atomic_write_csv(acc["eval_costs_by_config"], acc_files[3])
        state.mark_done("accounting", acc_sig, acc_files)
        _log("stage accounting: written")
    else:
        _log("stage accounting: up-to-date (skipped)")
    expl_files = sorted(set(expl_files + acc_files))

    # ---------------- Stage 4: main figures
    main_dir = dest / "figures" / "main"
    main_sig = sha256_json({"agg_sig": agg_sig, "stage": "figures_main_v1"})
    if not (resume and "figures_main" not in force_stages and state.is_up_to_date("figures_main", main_sig)):
        main_figs = generate_main_figures(agg, audit, manifest, main_dir)
        state.mark_done("figures_main", main_sig, main_figs)
        _log(f"stage figures_main: generated {len(main_figs)} main figures")
    else:
        main_figs = sorted(main_dir.glob("*.png"))
        _log(f"stage figures_main: up-to-date ({len(main_figs)} files skipped)")

    # ---------------- Stage 5: diagnostic atlas
    atlas_dir = dest / "figures" / "atlas"
    atlas_sig = sha256_json({"agg_sig": agg_sig, "acc_sig": acc_sig, "skip_atlas": skip_atlas, "stage": "figures_atlas_v1"})
    atlas_sections: dict[str, list[Path]] = {}
    if skip_atlas:
        _log("stage figures_atlas: skipped by flag")
    elif not (resume and "figures_atlas" not in force_stages and state.is_up_to_date("figures_atlas", atlas_sig)):
        atlas_sections = generate_diagnostic_atlas(agg, audit, acc, atlas_dir)
        all_atlas_paths = [p for paths in atlas_sections.values() for p in paths]
        state.mark_done("figures_atlas", atlas_sig, all_atlas_paths)
        _log(f"stage figures_atlas: generated {len(all_atlas_paths)} atlas figures across {len(atlas_sections)} sections")
    else:
        if atlas_dir.is_dir():
            for sub in sorted(p for p in atlas_dir.iterdir() if p.is_dir()):
                atlas_sections[sub.name] = sorted(sub.glob("*.png"))
        _log("stage figures_atlas: up-to-date (skipped)")

    # ---------------- Stage 6: provenance, INDEX.md, package_manifest.json, COMPLETE/PARTIAL marker
    provenance = {
        "schema": "piv_postproc_provenance_v1",
        "scientific": not smoke,
        "strict_complete": strict_complete,
        "allow_partial": allow_partial,
        "code": _git_info(),
        "manifest_path": manifest.get("_manifest_path"),
        "manifest_sha256": manifest.get("_manifest_sha256"),
        "observation_spec_path": manifest.get("observation_spec_path"),
        "observation_spec_sha256": manifest.get("observation_spec_sha256"),
        "normalization_stats_path": manifest.get("normalization_stats_path"),
        "normalization_stats_sha256": manifest.get("normalization_stats_sha256"),
        "authorized_roles": manifest.get("authorized_roles"),
        "expected_replicas": manifest.get("expected_replicas"),
        "checkpoints_provenance": [
            {
                "run_id": r["run_id"],
                "checkpoint_kind": r["checkpoint_kind"],
                "training_git_sha": r["training_git_sha"],
                "eval_git_sha": r["eval_git_sha"],
                "checkpoint_epoch": r["checkpoint_epoch"],
                "checkpoint_file_sha256": r["checkpoint_file_sha256"],
                "checkpoint_weights_sha256": r["checkpoint_weights_sha256"],
                "spec_sha256": r["spec_sha256"],
            }
            for _, r in audit["table"].iterrows()
        ],
        "warnings": agg["warnings"],
    }
    atomic_write_json(dest / "provenance.json", provenance)
    atomic_write_text(
        dest / "INDEX.md",
        _render_index_md(
            manifest=manifest,
            audit=audit,
            conf_files=conf_files,
            expl_files=expl_files,
            main_figs=main_figs,
            atlas_sections=atlas_sections,
            dest=dest,
            strict_complete=strict_complete,
            smoke=smoke,
        ),
    )

    # Package file manifest (exclude .state and package_manifest.json itself)
    all_files = sorted(
        p
        for p in dest.rglob("*")
        if p.is_file() and ".state" not in p.relative_to(dest).parts and p.name != "package_manifest.json"
    )
    pkg_manifest = {
        "schema": "piv_postproc_package_manifest_v1",
        "scientific": not smoke,
        "strict_complete": strict_complete,
        "manifest_sha256": manifest["_manifest_sha256"],
        "n_files": len(all_files),
        "files": {p.relative_to(dest).as_posix(): sha256_file(p) for p in all_files},
    }
    atomic_write_json(dest / "package_manifest.json", pkg_manifest)

    if strict_complete and not smoke:
        (dest / "PARTIAL_DIAGNOSTIC.json").unlink(missing_ok=True)
        (dest / "BLOCK_REPORT.md").unlink(missing_ok=True)
        atomic_write_json(
            dest / "COMPLETE.json",
            {
                "ok": True,
                "scientific": True,
                "strict_complete": True,
                "n_runs_total": audit["summary"]["n_runs_total"],
                "n_budget_exhausted_runs": audit["summary"]["n_budget_exhausted_runs"],
                "note": audit["summary"]["budget_exhausted_note"],
            },
        )
        _log("POSTPROC_COMPLETE (strict scientific package)")
    else:
        (dest / "COMPLETE.json").unlink(missing_ok=True)
        atomic_write_json(
            dest / ("SMOKE_OK.json" if smoke else "PARTIAL_DIAGNOSTIC.json"),
            {
                "ok": True,
                "scientific": False if smoke else False,
                "strict_complete": strict_complete,
                "reason": "non_scientific_smoke" if smoke else "partial_or_incomplete_runs_diagnostic_only",
                "training_status_counts": audit["summary"]["training_status_counts"],
                "eval_status_counts": audit["summary"]["eval_status_counts"],
            },
        )
        _log("POSTPROC_PARTIAL_OR_SMOKE_OK (not declared complete scientific package)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="Path to piv_postproc_manifest_v1 JSON/YAML")
    ap.add_argument("--dest", type=Path, required=True, help="Output package directory")
    ap.add_argument("--no-resume", action="store_true", help="Recompute all stages from scratch")
    ap.add_argument("--force-stage", action="append", default=[], help="Force recomputation of specific stage(s)")
    ap.add_argument("--allow-partial", action="store_true", help="Generate diagnostic package even if runs are missing/incomplete")
    ap.add_argument("--smoke", action="store_true", help="Non-scientific smoke/test mode (never writes COMPLETE.json)")
    ap.add_argument("--n-boot", type=int, default=300, help="Hierarchical block bootstrap resamples")
    ap.add_argument("--skip-atlas", action="store_true", help="Only build confirmatory tables and main figures")
    args = ap.parse_args(argv)
    return run_pipeline(
        manifest_path=args.manifest,
        dest=args.dest,
        resume=not args.no_resume,
        allow_partial=args.allow_partial,
        smoke=args.smoke,
        n_boot=args.n_boot,
        skip_atlas=args.skip_atlas,
        force_stages=set(args.force_stage),
    )


if __name__ == "__main__":
    raise SystemExit(main())
