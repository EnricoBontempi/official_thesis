#!/usr/bin/env python3
"""Read-only O0 check of the delivered S6 tables and figures. Changes nothing."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/month2/s6_sparse_eval/STOP_E_delivery"


def main() -> int:
    out: list[str] = ["# Controllo O0 della consegna S6 (STOP E)", "", "Lettura sola, `scripts/31_o0_s6_delivery_check.py`.", ""]
    ok = True
    rows = list(csv.DictReader(open(D / "tables/neural_per_condition_all.csv")))
    runs = sorted({r["run_id"] for r in rows})
    out.append(f"- Righe per condizione: {len(rows)}, run: {', '.join(runs)}.")
    for run in runs:
        part = [r for r in rows if r["run_id"] == run]
        by_role = Counter(r["role"] for r in part)
        conds = {role: len({r["sim_id"] for r in part if r["role"] == role}) for role in by_role}
        ok &= conds == {"train_fit": 20, "seen_train_eval": 20, "val_eval": 3}
        covs = {r["site_fraction"] for r in part}
        ckpts = {r["checkpoint"] for r in part}
        reps = {str(int(float(r["mask_replica_id"]))) for r in part}
        ok &= covs == {"0.01", "0.05", "0.2"} and ckpts == {"best", "last"} and reps == {"0", "1", "2"}
        out.append(f"  - {run}: condizioni {conds}, coperture {sorted(covs)}, checkpoint {sorted(ckpts)}, repliche {sorted(reps)}.")
    regional = [c for c in rows[0] if any(t in c for t in ("high", "peak", "jet"))]
    bad = {c: sum(1 for r in rows if r[c] in ("", "nan", "NaN")) for c in regional}
    bad = {c: n for c, n in bad.items() if n}
    ok &= not bad
    out.append(f"- Colonne regionali presenti: {len(regional)} ({', '.join(c for c in regional if c.endswith('_K'))}). Valori mancanti nella tabella per condizione: {bad or 'nessuno'}.")
    val = list(csv.DictReader(open(D / "tables/val_coverage_best_last.csv")))
    n_frames = {r["n_frames"] for r in val}
    ok &= n_frames == {"369"}
    out.append(f"- VAL: {len(val)} righe di copertura, `n_frames` = {sorted(n_frames)} (123 frame × 3 repliche).")
    figs = sorted(p.name for p in (D / "figures").glob("*.png"))
    ok &= len(figs) == 12
    out.append(f"- Figure presenti: {len(figs)} (tre per run).")
    out += [
        "",
        "## Esito",
        "",
        f"Tabelle: conteggi, condizioni, repliche e colonne regionali coerenti con il contratto: **{'sì' if ok else 'NO'}**.",
        "",
        "## Limiti delle figure (osservati guardandole, non rigenerati)",
        "",
        "- `*_train_loss_curves.png`: loss di training su scala lineare, una sola curva, senza VAL/seen né best/last; dopo circa 100 epoche la curva è indistinguibile dallo zero. Non permette di leggere il plateau.",
        "- `*_val_frame1000_aligned_vs_swapped.png`: ogni pannello ha la propria barra colore, e l'errore assoluto allineato e scambiato usa scale diverse (circa 600 e 1400 K). Serve a vedere il contrasto dello swap, non a confrontare le ampiezze fra pannelli.",
        "- Le figure di O5 (`results/month2/opus_data_study/o5/package/figures/`) usano scale comuni dichiarate. Le figure S6 non sono state rigenerate: lo script S6 e i dati storici non sono stati toccati.",
    ]
    (ROOT / "results/month2/opus_data_study/o0_checks/S6_DELIVERY_CHECK.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
