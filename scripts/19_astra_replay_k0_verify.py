#!/usr/bin/env python3
"""Astra replay K=0 verification from historical artefacts. NO inference, NO training.

Replaces the failed replay inside replay_k0.json (filter checkpoint=="best" never matched the
S6 summary, which only has `last`/`best_swapped`). Original replay_k0.json files are not touched.

Compares, per run x role x replica, Astra free macro RMSE (frame -> condition -> mean over
conditions) against historical sources for the ORIGINAL S5 best checkpoint:

  A. S6 per_frame_best_cov05_<role>_rep<r>.csv.gz   (all roles/replicas; also frame-level)
  B. S5 eval_history.json @ best epoch              (VAL: 3 replicas; seen: replica 0 only)
  C. S5 evals/epoch_<best>/<role>_per_frame.csv.gz  (VAL: 3 replicas; seen: replica 0 only)

A missing source is reported as MISSING, never as PASS. Overall PASS needs source A for all
24 (run, role, replica) cells within atol AND every available B/C cell within atol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = ("s5_sparse_unet_pon_seed0", "s5_sparse_unet_poff_seed0",
        "s5_adapted_glu_pon_seed0", "s5_adapted_glu_poff_seed0")
ROLES = ("seen_train_eval", "val_eval")
REPS = (0, 1, 2)
ATOL = 0.05


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def macro(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("sim_id")[col].mean().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra-root", required=True, help=".../level_structure_v1")
    ap.add_argument("--s6-root", required=True, help=".../official_thesis_s6_results")
    ap.add_argument("--s5-runs-root", required=True, help=".../official_thesis_s5_runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    astra, s6r, s5r = Path(a.astra_root), Path(a.s6_root), Path(a.s5_runs_root)

    cells, files = [], {}

    def reg(p: Path) -> str:
        files[str(p)] = sha(p)
        return str(p)

    for run in RUNS:
        prov = json.load(open(astra / run / "provenance.json"))
        s5sum = json.load(open(s5r / run / "summary.json"))
        hist = json.load(open(s5r / run / "eval_history.json"))
        reg(astra / run / "provenance.json"); reg(s5r / run / "summary.json"); reg(s5r / run / "eval_history.json")
        ep = int(prov["checkpoint_epoch"])
        epoch_ok = ep == int(s5sum["best_val_epoch"])
        h = [x for x in hist if int(x["epoch"]) == ep]
        hrow = h[0] if len(h) == 1 else None
        for role in ROLES:
            ast = pd.read_csv(astra / run / f"per_frame_{role}.csv.gz")
            reg(astra / run / f"per_frame_{role}.csv.gz")
            ast = ast[ast.treatment == "free"]
            for rep in REPS:
                ag = ast[ast.replica_id == rep]
                mine = macro(ag, "all_rmse_K")
                base = dict(run=run, role=role, replica=rep, checkpoint="best", epoch=ep,
                            epoch_matches_s5_best_val_epoch=epoch_ok, n_frames_astra=len(ag),
                            astra_macro_rmse_K=mine, atol_K=ATOL)
                # --- A: S6 per-frame best 5%
                pa = s6r / run / run / f"per_frame_best_cov05_{role}_rep{rep}.csv.gz"
                if pa.exists():
                    s6 = pd.read_csv(pa); reg(pa)
                    keys_ok = (set(s6.variant) == {"best"} and set(s6.mask_replica_id.astype(int)) == {rep}
                               and set(s6.role) == {role} and np.allclose(s6.site_fraction.astype(float), 0.05)
                               and set(s6.run_id) == {run})
                    m = ag.merge(s6[["sim_id", "frame_index", "rmse_K"]], on=["sim_id", "frame_index"], how="outer",
                                 indicator=True)
                    pairs_ok = bool((m["_merge"] == "both").all())
                    fmax = float((m["all_rmse_K"] - m["rmse_K"]).abs().max()) if pairs_ok else None
                    ref = macro(s6, "rmse_K")
                    cells.append({**base, "source": "A_S6_per_frame_best_cov05", "source_file": str(pa),
                                  "keys_checked": "variant=best,mask_replica_id,role,site_fraction=0.05,run_id,(sim_id,frame_index)",
                                  "keys_ok": bool(keys_ok and pairs_ok), "ref_macro_rmse_K": ref,
                                  "delta_K": mine - ref, "max_abs_frame_delta_K": fmax,
                                  "status": "PASS" if (keys_ok and pairs_ok and abs(mine - ref) <= ATOL
                                                       and fmax <= ATOL) else "FAIL"})
                else:
                    cells.append({**base, "source": "A_S6_per_frame_best_cov05", "source_file": str(pa),
                                  "status": "MISSING"})
                # --- B: S5 eval_history @ best epoch
                ref_b = None
                if hrow is not None:
                    if role == "val_eval":
                        v = [x for x in hrow["val_per_replica"] if int(x["replica_id"]) == rep]
                        ref_b = float(v[0]["mean_condition_rmse_K"]) if v else None
                    elif rep == 0:
                        ref_b = float(hrow["seen_mean_condition_rmse_K"])  # seen: replica-0 only
                if ref_b is None:
                    cells.append({**base, "source": "B_S5_eval_history", "status": "MISSING",
                                  "note": "no per-replica seen figure in eval_history (replica 0 only)" if role == "seen_train_eval"
                                  else "epoch/replica not found"})
                else:
                    cells.append({**base, "source": "B_S5_eval_history", "source_file": str(s5r / run / "eval_history.json"),
                                  "keys_checked": f"epoch={ep}," + ("val_per_replica.replica_id" if role == "val_eval" else "seen_mean_condition_rmse_K(rep0 only)"),
                                  "ref_macro_rmse_K": ref_b, "delta_K": mine - ref_b,
                                  "status": "PASS" if abs(mine - ref_b) <= ATOL else "FAIL"})
                # --- C: S5 evals/epoch_<best> per-frame
                pc = s5r / run / "evals" / f"epoch_{ep:04d}" / f"{role}_per_frame.csv.gz"
                ref_c = None
                if pc.exists():
                    c = pd.read_csv(pc); reg(pc)
                    cr = c[c.mask_replica_id.astype(int) == rep]
                    if len(cr):
                        ref_c = macro(cr, "rmse_K")
                if ref_c is None:
                    cells.append({**base, "source": "C_S5_evals_epoch_per_frame", "source_file": str(pc),
                                  "status": "MISSING",
                                  "note": f"no rows with mask_replica_id={rep} in this file" if pc.exists() else "file missing"})
                else:
                    cells.append({**base, "source": "C_S5_evals_epoch_per_frame", "source_file": str(pc),
                                  "keys_checked": f"mask_replica_id={rep},(sim_id)",
                                  "ref_macro_rmse_K": ref_c, "delta_K": mine - ref_c,
                                  "status": "PASS" if abs(mine - ref_c) <= ATOL else "FAIL"})

    df = pd.DataFrame(cells)
    a_cells = df[df.source == "A_S6_per_frame_best_cov05"]
    others = df[(df.source != "A_S6_per_frame_best_cov05") & (df.status != "MISSING")]
    summary = {
        "atol_K": ATOL,
        "cells_A_expected": len(RUNS) * len(ROLES) * len(REPS),
        "cells_A_pass": int((a_cells.status == "PASS").sum()),
        "cells_BC_available": int(len(others)), "cells_BC_pass": int((others.status == "PASS").sum()),
        "cells_BC_missing": df[(df.source != "A_S6_per_frame_best_cov05") & (df.status == "MISSING")][
            ["run", "role", "replica", "source"]].to_dict("records"),
        "epoch_matches_all": bool(df.epoch_matches_s5_best_val_epoch.all()),
        "max_abs_delta_K": float(df.delta_K.abs().max()),
        "max_abs_frame_delta_K": float(a_cells.max_abs_frame_delta_K.max()),
        "overall_pass": bool((a_cells.status == "PASS").all() and len(a_cells) == 24
                             and (others.status == "PASS").all()
                             and df.epoch_matches_s5_best_val_epoch.all()),
        "caveat": "S6 per_frame_best files carry no epoch column: best-checkpoint identity rests on the value "
                  "match with S5 eval_history/evals at the best epoch (sources B/C) plus provenance epoch == "
                  "S5 summary best_val_epoch. Seen replicas 1/2 are covered by source A only.",
        "source_sha256": files,
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "replay_k0_verification_cells.csv", index=False)
    (out / "replay_k0_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, indent=2))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
