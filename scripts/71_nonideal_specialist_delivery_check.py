#!/usr/bin/env python3
"""Read-only delivery check of the 30-specialist numeric package. No training, no inference, no GPU.

Re-derives from the frozen artefacts (freeze.json, 60 x gate.json / COMPLETE.json / provenance.json,
per-frame CSVs, aggregated tables, manifest) what the aggregation reported:

* 30 valid runs, expected ids, terminal states, SHAs of training and post-processing;
* 60 best/last evaluations: gate.json all_pass, COMPLETE.json ok with 6174 rows, weights hash equal to freeze,
  evaluation SHA, specification hash, observation-matrix hash and cache hashes identical across all 60;
* 720 rows of tables/per_run_scenario.csv: complete, unique keys, finite, gain/delta/clean-cost arithmetic,
  baseline columns constant per (family, seed, cell, role) across training regimes and checkpoints;
* RMSE of every aggregated row re-reduced from the per-frame CSVs (mean over frames, then conditions, then
  mask replicas) and compared with the table;
* hashes of the package files against package_manifest.json, and of the baseline per-frame CSV.

Writes <out>/delivery_check.json and delivery_check.md. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REGIMES = ["N1", "N2", "G1", "G1-eq", "N1G1"]
CELLS = ["R0", "N1", "N2", "G1", "G1-eq", "N1G1"]
ROLES = ["seen_train_eval", "val_eval"]
CKPTS = ["best", "last"]
FAMS = {"glu": "adapted_glu", "unet": "sparse_unet"}
N_ROWS_EVAL = 6174
TOL_ROW_K = 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_ids() -> list[str]:
    out = []
    for fam in FAMS:
        for reg in REGIMES:
            for s in (0, 1, 2):
                out.append(f"NI-{fam}-on-{reg.replace('-', '')}-s{s}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True, help="campaign package root (freeze.json, eval/, package/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--training-sha", required=True)
    ap.add_argument("--post-sha", required=True)
    ap.add_argument("--baseline-per-frame", type=Path, required=True)
    ap.add_argument("--baseline-sha256", required=True)
    ap.add_argument("--skip-rereduce", action="store_true")
    a = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    dest, ev, pkg = a.dest, a.dest / "eval", a.dest / "package"
    freeze = json.loads((dest / "freeze.json").read_text())
    ids = run_ids()
    chk("freeze: 30 expected run ids", sorted(freeze["runs"]) == sorted(ids), f"{len(freeze['runs'])} runs")
    chk("freeze: n_runs/n_valid = 30, problems empty",
        freeze["n_runs"] == 30 and freeze["n_valid"] == 30 and not freeze["problems"],
        f"n_runs={freeze['n_runs']} n_valid={freeze['n_valid']} problems={freeze['problems']}")
    chk("freeze: training SHA", freeze.get("training_git_sha", freeze.get("git_sha")) == a.training_sha, a.training_sha)
    chk("freeze: post-processing SHA", freeze["post_processing_git_sha"] == a.post_sha, a.post_sha)
    states = defaultdict(int)
    for rid in ids:
        r = freeze["runs"][rid]
        states[r["state"]] += 1
        chk(f"{rid}: valid, no problems, terminal state", r["valid"] and not r["problems"] and r["state"] in ("plateau", "budget_exhausted"),
            r["state"])
    chk("freeze: 30 plateau, 0 budget_exhausted", states.get("plateau") == 30 and freeze["n_budget_exhausted_valid"] == 0, dict(states).__repr__())
    ext = [rid for rid in ids if freeze["runs"][rid]["ledger"].get("extension")]
    n_resumes = {rid: freeze["runs"][rid]["ledger"]["attempts"] for rid in ids}

    # 60 evaluations
    mats, caches, specs = set(), set(), set()
    n_gate = n_complete = n_prov = 0
    max_gate = 0.0
    for rid in ids:
        for ck in CKPTS:
            d = ev / rid / ck
            try:
                gate = json.loads((d / "gate.json").read_text())
                comp = json.loads((d / "COMPLETE.json").read_text())
                prov = json.loads((d / "provenance.json").read_text())
            except Exception as exc:  # noqa: BLE001
                chk(f"{rid}/{ck}: artefacts readable", False, repr(exc))
                continue
            ok_g = bool(gate["all_pass"]) and gate["n_within_tolerance"] == gate["n_comparisons"] == gate["n_expected"]
            n_gate += ok_g
            max_gate = max(max_gate, float(gate["max_abs_delta_K"]))
            ok_c = comp.get("ok") is True and comp.get("n_rows") == N_ROWS_EVAL and comp.get("gate_all_pass") is True
            n_complete += ok_c
            fz = freeze["runs"][rid][ck]
            ok_p = (prov["checkpoint_weights_sha256"] == fz["weights_sha256"] and prov["checkpoint_file_sha256"] == fz["file_sha256"]
                    and prov["eval_git_sha"] == a.post_sha and prov["spec_sha256"] == freeze["degradation_spec_sha256"]
                    and prov["weights_changed"] is False and prov["n_rows"] == N_ROWS_EVAL and prov["checkpoint_epoch"] == fz["epoch"]
                    and prov["train_regime"] == freeze["runs"][rid]["regime"])
            n_prov += ok_p
            mats.add(prov["matrix_sha256"])
            caches.add(json.dumps(prov["consumed"], sort_keys=True))
            specs.add(prov["spec_sha256"])
            if not (ok_g and ok_c and ok_p):
                chk(f"{rid}/{ck}: gate/complete/provenance", False, f"gate={ok_g} complete={ok_c} prov={ok_p}")
    chk("60/60 gate.json all_pass (5/5 comparisons within 0.05 K)", n_gate == 60, f"{n_gate}/60, max |delta| {max_gate:.4f} K")
    chk("60/60 COMPLETE.json ok, 6174 rows, gate_all_pass", n_complete == 60, f"{n_complete}/60")
    chk("60/60 provenance = freeze (weights, file, epoch, regime, spec, eval SHA, weights unchanged)", n_prov == 60, f"{n_prov}/60")
    chk("one observation-matrix hash, one cache identity, one spec hash across 60 evaluations",
        len(mats) == 1 and len(caches) == 1 and len(specs) == 1, f"{len(mats)} matrix, {len(caches)} cache, {len(specs)} spec")

    # aggregated table
    rows = list(csv.DictReader(open(pkg / "tables" / "per_run_scenario.csv")))
    chk("per_run_scenario.csv has 720 rows", len(rows) == 720, str(len(rows)))
    keys = [(r["run_id"], r["checkpoint_kind"], r["cell"], r["role"]) for r in rows]
    chk("keys unique", len(set(keys)) == len(keys), f"{len(set(keys))} unique")
    expected = {(rid, ck, c, ro) for rid in ids for ck in CKPTS for c in CELLS for ro in ROLES}
    chk("keys = 30 runs x 2 checkpoints x 6 cells x 2 roles", set(keys) == expected)
    num = ["rmse_K", "base_rmse_K", "gain_vs_baseline_K", "delta_vs_baseline_K", "clean_cost_K", "base_R0_rmse_K", "spec_R0_rmse_K"]
    finite = all(math.isfinite(float(r[k])) for r in rows for k in num)
    chk("finite values", finite)
    arith = True
    worst = 0.0
    base_ref: dict = {}
    base_const = True
    for r in rows:
        rm, b = float(r["rmse_K"]), float(r["base_rmse_K"])
        g, dl, cc = float(r["gain_vs_baseline_K"]), float(r["delta_vs_baseline_K"]), float(r["clean_cost_K"])
        worst = max(worst, abs(g - (b - rm)), abs(dl - (rm - b)), abs(cc - (float(r["spec_R0_rmse_K"]) - float(r["base_R0_rmse_K"]))))
        k = (r["family"], r["model_seed"], r["cell"], r["role"])
        base_ref.setdefault(k, set()).add(round(b, 9))
        if r["cell"] == "R0":
            arith &= abs(rm - float(r["spec_R0_rmse_K"])) < 1e-9 and abs(b - float(r["base_R0_rmse_K"])) < 1e-9
    base_const = all(len(v) == 1 for v in base_ref.values())
    chk("gain = base - spec, delta = spec - base, clean_cost = spec_R0 - base_R0 (<1e-9 K)", worst < 1e-9 and arith, f"max {worst:.2e}")
    chk("baseline RMSE identical across regimes and checkpoints for each (family, seed, cell, role)", base_const,
        f"{len(base_ref)} baseline groups (expect 2 families x 3 seeds x 6 cells x 2 roles = 72)")
    chk("72 baseline groups", len(base_ref) == 72, str(len(base_ref)))

    # package manifest
    man = json.loads((pkg / "package_manifest.json").read_text())
    bad = [f for f, h in man["files"].items() if sha256(pkg / f) != h]
    chk("package_manifest hashes match files", not bad, f"{len(man['files'])} files; mismatch: {bad}")
    chk("package_manifest freeze hash = sha256(freeze.json)", man["freeze_sha256"] == sha256(dest / "freeze.json"))
    bsha = sha256(a.baseline_per_frame)
    chk("baseline per-frame CSV sha256 = expected = package manifest", bsha == a.baseline_sha256 == man["baselines"], bsha)
    fi = json.loads((pkg / "fields_index.json").read_text())
    chk("fields_index.json readable", isinstance(fi, dict), f"keys {list(fi)[:4]}")

    # re-reduce RMSE from per-frame CSVs
    rr = {"n": 0, "max_abs_diff": 0.0, "ok": True}
    if not a.skip_rereduce:
        tab = {(r["run_id"], r["checkpoint_kind"], r["cell"], r["role"]): float(r["rmse_K"]) for r in rows}
        for rid in ids:
            for ck in CKPTS:
                for c in CELLS:
                    for ro in ROLES:
                        reps = []
                        for k in (0, 1, 2):
                            p = ev / rid / ck / "per_frame" / c / ro / f"rep{k}.csv"
                            per = defaultdict(list)
                            with open(p) as f:
                                for row in csv.DictReader(f):
                                    per[row["sim_id"]].append(float(row["rmse_K"]))
                            reps.append(sum(sum(v) / len(v) for v in per.values()) / len(per))
                        val = sum(reps) / 3
                        d = abs(val - tab[(rid, ck, c, ro)])
                        rr["max_abs_diff"] = max(rr["max_abs_diff"], d)
                        rr["n"] += 1
        rr["ok"] = rr["n"] == 720 and rr["max_abs_diff"] < TOL_ROW_K
        chk("720 aggregated RMSE = re-reduction of per-frame CSVs (frames -> conditions -> replicas)", rr["ok"],
            f"{rr['n']} rows, max |diff| {rr['max_abs_diff']:.2e} K (tol {TOL_ROW_K})")

    all_ok = all(ok for _, ok, _ in checks)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "nonideal_specialist_delivery_check_v1",
        "all_ok": all_ok,
        "training_git_sha": a.training_sha,
        "post_processing_git_sha": a.post_sha,
        "runs_with_extension": ext,
        "attempts_per_run": n_resumes,
        "states": dict(states),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
    }
    (a.out / "delivery_check.json").write_text(json.dumps(report, indent=1))
    md = ["# Delivery check of the 30-specialist numeric package", "",
          "Read-only. No inference. Failed checks are listed first.", "", f"**Overall: {'OK' if all_ok else 'FAILED'}**", ""]
    md += ["| check | result | detail |", "|---|---|---|"]
    for n, ok, d in sorted(checks, key=lambda t: t[1]):
        if ok and n.startswith("NI-") or (ok and ": valid, no problems" in n):
            continue
        md.append(f"| {n} | {'ok' if ok else 'FAIL'} | {d} |")
    md.append(f"\nThe 30 per-run `valid/terminal state` checks are included in the JSON ({sum(1 for n,_,_ in checks if ': valid, no problems' in n)} entries).")
    (a.out / "delivery_check.md").write_text("\n".join(md) + "\n")
    print("DELIVERY_CHECK_OK" if all_ok else "DELIVERY_CHECK_FAILED")
    for n, ok, d in checks:
        if not (ok and ": valid, no problems" in n):
            print(("ok  " if ok else "FAIL"), n, "|", d)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
