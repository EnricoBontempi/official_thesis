#!/usr/bin/env python3
"""Evaluate ONE frozen specialist (best and last) on all six degradation cells (Block II protocol, unchanged).

Reuses the S7/Block II evaluation code. Gate: R0 macro within 0.05 K of the training-loop metric at the evaluated epoch
(VAL replicas 0/1/2 and seen replica 0). A failed gate keeps the outputs as diagnostic and does not run the degraded cells.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.nonideal_observations import file_sha256, load_degradation_spec, load_velocity_scale  # noqa: E402
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402
from combustion_recon.evaluation.s7_eval_plan import gate_rows, loop_reference, summarize_gate  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] specialist-eval {m}", flush=True)


def _wj(p: Path, o: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/nonideal_specialist_matrix.json")
    ap.add_argument("--s7-eval-matrix", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json")
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/nonideal/degradation_spec_v1_12models.yaml")
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=("full", "smoke"), default="full")
    ap.add_argument("--smoke-frames", type=int, default=8)
    ap.add_argument("--train-root", type=Path, default=None, help="smoke only: directory holding the run (default: matrix out_root)")
    a = ap.parse_args()
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.out_root.name.lower()):
        raise SystemExit("smoke mode needs an out-root named '*smoke*'; full must not")
    import torch

    if a.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    n60 = _load("n60_for_spec", "scripts/60_nonideal_eval.py")
    s7e = n60._load("s7e_for_spec", "scripts/38_s7_eval.py")
    s6 = s7e._s6()
    freeze = json.loads(a.freeze.read_text())
    fr = freeze["runs"][a.run_id]
    if not fr.get("valid"):
        raise SystemExit(f"{a.run_id} is not a valid frozen run: {fr.get('problems')}")
    matrix_run = next(r for r in json.loads(a.matrix.read_text())["new_runs"] if r["run_id"] == a.run_id)
    spec = load_degradation_spec(a.spec)
    spec_sha = file_sha256(a.spec)
    if spec_sha != freeze["degradation_spec_sha256"]:
        raise SystemExit("evaluation spec differs from the training spec")
    smatrix = s7e.load_matrix(a.s7_eval_matrix)
    if a.train_root is not None and not smoke:
        raise SystemExit("--train-root is for smoke only")
    run_dir = (a.train_root or Path(json.loads(a.matrix.read_text())["out_root"])) / a.run_id
    ident = json.loads((run_dir / "run_identity.json").read_text())
    entry = {"run_id": a.run_id, "pool_id": "C149", "family": fr["family"], "p_conditioning": True, "training_dir": str(run_dir),
             "training_git_shas": [ident["git_sha"]], "model_seed": fr["seed"]}
    family = fr["family"]
    scale = load_velocity_scale(smatrix["shared_inputs"]["stats"], expected_sha256=smatrix["shared_inputs"]["stats_sha256"])
    consumed = s7e._verify_inputs(entry, smatrix)
    normalizer = Month2FitOnlyNormalizer(smatrix["shared_inputs"]["stats"])
    smoke_frames = int(a.smoke_frames) if smoke else None
    datasets, expected = n60._build_datasets(s7e, entry, smatrix, consumed, normalizer, smoke_frames, ne.EVAL_ROLES)
    out_run = a.out_root / a.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    model, _ = s6._load_model(family, consumed["cache_root"], device)
    query = make_synthetic_query_grid(datasets["val_eval"].x, datasets["val_eval"].y).to(device) if family == "adapted_glu" else None
    keep = {role: n60._figure_frames(datasets[role], role) for role in ne.EVAL_ROLES}
    results: dict[str, Any] = {}
    for kind in ("best", "last"):
        info = fr[kind]
        kdir = out_run / kind
        if kind == "last" and fr.get("last_same_as_best") and not smoke:
            _wj(kdir / "SAME_AS_BEST.json", {"reason": "identical weights (same sha256): best rows are the last rows", "weights_sha256": info["weights_sha256"]})
            results[kind] = "same_as_best"
            continue
        if file_sha256(Path(info["path"])) != info["file_sha256"]:
            raise SystemExit(f"{kind}.pt changed since the freeze")
        s6.load_sparse_scientific_checkpoint(Path(info["path"]), model=model, require_resumable=True, restore_rng=False)
        model.to(device).eval()

        def forward(batch, model=model):
            return s6._forward(model, family, batch, p_conditioning=True, query_coords_phys=query, device=device)

        tags = {"run_id": a.run_id, "family": family, "variant": "p_on", "model_seed": int(fr["seed"]), "checkpoint_kind": kind,
                "checkpoint_epoch": int(info["epoch"]), "checkpoint_file_sha256": info["file_sha256"],
                "checkpoint_weights_sha256": info["weights_sha256"], "train_regime": fr["regime"], "spec_sha256": spec_sha,
                "eval_git_sha": a.eval_git_sha, "experiment_id": "nonideal_specialist_eval_v1", "scientific": not smoke}
        p_check = n60._p_checks(s6, model, family, datasets["seen_train_eval"], device, query, True, torch)
        rows: list[dict[str, Any]] = []
        costs = []

        def run_cell(cell, role, rep):
            res = ne.evaluate_cell(ds=datasets[role], role=role, replica_id=rep, cell=cell, spec=spec, scale=scale, family=family,
                                   p_conditioning=True, forward_fn=forward, normalizer=normalizer, run_tags=tags,
                                   keep_fields_for=keep[role] if kind == "best" else None)
            if res["n_frames"] != expected[role]:
                raise SystemExit(f"{cell.cell_id} {role} rep{rep}: {res['n_frames']} != {expected[role]}")
            d = kdir / "per_frame" / cell.cell_id / role
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(res["rows"]).to_csv(d / f"rep{rep}.csv", index=False)
            rows.extend(res["rows"])
            costs.append({"cell": cell.cell_id, "role": role, "replica": rep, "n": res["n_frames"], "wall_s": res["wall_s"]})
            for (sim, frm), f in res["kept"].items():
                for k, v in f.items():
                    fields[f"{cell.cell_id}|{role}|{sim}|{frm}|{k}"] = v

        fields: dict[str, np.ndarray] = {}
        for role in ne.EVAL_ROLES:
            for rep in ne.REPLICAS:
                run_cell(spec.cells["R0"], role, rep)
        r0 = pd.DataFrame(rows)
        macro = {(role, rep): ne.macro_rmse(r0[(r0["role"] == role) & (r0["replica"] == rep)]) for role in ne.EVAL_ROLES for rep in ne.REPLICAS}
        rec = fr["loop_at_best"] if kind == "best" else fr["loop_at_last"]
        ref = loop_reference([rec], int(info["epoch"]))
        grows = gate_rows(checkpoint=kind, epoch=int(info["epoch"]), o5_macro={("seen_train_eval", 0): macro[("seen_train_eval", 0)],
                          **{("val_eval", r): macro[("val_eval", r)] for r in ne.REPLICAS}}, reference=ref)
        gate = {**summarize_gate(grows, n_checkpoints=1), "rows": grows, "reference": "training-loop metric at the evaluated epoch",
                "p_check": p_check}
        _wj(kdir / "gate.json", gate)
        _log(f"{kind}: gate all_pass={gate['all_pass']} max|d|={gate['max_abs_delta_K']}")
        if not gate["all_pass"] and not smoke:
            _wj(kdir / "GATE_FAILED.json", {"run_id": a.run_id, "kind": kind, "gate": gate})
            results[kind] = "gate_failed"
            continue
        for cid in ne.CELL_ORDER[1:]:
            for role in ne.EVAL_ROLES:
                for rep in ne.REPLICAS:
                    run_cell(spec.cells[cid], role, rep)
        if fields:
            np.savez_compressed(kdir / "fields_figure_frames.npz", **fields)
        pd.DataFrame(costs).to_csv(kdir / "setting_costs.csv", index=False)
        frames = pd.concat([pd.read_csv(p) for p in sorted((kdir / "per_frame").rglob("rep*.csv"))], ignore_index=True)
        if len(frames) != len(rows) or frames.duplicated(ne.KEY_COLS).any():
            raise SystemExit("readback mismatch")
        if not np.isfinite(frames[["rmse_K", "mae_K", "bias_K"]].to_numpy(float)).all():
            raise SystemExit("non-finite metrics")
        if not smoke:
            ne.check_run_frame_table(frames, family=family, expect_diag=False)
        _wj(kdir / "provenance.json", {**tags, "consumed": {k: str(v) for k, v in consumed.items()}, "spec_sha256": spec_sha,
                                       "matrix_sha256": file_sha256(a.matrix), "n_rows": len(frames), "weights_changed": False,
                                       "total_wall_s": float(sum(c["wall_s"] for c in costs)),
                                       "max_memory_bytes": int(torch.cuda.max_memory_allocated())})
        _wj(kdir / ("SMOKE_OK.json" if smoke else "COMPLETE.json"), {"ok": True, "n_rows": len(frames), "gate_all_pass": gate["all_pass"]})
        results[kind] = "complete"
        _log(f"{kind}: done {len(frames)} rows")
    _wj(out_run / "RUN_DONE.json", {"run_id": a.run_id, "results": results, "regime": fr["regime"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
