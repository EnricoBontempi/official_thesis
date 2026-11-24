#!/usr/bin/env python3
"""Block II task: evaluate ONE frozen best checkpoint on the six degradation cells.

Reuses the S7 evaluation path (input verification, checkpoint identity, datasets, forward) and the
tested operators of Block I. No weights, architectures or neighbour rules are changed. TEST is refused.

Modes
  full      all 6 cells x 3 replicas x (seen220 + VAL123), preceded by the R0 replay gate. COMPLETE.json only
            if the gate passes and the read-back checks pass; otherwise GATE_FAILED.json (+ CAMPAIGN_HALT.json).
  smoke     same code on a few frames per role, all six cells. Never COMPLETE. Output dir must contain 'smoke'.
  selftest  CPU, random-initialised model, seen_train_eval frames only (no checkpoint, no VAL). Not scientific.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.nonideal_observations import (  # noqa: E402
    file_sha256, load_degradation_spec, load_velocity_scale,
)
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402

EXPERIMENT_ID = "nonideal_block2_v1"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] nonideal {msg}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _figure_frames(ds, role: str) -> dict[tuple[str, int], bool]:
    """Rule fixed in the spec: VAL: all conditions; seen: first and last sorted sim_id; lowest frame_index each."""
    metas = [ds.frames[i] for i in ds.indices]
    conds = sorted({m["sim_id"] for m in metas})
    chosen = conds if role == "val_eval" else sorted({conds[0], conds[-1]})
    out = {}
    for c in chosen:
        fr = min(int(m["frame_index"]) for m in metas if m["sim_id"] == c)
        out[(c, fr)] = True
    return out


def _spec_entry(spec, run_id: str) -> dict[str, Any]:
    hits = [r for r in spec.raw["models"]["runs"] if r["run_id"] == run_id]
    if len(hits) != 1:
        raise SystemExit(f"spec does not contain exactly one {run_id}")
    return hits[0]


def _check_entry_matches_spec(entry: dict[str, Any], srun: dict[str, Any]) -> None:
    best = entry["checkpoints"]["best"]
    pairs = {
        "family": (entry["family"], srun["family"]),
        "p_conditioning": (bool(entry["p_conditioning"]), bool(srun["p_conditioning"])),
        "model_seed": (int(entry["model_seed"]), int(srun["model_seed"])),
        "epoch": (int(best["epoch"]), int(srun["best_epoch"])),
        "file_sha256": (best["file_sha256"], srun["best_file_sha256"]),
        "weights_sha256": (best["weights_sha256"], srun["best_weights_sha256"]),
    }
    bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"S7 matrix and spec disagree for {entry['run_id']}: {bad}")


def _inspect_best(s7e, entry, torch) -> dict[str, Any]:
    """Identity of the single best checkpoint (file sha, epoch, identity block, weights sha)."""
    spec = entry["checkpoints"]["best"]
    path = Path(entry["training_dir"]) / "checkpoints" / "best.pt"
    file_sha = s7e.sha256_file(path)
    if file_sha != spec["file_sha256"]:
        raise SystemExit(f"best file sha mismatch for {entry['run_id']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
    if epoch != int(spec["epoch"]):
        raise SystemExit(f"best epoch {epoch} != matrix {spec['epoch']}")
    ident = (payload.get("extra") or {}).get("run_identity") or {}
    cfg = payload.get("epoch_config") or {}
    if ident.get("run_id") != entry["run_id"] or ident.get("family") != entry["family"]:
        raise SystemExit("checkpoint identity (run/family) does not match the matrix")
    if bool(ident.get("p_conditioning")) != bool(entry["p_conditioning"]):
        raise SystemExit("checkpoint p_conditioning does not match the matrix")
    if int(cfg.get("seed", -1)) != int(entry["model_seed"]):
        raise SystemExit("checkpoint model seed does not match the matrix")
    digest = s7e.weights_sha256(payload["model_state_dict"])
    if digest != spec["weights_sha256"]:
        raise SystemExit("best weights sha mismatch")
    del payload
    return {"path": path, "epoch": epoch, "file_sha256": file_sha, "weights_sha256": digest}


def _build_datasets(s7e, entry, matrix, consumed, normalizer, smoke_frames, roles):
    cache_root = consumed["cache_root"]
    kwargs = cache_validation_kwargs(str(entry["pool_id"]))
    datasets, expected = {}, {}
    for role in roles:
        indices = None
        if smoke_frames is not None:
            frames = json.loads((cache_root / role / "frames.json").read_text())
            indices = s7e._smoke_positions(frames, list(range(len(frames))), smoke_frames, False)
        datasets[role] = SparseUNetCacheDataset(
            role, cache_root=cache_root, indices=indices, mask_mode="evaluation", site_fraction=0.05,
            eval_mask_seed=int(matrix["shared_inputs"]["eval_mask_seed"]), layout_id=str(matrix["shared_inputs"]["layout_id"]),
            replica_id=0, p_conditioning=bool(entry["p_conditioning"]), normalizer=normalizer,
            eval_mask_payload_root=matrix["shared_inputs"]["mask_root"], require_eval_mask_payload=True, **kwargs,
        )
        expected[role] = len(datasets[role]) if smoke_frames is not None else ne.ROLE_N_FRAMES[role]
        if len(datasets[role]) != expected[role]:
            raise SystemExit(f"{role}: {len(datasets[role])} frames, expected {expected[role]}")
    return datasets, expected


def _p_checks(s6, model, family, ds, device, query, p_conditioning, torch) -> dict[str, Any]:
    """p_off: the supplied p must not matter and the p channels must be exactly 0. p_on: p must matter."""
    from combustion_recon.training.sparse_dataset import collate_sparse_batch
    ds.replica_id, ds.site_fraction = 0, 0.05
    batch = collate_sparse_batch([ds[0], ds[1]])
    s6._attach_p(batch, ds)
    base = s6._forward(model, family, batch, p_conditioning=p_conditioning, query_coords_phys=query, device=device)
    res: dict[str, Any] = {"p_conditioning": bool(p_conditioning)}
    other = dict(batch)
    other["p_norm_tensor"] = torch.flip(batch["p_norm_tensor"], dims=[0]) + 0.5
    if family == "sparse_unet":
        x = batch["input_tensor"].clone()
        res["p_channels_exact_zero"] = bool(torch.all(x[:, -2:] == 0)) if not p_conditioning else None
        x2 = x.clone()
        x2[:, -2:] += 0.5
        other["input_tensor"] = x2
        if not p_conditioning:  # the original p_off path never feeds p: perturbing the channels must be impossible at eval
            res["note"] = "p_off input channels are zero by construction; perturbed-channel forward shown for information"
    alt = s6._forward(model, family, other, p_conditioning=p_conditioning, query_coords_phys=query, device=device)
    diff = float(np.max(np.abs(alt - base)))
    res["max_abs_change_normalized"] = diff
    res["ok"] = bool(diff == 0.0) if (not p_conditioning and family == "adapted_glu") else (bool(diff > 0.0) if p_conditioning else True)
    if not p_conditioning and family == "sparse_unet":
        res["ok"] = bool(res["p_channels_exact_zero"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/nonideal/degradation_spec_v1_12models.yaml")
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json")
    ap.add_argument("--s7-reference", type=Path,
                    default=ROOT / "results/month2/s7_c149_comparison/package/tables/global_by_replica.csv")
    ap.add_argument("--s7-per-frame-root", type=Path, default=Path("/home/bontempi/official_thesis_s7_c149_eval"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=("full", "smoke", "selftest"), default="full")
    ap.add_argument("--smoke-frames", type=int, default=8)
    ap.add_argument("--selftest-data-root", type=Path, default=None, help="selftest only: local data/ root")
    args = ap.parse_args()

    scientific = args.mode == "full"
    if (args.mode != "full") != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("smoke/selftest need an out-root named '*smoke*'; full must not")
    import torch

    selftest = args.mode == "selftest"
    if not selftest and (args.device != "cuda" or not torch.cuda.is_available()):
        raise SystemExit("CUDA required (selftest is the only CPU mode)")
    if not selftest:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cpu" if selftest else "cuda")

    spec = load_degradation_spec(args.spec)
    spec_sha = file_sha256(args.spec)
    base_sha = file_sha256(ROOT / spec.raw["base_spec"]["path"])
    if base_sha != spec.raw["base_spec"]["sha256"]:
        raise SystemExit("base spec hash mismatch")
    srun = _spec_entry(spec, args.run_id)
    family = srun["family"]
    p_cond = bool(srun["p_conditioning"])
    s7e = _load("s7e_nonideal", "scripts/38_s7_eval.py")
    s6 = s7e._s6()
    matrix = s7e.load_matrix(args.matrix)
    if selftest:
        if args.selftest_data_root is None:
            raise SystemExit("selftest needs --selftest-data-root")
        sh = matrix["shared_inputs"]
        sh["c149_cache_root"] = str(args.selftest_data_root / "processed/month2_dense_unet_cache_c149")
        sh["mask_root"] = str(args.selftest_data_root / "processed/month2_sparse_eval_masks")
        sh["stats"] = str(args.selftest_data_root / "manifests/train_norm_stats_month2_fit.json")
    entry = s7e._run_entry(matrix, args.run_id)
    _check_entry_matches_spec(entry, srun)
    scale = load_velocity_scale(matrix["shared_inputs"]["stats"], expected_sha256=matrix["shared_inputs"]["stats_sha256"])
    z = spec.raw["provenance"]["normalizer"]["velocity_zscore_ms"]
    if (z["Velocity_i"]["std"], z["Velocity_j"]["std"]) != scale.std:
        raise SystemExit("spec z-score differs from the frozen stats")

    out_run = args.out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    halt = args.out_root / "CAMPAIGN_HALT.json"
    if halt.exists():
        raise SystemExit("CAMPAIGN_HALT.json present: not starting")
    _log(f"start run={args.run_id} family={family} p_on={p_cond} mode={args.mode} sha={args.eval_git_sha}")

    roles = ("seen_train_eval",) if selftest else ne.EVAL_ROLES
    smoke_frames = None if args.mode == "full" else int(args.smoke_frames)
    if selftest:
        consumed = {"cache_root": Path(matrix["shared_inputs"]["c149_cache_root"]), "cache_aggregate_sha256": "selftest",
                    "cache_id": "selftest", "stats_sha256": scale.stats_sha256,
                    "eval_mask_manifest_file_sha256": "selftest", "eval_mask_aggregate_payload_sha256": "selftest"}
        inspected = {"epoch": -1, "file_sha256": "selftest", "weights_sha256": "selftest", "path": None}
    else:
        _log("phase identity: begin")
        consumed = s7e._verify_inputs(entry, matrix)
        inspected = _inspect_best(s7e, entry, torch)
        _log("phase identity: done")

    normalizer = Month2FitOnlyNormalizer(matrix["shared_inputs"]["stats"])
    datasets, expected = _build_datasets(s7e, entry, matrix, consumed, normalizer, smoke_frames, roles)
    model, _ = s6._load_model(family, consumed["cache_root"], device)
    if not selftest:
        payload = s6.load_sparse_scientific_checkpoint(inspected["path"], model=model, require_resumable=True, restore_rng=False)
        if int((payload.get("epoch_state") or {}).get("epoch", -1)) != int(srun["best_epoch"]):
            raise SystemExit("loaded epoch drifted")
    model.to(device).eval()
    ds0 = datasets[roles[0]]
    query = None
    if family == "adapted_glu":
        query = make_synthetic_query_grid(ds0.x, ds0.y).to(device)

    def forward(batch):
        return s6._forward(model, family, batch, p_conditioning=p_cond, query_coords_phys=query, device=device)

    run_tags = {
        "run_id": args.run_id, "family": family, "variant": "p_on" if p_cond else "p_off", "model_seed": int(srun["model_seed"]),
        "checkpoint_epoch": int(srun["best_epoch"]), "checkpoint_file_sha256": srun["best_file_sha256"],
        "checkpoint_weights_sha256": srun["best_weights_sha256"], "spec_sha256": spec_sha, "base_spec_sha256": base_sha,
        "eval_git_sha": args.eval_git_sha, "experiment_id": EXPERIMENT_ID, "scientific": bool(scientific),
    }
    p_check = _p_checks(s6, model, family, datasets[roles[0]], device, query, p_cond, torch)
    _log(f"p check: {json.dumps(p_check)}")
    if not p_check["ok"]:
        _write_json(out_run / "SMOKE_FAILED.json" if not scientific else out_run / "GATE_FAILED.json", {"p_check": p_check})
        raise SystemExit("p check failed")

    keep = {role: _figure_frames(datasets[role], role) for role in roles}
    all_rows: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {}

    def run_cell(cell, role, rep, *, diag=False):
        res = ne.evaluate_cell(
            ds=datasets[role], role=role, replica_id=rep, cell=cell, spec=spec, scale=scale, family=family,
            p_conditioning=p_cond, forward_fn=forward, normalizer=normalizer, run_tags=run_tags,
            diagnostic_s7_batching=diag, keep_fields_for=keep[role] if not diag else None,
        )
        if res["n_frames"] != expected[role]:
            raise SystemExit(f"{cell.cell_id} {role} rep{rep}: {res['n_frames']} frames != {expected[role]}")
        name = ne.DIAG_CELL if diag else cell.cell_id
        (out_run / "per_frame" / name / role).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res["rows"]).to_csv(out_run / "per_frame" / name / role / f"rep{rep}.csv", index=False)
        all_rows.extend(res["rows"])
        costs.append({"cell": name, "role": role, "replica": rep, "n": res["n_frames"], "wall_s": res["wall_s"]})
        for (sim, fr), f in res["kept"].items():
            for k, v in f.items():
                fields[f"{name}|{role}|{sim}|{fr}|{k}"] = v
        _log(f"cell {name} {role} rep{rep} n={res['n_frames']} wall={res['wall_s']:.1f}s")

    # ---- R0 replay and gate (before any degraded cell)
    r0 = spec.cells["R0"]
    for role in roles:
        for rep in ne.REPLICAS:
            run_cell(r0, role, rep)
    r0_df = pd.DataFrame(all_rows)
    reference = pd.read_csv(args.s7_reference)
    gate = ne.gate_against_reference(r0_df, reference, args.run_id)
    per_frame_diag = {}
    for role in roles:
        for rep in ne.REPLICAS:
            f = args.s7_per_frame_root / args.run_id / "per_frame" / "best" / "cov0.05" / role / f"rep{rep}.csv"
            if f.exists():
                disc = ne.per_frame_discrepancy(r0_df, pd.read_csv(f), role=role, replica=rep)
                disc.to_csv(out_run / f"r0_vs_s7_per_frame_{role}_rep{rep}.csv", index=False)
                d = disc["delta_K"].abs()
                per_frame_diag[f"{role}/rep{rep}"] = {"n_matched": int(d.notna().sum()), "max_abs_K": float(d.max()), "mean_abs_K": float(d.mean()),
                                                       "unmatched": int((disc["_merge"] != "both").sum())}
            else:
                per_frame_diag[f"{role}/rep{rep}"] = "s7 per-frame csv not reachable"
    gate["per_frame_vs_s7"] = per_frame_diag
    if family == "adapted_glu":
        for role in roles:
            for rep in ne.REPLICAS:
                run_cell(r0, role, rep, diag=True)
        d = pd.DataFrame(all_rows)
        a = d[d["cell"] == "R0"].set_index(["role", "condition", "frame_index", "replica"])["rmse_K"]
        b = d[d["cell"] == ne.DIAG_CELL].set_index(["role", "condition", "frame_index", "replica"])["rmse_K"]
        gate["glu_batch1_vs_s7_batching"] = {
            "max_abs_frame_delta_K": float((a - b.reindex(a.index)).abs().max()),
            "macro_delta_K_by_role_replica": {
                f"{r}/rep{k}": ne.macro_rmse(d[(d["cell"] == "R0") & (d["role"] == r) & (d["replica"] == k)])
                - ne.macro_rmse(d[(d["cell"] == ne.DIAG_CELL) & (d["role"] == r) & (d["replica"] == k)])
                for r in roles for k in ne.REPLICAS},
        }
    _write_json(out_run / "gate.json", gate)
    _log(f"gate all_pass={gate['all_pass']} max_abs_delta={gate['max_abs_delta_K']}")
    if scientific and not gate["all_pass"]:
        _write_json(out_run / "GATE_FAILED.json", {"run_id": args.run_id, "gate": gate})
        _write_json(halt, {"halted_by": args.run_id, "reason": "R0 replay gate failed", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        sid = os.environ.get("SLURM_ARRAY_JOB_ID")
        if sid:
            subprocess.run(["scancel", sid], check=False)
        return 3

    # ---- degraded cells
    for cid in ne.CELL_ORDER[1:]:
        if halt.exists():
            _write_json(out_run / "HALTED.json", {"run_id": args.run_id, "cell": cid})
            return 3
        for role in roles:
            for rep in ne.REPLICAS:
                run_cell(spec.cells[cid], role, rep)

    # ---- fields, read-back, markers
    if fields:
        np.savez_compressed(out_run / "fields_figure_frames.npz", **fields)
    pd.DataFrame(costs).to_csv(out_run / "setting_costs.csv", index=False)
    mem = int(torch.cuda.max_memory_allocated()) if not selftest else 0
    provenance = {
        **run_tags, "mode": args.mode, "frames_per_role": expected, "consumed": {k: str(v) for k, v in consumed.items()},
        "matrix_sha256": file_sha256(args.matrix), "s7_reference_sha256": file_sha256(args.s7_reference),
        "batching": {"sparse_unet": ne.UNET_BATCH, "adapted_glu": ne.GLU_BATCH}, "p_check": p_check,
        "numerics": {"precision": "fp32", "allow_tf32": False}, "max_memory_bytes": mem,
        "no_test_split": True, "weights_changed": False, "temperature_in_input": False,
        "total_wall_s": float(sum(c["wall_s"] for c in costs)),
    }
    _write_json(out_run / "provenance.json", provenance)
    _log("phase readback: begin")
    frames = pd.concat([pd.read_csv(p) for p in sorted((out_run / "per_frame").rglob("rep*.csv"))], ignore_index=True)
    if len(frames) != len(all_rows):
        raise SystemExit(f"readback rows {len(frames)} != written {len(all_rows)}")
    if frames.duplicated(ne.KEY_COLS).any():
        raise SystemExit("duplicate keys after readback")
    if not np.isfinite(frames[["rmse_K", "mae_K", "bias_K"]].to_numpy(float)).all():
        raise SystemExit("non-finite metrics after readback")
    for col in ("original_sites_hash", "surviving_sites_hash", "spec_sha256", "checkpoint_file_sha256"):
        if frames[col].isna().any() or (frames[col] == "").any():
            raise SystemExit(f"empty {col}")
    summary = {"ok": True, "run_id": args.run_id, "family": family, "n_rows": int(len(frames)), "gate": {k: gate[k] for k in ("all_pass", "max_abs_delta_K")}}
    if scientific:
        summary["checks"] = ne.check_run_frame_table(frames, family=family, expect_diag=(family == "adapted_glu"))
        _write_json(out_run / "COMPLETE.json", summary)
        _log("done COMPLETE.json")
    else:
        _write_json(out_run / ("SELFTEST_OK.json" if selftest else "SMOKE_OK.json"), {**summary, "scientific": False})
        _log("done SMOKE_OK.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
