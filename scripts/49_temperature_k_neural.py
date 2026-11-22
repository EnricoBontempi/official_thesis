#!/usr/bin/env python3
"""Scalar K corrections on one frozen S7 run. No training and no change to the free forward.

The forward is the S7 evaluator's: batch 8, shuffle off, fp32, TF32 off, the same
query grid and the same ``_forward``. Best and last are loaded from their own
checkpoints. Layout r is paired with velocity-mask replica r. Historical T9 is
written as a separate bridge and is not one of the three manifest layouts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.evaluation.temperature_k import (  # noqa: E402
    GATE_ATOL_K,
    N_FRAMES,
    PROTOCOL_RELATIVE,
    SWEEP_CHECKPOINTS,
    SWEEP_ROLES,
    gate_free_against_reference,
    layout_by_replica,
    load_frozen_manifest,
    prediction_sha256,
    rows_for_historical_t9,
    rows_for_prediction,
    sha256_file,
)

MANIFEST = ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] ksweep {message}", flush=True)


def _s7_reference(root: Path, run_id: str, checkpoint: str, role: str, replica: int) -> Path:
    return root / run_id / "per_frame" / checkpoint / "cov0.05" / role / f"rep{replica}.csv"


def _check_readback(frame: pd.DataFrame, *, n_expected: int) -> list[str]:
    issues = []
    primary = frame[frame["layout_kind"] == "primary"]
    if primary.duplicated(["sim_id", "frame_index", "K", "treatment"]).any():
        issues.append("duplicate primary rows")
    frames = primary[primary["treatment"] == "free"]
    n = frames[frames["K"] == 0][["sim_id", "frame_index"]].drop_duplicates().shape[0]
    if n != n_expected:
        issues.append(f"{n} frames, expected {n_expected}")
    if not np.isfinite(primary["full_rmse_K"].to_numpy(dtype=float)).all():
        issues.append("non-finite full RMSE")
    k0 = primary[(primary["K"] == 0) & primary["treatment"].isin(["free", "ao", "mean", "median"])]
    if not np.allclose(k0["delta_applied_K"].to_numpy(dtype=float), 0.0):
        issues.append("K=0 correction delta is not 0")
    for k, part in primary.groupby("K"):
        free = part[part["treatment"] == "free"].set_index(["sim_id", "frame_index"])
        for name in ("ao", "mean", "median", "oracle"):
            other = part[part["treatment"] == name].set_index(["sim_id", "frame_index"])
            delta = (other["full_centered_rmse_K"] - free["full_centered_rmse_K"]).abs()
            if float(delta.max()) > 1e-6:
                issues.append(f"centered RMSE moved at K={k} {name}")
            common = (other["common_centered_rmse_K"] - free["common_centered_rmse_K"]).abs()
            if float(common.max()) > 1e-6:
                issues.append(f"common centered RMSE moved at K={k} {name}")
    if primary["common_n"].ne(128 * 128 - 32).any():
        issues.append("common domain is not 16352 pixels")
    bridge = frame[frame["layout_kind"] == "t9_bridge"]
    if bridge.empty or set(bridge["K"]) != {9}:
        issues.append("historical T9 bridge missing or mixed into another K")
    if set(bridge["layout_id"]) & set(primary["layout_id"]):
        issues.append("historical T9 layout id collides with a primary layout")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--s7-eval-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--eval-git-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--smoke-frames", type=int, default=4)
    args = parser.parse_args()
    smoke = args.mode == "smoke"
    if smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("smoke mode needs an out-root whose name contains 'smoke'; full must not")
    if args.batch_size != 8:
        raise SystemExit("refusing a batch size other than the S7 batch of 8")
    protocol = (ROOT / PROTOCOL_RELATIVE).read_text()
    if "configs/observations/month2_temperature_anchors_nested_v1.json" not in protocol:
        raise SystemExit("protocol text no longer points at the frozen manifest")
    manifest = load_frozen_manifest(MANIFEST)
    layouts = layout_by_replica(manifest)

    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("the neural sweep requires CUDA; refusing before creating an output directory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    s7 = _load("s7_eval_k", "scripts/38_s7_eval.py")
    s6 = s7._s6()
    matrix = s7.load_matrix(args.matrix)
    entry = s7._run_entry(matrix, args.run_id)
    _log(f"start run={args.run_id} mode={args.mode} sha={args.eval_git_sha}")
    inspected = s7._inspect_checkpoints(entry, torch)
    consumed = s7._verify_inputs(entry, matrix)
    normalizer = Month2FitOnlyNormalizer(matrix["shared_inputs"]["stats"])
    sigma = float(normalizer.temperature_std)
    datasets, expected = s7._build_datasets(
        entry, matrix, consumed, normalizer, smoke_frames=int(args.smoke_frames) if smoke else None
    )
    family = str(entry["family"])
    model, _ = s6._load_model(family, consumed["cache_root"], device)
    query = None
    if family == "adapted_glu":
        query = s6.make_synthetic_query_grid(datasets["train_fit"].x, datasets["train_fit"].y).to(device)
    variant = "p_on" if entry["p_conditioning"] else "p_off"
    out_run = args.out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    gates: list[dict[str, Any]] = []
    issues: list[str] = []
    for name in SWEEP_CHECKPOINTS:
        spec = entry["checkpoints"][name]
        payload = s6.load_sparse_scientific_checkpoint(
            inspected[name]["path"], model=model, require_resumable=True, restore_rng=False
        )
        if int((payload.get("epoch_state") or {}).get("epoch", -1)) != int(spec["epoch"]):
            raise SystemExit(f"{name} epoch drifted from the matrix")
        model.to(device)
        model.eval()
        _log(f"checkpoint {name} epoch={spec['epoch']} reuse_recorded={spec.get('reuse_forward_from')}")
        for role in SWEEP_ROLES:
            for replica in (0, 1, 2):
                layout = layouts[int(replica)]
                order = layout["site_order_flat"]
                ds = datasets[role]
                ds.replica_id = int(replica)
                ds.site_fraction = 0.05
                ds.mask_mode = "evaluation"
                loader = s6.DataLoader(
                    ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=s6.collate_sparse_batch
                )
                rows: list[dict[str, Any]] = []
                residual_rows: list[np.ndarray] = []
                id_rows: list[tuple[str, int]] = []
                oracle_rows: list[float] = []
                sha_rows: list[str] = []
                for batch in loader:
                    if family == "adapted_glu":
                        s6._attach_p(batch, ds)
                    pred_n = s6._forward(
                        model, family, batch, p_conditioning=bool(entry["p_conditioning"]),
                        query_coords_phys=query, device=device,
                    )
                    true_batch = batch["target_T_K"].numpy()
                    for i, meta in enumerate(batch["metadata"]):
                        pred_K = s6.denormalize_temperature(pred_n[i, 0], normalizer)
                        true_K = true_batch[i, 0]
                        s7_row = s6.evaluate_frame_prediction(
                            pred_K, true_K, train_sigma_T=sigma, sim_id=str(meta["sim_id"]),
                            split=str(meta.get("split", role)), role=role, frame_index=int(meta["frame_index"]),
                            method=family, variant=variant, p_conditioning=bool(entry["p_conditioning"]),
                            K=0, site_fraction=0.05, model_seed=int(entry["model_seed"]),
                            mask_replica_id=int(replica), layout_id=str(ds.layout_id),
                            task_mode="temperature_free", checkpoint_kind=name, run_id=args.run_id,
                            mask_hash=str(meta.get("mask_hash", "")),
                        )
                        identity = {
                            "run_id": args.run_id,
                            "method": family,
                            "family": family,
                            "p_conditioning": bool(entry["p_conditioning"]),
                            "model_seed": int(entry["model_seed"]),
                            "checkpoint": name,
                            "epoch": int(spec["epoch"]),
                            "role": role,
                            "mask_replica_id": int(replica),
                            "sim_id": str(meta["sim_id"]),
                            "frame_index": int(meta["frame_index"]),
                            "velocity_mask_hash": str(meta.get("mask_hash", "")),
                            "t_order_sha256": layout["order_sha256"],
                            "prediction_sha256": prediction_sha256(pred_K),
                            "s7_rmse_K": float(s7_row["rmse_K"]),
                            "file_sha256": spec["file_sha256"],
                            "weights_sha256": spec["weights_sha256"],
                        }
                        built = rows_for_prediction(
                            pred_K, true_K, order, identity=identity,
                            layout_id=f"nested_r{replica}", layout_kind="primary",
                        )
                        built.extend(rows_for_historical_t9(pred_K, true_K, identity=identity))
                        rows.extend(built)
                        sites = np.asarray(order, dtype=np.int64)
                        residual_rows.append(np.asarray(true_K, dtype=np.float64).ravel()[sites] - np.asarray(pred_K, dtype=np.float64).ravel()[sites])
                        id_rows.append((str(meta["sim_id"]), int(meta["frame_index"])))
                        oracle_rows.append(float(built[0]["delta_oracle_K"]))
                        sha_rows.append(identity["prediction_sha256"])
                frame = pd.DataFrame(rows)
                n_expected = expected[role] if smoke else N_FRAMES[role]
                dest = out_run / "per_frame" / name / role / f"rep{replica}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(dest, index=False)
                reread = pd.read_csv(dest)
                if len(reread) != len(frame):
                    issues.append(f"{name} {role} rep{replica}: readback row count {len(reread)} != {len(frame)}")
                issues.extend(f"{name} {role} rep{replica}: {item}" for item in _check_readback(reread, n_expected=n_expected))
                resid_path = out_run / "residuals" / name / role / f"rep{replica}.npz"
                resid_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    resid_path,
                    sim_id=np.array([item[0] for item in id_rows]),
                    frame_index=np.array([item[1] for item in id_rows], dtype=np.int64),
                    residual_K=np.stack(residual_rows),
                    oracle_delta_K=np.array(oracle_rows, dtype=np.float64),
                    prediction_sha256=np.array(sha_rows),
                    order_sha256=np.array(layout["order_sha256"]),
                )
                reference = _s7_reference(args.s7_eval_root, args.run_id, name, role, replica)
                if not reference.is_file():
                    gate = {"ok": False, "replay_executed": False, "reason": "missing S7 reference", "reference_path": str(reference)}
                else:
                    ref = pd.read_csv(reference)
                    if smoke:
                        keep = set(zip(frame["sim_id"].astype(str), frame["frame_index"].astype(int)))
                        ref = ref[ref.apply(lambda row: (str(row["sim_id"]), int(row["frame_index"])) in keep, axis=1)]
                    free = reread[(reread["treatment"] == "free") & (reread["K"] == 0) & (reread["layout_kind"] == "primary")].copy()
                    free["sim_id"] = free["sim_id"].astype(str)
                    free["frame_index"] = free["frame_index"].astype(int)
                    gate = gate_free_against_reference(
                        free, ref, reference_path=str(reference), reference_sha256=sha256_file(reference), atol_K=GATE_ATOL_K
                    )
                    if "mask_hash" not in ref.columns:
                        gate["ok"] = False
                        gate["reason"] = "S7 reference has no mask_hash"
                    else:
                        merged = free.merge(
                            ref[["sim_id", "frame_index", "mask_hash"]].assign(
                                sim_id=lambda d: d["sim_id"].astype(str), frame_index=lambda d: d["frame_index"].astype(int)
                            ),
                            on=["sim_id", "frame_index"], how="left",
                        )
                        mismatch = merged["velocity_mask_hash"].astype(str) != merged["mask_hash"].astype(str)
                        gate["mask_hash_mismatch"] = int(mismatch.sum())
                        if int(mismatch.sum()):
                            gate["ok"] = False
                            gate["reason"] = "velocity mask_hash differs from the S7 CSV"
                per_frame = gate.pop("per_frame", None)
                gate.update({"checkpoint": name, "role": role, "replica": int(replica)})
                gates.append({key: value for key, value in gate.items() if key != "per_frame"})
                gate_dir = out_run / "gate"
                gate_dir.mkdir(parents=True, exist_ok=True)
                (gate_dir / f"{name}_{role}_rep{replica}.json").write_text(json.dumps(gate, indent=2) + "\n")
                if per_frame is not None:
                    per_frame.to_csv(gate_dir / f"{name}_{role}_rep{replica}_frames.csv", index=False)
                _log(
                    f"{name} {role} rep{replica} frames={n_expected} gate={gate.get('ok')} "
                    f"macro_delta={gate.get('macro_delta_K')} max_frame={gate.get('max_abs_frame_delta_K')}"
                )
        torch.cuda.empty_cache()

    provenance = {
        "run_id": args.run_id,
        "family": family,
        "p_conditioning": bool(entry["p_conditioning"]),
        "model_seed": int(entry["model_seed"]),
        "eval_git_sha": args.eval_git_sha,
        "manifest_file_sha256": sha256_file(MANIFEST),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "matrix_sha256": sha256_file(args.matrix),
        "checkpoints": {name: entry["checkpoints"][name] for name in SWEEP_CHECKPOINTS},
        "numerics": {"precision": "fp32", "allow_tf32": False, "batch_size": 8},
        "roles": list(SWEEP_ROLES),
        "no_test_split": True,
        "no_best_seen": True,
        "historical_t9": "separate bridge, not a manifest layout",
        "protocol_yaml_not_parsed": True,
        "gates": gates,
        "readback_issues": issues,
    }
    (out_run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    ok = all(item.get("ok") for item in gates) and not issues
    marker = "SMOKE_OK.json" if smoke and ok else "COMPLETE.json" if ok else "GATE_FAILED.json"
    (out_run / marker).write_text(json.dumps({"ok": ok, "n_gates": len(gates), "n_issues": len(issues)}, indent=2) + "\n")
    _log(f"done {marker}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
