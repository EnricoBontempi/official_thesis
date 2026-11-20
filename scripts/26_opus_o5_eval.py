#!/usr/bin/env python3
"""Opus O5 final evaluation for the eight scientific-stream runs.

Reuses ``evaluate_setting`` from the S5 evaluator. Does not refit, resume,
change selection, or touch TEST. Historical S5 runs are not rows of this matrix.
"""

from __future__ import annotations

import argparse
import hashlib
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
from combustion_recon.data.sparse_eval_masks import (  # noqa: E402
    EvalMaskContractError,
    check_eval_mask_contract,
)
from combustion_recon.evaluation.opus_o5_plan import (  # noqa: E402
    CHECKPOINTS,
    COVERAGES,
    REPLICAS,
    ROLE_N_FRAMES,
    ROLES,
    reuse_plan,
)
from combustion_recon.training.dense_cache import (  # noqa: E402
    cache_validation_kwargs,
    verify_cache_payload_hashes,
)
from combustion_recon.training.scientific_epoch_sampler import (  # noqa: E402
    cache_indices_for_pool,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402


def _s6():
    path = ROOT / "scripts" / "17_s6_sparse_eval_and_swap.py"
    spec = importlib.util.spec_from_file_location("s6_sparse_eval_o5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log(message: str) -> None:
    """Timestamped, flushed progress line. Operational only."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{stamp}] o5 {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        array = state_dict[key].detach().contiguous().cpu().numpy()
        digest.update(str(key).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "month2_opus_o5_v1":
        raise SystemExit(f"unexpected O5 experiment_id: {payload.get('experiment_id')}")
    return payload


def _run_entry(matrix: dict[str, Any], run_id: str) -> dict[str, Any]:
    matches = [row for row in matrix["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise SystemExit(f"matrix does not contain exactly one {run_id}")
    return matches[0]


def _tag_rows(rows: list[dict[str, Any]], tags: dict[str, Any]) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        item = dict(row)
        item.pop("_spatial_bias_map", None)
        item.update(tags)
        tagged.append(item)
    return tagged


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if "_spatial_bias_map" in frame.columns:
        frame = frame.drop(columns=["_spatial_bias_map"])
    frame.to_csv(path, index=False)


def _save_val_frame1000(s6, model, ds, normalizer, device, out_path: Path, tags: dict[str, Any]) -> int:
    positions = [
        j
        for j, idx in enumerate(ds.indices)
        if int(ds.frames[idx]["frame_index"]) == 1000
    ]
    if len(positions) != 3:
        raise RuntimeError(f"VAL frame 1000 count is {len(positions)}, expected 3")
    items = [ds[j] for j in positions]
    batch = s6.collate_sparse_batch(items)
    pred_n = s6._forward(
        model,
        "sparse_unet",
        batch,
        p_conditioning=bool(ds.p_conditioning),
        query_coords_phys=None,
        device=device,
    )
    pred = np.stack(
        [s6.denormalize_temperature(pred_n[i, 0], normalizer) for i in range(len(items))],
        axis=0,
    ).astype(np.float32)
    truth = batch["target_T_K"].numpy()[:, 0].astype(np.float32)
    sim_id = np.array([str(item["metadata"]["sim_id"]) for item in items])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pred_T_K=pred,
        true_T_K=truth,
        sim_id=sim_id,
        frame_index=np.array([1000, 1000, 1000], dtype=np.int64),
        **{key: np.array(value) for key, value in tags.items()},
    )
    return len(positions)


def _verify_inputs(entry: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    shared = matrix["shared_inputs"]
    pool = str(entry["pool_id"])
    if pool == "C149":
        cache_root = Path(shared["c149_cache_root"])
        hash_file = Path(shared["c149_hash_file"])
        expect_agg = shared["c149_cache_aggregate_sha256"]
        expect_id = shared["c149_cache_id"]
    else:
        cache_root = Path(shared["historical_cache_root"])
        hash_file = Path(shared["historical_hash_file"])
        expect_agg = shared["historical_cache_aggregate_sha256"]
        expect_id = shared["historical_cache_id"]
    expected = json.loads(hash_file.read_text())
    verified = verify_cache_payload_hashes(expected, cache_root=cache_root)
    if not verified["ok"] or verified["actual_aggregate_sha256"] != expect_agg:
        raise SystemExit(f"cache hash mismatch for {entry['run_id']}: {verified['mismatches']}")
    manifest = json.loads((cache_root / "manifest.json").read_text())
    if manifest.get("cache_id") != expect_id:
        raise SystemExit(f"cache id {manifest.get('cache_id')} != {expect_id}")
    stats = Path(shared["stats"])
    stats_sha = sha256_file(stats)
    if stats_sha != shared["stats_sha256"]:
        raise SystemExit("normalizer sha mismatch")
    mask_root = Path(shared["mask_root"])
    try:
        mask_report = check_eval_mask_contract(
            manifest_path=mask_root / "manifest.json",
            payload_root=mask_root,
            expected_manifest_file_sha256=shared["eval_mask_manifest_file_sha256"],
            expected_aggregate_payload_sha256=shared["eval_mask_aggregate_payload_sha256"],
        )
    except EvalMaskContractError as exc:
        raise SystemExit(f"eval-mask contract failed: {exc.reason}") from exc
    identity_path = Path(entry["training_dir"]) / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    if identity.get("git_sha") != entry["training_git_sha"]:
        raise SystemExit("training git sha does not match the matrix")
    if identity.get("pool_id") != pool:
        raise SystemExit("training pool_id does not match the matrix")
    return {
        "cache_root": cache_root,
        "cache_aggregate_sha256": verified["actual_aggregate_sha256"],
        "cache_id": expect_id,
        "stats_sha256": stats_sha,
        "eval_mask_manifest_file_sha256": mask_report["eval_mask_manifest_file_sha256"],
        "eval_mask_aggregate_payload_sha256": mask_report["eval_mask_aggregate_payload_sha256"],
    }


def _inspect_checkpoints(entry: dict[str, Any], torch) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    weight_shas: dict[str, str] = {}
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        path = Path(entry["training_dir"]) / "checkpoints" / f"{name}.pt"
        file_sha = sha256_file(path)
        if file_sha != spec["file_sha256"]:
            raise SystemExit(f"{name} file sha mismatch for {entry['run_id']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
        if epoch != int(spec["epoch"]):
            raise SystemExit(f"{name} epoch {epoch} != matrix {spec['epoch']}")
        if payload.get("checkpoint_kind") != "sparse_scientific_epoch_complete_v1":
            raise SystemExit(f"{name} is not a complete epoch checkpoint")
        digest = weights_sha256(payload["model_state_dict"])
        if digest != spec["weights_sha256"]:
            raise SystemExit(f"{name} weight sha mismatch for {entry['run_id']}")
        weight_shas[name] = digest
        found[name] = {"path": path, "epoch": epoch, "file_sha256": file_sha, "weights_sha256": digest}
        del payload
    declared = {name: entry["checkpoints"][name].get("reuse_forward_from") for name in CHECKPOINTS}
    actual = reuse_plan(weight_shas)
    if declared != actual:
        raise SystemExit(f"reuse plan mismatch: matrix {declared} runtime {actual}")
    return found


def _smoke_positions(frames: list[dict[str, Any]], indices: list[int], n: int, keep_frame_1000: bool) -> list[int]:
    """Evenly spaced cache rows for the non-scientific smoke subset."""
    take = sorted({int(round(v)) for v in np.linspace(0, len(indices) - 1, num=min(n, len(indices)))})
    chosen = [indices[i] for i in take]
    if keep_frame_1000:
        chosen += [i for i in indices if int(frames[i]["frame_index"]) == 1000 and i not in chosen]
    return sorted(set(chosen))


def _build_datasets(entry, matrix, consumed, normalizer, *, smoke_frames: int | None):
    cache_root = consumed["cache_root"]
    train_frames = json.loads((cache_root / "train_fit" / "frames.json").read_text())
    ref760 = cache_indices_for_pool(train_frames, "A38")
    if len(ref760) != 760:
        raise SystemExit(f"reference760 resolved {len(ref760)} rows")
    cache_kwargs = cache_validation_kwargs(str(entry["pool_id"]))
    datasets: dict[str, SparseUNetCacheDataset] = {}
    expected: dict[str, int] = {}
    for role in ROLES:
        indices = ref760 if role == "train_fit" else None
        if smoke_frames is not None:
            frames = json.loads((cache_root / role / "frames.json").read_text())
            pool = ref760 if indices is not None else list(range(len(frames)))
            indices = _smoke_positions(frames, pool, smoke_frames, role == "val_eval")
        datasets[role] = SparseUNetCacheDataset(
            role,
            cache_root=cache_root,
            indices=indices,
            mask_mode="evaluation",
            site_fraction=0.05,
            eval_mask_seed=int(matrix["shared_inputs"]["eval_mask_seed"]),
            layout_id=str(matrix["shared_inputs"]["layout_id"]),
            replica_id=0,
            p_conditioning=bool(entry["p_conditioning"]),
            normalizer=normalizer,
            eval_mask_payload_root=matrix["shared_inputs"]["mask_root"],
            require_eval_mask_payload=True,
            **cache_kwargs,
        )
        expected[role] = len(datasets[role]) if smoke_frames is not None else ROLE_N_FRAMES[role]
        if len(datasets[role]) != expected[role]:
            raise SystemExit(f"{role} has {len(datasets[role])} frames, expected {expected[role]}")
    return datasets, expected


def verify_outputs(out_run: Path, entry: dict[str, Any], expected: dict[str, int], run_id: str) -> dict[str, Any]:
    """Read every written file back. COMPLETE or SMOKE_OK only follows this check."""
    n_csv = 0
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        source = spec.get("reuse_forward_from")
        for coverage in COVERAGES:
            cov = f"{float(coverage):.2f}"
            for role in ROLES:
                for replica in REPLICAS:
                    path = out_run / "per_frame" / name / f"cov{cov}" / role / f"rep{replica}.csv"
                    if not path.is_file():
                        raise SystemExit(f"missing output {path}")
                    frame = pd.read_csv(path)
                    n_csv += 1
                    if len(frame) != expected[role]:
                        raise SystemExit(f"{path}: {len(frame)} rows, expected {expected[role]}")
                    if not np.isfinite(frame["rmse_K"].to_numpy(dtype=float)).all():
                        raise SystemExit(f"{path}: non-finite rmse_K")
                    if set(frame["checkpoint_kind"]) != {name}:
                        raise SystemExit(f"{path}: wrong checkpoint_kind")
                    if set(frame["weights_sha256"]) != {spec["weights_sha256"]}:
                        raise SystemExit(f"{path}: weights_sha256 drifted from the matrix")
                    if set(frame["file_sha256"]) != {spec["file_sha256"]}:
                        raise SystemExit(f"{path}: file_sha256 drifted from the matrix")
                    if set(frame["run_id"]) != {run_id}:
                        raise SystemExit(f"{path}: wrong run_id")
                    if source:
                        src = pd.read_csv(
                            out_run / "per_frame" / source / f"cov{cov}" / role / f"rep{replica}.csv"
                        )
                        if not np.array_equal(src["rmse_K"].to_numpy(), frame["rmse_K"].to_numpy()):
                            raise SystemExit(f"{path}: shared forward differs from {source}")
                        if set(frame["shared_forward_from"].fillna("")) != {source}:
                            raise SystemExit(f"{path}: shared_forward_from not recorded")
        field = out_run / "fields" / f"val_frame1000_{name}_cov05_rep0.npz"
        with np.load(field, allow_pickle=True) as data:
            if data["pred_T_K"].shape != data["true_T_K"].shape or data["pred_T_K"].shape[0] != 3:
                raise SystemExit(f"{field}: bad shape")
            if not np.isfinite(data["pred_T_K"]).all():
                raise SystemExit(f"{field}: non-finite prediction")
    expected_n = len(CHECKPOINTS) * len(COVERAGES) * len(ROLES) * len(REPLICAS)
    if n_csv != expected_n:
        raise SystemExit(f"{n_csv} csv files, expected {expected_n}")
    prov = json.loads((out_run / "provenance.json").read_text())
    if prov["run_id"] != run_id:
        raise SystemExit("provenance run_id mismatch")
    return {"n_csv": n_csv, "n_fields": len(CHECKPOINTS)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--eval-git-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-scientific subset with the real entrypoint; writes SMOKE_OK.json, never COMPLETE.json",
    )
    parser.add_argument("--smoke-frames", type=int, default=6)
    args = parser.parse_args()

    in_smoke_dir = "smoke" in args.out_root.name.lower()
    if args.smoke != in_smoke_dir:
        raise SystemExit("--smoke requires an out-root whose final directory name contains 'smoke'; a scientific out-root must not")
    smoke_frames = int(args.smoke_frames) if args.smoke else None

    _log(f"start run={args.run_id} smoke={bool(args.smoke)} sha={args.eval_git_sha}")
    matrix = load_matrix(args.matrix)
    entry = _run_entry(matrix, args.run_id)
    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("O5 requires CUDA; refusing before creating an output directory")
    device = torch.device("cuda")
    _log("phase checkpoints: begin (file and weight hashes)")
    inspected = _inspect_checkpoints(entry, torch)
    _log("phase checkpoints: done")
    _log(f"phase cache: begin (payload hashes, masks, stats) pool={entry['pool_id']}")
    consumed = _verify_inputs(entry, matrix)
    _log("phase cache: done")

    # Everything that can fail on imports, API or data happens before the run directory exists.
    _log("phase dataset: begin (normalizer, datasets, model)")
    s6 = _s6()
    normalizer = Month2FitOnlyNormalizer(matrix["shared_inputs"]["stats"])
    datasets, expected = _build_datasets(entry, matrix, consumed, normalizer, smoke_frames=smoke_frames)
    model, _ = s6._load_model("sparse_unet", consumed["cache_root"], device)

    _log("phase dataset: done")
    out_run = args.out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    stored_rows: dict[str, dict[tuple, list[dict[str, Any]]]] = {}
    stored_fields: dict[str, Path] = {}
    completeness: list[dict[str, Any]] = []
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        source = spec.get("reuse_forward_from")
        field_path = out_run / "fields" / f"val_frame1000_{name}_cov05_rep0.npz"
        if source:
            if source not in stored_rows:
                raise SystemExit(f"reuse source {source} was not evaluated")
            for key, rows in stored_rows[source].items():
                retagged = _tag_rows(
                    rows,
                    {
                        "checkpoint_kind": name,
                        "epoch": int(spec["epoch"]),
                        "criterion": spec["criterion"],
                        "file_sha256": spec["file_sha256"],
                        "weights_sha256": spec["weights_sha256"],
                        "shared_forward_from": source,
                        "model_seed": int(entry["model_seed"]),
                    },
                )
                coverage, role, replica = key
                path = out_run / "per_frame" / name / f"cov{coverage}" / role / f"rep{replica}.csv"
                _write_rows(path, retagged)
                completeness.append(
                    {"checkpoint": name, "coverage": coverage, "role": role, "replica": replica, "n": len(retagged), "shared_forward_from": source}
                )
            copied = np.load(stored_fields[source], allow_pickle=True)
            field_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                field_path,
                pred_T_K=copied["pred_T_K"],
                true_T_K=copied["true_T_K"],
                sim_id=copied["sim_id"],
                frame_index=copied["frame_index"],
                checkpoint_kind=np.array(name),
                run_id=np.array(args.run_id),
                site_fraction=np.array("0.05"),
                replica_id=np.array("0"),
                shared_forward_from=np.array(source),
            )
            continue

        _log(f"phase forward: begin checkpoint={name}")
        payload = s6.load_sparse_scientific_checkpoint(
            inspected[name]["path"], model=model, require_resumable=True, restore_rng=False
        )
        if int((payload.get("epoch_state") or {}).get("epoch", -1)) != int(spec["epoch"]):
            raise SystemExit("loaded epoch drifted from the inspected checkpoint")
        model.to(device)
        model.eval()
        stored_rows[name] = {}
        for coverage in COVERAGES:
            _log(f"forward checkpoint={name} coverage={coverage:.2f}")
            for role in ROLES:
                for replica in REPLICAS:
                    out = s6.evaluate_setting(
                        model=model,
                        family="sparse_unet",
                        p_conditioning=bool(entry["p_conditioning"]),
                        ds=datasets[role],
                        role=role,
                        replica_id=int(replica),
                        site_fraction=float(coverage),
                        method="sparse_unet",
                        variant="p_on" if entry["p_conditioning"] else "p_off",
                        checkpoint_kind=name,
                        run_id=args.run_id,
                        normalizer=normalizer,
                        query_coords_phys=None,
                        device=device,
                        batch_size=int(args.batch_size),
                    )
                    if int(out["n_frames"]) != expected[role]:
                        raise SystemExit(
                            f"{name} {role} coverage {coverage} replica {replica}: {out['n_frames']} frames"
                        )
                    tags = {
                        "pool_id": entry["pool_id"],
                        "model_seed": int(entry["model_seed"]),
                        "epoch": int(spec["epoch"]),
                        "criterion": spec["criterion"],
                        "file_sha256": spec["file_sha256"],
                        "weights_sha256": spec["weights_sha256"],
                        "shared_forward_from": "",
                        "training_git_sha": entry["training_git_sha"],
                        "eval_git_sha": args.eval_git_sha,
                    }
                    rows = _tag_rows(out["rows"], tags)
                    cov_label = f"{float(coverage):.2f}"
                    path = out_run / "per_frame" / name / f"cov{cov_label}" / role / f"rep{replica}.csv"
                    _write_rows(path, rows)
                    stored_rows[name][(cov_label, role, int(replica))] = rows
                    completeness.append(
                        {
                            "checkpoint": name,
                            "coverage": cov_label,
                            "role": role,
                            "replica": int(replica),
                            "n": len(rows),
                            "wall_s": out["wall_s"],
                            "mean_condition_rmse_K": out["mean_condition_rmse_K"],
                            "shared_forward_from": "",
                        }
                    )
        datasets["val_eval"].replica_id = 0
        datasets["val_eval"].site_fraction = 0.05
        _save_val_frame1000(
            s6,
            model,
            datasets["val_eval"],
            normalizer,
            device,
            field_path,
            {
                "checkpoint_kind": name,
                "run_id": args.run_id,
                "site_fraction": "0.05",
                "replica_id": "0",
            },
        )
        stored_fields[name] = field_path
        _log(f"phase forward: done checkpoint={name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    provenance = {
        "experiment_id": matrix["experiment_id"],
        "run_id": args.run_id,
        "smoke": bool(args.smoke),
        "smoke_frames_per_role": expected if args.smoke else None,
        "eval_git_sha": args.eval_git_sha,
        "training_git_sha": entry["training_git_sha"],
        "training_job_id": entry["job_id"],
        "training_dir": entry["training_dir"],
        "consumed": {key: str(value) for key, value in consumed.items()},
        "checkpoints": entry["checkpoints"],
        "reference760": "A38 frame indices inside train_fit, including B19 and C149",
        "loop_metrics_are_not_this_output": True,
        "no_test_split": True,
        "max_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (out_run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pd.DataFrame(completeness).to_csv(out_run / "setting_costs.csv", index=False)
    expected_n = len(CHECKPOINTS) * len(COVERAGES) * len(ROLES) * len(REPLICAS)
    if len(completeness) != expected_n:
        raise SystemExit(f"completeness rows {len(completeness)} != {expected_n}")
    _log("phase readback: begin")
    readback = verify_outputs(out_run, entry, expected, args.run_id)
    marker = "SMOKE_OK.json" if args.smoke else "COMPLETE.json"
    payload = {"ok": True, "n_settings": len(completeness), "run_id": args.run_id, **readback}
    if args.smoke:
        payload["scientific"] = False
        payload["frames_per_role"] = expected
    (out_run / marker).write_text(json.dumps(payload, indent=2) + "\n")
    _log(f"done {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
