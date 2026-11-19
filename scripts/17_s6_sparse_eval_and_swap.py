#!/usr/bin/env python3
"""S6 — offline best/last sparse eval at 1/5/20% + snapshot-swap on best@5%.

No training / no resume. Uses frozen S5 checkpoints only.
GappyPOD nested 1/20% is NOT refit here; matched 5% baselines are reused from tables.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import (  # noqa: E402
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
    build_glu_from_config,
    make_synthetic_query_grid,
)
from combustion_recon.models.unet import build_sparse_unet_from_config  # noqa: E402
from combustion_recon.training.dense_cache import (  # noqa: E402
    compute_cache_payload_hashes,
    validate_dense_cache_manifest,
)
from combustion_recon.training.snapshot_swap import (  # noqa: E402
    build_within_condition_swap_pairs,
    pairs_by_dest_cache_index,
)
from combustion_recon.training.sparse_dataset import (  # noqa: E402
    SparseUNetCacheDataset,
    collate_sparse_batch,
)
from combustion_recon.training.sparse_scientific_loop import (  # noqa: E402
    load_sparse_scientific_checkpoint,
)
from combustion_recon.training.sparse_view import (  # noqa: E402
    build_canonical_observation,
    build_sparse_unet_input,
)
from combustion_recon.training.trainer import write_json  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROLES = ("train_fit", "seen_train_eval", "val_eval")
COVERAGES = (0.01, 0.05, 0.20)
REPLICAS = (0, 1, 2)
RUNS = (
    {
        "job_id": 29567988,
        "run_id": "s5_sparse_unet_pon_seed0",
        "family": "sparse_unet",
        "p_conditioning": True,
        "label": "U-Net p_on",
    },
    {
        "job_id": 29567990,
        "run_id": "s5_sparse_unet_poff_seed0",
        "family": "sparse_unet",
        "p_conditioning": False,
        "label": "U-Net p_off",
    },
    {
        "job_id": 29567992,
        "run_id": "s5_adapted_glu_pon_seed0",
        "family": "adapted_glu",
        "p_conditioning": True,
        "label": "GLU p_on",
    },
    {
        "job_id": 29567994,
        "run_id": "s5_adapted_glu_poff_seed0",
        "family": "adapted_glu",
        "p_conditioning": False,
        "label": "GLU p_off",
    },
)


def _mean_condition_rmse(rows: list[dict[str, Any]], role: str) -> float | None:
    if not rows:
        return None
    agg = aggregate_frame_then_condition_then_split(rows)
    sel = agg["selection_mean_condition_rmse"]
    if role == "val_eval" and sel is not None and not getattr(sel, "empty", True):
        return float(sel["mean_condition_rmse_K"].iloc[0])
    pc = agg["per_condition"]
    if pc is None or getattr(pc, "empty", True) or "rmse_K" not in pc.columns:
        return None
    return float(pc["rmse_K"].mean())


def _load_model(family: str, cache_root: Path, device: torch.device):
    if family == "sparse_unet":
        model = build_sparse_unet_from_config(init_seed=0)
        queries = None
    else:
        model = build_glu_from_config(domain_geometry_path=cache_root / "geometry.npz")
        # queries filled later from any role dataset axes
        queries = None
    model.to(device)
    model.eval()
    return model, queries


@torch.no_grad()
def _forward(
    model,
    family: str,
    batch: dict[str, Any],
    *,
    p_conditioning: bool,
    query_coords_phys: torch.Tensor | None,
    device: torch.device,
) -> np.ndarray:
    if family == "sparse_unet":
        x = batch["input_tensor"].to(device)
        pred = model(x)
    else:
        assert query_coords_phys is not None
        tokens = batch["glu_tokens"]
        B = len(tokens)
        coords = torch.stack(
            [torch.from_numpy(np.asarray(t["coords_phys"], dtype=np.float32)) for t in tokens],
            dim=0,
        ).to(device)
        vals = torch.stack(
            [torch.from_numpy(np.asarray(t["values_norm"], dtype=np.float32)) for t in tokens],
            dim=0,
        ).to(device)
        sites = torch.stack(
            [torch.from_numpy(np.asarray(t["site_index"], dtype=np.int64)) for t in tokens],
            dim=0,
        ).to(device)
        p = batch["p_norm_tensor"].to(device)
        if not p_conditioning:
            p = torch.zeros_like(p)
        Y = query_coords_phys.unsqueeze(0).expand(B, -1, -1).contiguous()
        pred = model(
            coords,
            vals,
            sites,
            p,
            Y,
            p_conditioning=p_conditioning,
            decode_chunked=True,
            chunk_size=512,
            return_flat=False,
        )
    return pred.detach().cpu().numpy()


def _attach_p(batch: dict[str, Any], ds: SparseUNetCacheDataset) -> None:
    p_rows = []
    for meta in batch["metadata"]:
        idx = int(meta["cache_index"])
        p_rows.append([float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])])
    batch["p_norm_tensor"] = torch.tensor(p_rows, dtype=torch.float32)


def evaluate_setting(
    *,
    model,
    family: str,
    p_conditioning: bool,
    ds: SparseUNetCacheDataset,
    role: str,
    replica_id: int,
    site_fraction: float,
    method: str,
    variant: str,
    checkpoint_kind: str,
    run_id: str,
    normalizer: Month2FitOnlyNormalizer,
    query_coords_phys: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    ds.replica_id = int(replica_id)
    ds.site_fraction = float(site_fraction)
    ds.mask_mode = "evaluation"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sparse_batch,
    )
    rows: list[dict[str, Any]] = []
    sigma = float(normalizer.temperature_std)
    t0 = time.perf_counter()
    for batch in loader:
        if family == "adapted_glu":
            _attach_p(batch, ds)
        pred_n = _forward(
            model,
            family,
            batch,
            p_conditioning=p_conditioning,
            query_coords_phys=query_coords_phys,
            device=device,
        )
        true_K = batch["target_T_K"].numpy()
        for i, meta in enumerate(batch["metadata"]):
            pred_K = denormalize_temperature(pred_n[i, 0], normalizer)
            row = evaluate_frame_prediction(
                pred_K,
                true_K[i, 0],
                train_sigma_T=sigma,
                sim_id=str(meta["sim_id"]),
                split=str(meta.get("split", role)),
                role=role,
                frame_index=int(meta["frame_index"]),
                method=method,
                variant=variant,
                p_conditioning=bool(p_conditioning),
                K=0,
                site_fraction=float(site_fraction),
                model_seed=0,
                mask_replica_id=int(replica_id),
                layout_id=str(ds.layout_id),
                task_mode="temperature_free",
                base_channels=32,
                checkpoint_kind=checkpoint_kind,
                run_id=run_id,
                mask_hash=str(meta.get("mask_hash", "")),
            )
            rows.append(row)
    wall = time.perf_counter() - t0
    mean_cond = _mean_condition_rmse(rows, role)
    agg = aggregate_frame_then_condition_then_split(rows)
    return {
        "rows": rows,
        "mean_condition_rmse_K": mean_cond,
        "per_condition": agg["per_condition"],
        "wall_s": wall,
        "n_frames": len(rows),
    }


def build_sparse_swapped_batch_item(
    ds: SparseUNetCacheDataset,
    dest_i: int,
    *,
    pair_lookup: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Recipient mask/p/target; donor vi/vj at the same recipient sites."""
    dest = ds[dest_i]
    dest_idx = int(dest["metadata"]["cache_index"])
    pair = pair_lookup[dest_idx]
    src_idx = int(pair["src_cache_index"])
    sites = np.asarray(dest["observation"].sites, dtype=np.int64)
    vi_n = np.asarray(ds.vi[src_idx], dtype=np.float64)
    vj_n = np.asarray(ds.vj[src_idx], dtype=np.float64)
    vi_p, vj_p = ds._phys_velocities(src_idx)
    obs = build_canonical_observation(
        sites=sites,
        vi_phys=vi_p,
        vj_phys=vj_p,
        vi_norm=vi_n,
        vj_norm=vj_n,
        x_coords=ds.x,
        y_coords=ds.y,
        site_fraction=float(ds.site_fraction),
        layout_id=str(ds.layout_id),
        velocity_mask_seed=int(dest["metadata"]["velocity_mask_seed"]),
    )
    x = build_sparse_unet_input(
        sites=obs.sites,
        velocity_values_norm=obs.velocity_values_norm,
        x_scaled=ds.x_scaled,
        y_scaled=ds.y_scaled,
        nh3_norm=float(ds.p_norm[dest_idx, 0]),
        phi_norm=float(ds.p_norm[dest_idx, 1]),
        p_conditioning=bool(ds.p_conditioning),
    )
    y = np.asarray(ds.t_norm[dest_idx], dtype=np.float32)[None, ...]
    t_K = np.asarray(ds.t_K[dest_idx], dtype=np.float32)[None, ...]
    t_K_src = np.asarray(ds.t_K[src_idx], dtype=np.float32)[None, ...]
    meta = dict(dest["metadata"])
    meta.update(
        {
            "swap_src_cache_index": src_idx,
            "swap_src_frame_index": int(pair["src_frame_index"]),
            "swap_dest_cache_index": dest_idx,
            "swap_dest_frame_index": int(pair["dest_frame_index"]),
            "observation_mode": "snapshot_swapped_sparse",
            "p_conditioning": bool(ds.p_conditioning),
        }
    )
    return {
        "model_inputs": {
            "input_tensor": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "p_conditioning": bool(ds.p_conditioning),
            "channel_order": list(dest["model_inputs"].get("channel_order", [])),
            "K": 0,
            "site_fraction": float(ds.site_fraction),
            "task_mode": "temperature_free",
        },
        "targets": {
            "temperature_normalized": torch.from_numpy(np.array(y, dtype=np.float32, copy=True)),
            "temperature": torch.from_numpy(np.array(t_K, dtype=np.float32, copy=True)),
            "temperature_donor": torch.from_numpy(np.array(t_K_src, dtype=np.float32, copy=True)),
        },
        "observation": obs,
        "glu_tokens": obs.to_glu_tokens(),
        "baseline_physical": obs.to_baseline_physical(),
        "metadata": meta,
    }


@torch.no_grad()
def evaluate_swap(
    *,
    model,
    family: str,
    p_conditioning: bool,
    ds: SparseUNetCacheDataset,
    role: str,
    pair_lookup: dict[int, dict[str, Any]],
    method: str,
    variant: str,
    run_id: str,
    normalizer: Month2FitOnlyNormalizer,
    query_coords_phys: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    ds.replica_id = 0
    ds.site_fraction = 0.05
    ds.mask_mode = "evaluation"
    items = [build_sparse_swapped_batch_item(ds, i, pair_lookup=pair_lookup) for i in range(len(ds))]
    rows: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    sigma = float(normalizer.temperature_std)
    t0 = time.perf_counter()
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        batch = collate_sparse_batch(batch_items)
        # restore donor T for reference metrics
        donor_T = torch.stack(
            [b["targets"]["temperature_donor"] for b in batch_items], dim=0
        ).numpy()
        if family == "adapted_glu":
            _attach_p(batch, ds)
        pred_n = _forward(
            model,
            family,
            batch,
            p_conditioning=p_conditioning,
            query_coords_phys=query_coords_phys,
            device=device,
        )
        true_K = batch["target_T_K"].numpy()
        for i, meta in enumerate(batch["metadata"]):
            pred_K = denormalize_temperature(pred_n[i, 0], normalizer)
            row = evaluate_frame_prediction(
                pred_K,
                true_K[i, 0],
                train_sigma_T=sigma,
                sim_id=str(meta["sim_id"]),
                split=str(meta.get("split", role)),
                role=role,
                frame_index=int(meta["frame_index"]),
                method=method,
                variant=variant,
                p_conditioning=bool(p_conditioning),
                K=0,
                site_fraction=0.05,
                model_seed=0,
                mask_replica_id=0,
                layout_id=str(ds.layout_id),
                task_mode="temperature_free",
                base_channels=32,
                checkpoint_kind="best",
                run_id=run_id,
                observation_mode="snapshot_swapped_sparse",
                mask_hash=str(meta.get("mask_hash", "")),
                swap_src_frame_index=int(meta["swap_src_frame_index"]),
                swap_src_cache_index=int(meta["swap_src_cache_index"]),
            )
            rows.append(row)
            drow = evaluate_frame_prediction(
                donor_T[i, 0],
                true_K[i, 0],
                train_sigma_T=sigma,
                sim_id=str(meta["sim_id"]),
                split=str(meta.get("split", role)),
                role=role,
                frame_index=int(meta["frame_index"]),
                method="T_donor_vs_T_dest",
                variant="reference_only",
                p_conditioning=False,
                K=0,
                site_fraction=0.05,
                model_seed=0,
                mask_replica_id=0,
                layout_id=str(ds.layout_id),
                task_mode="temperature_free",
                base_channels=0,
                checkpoint_kind="n/a",
                run_id=run_id,
                observation_mode="donor_target_reference",
            )
            donor_rows.append(drow)
    wall = time.perf_counter() - t0
    return {
        "rows": rows,
        "donor_reference_rows": donor_rows,
        "mean_condition_rmse_K": _mean_condition_rmse(rows, role),
        "donor_ref_mean_condition_rmse_K": _mean_condition_rmse(donor_rows, role),
        "wall_s": wall,
        "n_frames": len(rows),
    }


def _save_frame1000_figure(
    *,
    out_path: Path,
    true_K: np.ndarray,
    pred_aligned: np.ndarray,
    pred_swapped: np.ndarray | None,
    title: str,
) -> None:
    panels = [("T true", true_K), ("pred aligned", pred_aligned)]
    if pred_swapped is not None:
        panels.append(("pred swapped", pred_swapped))
        panels.append(("|err| aligned", np.abs(pred_aligned - true_K)))
        panels.append(("|err| swapped", np.abs(pred_swapped - true_K)))
    else:
        panels.append(("|err| aligned", np.abs(pred_aligned - true_K)))
    vmin = float(np.min(true_K))
    vmax = float(np.max(true_K))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (lab, data) in zip(axes, panels):
        if lab.startswith("|err|"):
            im = ax.imshow(data, origin="lower", cmap="magma")
        else:
            im = ax.imshow(data, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tip", type=Path, default=ROOT)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--mask-root", type=Path, default=None)
    p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--baseline-summary", type=Path, default=None)
    p.add_argument("--u5-swap-pairs", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--only-run-id",
        action="append",
        default=None,
        help="Optional filter; repeatable. Default: all four S5 runs.",
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        choices=("best", "last"),
        default=["best", "last"],
        help="Which checkpoints to evaluate at 1/5/20%. Swap always uses best.pt.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip coverage tags whose per_frame_*.csv.gz already exists.",
    )
    p.add_argument("--skip-swap", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--expect-git-sha", type=str, default="38ac7fe40178982ac1bd33c5d5602a5e7d928d3b")
    args = p.parse_args()
    ck_eval = tuple(dict.fromkeys(args.checkpoints))  # preserve order, unique

    tip = Path(args.tip)
    runs_root = Path(args.runs_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root or (tip / "data/processed/month2_dense_unet_cache"))
    mask_root = Path(args.mask_root or (tip / "data/processed/month2_sparse_eval_masks"))
    stats = Path(args.stats or (tip / "data/manifests/train_norm_stats_month2_fit.json"))
    baseline_summary = Path(
        args.baseline_summary or (tip / "results/tables/month2_baseline_summary.csv")
    )
    u5_pairs_path = Path(
        args.u5_swap_pairs
        or (
            tip
            / "results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/swap_pairs.json"
        )
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise SystemExit("CUDA requested but unavailable")

    # Provenance
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(
        manifest,
        cache_root=cache_root,
        require_roles=ROLES,
    )
    payload_hashes = compute_cache_payload_hashes(cache_root)
    mask_manifest = json.loads((mask_root / "manifest.json").read_text())
    counts = {r: len(json.loads((cache_root / r / "frames.json").read_text())) for r in ROLES}
    expected_counts = {"train_fit": 760, "seen_train_eval": 220, "val_eval": 123}
    if counts != expected_counts:
        raise SystemExit(f"split count mismatch: {counts} != {expected_counts}")

    prov = {
        "s6_script": str(Path(__file__).resolve()),
        "tip": str(tip),
        "runs_root": str(runs_root),
        "out_root": str(out_root),
        "cache_root": str(cache_root),
        "cache_aggregate_sha256": payload_hashes.get("aggregate_sha256"),
        "geometry_sha256": payload_hashes.get("files", {}).get("geometry.npz"),
        "mask_root": str(mask_root),
        "mask_aggregate_payload_sha256": mask_manifest.get("aggregate_payload_sha256"),
        "mask_manifest_sha256": sha256_file(mask_root / "manifest.json"),
        "stats_sha256": sha256_file(stats),
        "split_counts": counts,
        "expect_git_sha": args.expect_git_sha,
        "device": str(device),
        "gappypod_1_20_status": "open_missing_fitted_pod_artifacts_no_silent_refit",
        "note_eval_history_json": (
            "standalone eval_history.json may omit final early-stop entry; "
            "last.pt embeds complete history — S6 re-evaluates from checkpoints"
        ),
    }
    write_json(out_root / "provenance.json", prov)

    # Baseline reuse @5% only
    baseline_note = {
        "path": str(baseline_summary) if baseline_summary.exists() else None,
        "matched_5pct_available": False,
        "gappypod_1_20": "not_available_no_refit",
        "dense_comparison": "kept_separate_see_docs_month2_03",
    }
    if baseline_summary.exists():
        bdf = pd.read_csv(baseline_summary)
        b5 = bdf[np.isclose(bdf["site_fraction"], 0.05) & (bdf["K"] == 0)].copy()
        b5.to_csv(out_root / "baselines_matched_5pct_reuse.csv", index=False)
        baseline_note["matched_5pct_available"] = True
        baseline_note["n_rows"] = int(len(b5))
        # explicit absence of 1/20
        missing = bdf[bdf["site_fraction"].isin([0.01, 0.20]) & (bdf["method"].astype(str).str.contains("GappyPOD"))]
        baseline_note["gappypod_rows_at_1_or_20"] = int(len(missing))
    write_json(out_root / "baseline_reuse_note.json", baseline_note)

    normalizer = Month2FitOnlyNormalizer(stats)
    only = set(args.only_run_id) if args.only_run_id else None
    summary_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    # Load U5 pairs once
    swap_pairs_all = None
    if not args.skip_swap:
        if not u5_pairs_path.exists():
            raise SystemExit(f"missing U5 swap pairs: {u5_pairs_path}")
        swap_pairs_all = json.loads(u5_pairs_path.read_text())

    for run in RUNS:
        if only is not None and run["run_id"] not in only:
            continue
        run_dir = runs_root / run["run_id"]
        if not run_dir.exists():
            raise SystemExit(f"missing run dir {run_dir}")
        identity = json.loads((run_dir / "run_identity.json").read_text())
        if identity.get("git_sha") != args.expect_git_sha:
            raise SystemExit(
                f"{run['run_id']} git_sha {identity.get('git_sha')} != {args.expect_git_sha}"
            )
        # cache file-level checks (geometry + a couple payloads)
        eng_files = (identity.get("cache_payload_sha256") or {}).get("files") or {}
        for rel in ("geometry.npz", "train_fit/vi_norm.npy", "val_eval/target_T_K.npy"):
            got = (payload_hashes.get("files") or {}).get(rel)
            exp = eng_files.get(rel)
            if exp and got and exp != got:
                raise SystemExit(f"cache file hash mismatch {rel}: {got} != {exp}")

        progress = json.loads((run_dir / "progress.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        cost_rows.append(
            {
                "run_id": run["run_id"],
                "label": run["label"],
                "job_id": run["job_id"],
                "family": run["family"],
                "p_conditioning": run["p_conditioning"],
                "status": summary.get("status"),
                "stop_reason": summary.get("stop_reason"),
                "epochs_completed": summary.get("epochs_completed"),
                "updates": summary.get("updates"),
                "elapsed_s_train": summary.get("elapsed_s"),
                "best_val_epoch": summary.get("best_val_epoch"),
                "best_val_metric": summary.get("best_val_metric"),
                "clip_events": summary.get("clip_events"),
            }
        )

        out_run = out_root / run["run_id"]
        out_run.mkdir(parents=True, exist_ok=True)
        write_json(
            out_run / "run_provenance.json",
            {
                "run_identity": identity,
                "progress": progress,
                "summary_keys": sorted(summary.keys()),
            },
        )

        model, _ = _load_model(run["family"], cache_root, device)
        # Build datasets / queries
        ds_by_role: dict[str, SparseUNetCacheDataset] = {}
        for role in ROLES:
            ds_by_role[role] = SparseUNetCacheDataset(
                role,  # type: ignore[arg-type]
                cache_root=cache_root,
                mask_mode="evaluation",
                site_fraction=0.05,
                eval_mask_seed=20261116,
                layout_id="random_sparse_primary",
                p_conditioning=bool(run["p_conditioning"]),
                normalizer=normalizer,
                eval_mask_payload_root=mask_root,
            )
        q = None
        if run["family"] == "adapted_glu":
            q = make_synthetic_query_grid(ds_by_role["train_fit"].x, ds_by_role["train_fit"].y)
            q = q.to(device)

        method = f"{run['family']}_{'pon' if run['p_conditioning'] else 'poff'}"
        best_epoch_for_swap: int | None = None
        for ck_name in ck_eval:
            ck_path = run_dir / "checkpoints" / f"{ck_name}.pt"
            if not ck_path.exists():
                raise SystemExit(f"missing checkpoint {ck_path}")
            payload = load_sparse_scientific_checkpoint(
                ck_path, model=model, require_resumable=True, restore_rng=False
            )
            model.to(device)
            model.eval()
            ck_epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
            if ck_name == "best":
                best_epoch_for_swap = ck_epoch
            for cov in COVERAGES:
                for role in ROLES:
                    per_rep_scores = []
                    all_rows: list[dict[str, Any]] = []
                    for rid in REPLICAS:
                        tag = f"{ck_name}_cov{int(cov*100):02d}_{role}_rep{rid}"
                        frame_path = out_run / f"per_frame_{tag}.csv.gz"
                        if args.skip_existing and frame_path.exists():
                            print(f"[{run['run_id']}] skip_existing {tag}", flush=True)
                            prev = pd.read_csv(frame_path)
                            # Reconstruct mean-condition for summary continuity.
                            tmp_rows = prev.to_dict(orient="records")
                            score = _mean_condition_rmse(tmp_rows, role)
                            per_rep_scores.append(score)
                            all_rows.extend(tmp_rows)
                            summary_rows.append(
                                {
                                    "run_id": run["run_id"],
                                    "label": run["label"],
                                    "family": run["family"],
                                    "p_conditioning": run["p_conditioning"],
                                    "checkpoint": ck_name,
                                    "checkpoint_epoch": ck_epoch,
                                    "site_fraction": cov,
                                    "role": role,
                                    "replica_id": rid,
                                    "mean_condition_rmse_K": score,
                                    "n_frames": len(tmp_rows),
                                    "wall_s": None,
                                    "skipped_existing": True,
                                    "primary_contrast": bool(
                                        ck_name == "best"
                                        and math.isclose(cov, 0.05)
                                        and role == "val_eval"
                                    ),
                                }
                            )
                            continue
                        print(
                            f"[{run['run_id']}] {tag} epoch_ckpt={ck_epoch}",
                            flush=True,
                        )
                        out = evaluate_setting(
                            model=model,
                            family=run["family"],
                            p_conditioning=bool(run["p_conditioning"]),
                            ds=ds_by_role[role],
                            role=role,
                            replica_id=rid,
                            site_fraction=float(cov),
                            method=method,
                            variant=ck_name,
                            checkpoint_kind=ck_name,
                            run_id=run["run_id"],
                            normalizer=normalizer,
                            query_coords_phys=q,
                            device=device,
                            batch_size=int(args.batch_size),
                        )
                        per_rep_scores.append(out["mean_condition_rmse_K"])
                        all_rows.extend(out["rows"])
                        pd.DataFrame(out["rows"]).to_csv(
                            frame_path,
                            index=False,
                            compression="gzip",
                        )
                        if out["per_condition"] is not None and not getattr(
                            out["per_condition"], "empty", True
                        ):
                            out["per_condition"].to_csv(
                                out_run / f"per_condition_{tag}.csv", index=False
                            )
                        summary_rows.append(
                            {
                                "run_id": run["run_id"],
                                "label": run["label"],
                                "family": run["family"],
                                "p_conditioning": run["p_conditioning"],
                                "checkpoint": ck_name,
                                "checkpoint_epoch": ck_epoch,
                                "site_fraction": cov,
                                "role": role,
                                "replica_id": rid,
                                "mean_condition_rmse_K": out["mean_condition_rmse_K"],
                                "n_frames": out["n_frames"],
                                "wall_s": out["wall_s"],
                                "primary_contrast": bool(
                                    ck_name == "best"
                                    and math.isclose(cov, 0.05)
                                    and role == "val_eval"
                                ),
                            }
                        )
                    # mean over replicas for this cov/role/ckpt
                    scores = [s for s in per_rep_scores if s is not None]
                    summary_rows.append(
                        {
                            "run_id": run["run_id"],
                            "label": run["label"],
                            "family": run["family"],
                            "p_conditioning": run["p_conditioning"],
                            "checkpoint": ck_name,
                            "checkpoint_epoch": ck_epoch,
                            "site_fraction": cov,
                            "role": role,
                            "replica_id": "mean",
                            "mean_condition_rmse_K": float(np.mean(scores)) if scores else None,
                            "n_frames": len(all_rows),
                            "wall_s": None,
                            "primary_contrast": bool(
                                ck_name == "best"
                                and math.isclose(cov, 0.05)
                                and role == "val_eval"
                            ),
                        }
                    )

        # Snapshot swap on best@5% replica 0, seen+VAL (independent of --checkpoints)
        if not args.skip_swap:
            assert swap_pairs_all is not None
            best_path = run_dir / "checkpoints" / "best.pt"
            if not best_path.exists():
                raise SystemExit(f"missing checkpoint {best_path}")
            best_payload = load_sparse_scientific_checkpoint(
                best_path, model=model, require_resumable=True, restore_rng=False
            )
            model.to(device)
            model.eval()
            if best_epoch_for_swap is None:
                best_epoch_for_swap = int(
                    (best_payload.get("epoch_state") or {}).get("epoch", -1)
                )
            for role in ("seen_train_eval", "val_eval"):
                pairs = swap_pairs_all[role]
                lookup = pairs_by_dest_cache_index(pairs)
                write_json(out_run / f"swap_pairs_{role}.json", pairs)
                swap_sum_path = out_run / f"swap_summary_{role}.json"
                if args.skip_existing and swap_sum_path.exists():
                    print(f"[{run['run_id']}] skip_existing swap {role}", flush=True)
                    continue
                print(f"[{run['run_id']}] swap best@5% rep0 {role}", flush=True)
                sw = evaluate_swap(
                    model=model,
                    family=run["family"],
                    p_conditioning=bool(run["p_conditioning"]),
                    ds=ds_by_role[role],
                    role=role,
                    pair_lookup=lookup,
                    method=method,
                    variant="best_swapped",
                    run_id=run["run_id"],
                    normalizer=normalizer,
                    query_coords_phys=q,
                    device=device,
                    batch_size=int(args.batch_size),
                )
                pd.DataFrame(sw["rows"]).to_csv(
                    out_run / f"per_frame_best_swap_{role}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                pd.DataFrame(sw["donor_reference_rows"]).to_csv(
                    out_run / f"per_frame_Tdonor_ref_{role}.csv.gz",
                    index=False,
                    compression="gzip",
                )
                write_json(
                    swap_sum_path,
                    {
                        "role": role,
                        "site_fraction": 0.05,
                        "replica_id": 0,
                        "swapped_mean_condition_rmse_K": sw["mean_condition_rmse_K"],
                        "donor_T_ref_mean_condition_rmse_K": sw[
                            "donor_ref_mean_condition_rmse_K"
                        ],
                        "wall_s": sw["wall_s"],
                        "n_frames": sw["n_frames"],
                    },
                )
                summary_rows.append(
                    {
                        "run_id": run["run_id"],
                        "label": run["label"],
                        "family": run["family"],
                        "p_conditioning": run["p_conditioning"],
                        "checkpoint": "best_swapped",
                        "checkpoint_epoch": best_epoch_for_swap,
                        "site_fraction": 0.05,
                        "role": role,
                        "replica_id": 0,
                        "mean_condition_rmse_K": sw["mean_condition_rmse_K"],
                        "donor_T_ref_mean_condition_rmse_K": sw[
                            "donor_ref_mean_condition_rmse_K"
                        ],
                        "n_frames": sw["n_frames"],
                        "wall_s": sw["wall_s"],
                        "primary_contrast": False,
                    }
                )

                if not args.skip_figures and role == "val_eval":
                    idxs = [
                        i
                        for i in range(len(ds_by_role[role]))
                        if int(ds_by_role[role].get_frame_meta(i)["frame_index"]) == 1000
                    ]
                    if idxs:
                        i0 = idxs[0]
                        ds_by_role[role].replica_id = 0
                        ds_by_role[role].site_fraction = 0.05
                        aligned_item = ds_by_role[role][i0]
                        swapped_item = build_sparse_swapped_batch_item(
                            ds_by_role[role], i0, pair_lookup=lookup
                        )
                        pred_al = pred_sw = true_K = None
                        for mode, item in (("aligned", aligned_item), ("swapped", swapped_item)):
                            batch = collate_sparse_batch([item])
                            if run["family"] == "adapted_glu":
                                _attach_p(batch, ds_by_role[role])
                            pred_n = _forward(
                                model,
                                run["family"],
                                batch,
                                p_conditioning=bool(run["p_conditioning"]),
                                query_coords_phys=q,
                                device=device,
                            )
                            pred_K = denormalize_temperature(pred_n[0, 0], normalizer)
                            if mode == "aligned":
                                pred_al = pred_K
                                true_K = item["targets"]["temperature"].numpy()[0]
                            else:
                                pred_sw = pred_K
                        _save_frame1000_figure(
                            out_path=out_run / "figures" / "val_frame1000_aligned_vs_swapped.png",
                            true_K=true_K,
                            pred_aligned=pred_al,
                            pred_swapped=pred_sw,
                            title=f"{run['label']} best @5% rep0 frame_index=1000",
                        )

        # training curves figure
        if not args.skip_figures:
            curves_path = run_dir / "curves.csv.gz"
            if curves_path.exists():
                cdf = pd.read_csv(curves_path)
                fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
                if "epoch" in cdf.columns and "epoch_mean_loss" in cdf.columns:
                    ax.plot(cdf["epoch"], cdf["epoch_mean_loss"], lw=1.0)
                    ax.set_xlabel("epoch")
                    ax.set_ylabel("TRAIN mean loss")
                    ax.set_title(f"{run['label']} curves")
                    fig.savefig(out_run / "figures" / "train_loss_curves.png", dpi=140)
                plt.close(fig)
            # LR / best markers from eval_history in last.pt
            last_payload = torch.load(
                run_dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
            )
            eh = last_payload.get("eval_history") or []
            if eh:
                edf = pd.DataFrame(eh)
                fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
                ax.plot(edf["epoch"], edf["val_mean_condition_rmse_K"], label="VAL S", lw=1.0)
                be = summary.get("best_val_epoch")
                if be is not None:
                    ax.axvline(be, color="C1", ls="--", label=f"best@{be}")
                ax.set_xlabel("epoch")
                ax.set_ylabel("VAL mean-condition RMSE [K]")
                ax.legend(fontsize=8)
                ax.set_title(f"{run['label']} VAL selection metric")
                (out_run / "figures").mkdir(parents=True, exist_ok=True)
                fig.savefig(out_run / "figures" / "val_selection_curve.png", dpi=140)
                plt.close(fig)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out_root / "summary_all_settings.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(out_root / "training_cost_from_s5.csv", index=False)

    # Primary contrast table: best VAL @5% mean over replicas
    prim = sdf[
        (sdf["checkpoint"] == "best")
        & np.isclose(sdf["site_fraction"].astype(float), 0.05)
        & (sdf["role"] == "val_eval")
        & (sdf["replica_id"].astype(str) == "mean")
    ].copy()
    prim.to_csv(out_root / "primary_best_val_5pct.csv", index=False)

    # Coverage curves for best on VAL
    cov = sdf[
        (sdf["checkpoint"] == "best")
        & (sdf["role"] == "val_eval")
        & (sdf["replica_id"].astype(str) == "mean")
    ].copy()
    cov.to_csv(out_root / "best_val_coverage_curve.csv", index=False)

    write_json(
        out_root / "report_stub.json",
        {
            "status": "s6_eval_complete",
            "primary_table": "primary_best_val_5pct.csv",
            "baseline_note": baseline_note,
            "provenance": prov,
            "n_summary_rows": int(len(sdf)),
        },
    )
    print(json.dumps({"status": "s6_eval_complete", "out_root": str(out_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
