#!/usr/bin/env python3
"""U5 — snapshot-swap control on frozen U4 best checkpoint (no training)."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import sha256_file
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.models.baselines import (
    GlobalTrainMeanPredictor,
    ParamConditionedMeanPredictor,
)
from combustion_recon.models.unet import build_dense_unet_from_config
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DenseUNetCacheDataset,
    compute_cache_payload_hashes,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.epoch_loop import (
    apply_tf32_policy,
    load_epoch_checkpoint,
)
from combustion_recon.training.snapshot_swap import (
    build_swapped_input_from_cache,
    build_within_condition_swap_pairs,
    pairs_by_dest_cache_index,
)
from combustion_recon.training.trainer import write_json

# Declared BEFORE aligned replay comparison (same Engaging GPU preferred).
ALIGNED_REPLAY_ATOL_K = 0.05
ALIGNED_REPLAY_RTOL = 1e-5

# U4 recorded macros at best epoch 160 (selection.json).
U4_REF_VAL_MACRO = 26.931456559998264
U4_REF_SEEN_MACRO = 37.675490288478024
U4_BEST_EPOCH = 160
U4_BEST_CKPT_SHA256 = "6d456abcb5653c16a7a90df40eda377caf42ffb94fd829a65f4c06be58a03634"

# U3 dense GappyPOD selected λ (no re-search).
U3_GAPPY_PON_LAMBDA = 0.01
PARAM_MEAN_RIDGE = 0.01
PARAM_MEAN_BASIS = "piecewise_phi1"

VAL_FIGURE_FRAME_INDEX = 1000


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _macro_from_rows(rows: list[dict[str, Any]], role: str) -> float:
    agg = aggregate_frame_then_condition_then_split(rows)
    if role == "val_eval":
        sel = agg["selection_mean_condition_rmse"]
        return float(sel["mean_condition_rmse_K"].iloc[0])
    return float(agg["per_condition"]["rmse_K"].mean())


@torch.no_grad()
def eval_unet_role(
    *,
    model: torch.nn.Module,
    ds: DenseUNetCacheDataset,
    device: torch.device,
    normalizer: Month2FitOnlyNormalizer,
    mode: str,
    pair_lookup: dict[int, dict[str, Any]] | None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    model.eval()
    sigma = float(normalizer.temperature_std)
    rows: list[dict[str, Any]] = []
    n = len(ds)
    for start in range(0, n, batch_size):
        batch_items = []
        for i in range(start, min(start + batch_size, n)):
            if mode == "aligned":
                batch_items.append(ds[i])
            elif mode == "swapped":
                assert pair_lookup is not None
                batch_items.append(
                    build_swapped_input_from_cache(ds, i, pair_lookup=pair_lookup)
                )
            else:
                raise ValueError(mode)
        x = torch.stack(
            [b["model_inputs"]["input_tensor"] for b in batch_items], dim=0
        ).to(device)
        pred_n = model(x).detach().cpu().numpy()
        for j, item in enumerate(batch_items):
            meta = item["metadata"]
            true_K = item["targets"]["temperature"].numpy()[0]
            pred_K = denormalize_temperature(pred_n[j, 0], normalizer)
            row = evaluate_frame_prediction(
                pred_K,
                true_K,
                train_sigma_T=sigma,
                sim_id=str(meta["sim_id"]),
                split=str(meta["split"]),
                role=str(ds.role),
                frame_index=int(meta["frame_index"]),
                method="DenseUNet",
                variant="p_on",
                p_conditioning=True,
                K=0,
                site_fraction=1.0,
                model_seed=0,
                mask_replica_id=0,
                layout_id="dense_full_grid",
                task_mode="temperature_free",
                base_channels=32,
                snapshot_control=mode,
                checkpoint_epoch=U4_BEST_EPOCH,
            )
            if mode == "swapped":
                row["src_frame_index"] = int(meta["src_frame_index"])
                row["src_cache_index"] = int(meta["src_cache_index"])
            rows.append(row)
    return rows


def _predict_field_unet(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    normalizer: Month2FitOnlyNormalizer,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_n = model(x.unsqueeze(0).to(device)).detach().cpu().numpy()[0, 0]
    return denormalize_temperature(pred_n, normalizer)


def build_baselines_on_train(
    *,
    cache_root: Path,
    normalizer: Month2FitOnlyNormalizer,
) -> tuple[GlobalTrainMeanPredictor, ParamConditionedMeanPredictor, Any]:
    """Reconstruct U3-selected baseline objects (same hyperparams; no λ search)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    u3 = import_module("08_dense_linear_reference")
    from combustion_recon.models.baselines import GappyPODReconstructor

    train_frames = u3.dense_physical_frames_from_cache(
        "train_fit", cache_root=cache_root, normalizer=normalizer
    )
    global_mean = GlobalTrainMeanPredictor.fit([f["T"] for f in train_frames])
    by_cond: dict[str, list[np.ndarray]] = {}
    ops: dict[str, tuple[float, float]] = {}
    for fr in train_frames:
        by_cond.setdefault(fr["sim_id"], []).append(fr["T"])
        ops[fr["sim_id"]] = (float(fr["nh3"]), float(fr["phi"]))
    cond_means = [np.mean(np.stack(v, 0), 0) for _, v in sorted(by_cond.items())]
    cond_ops = [ops[sid] for sid, _ in sorted(by_cond.items())]
    param_mean = ParamConditionedMeanPredictor.fit(
        cond_means,
        cond_ops,
        ridge_alpha=PARAM_MEAN_RIDGE,
        basis=PARAM_MEAN_BASIS,
    )
    pod = GappyPODReconstructor.fit(
        train_frames,
        R=32,
        ridge_lambda=U3_GAPPY_PON_LAMBDA,
        w_T=5.0,
        use_param_mean=True,
        param_mean_predictor=param_mean,
        n_velocity_components=2,
        seed=0,
    )
    return global_mean, param_mean, pod


def eval_baselines_role(
    *,
    role: str,
    cache_root: Path,
    normalizer: Month2FitOnlyNormalizer,
    global_mean: GlobalTrainMeanPredictor,
    param_mean: ParamConditionedMeanPredictor,
    pod: Any,
) -> dict[str, list[dict[str, Any]]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    u3 = import_module("08_dense_linear_reference")
    from combustion_recon.models.month2_baseline_adapters import (
        predict_gappy,
        predict_global_mean,
        predict_param_mean,
    )

    frames = u3.dense_physical_frames_from_cache(
        role, cache_root=cache_root, normalizer=normalizer
    )
    sigma = float(normalizer.temperature_std)
    out: dict[str, list[dict[str, Any]]] = {
        "GlobalMean": [],
        "ParamMean": [],
        "GappyPOD_dense_p_on": [],
    }
    for fr in frames:
        mi = u3.dense_model_inputs_from_frame(fr, p_conditioning=True)
        preds = {
            "GlobalMean": predict_global_mean(global_mean),
            "ParamMean": predict_param_mean(param_mean, mi),
            "GappyPOD_dense_p_on": predict_gappy(
                pod,
                mi,
                use_temperature_block=False,
                anchor_offset=False,
                p_conditioning=True,
            ),
        }
        for method, pred in preds.items():
            out[method].append(
                evaluate_frame_prediction(
                    pred,
                    fr["T"],
                    train_sigma_T=sigma,
                    sim_id=fr["sim_id"],
                    split=fr["split"],
                    role=role,
                    frame_index=int(fr["frame_index"]),
                    method=method,
                    variant="p_on",
                    p_conditioning=True,
                    K=0,
                    site_fraction=1.0,
                    model_seed=0,
                    mask_replica_id=0,
                    layout_id="dense_full_grid",
                    task_mode="temperature_free",
                    snapshot_control="aligned_baseline",
                    **(
                        {"R": 32, "ridge_lambda": U3_GAPPY_PON_LAMBDA}
                        if "Gappy" in method
                        else {}
                    ),
                )
            )
    return out


def paired_delta_tables(
    aligned_rows: list[dict[str, Any]],
    swapped_rows: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = pd.DataFrame(aligned_rows)
    s = pd.DataFrame(swapped_rows)
    keys = ["role", "sim_id", "frame_index"]
    merged = a.merge(s, on=keys, suffixes=("_aligned", "_swapped"))
    if len(merged) != len(a) or len(merged) != len(s):
        raise RuntimeError(
            f"aligned/swapped pairing mismatch: {len(a)} vs {len(s)} → {len(merged)}"
        )
    for m in metric_keys:
        merged[f"delta_{m}"] = merged[f"{m}_swapped"] - merged[f"{m}_aligned"]
    # Aggregate frame → condition (mean of deltas and of metrics)
    gcols = ["role", "sim_id"]
    agg_spec = {}
    for m in metric_keys:
        agg_spec[f"{m}_aligned"] = "mean"
        agg_spec[f"{m}_swapped"] = "mean"
        agg_spec[f"delta_{m}"] = "mean"
    per_cond = merged.groupby(gcols, as_index=False).agg(agg_spec)
    # role macro: mean over conditions
    per_role = per_cond.groupby("role", as_index=False).mean(numeric_only=True)
    return merged, per_cond, per_role


def make_figures(
    *,
    out_dir: Path,
    cache_root: Path,
    model: torch.nn.Module,
    device: torch.device,
    normalizer: Month2FitOnlyNormalizer,
    pair_lookup_val: dict[int, dict[str, Any]],
    global_mean: GlobalTrainMeanPredictor,
    param_mean: ParamConditionedMeanPredictor,
    pod: Any,
    eval_history_path: Path,
) -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    u3 = import_module("08_dense_linear_reference")
    from combustion_recon.models.month2_baseline_adapters import (
        predict_gappy,
        predict_global_mean,
        predict_param_mean,
    )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    ds = DenseUNetCacheDataset("val_eval", cache_root=cache_root, p_conditioning=True)
    # Fixed frame index 1000 per VAL condition (roadmap).
    targets = []
    for i, meta in enumerate(ds.frames):
        if int(meta["frame_index"]) == VAL_FIGURE_FRAME_INDEX:
            targets.append((i, meta))
    if len(targets) != 3:
        raise RuntimeError(
            f"expected 3 VAL frames at index {VAL_FIGURE_FRAME_INDEX}, got {len(targets)}"
        )

    # Common scales from these panels
    fields_T = []
    panel_data = []
    for i, meta in targets:
        item = ds[i]
        true_K = item["targets"]["temperature"].numpy()[0]
        x_al = item["model_inputs"]["input_tensor"]
        swapped = build_swapped_input_from_cache(ds, i, pair_lookup=pair_lookup_val)
        x_sw = swapped["model_inputs"]["input_tensor"]
        pred_al = _predict_field_unet(model, x_al, device=device, normalizer=normalizer)
        pred_sw = _predict_field_unet(model, x_sw, device=device, normalizer=normalizer)
        fr = {
            "T": true_K.astype(np.float64),
            "vi": normalizer.denormalize_field(
                "Velocity_i", np.asarray(ds.vi[ds.indices[i]], dtype=np.float64)
            ),
            "vj": normalizer.denormalize_field(
                "Velocity_j", np.asarray(ds.vj[ds.indices[i]], dtype=np.float64)
            ),
            "nh3": float(meta["nh3_fraction"]),
            "phi": float(meta["phi"]),
            "sim_id": meta["sim_id"],
            "frame_index": int(meta["frame_index"]),
            "split": meta["split"],
            "role": "val_eval",
        }
        mi = u3.dense_model_inputs_from_frame(fr, p_conditioning=True)
        pred_gm = predict_global_mean(global_mean)
        pred_pm = predict_param_mean(param_mean, mi)
        pred_gp = predict_gappy(
            pod, mi, use_temperature_block=False, anchor_offset=False, p_conditioning=True
        )
        panel_data.append(
            {
                "sim_id": meta["sim_id"],
                "true": true_K,
                "ParamMean": pred_pm,
                "GappyPOD": pred_gp,
                "UNet_aligned": pred_al,
                "UNet_swapped": pred_sw,
                "GlobalMean": pred_gm,
            }
        )
        fields_T.extend([true_K, pred_pm, pred_gp, pred_al, pred_sw])

    t_lo = float(np.min([f.min() for f in fields_T]))
    t_hi = float(np.max([f.max() for f in fields_T]))
    err_maps = []
    for p in panel_data:
        for key in ("ParamMean", "GappyPOD", "UNet_aligned", "UNet_swapped"):
            err_maps.append(p[key] - p["true"])
    e_abs = float(np.max([np.max(np.abs(e)) for e in err_maps]))
    e_lim = e_abs

    for p in panel_data:
        sim = str(p["sim_id"]).replace(".h5", "")
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), constrained_layout=True)
        recon_keys = [
            ("true", "T true"),
            ("ParamMean", "ParamMean"),
            ("GappyPOD", "GappyPOD dense"),
            ("UNet_aligned", "U-Net aligned"),
            ("UNet_swapped", "U-Net swapped"),
        ]
        for ax, (key, title) in zip(axes[0], recon_keys):
            im = ax.imshow(p[key], origin="lower", cmap="inferno", vmin=t_lo, vmax=t_hi)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02, label="T [K]")
        err_keys = [
            ("ParamMean", "err ParamMean"),
            ("GappyPOD", "err GappyPOD"),
            ("UNet_aligned", "err U-Net aligned"),
            ("UNet_swapped", "err U-Net swapped"),
            ("UNet_swapped", "err swapped−aligned"),
        ]
        # last panel: swapped - aligned prediction difference (not vs truth)
        for col, (key, title) in enumerate(err_keys):
            ax = axes[1, col]
            if title.startswith("err swapped"):
                data = p["UNet_swapped"] - p["UNet_aligned"]
            else:
                data = p[key] - p["true"]
            im2 = ax.imshow(
                data, origin="lower", cmap="coolwarm", vmin=-e_lim, vmax=e_lim
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im2, ax=axes[1].tolist(), fraction=0.046, pad=0.02, label="ΔT [K]")
        fig.suptitle(
            f"VAL frame_index={VAL_FIGURE_FRAME_INDEX} · {sim}\n"
            f"T scale [{t_lo:.0f},{t_hi:.0f}] K · error scale ±{e_lim:.1f} K (common)",
            fontsize=11,
        )
        outp = fig_dir / f"val_frame{VAL_FIGURE_FRAME_INDEX}_{sim}_panels.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

    # Mean absolute error maps over all VAL frames (aligned and swapped)
    ds_val = DenseUNetCacheDataset("val_eval", cache_root=cache_root, p_conditioning=True)
    acc_al = np.zeros((128, 128), dtype=np.float64)
    acc_sw = np.zeros((128, 128), dtype=np.float64)
    n = 0
    for i in range(len(ds_val)):
        item = ds_val[i]
        true = item["targets"]["temperature"].numpy()[0]
        pred_al = _predict_field_unet(
            model, item["model_inputs"]["input_tensor"], device=device, normalizer=normalizer
        )
        swapped = build_swapped_input_from_cache(
            ds_val, i, pair_lookup=pair_lookup_val
        )
        pred_sw = _predict_field_unet(
            model,
            swapped["model_inputs"]["input_tensor"],
            device=device,
            normalizer=normalizer,
        )
        acc_al += np.abs(pred_al - true)
        acc_sw += np.abs(pred_sw - true)
        n += 1
    mean_al = acc_al / n
    mean_sw = acc_sw / n
    vmax = float(max(mean_al.max(), mean_sw.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        [mean_al, mean_sw, mean_sw - mean_al],
        ["mean |err| aligned", "mean |err| swapped", "Δ mean|err| (sw−al)"],
    ):
        if title.startswith("Δ"):
            lim = float(np.max(np.abs(data)))
            im = ax.imshow(data, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(data, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    outp = fig_dir / "val_mean_abs_error_aligned_vs_swapped.png"
    fig.savefig(outp, dpi=160)
    plt.close(fig)
    paths.append(str(outp.relative_to(out_dir)))

    # Quantitative per-condition bar chart (filled later by caller via CSV if needed)
    # U4 curves
    if eval_history_path.is_file():
        eh = json.loads(eval_history_path.read_text())
        ep = [e["epoch"] for e in eh]
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(ep, [e["val_mean_condition_rmse_K"] for e in eh], label="VAL")
        ax.plot(ep, [e["seen_mean_condition_rmse_K"] for e in eh], label="seen")
        ax.plot(ep, [e["train_mean_condition_rmse_K"] for e in eh], label="TRAIN")
        ax.axvline(U4_BEST_EPOCH, color="k", ls="--", lw=1, label=f"best@{U4_BEST_EPOCH}")
        ax.axvline(730, color="gray", ls=":", lw=1, label="early stop@730")
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro RMSE [K]")
        ax.set_title("U4 curves (p_on seed 0)")
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(ep, [e["lr"] for e in eh], color="tab:purple", alpha=0.5, label="lr")
        ax2.set_ylabel("lr")
        outp = fig_dir / "u4_curves_train_seen_val_lr.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

    # scale declaration
    write_json(
        fig_dir / "figure_scales.json",
        {
            "val_frame_index": VAL_FIGURE_FRAME_INDEX,
            "T_vmin_K": t_lo,
            "T_vmax_K": t_hi,
            "error_symmetric_limit_K": e_lim,
            "mean_abs_error_vmax_K": vmax,
            "note": "scales fixed from the three VAL frame-1000 panels before export",
        },
    )
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--expect-cache-hashes", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to U4 best.pt (epoch 160)",
    )
    p.add_argument("--expect-ckpt-sha256", type=str, default=U4_BEST_CKPT_SHA256)
    p.add_argument(
        "--u4-run-dir",
        type=Path,
        default=ROOT / "results/month2_dense_unet/u4_pon_seed0_engaging_v1",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refuse overwrite of existing {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Declare tolerance before comparison.
    tolerances = {
        "aligned_replay_atol_K": ALIGNED_REPLAY_ATOL_K,
        "aligned_replay_rtol": ALIGNED_REPLAY_RTOL,
        "declared_before_replay": True,
        "u4_ref_val_macro_K": U4_REF_VAL_MACRO,
        "u4_ref_seen_macro_K": U4_REF_SEEN_MACRO,
    }
    write_json(out_dir / "tolerances_declared.json", tolerances)

    tf32 = apply_tf32_policy(allow_tf32=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    cache_root = Path(args.cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    validate_dense_cache_manifest(
        manifest,
        cache_root=cache_root,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
    )
    expected_hashes = json.loads(Path(args.expect_cache_hashes).read_text())
    verified = verify_cache_payload_hashes(expected_hashes, cache_root=cache_root)
    if not verified["ok"]:
        raise SystemExit(f"cache payload hash mismatch: {verified['mismatches']}")

    ckpt_path = Path(args.checkpoint)
    ckpt_sha = _sha256_bytes(ckpt_path)
    if ckpt_sha != args.expect_ckpt_sha256:
        raise SystemExit(
            f"checkpoint sha mismatch: got {ckpt_sha}, expected {args.expect_ckpt_sha256}"
        )

    normalizer = Month2FitOnlyNormalizer()
    model = build_dense_unet_from_config(init_seed=0, base_channels=32)
    load_epoch_checkpoint(
        ckpt_path,
        model=model,
        map_location="cpu",
        require_resumable=True,
        restore_rng=False,
    )
    model.to(device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # Pairings (deterministic; saved before metrics interpretation)
    pairings: dict[str, list[dict[str, Any]]] = {}
    lookups: dict[str, dict[int, dict[str, Any]]] = {}
    for role in ("seen_train_eval", "val_eval"):
        ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=True)
        pairs = build_within_condition_swap_pairs(ds.frames, role=role)
        pairings[role] = pairs
        lookups[role] = pairs_by_dest_cache_index(pairs)
    write_json(out_dir / "swap_pairs.json", pairings)

    # Aligned replay
    aligned_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("seen_train_eval", "val_eval"):
        ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=True)
        aligned_rows[role] = eval_unet_role(
            model=model,
            ds=ds,
            device=device,
            normalizer=normalizer,
            mode="aligned",
            pair_lookup=None,
            batch_size=args.batch_size,
        )
    val_macro = _macro_from_rows(aligned_rows["val_eval"], "val_eval")
    seen_macro = _macro_from_rows(aligned_rows["seen_train_eval"], "seen_train_eval")
    replay = {
        "val_macro_rmse_K": val_macro,
        "seen_macro_rmse_K": seen_macro,
        "val_abs_diff_vs_u4": abs(val_macro - U4_REF_VAL_MACRO),
        "seen_abs_diff_vs_u4": abs(seen_macro - U4_REF_SEEN_MACRO),
        "val_ok": abs(val_macro - U4_REF_VAL_MACRO)
        <= ALIGNED_REPLAY_ATOL_K
        + ALIGNED_REPLAY_RTOL * abs(U4_REF_VAL_MACRO),
        "seen_ok": abs(seen_macro - U4_REF_SEEN_MACRO)
        <= ALIGNED_REPLAY_ATOL_K
        + ALIGNED_REPLAY_RTOL * abs(U4_REF_SEEN_MACRO),
    }
    write_json(out_dir / "aligned_replay.json", replay)
    if not (replay["val_ok"] and replay["seen_ok"]):
        write_json(
            out_dir / "STOP_aligned_replay_failed.json",
            {"replay": replay, "tolerances": tolerances},
        )
        print(json.dumps({"status": "aligned_replay_failed", "replay": replay}, indent=2))
        return 3

    # Swapped eval
    swapped_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("seen_train_eval", "val_eval"):
        ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=True)
        swapped_rows[role] = eval_unet_role(
            model=model,
            ds=ds,
            device=device,
            normalizer=normalizer,
            mode="swapped",
            pair_lookup=lookups[role],
            batch_size=args.batch_size,
        )

    metric_keys = (
        "rmse_K",
        "mean_signed_error_K",
        "abs_bias_K",
        "centered_rmse_K",
        "jet_rmse_K",
        "high_rmse_K",
        "jet_mae_K",
        "high_mae_K",
        "high_n_voxels",
    )
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_blocks = {}
    for role in ("seen_train_eval", "val_eval"):
        # Persist frame tables separately (homogeneous configs)
        pd.DataFrame(aligned_rows[role]).to_csv(
            tables_dir / f"{role}_unet_aligned_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(swapped_rows[role]).to_csv(
            tables_dir / f"{role}_unet_swapped_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        per_frame, per_cond, per_role = paired_delta_tables(
            aligned_rows[role], swapped_rows[role], metric_keys=metric_keys
        )
        per_frame.to_csv(
            tables_dir / f"{role}_paired_delta_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        per_cond.to_csv(
            tables_dir / f"{role}_paired_delta_per_condition.csv.gz",
            index=False,
            compression="gzip",
        )
        per_role.to_csv(
            tables_dir / f"{role}_paired_delta_per_role.csv.gz",
            index=False,
            compression="gzip",
        )
        # Also store aggregations without mixing modes
        for mode, rows in (("aligned", aligned_rows[role]), ("swapped", swapped_rows[role])):
            agg = aggregate_frame_then_condition_then_split(rows)
            agg["per_condition"].to_csv(
                tables_dir / f"{role}_unet_{mode}_per_condition.csv.gz",
                index=False,
                compression="gzip",
            )
        summary_blocks[role] = {
            "aligned_macro_rmse_K": _macro_from_rows(aligned_rows[role], role),
            "swapped_macro_rmse_K": _macro_from_rows(swapped_rows[role], role),
            "delta_macro_rmse_K": float(per_role["delta_rmse_K"].iloc[0]),
            "per_condition": per_cond.to_dict(orient="records"),
        }

    # Matched baselines (aligned only; reconstructed with U3-selected hyperparams)
    print("Fitting matched baselines (U3-selected hyperparams; no λ search)...", flush=True)
    global_mean, param_mean, pod = build_baselines_on_train(
        cache_root=cache_root, normalizer=normalizer
    )
    matched = {}
    for role in ("seen_train_eval", "val_eval"):
        base_rows = eval_baselines_role(
            role=role,
            cache_root=cache_root,
            normalizer=normalizer,
            global_mean=global_mean,
            param_mean=param_mean,
            pod=pod,
        )
        matched[role] = {}
        for method, rows in base_rows.items():
            pd.DataFrame(rows).to_csv(
                tables_dir / f"{role}_{method}_per_frame.csv.gz",
                index=False,
                compression="gzip",
            )
            agg = aggregate_frame_then_condition_then_split(rows)
            agg["per_condition"].to_csv(
                tables_dir / f"{role}_{method}_per_condition.csv.gz",
                index=False,
                compression="gzip",
            )
            matched[role][method] = {
                "macro_rmse_K": _macro_from_rows(rows, role),
                "per_condition": agg["per_condition"][
                    ["sim_id", "rmse_K", "abs_bias_K", "centered_rmse_K", "jet_rmse_K", "high_rmse_K"]
                ].to_dict(orient="records")
                if not agg["per_condition"].empty
                else [],
            }
        matched[role]["DenseUNet_aligned"] = {
            "macro_rmse_K": summary_blocks[role]["aligned_macro_rmse_K"]
        }
        matched[role]["DenseUNet_swapped"] = {
            "macro_rmse_K": summary_blocks[role]["swapped_macro_rmse_K"]
        }

    # Verify U3 GappyPOD VAL macro reproduces selected ~70.46 K within loose tol
    gappy_val = matched["val_eval"]["GappyPOD_dense_p_on"]["macro_rmse_K"]
    matched_check = {
        "gappy_val_macro_rmse_K": gappy_val,
        "u3_selected_val": 70.46287221326743,
        "abs_diff": abs(gappy_val - 70.46287221326743),
        "ok_within_0_5K": abs(gappy_val - 70.46287221326743) <= 0.5,
        "note": "reconstructed with selected λ=0.01; not a new search",
    }
    write_json(out_dir / "matched_baseline_check.json", matched_check)

    fig_paths: list[str] = []
    if not args.skip_figures:
        fig_paths = make_figures(
            out_dir=out_dir,
            cache_root=cache_root,
            model=model,
            device=device,
            normalizer=normalizer,
            pair_lookup_val=lookups["val_eval"],
            global_mean=global_mean,
            param_mean=param_mean,
            pod=pod,
            eval_history_path=Path(args.u4_run_dir) / "eval_history.json",
        )

    # Comparison bar data for VAL conditions
    val_cond = pd.read_csv(tables_dir / "val_eval_paired_delta_per_condition.csv.gz")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sims = [s.replace(".h5", "") for s in val_cond["sim_id"]]
    x = np.arange(len(sims))
    w = 0.35
    ax.bar(x - w / 2, val_cond["rmse_K_aligned"], w, label="U-Net aligned")
    ax.bar(x + w / 2, val_cond["rmse_K_swapped"], w, label="U-Net swapped")
    # overlay baselines macros per condition from matched tables
    for method, marker in (
        ("ParamMean", "o"),
        ("GappyPOD_dense_p_on", "s"),
        ("GlobalMean", "^"),
    ):
        pc = pd.read_csv(tables_dir / f"val_eval_{method}_per_condition.csv.gz")
        ax.scatter(
            x,
            [
                float(pc.loc[pc["sim_id"] == sid, "rmse_K"].iloc[0])
                for sid in val_cond["sim_id"]
            ],
            marker=marker,
            label=method,
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sims, rotation=15)
    ax.set_ylabel("RMSE [K]")
    ax.set_title("VAL per-condition RMSE (aligned/swapped + matched baselines)")
    ax.legend(fontsize=7)
    outp = fig_dir / "val_per_condition_rmse_comparison.png"
    fig.savefig(outp, dpi=160)
    plt.close(fig)
    fig_paths.append(str(outp.relative_to(out_dir)))

    env = {
        "git_commit": _git_head(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "device": str(device),
        "tf32_policy": tf32,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "cache_payload_sha256": verified["actual_aggregate_sha256"],
        "peak_rss_GB": _rss_gb(),
        "elapsed_s": time.perf_counter() - t0,
    }
    summary = {
        "experiment_id": "u5_pon_seed0_snapshot_control_v1",
        "status": "completed",
        "aligned_replay": replay,
        "tolerances": tolerances,
        "unet_swap_summary": summary_blocks,
        "matched_baselines": matched,
        "matched_baseline_check": matched_check,
        "figures": fig_paths,
        "environment": env,
        "pairing_counts": {k: len(v) for k, v in pairings.items()},
        "limits": [
            "VAL used for U4 selection/scheduler/early-stop — not an independent final test",
            "Single seed; no robustness claim",
            "Swap degradation ≠ physical causality or information ceiling",
            "Sparse 5% baselines excluded from matched dense ranking",
        ],
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "run_manifest.json", {**summary, "pairings_path": "swap_pairs.json"})

    # Auto-report (numeric only)
    lines = [
        "# U5 auto-report — snapshot control (numeric)",
        "",
        f"- Commit: `{env['git_commit']}`",
        f"- Checkpoint sha256: `{ckpt_sha}` (epoch {U4_BEST_EPOCH})",
        f"- Device: {env['gpu_name']} ({env['device']})",
        f"- Aligned replay VAL/seen: {val_macro:.6f} / {seen_macro:.6f} K "
        f"(Δ vs U4 {replay['val_abs_diff_vs_u4']:.4g} / {replay['seen_abs_diff_vs_u4']:.4g}; "
        f"atol={ALIGNED_REPLAY_ATOL_K} K) → OK",
        "",
        "## Macro RMSE [K]",
        "",
        "| Role | UNet aligned | UNet swapped | Δ (sw−al) | ParamMean | GappyPOD dense | GlobalMean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("val_eval", "seen_train_eval"):
        lines.append(
            "| {role} | {a:.3f} | {s:.3f} | {d:+.3f} | {pm:.3f} | {gp:.3f} | {gm:.3f} |".format(
                role=role,
                a=summary_blocks[role]["aligned_macro_rmse_K"],
                s=summary_blocks[role]["swapped_macro_rmse_K"],
                d=summary_blocks[role]["delta_macro_rmse_K"],
                pm=matched[role]["ParamMean"]["macro_rmse_K"],
                gp=matched[role]["GappyPOD_dense_p_on"]["macro_rmse_K"],
                gm=matched[role]["GlobalMean"]["macro_rmse_K"],
            )
        )
    lines.extend(["", "## VAL per-condition Δ RMSE (sw−al)", ""])
    for rec in summary_blocks["val_eval"]["per_condition"]:
        lines.append(
            f"- {rec['sim_id']}: aligned {rec['rmse_K_aligned']:.3f} → swapped "
            f"{rec['rmse_K_swapped']:.3f} (Δ {rec['delta_rmse_K']:+.3f} K); "
            f"Δ centered {rec['delta_centered_rmse_K']:+.3f}; "
            f"Δ |bias| {rec['delta_abs_bias_K']:+.3f}"
        )
    lines.extend(["", "## Figures", ""])
    for fp in fig_paths:
        lines.append(f"- `{fp}`")
    (out_dir / "auto_report.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "summary": summary_blocks}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
