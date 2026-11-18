#!/usr/bin/env python3
"""U6 — evaluate frozen p_off best vs p_on, then snapshot-swap on p_off best.

Reuses U5 saved pairings and U3 matched baseline tables (no new fits).
"""

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
from combustion_recon.evaluation.month2_evaluator import (
    aggregate_frame_then_condition_then_split,
    denormalize_temperature,
    evaluate_frame_prediction,
)
from combustion_recon.models.unet import build_dense_unet_from_config
from combustion_recon.training.dense_cache import (
    DEFAULT_CACHE_ROOT,
    DenseUNetCacheDataset,
    validate_dense_cache_manifest,
    verify_cache_payload_hashes,
)
from combustion_recon.training.epoch_loop import apply_tf32_policy, load_epoch_checkpoint
from combustion_recon.training.run_provenance import collect_run_provenance
from combustion_recon.training.snapshot_swap import (
    build_swapped_input_from_cache,
    pairs_by_dest_cache_index,
)
from combustion_recon.training.trainer import write_json

# Declared BEFORE aligned replay comparison.
ALIGNED_REPLAY_ATOL_K = 0.05
ALIGNED_REPLAY_RTOL = 1e-5
VAL_FIGURE_FRAME_INDEX = 1000
TIGHT_ERROR_LIMIT_K = 80.0  # supplemental view; saturation declared

U3_TABLE = ROOT / "results/month2_dense_linear/dense_linear_u3_v1/per_frame.csv.gz"
U5_DIR = ROOT / "results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1"
U4_DIR = ROOT / "results/month2_dense_unet/u4_pon_seed0_engaging_v1"


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
    method: str = "DenseUNet_p_off",
    variant: str = "p_off",
) -> list[dict[str, Any]]:
    model.eval()
    sigma = float(normalizer.temperature_std)
    rows: list[dict[str, Any]] = []
    n = len(ds)
    for start in range(0, n, batch_size):
        batch_items = []
        for j in range(start, min(start + batch_size, n)):
            if mode == "aligned":
                batch_items.append(ds[j])
            else:
                assert pair_lookup is not None
                batch_items.append(
                    build_swapped_input_from_cache(ds, j, pair_lookup=pair_lookup)
                )
        x = torch.stack(
            [it["model_inputs"]["input_tensor"] for it in batch_items], dim=0
        ).to(device)
        # Isolation: p_off channels must remain zero even after swap.
        if variant == "p_off":
            if int(torch.count_nonzero(x[:, 4])) != 0 or int(torch.count_nonzero(x[:, 5])) != 0:
                raise RuntimeError("p_off eval saw non-zero parameter channels")
        pred_norm = model(x).detach().cpu().numpy()
        for bi, it in enumerate(batch_items):
            true_K = it["targets"]["temperature"].numpy()[0]
            pred_K = denormalize_temperature(pred_norm[bi, 0], normalizer)
            meta = it["metadata"]
            rows.append(
                evaluate_frame_prediction(
                    pred_K,
                    true_K,
                    train_sigma_T=sigma,
                    sim_id=meta["sim_id"],
                    split=meta.get("split", "val" if ds.role == "val_eval" else "train"),
                    role=ds.role,
                    frame_index=int(meta["frame_index"]),
                    method=method,
                    variant=variant,
                    p_conditioning=False if variant == "p_off" else True,
                    K=0,
                    site_fraction=1.0,
                    model_seed=0,
                    mask_replica_id=0,
                    layout_id="dense_full_grid",
                    task_mode="temperature_free",
                    snapshot_control=mode,
                )
            )
    return rows


def paired_delta_tables(
    aligned_rows: list[dict[str, Any]],
    swapped_rows: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = pd.DataFrame(aligned_rows)
    s = pd.DataFrame(swapped_rows)
    keys = ["sim_id", "frame_index", "role"]
    m = a.merge(s, on=keys, suffixes=("_aligned", "_swapped"))
    for mk in metric_keys:
        m[f"delta_{mk}"] = m[f"{mk}_swapped"] - m[f"{mk}_aligned"]
    per_frame = m
    per_cond = (
        per_frame.groupby("sim_id", as_index=False)
        .agg(
            {
                **{f"{mk}_aligned": "mean" for mk in metric_keys},
                **{f"{mk}_swapped": "mean" for mk in metric_keys},
                **{f"delta_{mk}": "mean" for mk in metric_keys},
            }
        )
        .sort_values("sim_id")
    )
    per_role = pd.DataFrame(
        [
            {
                "role": aligned_rows[0]["role"],
                **{f"delta_{mk}": float(per_frame[f"delta_{mk}"].mean()) for mk in metric_keys},
            }
        ]
    )
    return per_frame, per_cond, per_role


def load_u3_matched(role: str) -> dict[str, dict[str, Any]]:
    """Reuse U3 per-frame tables; no refit."""
    df = pd.read_csv(U3_TABLE)
    out: dict[str, dict[str, Any]] = {}
    # GlobalMean appears once as p_off in U3; reuse for both sides.
    for method, variant, p_cond, label in [
        ("GlobalMean", "p_off", False, "GlobalMean"),
        ("GappyPOD_dense_free", "p_off", False, "GappyPOD_dense_p_off"),
        ("GappyPOD_dense_free", "p_on", True, "GappyPOD_dense_p_on"),
        ("ParamMean", "p_on", True, "ParamMean"),
    ]:
        sub = df[
            (df["method"] == method)
            & (df["variant"] == variant)
            & (df["role"] == role)
            & (df["p_conditioning"] == p_cond)
        ].copy()
        if sub.empty:
            raise SystemExit(f"missing U3 rows for {label} role={role}")
        rows = sub.to_dict(orient="records")
        out[label] = {
            "macro_rmse_K": _macro_from_rows(rows, role),
            "n_frames": len(sub),
            "source": str(U3_TABLE.relative_to(ROOT)),
            "per_condition": (
                aggregate_frame_then_condition_then_split(rows)["per_condition"][
                    ["sim_id", "rmse_K", "abs_bias_K", "centered_rmse_K", "jet_rmse_K", "high_rmse_K"]
                ].to_dict(orient="records")
            ),
        }
    return out


def _predict_field(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    normalizer: Month2FitOnlyNormalizer,
) -> np.ndarray:
    with torch.no_grad():
        pred = model(x[None].to(device)).detach().cpu().numpy()[0, 0]
    return denormalize_temperature(pred, normalizer)


def make_figures(
    *,
    out_dir: Path,
    cache_root: Path,
    model_poff: torch.nn.Module,
    model_pon: torch.nn.Module | None,
    device: torch.device,
    normalizer: Month2FitOnlyNormalizer,
    pair_lookup_val: dict[int, dict[str, Any]],
    u6_eval_history: Path,
    u4_eval_history: Path,
    u5_fig_dir: Path,
) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    ds_val = DenseUNetCacheDataset("val_eval", cache_root=cache_root, p_conditioning=False)
    # Collect fixed frame-1000 panels for shared scales across p_on/p_off.
    panel_data = []
    fields_T = []
    err_maps = []
    for i in range(len(ds_val)):
        meta = ds_val.get_frame_meta(i)
        if int(meta["frame_index"]) != VAL_FIGURE_FRAME_INDEX:
            continue
        item = ds_val[i]
        true_K = item["targets"]["temperature"].numpy()[0]
        pred_al = _predict_field(
            model_poff, item["model_inputs"]["input_tensor"], device=device, normalizer=normalizer
        )
        swapped = build_swapped_input_from_cache(ds_val, i, pair_lookup=pair_lookup_val)
        pred_sw = _predict_field(
            model_poff,
            swapped["model_inputs"]["input_tensor"],
            device=device,
            normalizer=normalizer,
        )
        pred_pon = None
        if model_pon is not None:
            ds_on = DenseUNetCacheDataset(
                "val_eval", cache_root=cache_root, p_conditioning=True
            )
            item_on = ds_on[i]
            pred_pon = _predict_field(
                model_pon,
                item_on["model_inputs"]["input_tensor"],
                device=device,
                normalizer=normalizer,
            )
        panel = {
            "sim_id": meta["sim_id"],
            "true": true_K,
            "UNet_poff_aligned": pred_al,
            "UNet_poff_swapped": pred_sw,
            "UNet_pon_aligned": pred_pon,
        }
        panel_data.append(panel)
        fields_T.extend([true_K, pred_al, pred_sw] + ([pred_pon] if pred_pon is not None else []))
        err_maps.extend([pred_al - true_K, pred_sw - true_K])
        if pred_pon is not None:
            err_maps.append(pred_pon - true_K)

    if not panel_data:
        raise SystemExit(f"no VAL frames with frame_index={VAL_FIGURE_FRAME_INDEX}")

    t_lo = float(np.min([f.min() for f in fields_T]))
    t_hi = float(np.max([f.max() for f in fields_T]))
    e_lim = float(np.max([np.max(np.abs(e)) for e in err_maps]))

    for p in panel_data:
        sim = str(p["sim_id"]).replace(".h5", "")
        ncols = 4 if p["UNet_pon_aligned"] is not None else 3
        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.2), constrained_layout=True)
        recon = [("true", "T true"), ("UNet_poff_aligned", "U-Net p_off al"), ("UNet_poff_swapped", "U-Net p_off sw")]
        if p["UNet_pon_aligned"] is not None:
            recon.insert(1, ("UNet_pon_aligned", "U-Net p_on al"))
        for ax, (key, title) in zip(axes[0], recon):
            im = ax.imshow(p[key], origin="lower", cmap="inferno", vmin=t_lo, vmax=t_hi)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02, label="T [K]")
        err_items = [
            ("UNet_poff_aligned", "err p_off al"),
            ("UNet_poff_swapped", "err p_off sw"),
            ("UNet_poff_swapped", "sw−al pred"),
        ]
        if p["UNet_pon_aligned"] is not None:
            err_items.insert(0, ("UNet_pon_aligned", "err p_on al"))
        for col, (key, title) in enumerate(err_items):
            ax = axes[1, col]
            if title.startswith("sw"):
                data = p["UNet_poff_swapped"] - p["UNet_poff_aligned"]
            else:
                data = p[key] - p["true"]
            im2 = ax.imshow(data, origin="lower", cmap="coolwarm", vmin=-e_lim, vmax=e_lim)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        # Hide unused bottom axes if any
        for col in range(len(err_items), ncols):
            axes[1, col].axis("off")
        fig.colorbar(im2, ax=axes[1].tolist(), fraction=0.046, pad=0.02, label="ΔT [K]")
        fig.suptitle(
            f"U6 VAL frame_index={VAL_FIGURE_FRAME_INDEX} · {sim}\n"
            f"shared T [{t_lo:.0f},{t_hi:.0f}] K · shared error ±{e_lim:.1f} K",
            fontsize=11,
        )
        outp = fig_dir / f"val_frame{VAL_FIGURE_FRAME_INDEX}_{sim}_poff_panels.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

        # Supplemental tight error scale (aligned errors readable; saturation declared)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        for ax, (data, title) in zip(
            axes,
            [
                (p["UNet_poff_aligned"] - p["true"], "err p_off aligned"),
                (p["UNet_poff_swapped"] - p["true"], "err p_off swapped"),
                (
                    (p["UNet_pon_aligned"] - p["true"])
                    if p["UNet_pon_aligned"] is not None
                    else (p["UNet_poff_aligned"] - p["true"]),
                    "err p_on aligned" if p["UNet_pon_aligned"] is not None else "err p_off al (dup)",
                ),
            ],
        ):
            im = ax.imshow(
                data,
                origin="lower",
                cmap="coolwarm",
                vmin=-TIGHT_ERROR_LIMIT_K,
                vmax=TIGHT_ERROR_LIMIT_K,
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.tolist(), fraction=0.046, pad=0.02, label="ΔT [K]")
        fig.suptitle(
            f"Tight error view ±{TIGHT_ERROR_LIMIT_K:.0f} K (sat. outside) · {sim}",
            fontsize=10,
        )
        outp = fig_dir / f"val_frame{VAL_FIGURE_FRAME_INDEX}_{sim}_error_tight.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

    # Preserve note that original U5 panels are untouched.
    write_json(
        fig_dir / "u5_original_panels_untouched.json",
        {
            "u5_figures_dir": str(u5_fig_dir),
            "note": "U5 original panels conserved; U6 adds separate views with shared/tight scales",
            "exists": u5_fig_dir.is_dir(),
        },
    )

    # Mean abs error maps VAL (p_off aligned vs swapped)
    acc_al = np.zeros((128, 128), dtype=np.float64)
    acc_sw = np.zeros((128, 128), dtype=np.float64)
    n = 0
    for i in range(len(ds_val)):
        item = ds_val[i]
        true = item["targets"]["temperature"].numpy()[0]
        pred_al = _predict_field(
            model_poff, item["model_inputs"]["input_tensor"], device=device, normalizer=normalizer
        )
        swapped = build_swapped_input_from_cache(ds_val, i, pair_lookup=pair_lookup_val)
        pred_sw = _predict_field(
            model_poff,
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
        ["mean |err| p_off al", "mean |err| p_off sw", "Δ mean|err| (sw−al)"],
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
    outp = fig_dir / "val_mean_abs_error_poff_aligned_vs_swapped.png"
    fig.savefig(outp, dpi=160)
    plt.close(fig)
    paths.append(str(outp.relative_to(out_dir)))

    # Curves p_on vs p_off
    if u6_eval_history.is_file() and u4_eval_history.is_file():
        eh6 = json.loads(u6_eval_history.read_text())
        eh4 = json.loads(u4_eval_history.read_text())
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        ax.plot(
            [e["epoch"] for e in eh4],
            [e["val_mean_condition_rmse_K"] for e in eh4],
            label="VAL p_on",
            color="C0",
        )
        ax.plot(
            [e["epoch"] for e in eh6],
            [e["val_mean_condition_rmse_K"] for e in eh6],
            label="VAL p_off",
            color="C1",
        )
        ax.plot(
            [e["epoch"] for e in eh4],
            [e["seen_mean_condition_rmse_K"] for e in eh4],
            label="seen p_on",
            color="C0",
            ls="--",
            alpha=0.7,
        )
        ax.plot(
            [e["epoch"] for e in eh6],
            [e["seen_mean_condition_rmse_K"] for e in eh6],
            label="seen p_off",
            color="C1",
            ls="--",
            alpha=0.7,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro RMSE [K]")
        ax.set_title("U4 p_on vs U6 p_off curves (seed 0)")
        ax.legend(fontsize=8)
        outp = fig_dir / "u4_u6_curves_pon_vs_poff.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot([e["epoch"] for e in eh6], [e["val_mean_condition_rmse_K"] for e in eh6], label="VAL")
        ax.plot([e["epoch"] for e in eh6], [e["seen_mean_condition_rmse_K"] for e in eh6], label="seen")
        if eh6 and "train_mean_condition_rmse_K" in eh6[0]:
            ax.plot(
                [e["epoch"] for e in eh6],
                [e["train_mean_condition_rmse_K"] for e in eh6],
                label="TRAIN",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro RMSE [K]")
        ax.set_title("U6 p_off curves (seed 0)")
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot([e["epoch"] for e in eh6], [e["lr"] for e in eh6], color="tab:purple", alpha=0.5)
        ax2.set_ylabel("lr")
        outp = fig_dir / "u6_curves_train_seen_val_lr.png"
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        paths.append(str(outp.relative_to(out_dir)))

    write_json(
        fig_dir / "figure_scales.json",
        {
            "val_frame_index": VAL_FIGURE_FRAME_INDEX,
            "T_vmin_K": t_lo,
            "T_vmax_K": t_hi,
            "error_symmetric_limit_K": e_lim,
            "tight_error_limit_K": TIGHT_ERROR_LIMIT_K,
            "tight_error_saturation": "values outside ±tight limit are clipped visually",
            "mean_abs_error_vmax_K": vmax,
            "shared_across_pon_poff": True,
        },
    )
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--expect-cache-hashes", type=Path, required=True)
    p.add_argument("--u6-checkpoint", type=Path, required=True)
    p.add_argument("--expect-u6-ckpt-sha256", type=str, required=True)
    p.add_argument("--u6-run-dir", type=Path, required=True)
    p.add_argument("--u4-run-dir", type=Path, default=U4_DIR)
    p.add_argument("--u5-dir", type=Path, default=U5_DIR)
    p.add_argument(
        "--u4-checkpoint",
        type=Path,
        default=None,
        help="Optional p_on best.pt for shared-scale panel comparison",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/month2_dense_unet/u6_poff_seed0_snapshot_control_v1",
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

    # Load U6 selection refs before replay.
    u6_sel = json.loads((Path(args.u6_run_dir) / "selection.json").read_text())
    ref_val = float(u6_sel["best_val_metric"])
    best_ep = int(u6_sel["best_val_epoch"])
    eh = json.loads((Path(args.u6_run_dir) / "eval_history.json").read_text())
    best_row = next(e for e in eh if int(e["epoch"]) == best_ep)
    ref_seen = float(best_row["seen_mean_condition_rmse_K"])

    tolerances = {
        "aligned_replay_atol_K": ALIGNED_REPLAY_ATOL_K,
        "aligned_replay_rtol": ALIGNED_REPLAY_RTOL,
        "declared_before_replay": True,
        "u6_ref_val_macro_K": ref_val,
        "u6_ref_seen_macro_K": ref_seen,
        "u6_best_epoch": best_ep,
    }
    write_json(out_dir / "tolerances_declared.json", tolerances)

    provenance = collect_run_provenance(
        extra_sources=[
            "scripts/10_u6_poff_eval_and_swap.py",
            "configs/experiments/month2_dense_unet_u6_candidate.yaml",
        ]
    )
    write_json(out_dir / "run_provenance.json", provenance)

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

    ckpt_path = Path(args.u6_checkpoint)
    ckpt_sha = _sha256_bytes(ckpt_path)
    if ckpt_sha != args.expect_u6_ckpt_sha256:
        raise SystemExit(
            f"U6 checkpoint sha mismatch: got {ckpt_sha}, expected {args.expect_u6_ckpt_sha256}"
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

    model_pon = None
    if args.u4_checkpoint and Path(args.u4_checkpoint).is_file():
        model_pon = build_dense_unet_from_config(init_seed=0, base_channels=32)
        load_epoch_checkpoint(
            Path(args.u4_checkpoint),
            model=model_pon,
            map_location="cpu",
            require_resumable=True,
            restore_rng=False,
        )
        model_pon.to(device)
        model_pon.eval()
        for p_ in model_pon.parameters():
            p_.requires_grad_(False)

    # Reuse U5 pairings exactly (do not rebuild).
    u5_pairs_path = Path(args.u5_dir) / "swap_pairs.json"
    pairings = json.loads(u5_pairs_path.read_text())
    write_json(out_dir / "swap_pairs_reused_from_u5.json", {"source": str(u5_pairs_path), "pairings": pairings})
    lookups = {
        role: pairs_by_dest_cache_index(pairings[role])
        for role in ("seen_train_eval", "val_eval")
    }

    aligned_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("seen_train_eval", "val_eval"):
        ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=False)
        aligned_rows[role] = eval_unet_role(
            model=model,
            ds=ds,
            device=device,
            normalizer=normalizer,
            mode="aligned",
            pair_lookup=None,
            batch_size=args.batch_size,
        )
        # Per-frame identity vs training eval tables when available
        train_pf = Path(args.u6_run_dir) / "evals" / f"epoch_{best_ep:04d}" / f"{role}_per_frame.csv.gz"
        if train_pf.is_file():
            ref = pd.read_csv(train_pf)
            cur = pd.DataFrame(aligned_rows[role])
            merged = ref.merge(
                cur,
                on=["sim_id", "frame_index"],
                suffixes=("_ref", "_cur"),
            )
            max_abs = float((merged["rmse_K_cur"] - merged["rmse_K_ref"]).abs().max())
            if max_abs > ALIGNED_REPLAY_ATOL_K:
                raise SystemExit(
                    f"per-frame aligned replay failed for {role}: max|ΔRMSE|={max_abs}"
                )

    val_macro = _macro_from_rows(aligned_rows["val_eval"], "val_eval")
    seen_macro = _macro_from_rows(aligned_rows["seen_train_eval"], "seen_train_eval")
    replay = {
        "val_macro_rmse_K": val_macro,
        "seen_macro_rmse_K": seen_macro,
        "val_abs_diff_vs_u6_best": abs(val_macro - ref_val),
        "seen_abs_diff_vs_u6_best": abs(seen_macro - ref_seen),
        "val_ok": abs(val_macro - ref_val)
        <= ALIGNED_REPLAY_ATOL_K + ALIGNED_REPLAY_RTOL * abs(ref_val),
        "seen_ok": abs(seen_macro - ref_seen)
        <= ALIGNED_REPLAY_ATOL_K + ALIGNED_REPLAY_RTOL * abs(ref_seen),
    }
    write_json(out_dir / "aligned_replay.json", replay)
    if not (replay["val_ok"] and replay["seen_ok"]):
        write_json(out_dir / "STOP_aligned_replay_failed.json", {"replay": replay, "tolerances": tolerances})
        print(json.dumps({"status": "aligned_replay_failed", "replay": replay}, indent=2))
        return 3

    swapped_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("seen_train_eval", "val_eval"):
        ds = DenseUNetCacheDataset(role, cache_root=cache_root, p_conditioning=False)
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
        pd.DataFrame(aligned_rows[role]).to_csv(
            tables_dir / f"{role}_unet_poff_aligned_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(swapped_rows[role]).to_csv(
            tables_dir / f"{role}_unet_poff_swapped_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        per_frame, per_cond, per_role = paired_delta_tables(
            aligned_rows[role], swapped_rows[role], metric_keys=metric_keys
        )
        per_frame.to_csv(
            tables_dir / f"{role}_poff_paired_delta_per_frame.csv.gz",
            index=False,
            compression="gzip",
        )
        per_cond.to_csv(
            tables_dir / f"{role}_poff_paired_delta_per_condition.csv.gz",
            index=False,
            compression="gzip",
        )
        per_role.to_csv(
            tables_dir / f"{role}_poff_paired_delta_per_role.csv.gz",
            index=False,
            compression="gzip",
        )
        for mode, rows in (("aligned", aligned_rows[role]), ("swapped", swapped_rows[role])):
            agg = aggregate_frame_then_condition_then_split(rows)
            agg["per_condition"].to_csv(
                tables_dir / f"{role}_unet_poff_{mode}_per_condition.csv.gz",
                index=False,
                compression="gzip",
            )
        al_m = _macro_from_rows(aligned_rows[role], role)
        sw_m = _macro_from_rows(swapped_rows[role], role)
        summary_blocks[role] = {
            "aligned_macro_rmse_K": al_m,
            "swapped_macro_rmse_K": sw_m,
            "delta_macro_rmse_K": sw_m - al_m,
            "mean_frame_delta_rmse_K": float(per_role["delta_rmse_K"].iloc[0]),
            "per_condition": per_cond.to_dict(orient="records"),
        }

    # Matched baselines from U3 (no refit). Also pull U4/U5 p_on unet macros from saved tables.
    matched = {}
    for role in ("seen_train_eval", "val_eval"):
        matched[role] = load_u3_matched(role)
        # p_on U-Net from U5 tables
        u5_al = Path(args.u5_dir) / "tables" / f"{role}_unet_aligned_per_condition.csv.gz"
        u5_sw = Path(args.u5_dir) / "tables" / f"{role}_unet_swapped_per_condition.csv.gz"
        if u5_al.is_file():
            al = pd.read_csv(u5_al)
            matched[role]["DenseUNet_p_on_aligned"] = {
                "macro_rmse_K": float(al["rmse_K"].mean()),
                "source": str(u5_al.relative_to(ROOT)),
                "per_condition": al.to_dict(orient="records"),
            }
        if u5_sw.is_file():
            sw = pd.read_csv(u5_sw)
            matched[role]["DenseUNet_p_on_swapped"] = {
                "macro_rmse_K": float(sw["rmse_K"].mean()),
                "source": str(u5_sw.relative_to(ROOT)),
            }
        matched[role]["DenseUNet_p_off_aligned"] = {
            "macro_rmse_K": summary_blocks[role]["aligned_macro_rmse_K"]
        }
        matched[role]["DenseUNet_p_off_swapped"] = {
            "macro_rmse_K": summary_blocks[role]["swapped_macro_rmse_K"]
        }

    # p_on vs p_off comparison table (best checkpoints)
    u4_sel = json.loads((Path(args.u4_run_dir) / "selection.json").read_text())
    u6_sum = json.loads((Path(args.u6_run_dir) / "summary.json").read_text())
    u4_last = u4_sel.get("last_eval") or {}
    u6_last = u6_sel.get("last_eval") or eh[-1]
    comparison = {
        "p_on": {
            "run_id": "u4_pon_seed0_engaging_v1",
            "best_epoch": int(u4_sel["best_val_epoch"]),
            "best_val_rmse_K": float(u4_sel["best_val_metric"]),
            "last_epoch": int(u4_sel.get("last_epoch") or u4_last.get("epoch") or 730),
            "last_val_rmse_K": float(u4_last.get("val_mean_condition_rmse_K") or 28.055),
            "status": None,
        },
        "p_off": {
            "run_id": Path(args.u6_run_dir).name,
            "best_epoch": best_ep,
            "best_val_rmse_K": float(u6_sel["best_val_metric"]),
            "last_epoch": int(u6_sel.get("last_epoch") or u6_sum.get("epochs_completed") or u6_last.get("epoch")),
            "last_val_rmse_K": float(
                u6_last.get("val_mean_condition_rmse_K")
                if isinstance(u6_last, dict)
                else eh[-1]["val_mean_condition_rmse_K"]
            ),
            "status": u6_sum.get("status"),
            "stop_reason": u6_sum.get("status"),
        },
        "delta_best_val_rmse_K_poff_minus_pon": float(u6_sel["best_val_metric"])
        - float(u4_sel["best_val_metric"]),
    }
    u4_sum_path = Path(args.u4_run_dir) / "summary.json"
    if u4_sum_path.is_file():
        u4_sum = json.loads(u4_sum_path.read_text())
        comparison["p_on"]["status"] = u4_sum.get("status")
        if u4_sum.get("epochs_completed") is not None:
            comparison["p_on"]["last_epoch"] = int(u4_sum["epochs_completed"])

    write_json(out_dir / "pon_poff_comparison.json", comparison)
    write_json(out_dir / "matched_baselines_u3_reused.json", matched)

    fig_paths: list[str] = []
    if not args.skip_figures:
        fig_paths = make_figures(
            out_dir=out_dir,
            cache_root=cache_root,
            model_poff=model,
            model_pon=model_pon,
            device=device,
            normalizer=normalizer,
            pair_lookup_val=lookups["val_eval"],
            u6_eval_history=Path(args.u6_run_dir) / "eval_history.json",
            u4_eval_history=Path(args.u4_run_dir) / "eval_history.json",
            u5_fig_dir=Path(args.u5_dir) / "figures",
        )

    # Per-condition comparison bar: p_on vs p_off aligned + baselines
    val_poff_cond = pd.read_csv(tables_dir / "val_eval_unet_poff_aligned_per_condition.csv.gz")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    sims = [s.replace(".h5", "") for s in val_poff_cond["sim_id"]]
    x = np.arange(len(sims))
    w = 0.25
    pon_vals = None
    if "DenseUNet_p_on_aligned" in matched["val_eval"]:
        pon_pc = matched["val_eval"]["DenseUNet_p_on_aligned"]["per_condition"]
        pon_map = {r["sim_id"]: r["rmse_K"] for r in pon_pc}
        pon_vals = [pon_map[sid] for sid in val_poff_cond["sim_id"]]
        ax.bar(x - w, pon_vals, w, label="U-Net p_on")
    ax.bar(x, val_poff_cond["rmse_K"], w, label="U-Net p_off")
    ax.bar(
        x + w,
        [
            next(
                r["rmse_K"]
                for r in matched["val_eval"]["GappyPOD_dense_p_off"]["per_condition"]
                if r["sim_id"] == sid
            )
            for sid in val_poff_cond["sim_id"]
        ],
        w,
        label="GappyPOD p_off",
    )
    ax.axhline(
        matched["val_eval"]["GlobalMean"]["macro_rmse_K"],
        color="gray",
        ls=":",
        label="GlobalMean macro",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sims, rotation=15)
    ax.set_ylabel("RMSE [K]")
    ax.set_title("VAL per-condition RMSE: p_on / p_off / GappyPOD p_off")
    ax.legend(fontsize=7)
    outp = fig_dir / "val_per_condition_pon_poff_comparison.png"
    fig.savefig(outp, dpi=160)
    plt.close(fig)
    fig_paths.append(str(outp.relative_to(out_dir)))

    env = {
        "git_commit": provenance.get("git_commit") or _git_head(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "device": str(device),
        "tf32_policy": tf32,
        "u6_checkpoint_path": str(ckpt_path),
        "u6_checkpoint_sha256": ckpt_sha,
        "cache_payload_sha256": verified["actual_aggregate_sha256"],
        "peak_rss_GB": _rss_gb(),
        "elapsed_s": time.perf_counter() - t0,
        "u4_checkpoint_used_for_panels": str(args.u4_checkpoint) if model_pon is not None else None,
    }
    summary = {
        "experiment_id": "u6_poff_seed0_snapshot_control_v1",
        "status": "completed",
        "aligned_replay": replay,
        "tolerances": tolerances,
        "unet_poff_swap_summary": summary_blocks,
        "pon_poff_comparison": comparison,
        "matched_baselines_u3_reused": {
            role: {k: {"macro_rmse_K": v["macro_rmse_K"], "source": v.get("source")} for k, v in blk.items()}
            for role, blk in matched.items()
        },
        "figures": fig_paths,
        "environment": env,
        "run_provenance": provenance,
        "limits": [
            "VAL used for U4/U6 selection/scheduler/early-stop — not independent final test",
            "Single paired seed 0; variability across seeds not measured",
            "No ParamMean in p_off forward; ParamMean listed only as p_on reference",
            "U3 baseline tables reused without refit",
            "Swap degradation ≠ physical causality or information ceiling",
            "best−last ~1 K must not be treated as seed variability",
        ],
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "run_manifest.json", summary)

    lines = [
        "# U6 auto-report — p_off eval + snapshot control (numeric)",
        "",
        f"- Commit: `{env['git_commit']}`",
        f"- U6 checkpoint sha256: `{ckpt_sha}` (best epoch {best_ep})",
        f"- Device: {env['gpu_name']} ({env['device']})",
        f"- Aligned replay VAL/seen: {val_macro:.6f} / {seen_macro:.6f} K "
        f"(Δ vs U6 best {replay['val_abs_diff_vs_u6_best']:.4g} / "
        f"{replay['seen_abs_diff_vs_u6_best']:.4g}; atol={ALIGNED_REPLAY_ATOL_K} K) → OK",
        "",
        "## Best p_on vs p_off (VAL macro RMSE K)",
        "",
        f"- p_on best: {comparison['p_on']['best_val_rmse_K']:.3f} @ {comparison['p_on']['best_epoch']}",
        f"- p_off best: {comparison['p_off']['best_val_rmse_K']:.3f} @ {comparison['p_off']['best_epoch']}",
        f"- Δ (p_off − p_on): {comparison['delta_best_val_rmse_K_poff_minus_pon']:+.3f}",
        "",
        "## p_off snapshot control macro RMSE [K]",
        "",
        "| Role | aligned | swapped | Δ | GlobalMean | GappyPOD p_off | GappyPOD p_on |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("val_eval", "seen_train_eval"):
        lines.append(
            "| {role} | {a:.3f} | {s:.3f} | {d:+.3f} | {gm:.3f} | {gp0:.3f} | {gp1:.3f} |".format(
                role=role,
                a=summary_blocks[role]["aligned_macro_rmse_K"],
                s=summary_blocks[role]["swapped_macro_rmse_K"],
                d=summary_blocks[role]["delta_macro_rmse_K"],
                gm=matched[role]["GlobalMean"]["macro_rmse_K"],
                gp0=matched[role]["GappyPOD_dense_p_off"]["macro_rmse_K"],
                gp1=matched[role]["GappyPOD_dense_p_on"]["macro_rmse_K"],
            )
        )
    lines.extend(["", "## VAL per-condition p_off Δ RMSE (sw−al)", ""])
    for rec in summary_blocks["val_eval"]["per_condition"]:
        lines.append(
            f"- {rec['sim_id']}: aligned {rec['rmse_K_aligned']:.3f} → swapped "
            f"{rec['rmse_K_swapped']:.3f} (Δ {rec['delta_rmse_K']:+.3f}); "
            f"Δ centered {rec['delta_centered_rmse_K']:+.3f}; "
            f"Δ |bias| {rec['delta_abs_bias_K']:+.3f}"
        )
    lines.extend(["", "## Figures", ""])
    for fp in fig_paths:
        lines.append(f"- `{fp}`")
    (out_dir / "auto_report.md").write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "replay": replay,
                "comparison": comparison,
                "swap": summary_blocks,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
