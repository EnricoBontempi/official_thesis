#!/usr/bin/env python3
"""Diagnostic small overfit for sparse U-Net (S2). Weights are NOT for science."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.models.unet import SparseUNet  # noqa: E402
from combustion_recon.training.dense_cache import subset_indices_for_overfit  # noqa: E402
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402
from combustion_recon.training.sparse_epoch_trainer import (  # noqa: E402
    SparseEpochConfig,
    SparseEpochTrainer,
)
from combustion_recon.training.sparse_view import sites_from_training_mask  # noqa: E402
from combustion_recon.training.trainer import set_seed  # noqa: E402

CACHE = ROOT / "data/processed/month2_dense_unet_cache"
STATS = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
OUT = ROOT / "results/month2/sparse_s2_diagnostic_overfit"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    normalizer = Month2FitOnlyNormalizer(STATS)
    frames = json.loads((CACHE / "train_fit/frames.json").read_text())
    idxs = subset_indices_for_overfit(
        frames, n_frames=8, min_conditions=2, max_conditions=4, seed=7
    )

    fixed = {}
    frame_meta = []
    for ci in idxs:
        fr = frames[ci]
        sites, vel_seed, h = sites_from_training_mask(
            train_mask_seed=20261118,
            epoch=0,
            sim_id=fr["sim_id"],
            frame_index=int(fr["frame_index"]),
        )
        fixed[ci] = sites
        frame_meta.append(
            {
                "cache_index": ci,
                "sim_id": fr["sim_id"],
                "frame_index": int(fr["frame_index"]),
                "nh3_fraction": fr.get("nh3_fraction"),
                "phi": fr.get("phi"),
                "n_sites": int(sites.size),
                "mask_hash": h,
                "velocity_mask_seed": vel_seed,
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseUNet(base_channels=32, init_seed=0)
    n_params = model.count_parameters()

    # A) Fixed masks overfit
    ds_fixed = SparseUNetCacheDataset(
        "train_fit",
        cache_root=CACHE,
        indices=idxs,
        mask_mode="fixed",
        fixed_sites_by_cache_index=fixed,
        p_conditioning=True,
        normalizer=normalizer,
    )
    t0 = time.perf_counter()
    res_fixed = SparseEpochTrainer(
        model,
        ds_fixed,
        cfg=SparseEpochConfig(
            n_epochs=40,
            batch_size=4,
            model_seed=0,
            shuffle_seed=0,
            device=device,
            lr=1e-3,
        ),
        out_dir=OUT / "fixed_masks",
    ).train()

    # Diagnostic maps on first frame
    item = ds_fixed[0]
    model.eval()
    with torch.no_grad():
        pred_n = model(item["model_inputs"]["input_tensor"].unsqueeze(0).to(device)).cpu().numpy()[0, 0]
    pred_K = normalizer.denormalize_field("Temperature", pred_n)
    true_K = item["targets"]["temperature"].numpy()[0]
    err = pred_K - true_K
    np.savez_compressed(
        OUT / "diag_maps_frame0.npz",
        pred_K=pred_K.astype(np.float32),
        true_K=true_K.astype(np.float32),
        err_K=err.astype(np.float32),
        input_7ch=item["model_inputs"]["input_tensor"].numpy(),
        sim_id=frame_meta[0]["sim_id"],
        frame_index=frame_meta[0]["frame_index"],
    )

    # B) Variable masks (fresh init)
    model_v = SparseUNet(base_channels=32, init_seed=0)
    ds_var = SparseUNetCacheDataset(
        "train_fit",
        cache_root=CACHE,
        indices=idxs,
        mask_mode="training",
        train_mask_seed=20261118,
        epoch=0,
        p_conditioning=True,
        normalizer=normalizer,
    )
    res_var = SparseEpochTrainer(
        model_v,
        ds_var,
        cfg=SparseEpochConfig(
            n_epochs=15,
            batch_size=4,
            model_seed=0,
            shuffle_seed=0,
            device=device,
            lr=1e-3,
        ),
        out_dir=OUT / "variable_masks",
    ).train()

    report = {
        "status": "diagnostic_only",
        "weights_reusable_for_science": False,
        "device": device,
        "n_params_b32": n_params,
        "frames": frame_meta,
        "tolerances_declared": {"pred_atol": 1e-6, "bit_atol": 0.0},
        "fixed_masks": {
            "n_epochs": 40,
            "first_loss": res_fixed["history"][0]["mean_train_loss"],
            "last_loss": res_fixed["history"][-1]["mean_train_loss"],
            "updates": res_fixed["state"]["update"],
            "elapsed_s": res_fixed["elapsed_s"],
            "history": res_fixed["history"],
        },
        "variable_masks": {
            "n_epochs": 15,
            "first_loss": res_var["history"][0]["mean_train_loss"],
            "last_loss": res_var["history"][-1]["mean_train_loss"],
            "updates": res_var["state"]["update"],
            "elapsed_s": res_var["elapsed_s"],
            "history": res_var["history"],
            "note": "Do not expect near-zero loss with changing incomplete layouts",
        },
        "total_wall_s": time.perf_counter() - t0 + res_var["elapsed_s"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("n_params_b32", "fixed_masks", "variable_masks", "device")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
