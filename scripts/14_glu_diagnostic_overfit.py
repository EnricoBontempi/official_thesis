#!/usr/bin/env python3
"""Diagnostic GLU overfit on the same 8 TRAIN frames as S2 (full 16384 queries)."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.models.glu_reconstructor import (  # noqa: E402
    build_glu_from_config,
    make_synthetic_query_grid,
)
from combustion_recon.training.dense_cache import subset_indices_for_overfit  # noqa: E402
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402
from combustion_recon.training.sparse_view import sites_from_training_mask  # noqa: E402
from combustion_recon.training.trainer import set_seed  # noqa: E402

CACHE = ROOT / "data/processed/month2_dense_unet_cache"
STATS = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
OUT = ROOT / "results/month2/sparse_s3_glu_diagnostic_overfit"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalizer = Month2FitOnlyNormalizer(STATS)
    frames = json.loads((CACHE / "train_fit/frames.json").read_text())
    idxs = subset_indices_for_overfit(frames, n_frames=8, seed=7)
    fixed = {}
    meta = []
    for ci in idxs:
        fr = frames[ci]
        sites, vel_seed, h = sites_from_training_mask(
            train_mask_seed=20261118,
            epoch=0,
            sim_id=fr["sim_id"],
            frame_index=int(fr["frame_index"]),
        )
        fixed[ci] = sites
        meta.append(
            {
                "cache_index": ci,
                "sim_id": fr["sim_id"],
                "frame_index": int(fr["frame_index"]),
                "mask_hash": h,
                "velocity_mask_seed": vel_seed,
                "n_sites": int(sites.size),
            }
        )

    ds_fixed = SparseUNetCacheDataset(
        "train_fit",
        cache_root=CACHE,
        indices=idxs,
        mask_mode="fixed",
        fixed_sites_by_cache_index=fixed,
        p_conditioning=True,
        normalizer=normalizer,
    )
    queries = make_synthetic_query_grid(ds_fixed.x, ds_fixed.y).unsqueeze(0).to(device)

    def train_loop(ds, n_epochs: int, tag: str):
        set_seed(0)
        model = build_glu_from_config().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_params = model.count_parameters()
        names0 = {k for k, _ in model.named_parameters()}
        history = []
        t0 = time.perf_counter()
        updates = 0
        model.train()
        for epoch in range(n_epochs):
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
            ep = []
            for i in range(len(ds)):
                item = ds[i]
                idx = ds.indices[i]
                p = torch.tensor(
                    [[float(ds.p_norm[idx, 0]), float(ds.p_norm[idx, 1])]],
                    dtype=torch.float32,
                    device=device,
                )
                opt.zero_grad(set_to_none=True)
                pred = model(
                    torch.from_numpy(item["observation"].velocity_coords.astype(np.float32))
                    .unsqueeze(0)
                    .to(device),
                    torch.from_numpy(item["observation"].velocity_values_norm.astype(np.float32))
                    .unsqueeze(0)
                    .to(device),
                    torch.from_numpy(item["observation"].sites.astype(np.int64)).unsqueeze(0).to(device),
                    p,
                    queries,
                    p_conditioning=True,
                    decode_chunked=True,
                    chunk_size=512,
                )
                tgt = item["targets"]["temperature_normalized"].to(device)
                loss = torch.mean((pred - tgt) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch} i={i}")
                loss.backward()
                assert {k for k, _ in model.named_parameters()} == names0
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                opt.step()
                updates += 1
                ep.append({"loss": float(loss.detach().cpu()), "grad_norm": gn})
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_train_loss": float(np.mean([e["loss"] for e in ep])),
                    "mean_grad_norm": float(np.mean([e["grad_norm"] for e in ep])),
                    "mask_epoch_used": epoch if ds.mask_mode == "training" else 0,
                }
            )
        elapsed = time.perf_counter() - t0
        # diagnostic map frame0
        model.eval()
        item0 = ds[0]
        idx0 = ds.indices[0]
        with torch.no_grad():
            pred0 = model(
                torch.from_numpy(item0["observation"].velocity_coords.astype(np.float32))
                .unsqueeze(0)
                .to(device),
                torch.from_numpy(item0["observation"].velocity_values_norm.astype(np.float32))
                .unsqueeze(0)
                .to(device),
                torch.from_numpy(item0["observation"].sites.astype(np.int64)).unsqueeze(0).to(device),
                torch.tensor(
                    [[float(ds.p_norm[idx0, 0]), float(ds.p_norm[idx0, 1])]],
                    dtype=torch.float32,
                    device=device,
                ),
                queries,
                decode_chunked=True,
                chunk_size=512,
            )
        pred_n = pred0.squeeze(0).squeeze(0).cpu().numpy()
        true_K = item0["targets"]["temperature"].numpy()[0]
        pred_K = normalizer.denormalize_field("Temperature", pred_n)
        out_dir = OUT / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "diag_maps_frame0.npz",
            pred_K=pred_K.astype(np.float32),
            true_K=true_K.astype(np.float32),
            err_K=(pred_K - true_K).astype(np.float32),
        )
        ckpt = out_dir / "last.pt"
        torch.save({"model_state_dict": model.state_dict(), "diagnostic": True}, ckpt)
        return {
            "n_params": n_params,
            "updates": updates,
            "elapsed_s": elapsed,
            "history": history,
            "first_loss": history[0]["mean_train_loss"],
            "last_loss": history[-1]["mean_train_loss"],
            "geometry_meta": model.geometry_meta,
            "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "ckpt_path": str(ckpt),
        }

    fixed_res = train_loop(ds_fixed, n_epochs=8, tag="fixed_masks")
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
    var_res = train_loop(ds_var, n_epochs=5, tag="variable_masks")

    report = {
        "status": "diagnostic_only",
        "weights_reusable_for_science": False,
        "device": device,
        "frames": meta,
        "config": "configs/models/glu_minimal.yaml",
        "decode_chunk_size": 512,
        "full_queries": 16384,
        "tolerances": {"chunk_atol": 1e-5, "perm_atol": 1e-5},
        "fixed_masks": fixed_res,
        "variable_masks": var_res,
        "note": "Not S4 profiling; not VAL selection.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "device": device,
                "fixed": {k: fixed_res[k] for k in ("first_loss", "last_loss", "updates", "elapsed_s", "n_params")},
                "variable": {k: var_res[k] for k in ("first_loss", "last_loss", "updates", "elapsed_s")},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
