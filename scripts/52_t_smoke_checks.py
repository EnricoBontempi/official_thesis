#!/usr/bin/env python3
"""Non-scientific checks of the temperature-assisted path on the real cache (one family per call).

Isolation (hidden T, hidden velocity), p_off, K=0, token/channel positions, nested eval layouts, a short TRAIN overfit with
variable K, the positive control (observed T are really used after learning), and a real six-K evaluation read back from disk.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.t_anchors import H, W, K_VALUES, AnchorManifest, train_K_for_update  # noqa: E402
from combustion_recon.models.glu_assisted import build_assisted_glu_from_config  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.models.unet import ASSISTED_CHANNEL_ORDER, build_sparse_unet_assisted_from_config  # noqa: E402
from combustion_recon.training.assisted_dataset import AssistedDataset  # noqa: E402
from combustion_recon.training.assisted_trainer import AssistedTrainer  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.sparse_dataset import collate_sparse_batch  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import SparseScientificConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("unet", "glu"), required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, default=Path("/home/bontempi/projects/official_thesis_s5/data/manifests/train_norm_stats_month2_fit.json"))
    ap.add_argument("--anchor-manifest", type=Path, default=ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overfit-updates", type=int, default=250)
    args = ap.parse_args()
    fam = "sparse_unet" if args.family == "unet" else "adapted_glu"
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    anchors = AnchorManifest.load(args.anchor_manifest)
    normalizer = Month2FitOnlyNormalizer(args.stats)
    kw = cache_validation_kwargs("C149")
    rep: dict = {"family": fam, "checks": {}}

    def mk(role, mode, p_on, **extra):
        return AssistedDataset(role, cache_root=args.cache_root, mask_mode=mode, site_fraction=0.05, eval_mask_seed=20261116,
                               layout_id="random_sparse_primary", replica_id=1, p_conditioning=p_on, normalizer=normalizer,
                               eval_mask_payload_root=args.mask_root, require_eval_mask_payload=True, anchors=anchors,
                               train_mask_seed=20261118, **kw, **extra)

    val_on = mk("val_eval", "evaluation", True)
    val_off = mk("val_eval", "evaluation", False)
    # ---- positions / tokens / K=0 / p_off / channel names
    val_on.eval_K = 9
    it = val_on[5]
    x = it["model_inputs"]["input_tensor"].numpy()
    sites = anchors.sites(1, 9)
    tch, mt = x[5].ravel(), x[6].ravel()
    rep["checks"]["channels_named"] = it["model_inputs"]["channel_order"] == list(ASSISTED_CHANNEL_ORDER) and x.shape == (9, H, W)
    rep["checks"]["mask_positions_equal_manifest_prefix"] = bool(set(np.flatnonzero(mt)) == set(sites.tolist()) and mt.sum() == 9)
    t_true = np.asarray(val_on.t_norm[it["metadata"]["cache_index"]], dtype=np.float32).ravel()
    rep["checks"]["t_values_are_observed_normalized_targets"] = bool(np.array_equal(tch[sites], t_true[sites]) and not tch[np.setdiff1d(np.arange(H * W), sites)].any())
    g = it["glu_tokens"]
    rep["checks"]["glu_t_tokens"] = bool(g["t_site_index"].shape[0] == 9 and np.allclose(g["t_coords_phys"][:, 0], val_on.x[sites % W]) and
                                         np.allclose(g["t_coords_phys"][:, 1], val_on.y[sites // W]) and np.array_equal(g["t_values_norm"][:, 0], t_true[sites]))
    val_on.eval_K = 0
    z = val_on[5]
    rep["checks"]["K0_has_no_T"] = bool(not z["model_inputs"]["input_tensor"].numpy()[5:7].any() and z["glu_tokens"]["t_site_index"].shape[0] == 0)
    val_off.eval_K = 4
    xo = val_off[5]["model_inputs"]["input_tensor"].numpy()
    rep["checks"]["p_off_channels_zero_by_name"] = bool(not xo[7:9].any() and xo[6].sum() == 4)
    # ---- hidden-T and hidden-velocity isolation (mutate the stored arrays where nothing is observed)
    val_on.eval_K = 16
    a = val_on[7]
    hidden_t = np.setdiff1d(np.arange(H * W), anchors.sites(1, 16))
    ds2 = copy.copy(val_on)
    ds2.t_norm = np.array(val_on.t_norm, copy=True)
    idx = a["metadata"]["cache_index"]
    flat = ds2.t_norm[idx].reshape(-1)
    flat[hidden_t[::3]] += 5.0
    ds2.t_norm[idx] = flat.reshape(H, W)
    b = ds2[7]
    rep["checks"]["hidden_T_isolation_bitwise"] = bool(torch.equal(a["model_inputs"]["input_tensor"], b["model_inputs"]["input_tensor"]) and
                                                        all(np.array_equal(a["glu_tokens"][k], b["glu_tokens"][k]) for k in ("t_values_norm", "values_norm", "coords_phys")))
    obs_sites = a["observation"].sites
    hidden_v = np.setdiff1d(np.arange(H * W), obs_sites)
    ds3 = copy.copy(val_on)
    ds3.vi = np.array(val_on.vi, copy=True)
    fv = ds3.vi[idx].reshape(-1)
    fv[hidden_v[::5]] += 3.0
    ds3.vi[idx] = fv.reshape(H, W)
    c = ds3[7]
    rep["checks"]["hidden_velocity_isolation_bitwise"] = bool(torch.equal(a["model_inputs"]["input_tensor"], c["model_inputs"]["input_tensor"]))

    # ---- model + short overfit on fixed TRAIN frames with variable K, then positive control and six-K eval
    cfg = SparseScientificConfig(family=fam, p_conditioning=True, seed=11, device="cuda", batch_size=8, decode_chunked_eval=True)
    torch.manual_seed(11)
    if fam == "sparse_unet":
        model = build_sparse_unet_assisted_from_config(init_seed=11)
        queries = None
    else:
        model = build_assisted_glu_from_config(domain_geometry_path=args.cache_root / "geometry.npz")
        queries = make_synthetic_query_grid(val_on.x, val_on.y)
    model.to(dev)
    trainer = AssistedTrainer(model, cfg=cfg, normalizer=normalizer, n_train_samples=16, query_coords_phys=queries, anchors=anchors)
    tr = mk("train_fit", "training", True, indices=list(range(0, 2980, 2980 // 16))[:16])
    tr.set_epoch(0)
    trainer._set_lr(1e-3)
    losses = {}
    t0 = time.perf_counter()
    upd = 0
    while upd < args.overfit_updates:
        for start in range(0, 16, 8):
            upd += 1
            K = train_K_for_update(upd)
            items = [tr[(start + j, K)] for j in range(8)]
            batch = collate_sparse_batch(items)
            if fam == "adapted_glu":
                batch["p_norm_tensor"] = torch.tensor([[float(tr.p_norm[int(m["cache_index"]), 0]), float(tr.p_norm[int(m["cache_index"]), 1])] for m in batch["metadata"]])
            out = trainer._one_train_step(batch)
            if isinstance(out, str):
                raise SystemExit(f"train step failed: {out}")
            losses.setdefault(K, []).append(out[0])
    rep["overfit"] = {"updates": upd, "wall_s": time.perf_counter() - t0, "mean_loss_first20": float(np.mean([v for L in losses.values() for v in L[:3]])),
                      "mean_loss_last20": float(np.mean([v for L in losses.values() for v in L[-3:]])), "loss_by_K_last": {int(k): float(np.mean(v[-4:])) for k, v in losses.items()}}
    rep["checks"]["overfit_loss_decreases"] = bool(rep["overfit"]["mean_loss_last20"] < 0.7 * rep["overfit"]["mean_loss_first20"])
    # positive control on the TRAIN frames the model has seen: same frames, T true vs T shifted vs K=0
    trainer.eval_mode()
    tr.mask_mode = "evaluation"
    tr.replica_id = 1
    res = {}
    with torch.no_grad():
        for tag, K, shift in (("K32", 32, 0.0), ("K32_shifted_T", 32, 1.0), ("K0", 0, 0.0)):
            tr.eval_K = K
            items = [tr[j] for j in range(8)]
            batch = collate_sparse_batch(items)
            if shift:
                if fam == "sparse_unet":
                    batch["input_tensor"][:, 5] = batch["input_tensor"][:, 5] + shift * batch["input_tensor"][:, 6]
                else:
                    for t in batch["glu_tokens"]:
                        t["t_values_norm"] = t["t_values_norm"] + shift
            if fam == "adapted_glu":
                batch["p_norm_tensor"] = torch.tensor([[float(tr.p_norm[int(m["cache_index"]), 0]), float(tr.p_norm[int(m["cache_index"]), 1])] for m in batch["metadata"]])
            pred = trainer._forward_norm(batch, decode_chunked=True).cpu()
            y = batch["target_T_norm"]
            res[tag] = (pred, float(((pred - y) ** 2).mean()))
    rep["positive_control"] = {k: v[1] for k, v in res.items()}
    rep["positive_control"]["max_abs_pred_change_when_T_shifted"] = float((res["K32"][0] - res["K32_shifted_T"][0]).abs().max())
    rep["checks"]["observed_T_are_used_after_learning"] = bool(rep["positive_control"]["max_abs_pred_change_when_T_shifted"] > 1e-3 and res["K32"][1] < res["K0"][1])
    # ---- six-K evaluation on a VAL subset, read back from disk
    val_on.indices = val_on.indices[:: max(1, len(val_on.indices) // 12)][:12]
    out = trainer.evaluate_dataset(val_on, role="val_eval", replica_id=0, method=fam, variant="smoke", model_seed=11)
    import pandas as pd

    df = pd.DataFrame(out["rows"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv = args.out.with_suffix(".eval_rows.csv")
    df.to_csv(csv, index=False)
    back = pd.read_csv(csv)
    rep["eval_six_K"] = {"n_rows": len(back), "K_values": sorted(back["K"].unique().tolist()), "per_K_common_rmse_K": out["per_K_common_rmse_K"],
                         "macro_selection_value": out["mean_condition_rmse_K"], "finite": bool(np.isfinite(back["common_rmse_K"]).all())}
    rep["checks"]["eval_six_K_read_back"] = bool(rep["eval_six_K"]["K_values"] == list(K_VALUES) and rep["eval_six_K"]["finite"] and len(back) == 6 * len(val_on))
    rep["all_pass"] = bool(all(rep["checks"].values()))
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "eval_six_K"}, indent=2))
    return 0 if rep["all_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
