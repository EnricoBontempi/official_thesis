#!/usr/bin/env python3
"""S7 residual of S6: snapshot-swap of the FROZEN historical A38 GappyPOD (p_on/off), CPU only.

Context A38 (760-frame fit, historical cache): the fitted objects are the persisted S6 artifact, NOT
refitted and NOT replaced by the C149 fit. Same pairing (S6 U5 pairs), same observations (5%, mask replica 0,
recipient sites; donor velocity at those sites; recipient p and target) and the same donor reference as the
neural swap. The aligned rows are replayed here and compared with the persisted S6 aligned rows (reuse only if
identical). COMPLETE.json only after readback and checks.
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
from combustion_recon.data.month2_protocol import sha256_file  # noqa: E402
from combustion_recon.data.sparse_eval_masks import check_eval_mask_contract  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import evaluate_frame_prediction  # noqa: E402
from combustion_recon.models.baselines import GappyPODReconstructor, ParamConditionedMeanPredictor  # noqa: E402
from combustion_recon.models.month2_baseline_adapters import predict_gappy  # noqa: E402
from combustion_recon.training.dense_cache import verify_cache_payload_hashes  # noqa: E402
from combustion_recon.training.snapshot_swap import build_within_condition_swap_pairs  # noqa: E402

ROLES = ("seen_train_eval", "val_eval")
N_FRAMES = {"seen_train_eval": 220, "val_eval": 123}
U5_PAIRS = ROOT / "results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/swap_pairs.json"
KEYS = ["mean_vi", "mean_vj", "mean_vk", "mean_T", "modes", "modal_variances", "n_pix", "R", "ridge_lambda", "w_T",
        "scale_vi", "scale_vj", "scale_vk", "scale_T", "use_param_mean", "n_velocity_components", "global_mean_T", "eps_var"]


def _log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] s7gappyswap {m}", flush=True)


def _load_g15():
    spec = importlib.util.spec_from_file_location("s6gappy15b", ROOT / "scripts/15_gappypod_nested_coverage_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pod(z, prefix: str, pm) -> GappyPODReconstructor:
    d = {k: z[f"{prefix}_{k}"] for k in KEYS if f"{prefix}_{k}" in z.files}
    return GappyPODReconstructor(
        mean_vi=np.asarray(d["mean_vi"]), mean_vj=np.asarray(d["mean_vj"]), mean_vk=np.asarray(d["mean_vk"]),
        mean_T=np.asarray(d["mean_T"]), modes=np.asarray(d["modes"]), modal_variances=np.asarray(d["modal_variances"]),
        n_pix=int(d["n_pix"]), R=int(d["R"]), ridge_lambda=float(d["ridge_lambda"]), w_T=float(d["w_T"]),
        scale_vi=float(d["scale_vi"]), scale_vj=float(d["scale_vj"]), scale_vk=float(d["scale_vk"]), scale_T=float(d["scale_T"]),
        use_param_mean=bool(d["use_param_mean"]), n_velocity_components=int(d["n_velocity_components"]),
        param_mean_predictor=pm if bool(d["use_param_mean"]) else None,
        global_mean_T=np.asarray(d["global_mean_T"]) if "global_mean_T" in d else None, eps_var=float(d.get("eps_var", 1e-8)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--expect-artifact-sha256", required=True)
    ap.add_argument("--s6-per-frame-5pct", type=Path, required=True, help="persisted S6 aligned rows (per_frame_5pct_refit.csv.gz)")
    ap.add_argument("--cache-root", type=Path, required=True, help="historical (A38 context) dense cache")
    ap.add_argument("--cache-hash-file", type=Path, required=True)
    ap.add_argument("--expect-cache-aggregate-sha256", required=True)
    ap.add_argument("--mask-root", type=Path, required=True)
    ap.add_argument("--expect-mask-manifest-sha256", required=True)
    ap.add_argument("--expect-mask-aggregate-sha256", required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--expect-stats-sha256", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("--smoke needs an out-root named *smoke*; the scientific root must not be")
    if sha256_file(args.artifact) != args.expect_artifact_sha256:
        raise SystemExit("A38 Gappy artifact sha mismatch: the frozen object is not the recorded one")
    if sha256_file(args.stats) != args.expect_stats_sha256:
        raise SystemExit("normalizer sha mismatch")
    ver = verify_cache_payload_hashes(json.loads(args.cache_hash_file.read_text()), cache_root=args.cache_root)
    if not ver["ok"] or ver["actual_aggregate_sha256"] != args.expect_cache_aggregate_sha256:
        raise SystemExit("historical cache hash mismatch")
    check_eval_mask_contract(manifest_path=args.mask_root / "manifest.json", payload_root=args.mask_root,
                             expected_manifest_file_sha256=args.expect_mask_manifest_sha256,
                             expected_aggregate_payload_sha256=args.expect_mask_aggregate_sha256)
    g15 = _load_g15()
    z = np.load(args.artifact, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    pmm = meta["param_mean_meta"]
    pm = ParamConditionedMeanPredictor(coef=np.asarray(z["param_mean_coef"]), H=pmm["H"], W=pmm["W"], ridge_alpha=pmm["ridge_alpha"],
                                       basis=pmm["basis"], selected_basis=pmm["basis"], selected_ridge_alpha=pmm["ridge_alpha"],
                                       loco_rmse_K=pmm["loco_rmse_K"])
    pods = {False: _pod(z, "off", pm), True: _pod(z, "on", pm)}
    normalizer = Month2FitOnlyNormalizer(args.stats)
    sigma_T = float(normalizer.temperature_std)
    u5 = json.loads(U5_PAIRS.read_text())
    s6_aligned = pd.read_csv(args.s6_per_frame_5pct)
    out = args.out_root
    out.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {"context": "A38 historical frozen objects (760-frame fit); not the C149 fit",
                               "artifact_sha256": args.expect_artifact_sha256, "artifact_meta": {"git_sha": meta.get("git_sha"), "R": meta.get("R"),
                               "frozen_lambdas": meta.get("frozen_lambdas")}, "roles": {}}
    checks: list[str] = []
    for role in ROLES:
        frames = g15._load_role_physical(args.cache_root, role, normalizer)
        masks = g15._load_mask_index(args.mask_root, role)
        rows_meta = [{"role": role, "sim_id": f["sim_id"], "frame_index": f["frame"], "cache_index": f["cache_index"]} for f in frames]
        pairs = build_within_condition_swap_pairs(rows_meta, role=role)
        key = lambda ps: sorted((p["sim_id"], p["dest_frame_index"], p["src_frame_index"]) for p in ps)  # noqa: E731
        if key(pairs) != key(u5[role]) or len(pairs) != N_FRAMES[role]:
            raise SystemExit(f"{role}: pairing differs from the S6 U5 pairing")
        lookup = {p["dest_cache_index"]: p for p in pairs}
        use = frames if not args.smoke else frames[:: max(1, len(frames) // 12)][:12]
        for p_on in (False, True):
            name = "p_on" if p_on else "p_off"
            pod = pods[p_on]
            sw_rows, don_rows, al_rows = [], [], []
            for fr in use:
                src = frames[lookup[fr["cache_index"]]["src_cache_index"]]
                sites = g15._mask_sites(masks, (fr["sim_id"], int(fr["frame"]), 0), "sites_5pct")
                h5 = masks["hash_5pct"][masks["row_of"][(fr["sim_id"], int(fr["frame"]), 0)]]
                swapped = dict(fr)
                swapped["vi"], swapped["vj"] = src["vi"], src["vj"]       # donor velocity, recipient p/T/coordinates
                common = dict(train_sigma_T=sigma_T, sim_id=fr["sim_id"], split="val" if role == "val_eval" else "seen", role=role,
                              frame_index=int(fr["frame"]), K=0, site_fraction=0.05, model_seed=0, mask_replica_id=0,
                              layout_id="random_sparse_primary", task_mode="temperature_free")
                for kind, f_in, rows in (("swap", swapped, sw_rows), ("aligned", fr, al_rows)):
                    pred = predict_gappy(pod, g15._model_inputs_from_sites(f_in, sites, p_conditioning=p_on), p_conditioning=p_on)
                    row = evaluate_frame_prediction(pred, fr["T"], method="GappyPOD_free", variant="free" if kind == "aligned" else "swapped",
                                                    p_conditioning=p_on, ridge_lambda=float(pod.ridge_lambda), R=int(pod.R), **common)
                    row.pop("_spatial_bias_map", None)
                    row.update({"mask_hash": h5, "fit_pool": "A38_frozen_artifact", "artifact_sha256": args.expect_artifact_sha256,
                                "observation_mode": "snapshot_swapped_sparse" if kind == "swap" else "aligned", "eval_git_sha": args.eval_git_sha})
                    if kind == "swap":
                        row["swap_src_frame_index"] = int(src["frame"])
                    rows.append(row)
                drow = evaluate_frame_prediction(src["T"], fr["T"], method="T_donor_vs_T_dest", variant="reference_only",
                                                 p_conditioning=False, observation_mode="donor_target_reference", **common)
                drow.pop("_spatial_bias_map", None)
                don_rows.append(drow)
            mac = lambda r: float(pd.DataFrame(r).groupby("sim_id")["rmse_K"].mean().mean())  # noqa: E731
            ref = s6_aligned[(s6_aligned["role"] == role) & (s6_aligned["p_conditioning"].astype(str).str.lower() == str(p_on).lower())
                             & (s6_aligned["mask_replica_id"] == 0)]
            al_df = pd.DataFrame(al_rows)
            m = al_df[["sim_id", "frame_index", "rmse_K"]].merge(ref[["sim_id", "frame_index", "rmse_K"]], on=["sim_id", "frame_index"], suffixes=("_replay", "_s6"))
            if len(m) != len(al_df):
                checks.append(f"{role}/{name}: {len(al_df) - len(m)} aligned frames missing in the S6 rows")
            max_d = float((m["rmse_K_replay"] - m["rmse_K_s6"]).abs().max()) if len(m) else float("nan")
            if not np.isfinite(max_d) or max_d > 1e-6:
                checks.append(f"{role}/{name}: aligned replay differs from the persisted S6 rows (max {max_d})")
            summary["roles"][f"{role}/{name}"] = {"n_frames": len(sw_rows), "aligned_macro_K": mac(al_rows), "swapped_macro_K": mac(sw_rows),
                                                  "donor_reference_macro_K": mac(don_rows), "aligned_replay_vs_s6_max_abs_frame_delta_K": max_d}
            pd.DataFrame(sw_rows).to_csv(out / f"per_frame_swap_{role}_{name}.csv", index=False)
            pd.DataFrame(don_rows).to_csv(out / f"per_frame_donor_reference_{role}_{name}.csv", index=False)
            al_df.to_csv(out / f"per_frame_aligned_replay_{role}_{name}.csv", index=False)
            _log(f"{role}/{name}: aligned {mac(al_rows):.3f} swapped {mac(sw_rows):.3f} donor {mac(don_rows):.3f}")
    summary["checks_failed"] = checks
    (out / "swap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for k, v in summary["roles"].items():
        role, name = k.split("/")
        fr = pd.read_csv(out / f"per_frame_swap_{role}_{name}.csv")
        if len(fr) != v["n_frames"] or not np.isfinite(fr["rmse_K"].to_numpy()).all():
            checks.append(f"readback failed {k}")
        if not args.smoke and len(fr) != N_FRAMES[role]:
            checks.append(f"{k}: {len(fr)} frames")
    if checks:
        (out / "SWAP_FAILED.json").write_text(json.dumps({"ok": False, "checks_failed": checks}, indent=2) + "\n")
        _log(f"FAILED {checks}")
        return 3
    (out / ("SMOKE_OK.json" if args.smoke else "COMPLETE.json")).write_text(json.dumps({"ok": True, "roles": summary["roles"]}, indent=2) + "\n")
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
